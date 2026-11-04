"""Fresh-process TensorRT or PyTorch control, identical host-input pipeline."""
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING']='0'
import argparse
p=argparse.ArgumentParser()
p.add_argument('--variant',default='trt_graph',choices=['trt_graph','trt','auto','fused_full_auto','ort_graph','ort'])
p.add_argument('--dtype',default='bf16',choices=['bf16','fp16','fp32'])
p.add_argument('--batch',type=int,default=64)
p.add_argument('--devices',type=int,default=512)
p.add_argument('--cycles',type=int,default=3)
p.add_argument('--label',default='probe')
p.add_argument('--build-only',action='store_true')
p.add_argument('--profile',action='store_true')
p.add_argument('--artifact-root',default='artifacts/tensorrt')
p.add_argument('--ort-sdpa',type=int,default=0)
a=p.parse_args()
from pathlib import Path
import time,json,gc,subprocess,traceback
import numpy as np
import torch
import gpu_suite as g
from model import load_model
from tensorrt_backend import build_engine,EngineScore

A=Path(a.artifact_root)/a.label;A.mkdir(parents=True,exist_ok=True);g.A=A
kind=a.variant+'_'+a.dtype
build_metadata=None
dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[a.dtype]
# TensorRT's legacy default stream can insert device synchronization. Use the
# same explicit CUDA stream for TensorRT and every PyTorch comparison.
execution_stream=torch.cuda.Stream()
torch.cuda.set_stream(execution_stream)

class Compiled(torch.nn.Module):
    def __init__(self,s):
        super().__init__();self.s=torch.compile(s,mode='max-autotune',dynamic=False,fullgraph=True)
    def forward(self,x):
        torch.compiler.cudagraph_mark_step_begin();return self.s(x).clone()

def make(_):
    global build_metadata
    m,c=load_model()
    if a.variant.startswith('ort'):
        from onnxruntime_backend import OnnxScore
        from tensorrt_backend import export_onnx
        path=Path('artifacts/onnxruntime/exports')/(a.dtype+'.onnx')
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():export_onnx(m,path,a.dtype)
        s=OnnxScore(path,graph=a.variant=='ort_graph',profile_dir=A if a.profile else None,
                   sdpa_kernel=a.ort_sdpa)
    elif a.variant.startswith('trt'):
        path,build_metadata=build_engine(m,'artifacts/tensorrt/engines',a.dtype,a.batch)
        s=EngineScore(path,graph=a.variant=='trt_graph')
    else:
        m=m.cuda().to(dtype)
        if a.variant=='fused_full_auto':
            assert a.dtype=='bf16'
            from fused_ffn import replace_ffn
            m=replace_ffn(m,full=True)
        s=Compiled(g.Score(m,dtype).eval())
    if not a.build_only:
        sizes=sorted({a.batch,a.devices%a.batch}-{0})
        for size in sizes:
            z=torch.zeros(size,720,32,device='cuda')
            for _ in range(3):s(z)
        torch.cuda.synchronize()
        # Common warmed GPU conditioning, outside all latency timers. Keep cold
        # compilation and capture costs in setup/build fields, not inference.
        start=time.perf_counter()
        while time.perf_counter()-start<.35:
            s(z)
            torch.cuda.synchronize()
    return s,c

def validation(s):
    d=np.load('artifacts/validation_reference_720.npz')
    x=torch.from_numpy(d['input']).cuda()
    m,_=load_model();reference_bf=g.Score(m,dtype).eval()
    actual=torch.cat([s(part).clone() for part in x.split(min(8,a.batch))]).cpu().numpy()
    eager=torch.cat([reference_bf(part) for part in x.split(8)]).cpu().numpy()
    cpu=d['scores'];threshold=float(d['threshold'])
    records={}
    for label,reference in [('cpu_fp32',cpu),('same_dtype_pytorch',eager)]:
        diff=np.abs(actual-reference)
        records[label]=dict(max_abs=float(diff.max()),max_relative=float((diff/np.abs(reference)).max()),
                            alert_agreement=float(np.mean((actual>threshold)==(reference>threshold))),
                            threshold=threshold,alert_count=int((actual>threshold).sum()))
    assert np.isfinite(actual).all()
    if a.dtype=='fp32':assert np.allclose(actual,cpu,rtol=1e-4,atol=1e-5),records
    else:assert records['cpu_fp32']['max_relative']<=.005,records
    return dict(length=720,windows=len(actual),comparisons=records,
                reduced_precision_max_relative_score_budget=.005,fp32_rtol=1e-4,fp32_atol=1e-5)

g.scorer=make
g.PAUSE.write_text(str(os.getpid()))
try:
    # Match the already-established keepalive exclusion protocol.
    while subprocess.run(['pgrep','-f','python3.*burn.py'],capture_output=True).returncode==0:time.sleep(1)
    with torch.inference_mode():
        s,c=make(kind)
        if a.build_only:
            g.save('build',build_metadata);print(json.dumps(build_metadata),flush=True)
        else:
            g.save('validation',validation(s))
            if hasattr(s,'inspect'):
                s.inspect(A/'engine_inspector.json',a.batch)
                # Separate resident-input timing, no preprocessing or transfer.
                data=torch.from_numpy(np.load('artifacts/validation_reference_720.npz')['input']).cuda()
                data=data.repeat((max(1,(a.batch+63)//64),1,1))[:a.batch]
                for _ in range(5):s(data)
                torch.cuda.synchronize();wall=[];device=[]
                for _ in range(40):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    t=time.perf_counter();start.record();s(data);end.record();end.synchronize()
                    wall.append(1000*(time.perf_counter()-t));device.append(start.elapsed_time(end))
                g.save('resident_micro',dict(host_ms=g.distribution(wall),cuda_ms=g.distribution(device),
                                             host_samples_ms=wall,cuda_samples_ms=device,batch=a.batch))
            del s;gc.collect();torch.cuda.empty_cache()
            record=g.pipeline(kind,a.batch,'gpu_normalize',a.devices,a.cycles,profile=a.profile)
            if a.devices==3000 and a.dtype!='fp32':
                assert record['cpu_score_agreement']['max_relative']<=.005,record['cpu_score_agreement']
            record['tensorrt_build']=build_metadata
            g.save('result',record)
finally:
    g.PAUSE.unlink(missing_ok=True)
