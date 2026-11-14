"""Numerical checks and sequential measurement cells for profiler-led experiments."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def stats(values):
    import numpy as np
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x)>1 else 0.,
                median=float(np.median(x)), p99=float(np.percentile(x, 99)))


def backend_context(name):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return contextlib.nullcontext() if name == 'auto' else sdpa_kernel({
        'flash': SDPBackend.FLASH_ATTENTION, 'cudnn': SDPBackend.CUDNN_ATTENTION,
        'efficient': SDPBackend.EFFICIENT_ATTENTION}[name])


def make_cpu_reference(output):
    import numpy as np
    import torch
    from autoencoder.model import load_model, Scorer
    torch.set_num_threads(2)
    model, checkpoint = load_model()
    scorer = Scorer(model).eval()
    ring = np.random.default_rng(567).standard_normal((3000, 720, 32), dtype=np.float32)
    scale = (1/checkpoint['std'].numpy()).astype(np.float32)
    offset = (-checkpoint['mean'].numpy()*scale).astype(np.float32)
    scores = []
    begin = time.perf_counter()
    with torch.inference_mode():
        for left in range(0, 3000, 8):
            raw = ring[left:left+8]
            data = np.concatenate((raw[:, 137:], raw[:, :137]), axis=1)
            np.multiply(data, scale, out=data)
            np.add(data, offset, out=data)
            scores.append(scorer(torch.from_numpy(data)).numpy())
            if left % 400 == 0:
                print('CPU_REFERENCE', left, flush=True)
    values = np.concatenate(scores)
    assert values.shape == (3000,) and np.isfinite(values).all()
    target = Path('artifacts/cpu_scores_head137.npy')
    np.save(target, values)
    save(output/'cpu_reference.json', dict(source='FP32 CPU model; regenerated independently',
         windows=3000, head=137, numpy_seed=567, batch=8, threads=2,
         elapsed_s=time.perf_counter()-begin, torch=torch.__version__,
         checkpoint_sha256=hashlib.sha256(Path('artifacts/model.pt').read_bytes()).hexdigest(),
         scores_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


def error_metrics(actual, expected):
    diff = actual.float()-expected.float()
    return dict(max_abs=float(diff.abs().max()),
                normalized_rmse=float(diff.square().mean().sqrt()/expected.float().square().mean().sqrt()))


def measure(fn, samples=100, warmup_s=.08):
    import torch
    count = 0
    begin = time.perf_counter()
    while count < 10 or time.perf_counter()-begin < warmup_s:
        fn()
        count += 1
        if count % 16 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    warm = time.perf_counter()-begin
    values=[]
    for _ in range(samples):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record();fn();end.record();end.synchronize()
        values.append(start.elapsed_time(end))
    return dict(cuda_ms=stats(values), cuda_samples_ms=values, warmup_calls=count, warmup_s=warm)


def micro(output):
    import gc
    import numpy as np
    import torch
    import triton
    from torch.nn import functional as F
    from autoencoder.model import load_model
    from gpu.fused_ffn import _full_ffn
    from gpu.optimization_kernels import _norm_ffn
    torch.set_num_threads(2)
    torch.manual_seed(123)
    model,_ = load_model()
    model=model.cuda().bfloat16()
    block=model.encoder[0]
    original=torch.from_numpy(np.load('artifacts/validation_reference_720.npz')['input'])
    w1=block.ff1.weight.T.contiguous();w2=block.ff2.weight.T.contiguous()
    configs=[(16,128,4)] + [(bm,step,warps) for bm in [32,64,128]
                          for step in [32,64,128,256] for warps in [4,8]]
    records=[]
    def record(row):
        records.append(row);save(output/'micro.json',records)
        print('MICRO',row['batch'],row['variant'],row.get('config',row.get('backend')),
              row.get('cuda_ms',{}).get('median',row.get('error')),flush=True)
    with torch.inference_mode():
        for batch in [64,32,48,80,96]:
            data=original.repeat(((batch+len(original)-1)//len(original),1,1))[:batch].cuda().bfloat16()
            x=model.input(data)+model.pe
            q,k,v=block.qkv(block.norm1(x)).reshape(batch,720,3,4,32).permute(2,0,3,1,4).unbind(0)
            with backend_context('flash'):
                attention_ref=F.scaled_dot_product_attention(q,k,v)
            backends=['auto','flash','cudnn','efficient'];random.Random(410+batch).shuffle(backends)
            for backend in backends:
                row=dict(batch=batch,variant='attention',backend=backend,q_stride=list(q.stride()))
                try:
                    with backend_context(backend):
                        fn=lambda:F.scaled_dot_product_attention(q,k,v)
                        row['numerics']=error_metrics(fn(),attention_ref)
                        assert row['numerics']['normalized_rmse']<.01,row
                        row.update(measure(fn))
                except Exception as exc:
                    row.update(error=repr(exc),traceback=traceback.format_exc())
                record(row)
            raw=x+block.out(attention_ref.transpose(1,2).reshape(batch,720,128))
            x=block.norm2(raw);out=torch.empty_like(x)
            reference=block.ff2(F.gelu(block.ff1(x),approximate='none'))
            order=configs.copy();random.Random(611+batch).shuffle(order)
            for bm,step,warps in order:
                row=dict(batch=batch,variant='fixed',config=[bm,step,warps])
                try:
                    def fn():
                        return _full_ffn.fn[(triton.cdiv(batch*720,bm),)](
                            x,w1,block.ff1.bias,w2,block.ff2.bias,out,M=batch*720,
                            BM=bm,STEP=step,num_warps=warps,num_stages=1)
                    kernel=fn()
                    row['numerics']=error_metrics(out,reference)
                    assert row['numerics']['normalized_rmse']<.01,row
                    row.update(measure(fn))
                    row['resources']=dict(registers_per_thread=kernel.n_regs,spills=kernel.n_spills,
                        shared_bytes=kernel.metadata.shared,grid_blocks=triton.cdiv(batch*720,bm))
                except Exception as exc:
                    row.update(error=repr(exc),traceback=traceback.format_exc())
                record(row)
            norm_configs=[(32,64,4),(32,128,4),(64,64,4),(64,64,8),(64,128,4),(64,128,8),(128,64,8)]
            random.Random(912+batch).shuffle(norm_configs)
            for variant in ['norm','norm_residual']:
                residual=variant=='norm_residual'
                expected=raw+reference if residual else reference
                for bm,step,warps in norm_configs:
                    row=dict(batch=batch,variant=variant,config=[bm,step,warps])
                    try:
                        def fn():
                            return _norm_ffn[(triton.cdiv(batch*720,bm),)](raw,block.norm2.weight,block.norm2.bias,
                                w1,block.ff1.bias,w2,block.ff2.bias,out,M=batch*720,EPS=block.norm2.eps,
                                BM=bm,STEP=step,RESIDUAL=residual,num_warps=warps,num_stages=1,enable_fp_fusion=False)
                        kernel=fn();row['numerics']=error_metrics(out,expected)
                        assert row['numerics']['normalized_rmse']<.01,row
                        row.update(measure(fn))
                        row['resources']=dict(registers_per_thread=kernel.n_regs,spills=kernel.n_spills,
                            shared_bytes=kernel.metadata.shared,grid_blocks=triton.cdiv(batch*720,bm))
                    except Exception as exc:
                        row.update(error=repr(exc),traceback=traceback.format_exc())
                    record(row)
            del raw,x,data,out,reference,q,k,v,attention_ref
            gc.collect();torch.cuda.empty_cache()


def cell(a):
    import gc
    import numpy as np
    import torch
    from autoencoder.model import load_model
    from gpu import gpu_suite as g
    from gpu.fused_ffn import replace_ffn
    from gpu.optimization_kernels import replace_experimental
    model,checkpoint=load_model()
    model=model.cuda().bfloat16()
    config=tuple(map(int,a.config.split(',')))
    model=replace_ffn(model,full=True) if a.variant=='baseline' else replace_experimental(model,a.variant,config)
    score=g.Score(model,torch.bfloat16).eval()
    g.A=a.output;g.A.mkdir(parents=True,exist_ok=True)
    class Compiled(torch.nn.Module):
        def __init__(self,s):
            super().__init__();self.s=torch.compile(s,mode='max-autotune',fullgraph=True,dynamic=False)
        def forward(self,x):
            torch.compiler.cudagraph_mark_step_begin()
            return self.s(x).clone()
    with torch.inference_mode(),backend_context(a.backend):
        # Independent saved CPU references at both supported sequence lengths.
        validation=g.validation(score,'experiment_bf16')
        assert all(r['max_relative']<=.005 for r in validation),validation
        save(a.output/'validation.json',validation)
        compiled=Compiled(score)
        g.scorer=lambda _:(compiled,checkpoint)
        kind=f'{a.variant}_{a.backend}_bf16'
        result=g.pipeline(kind,a.batch,'gpu_normalize',a.devices,a.cycles,
                          warmup_cycles=a.warmup_cycles,profile=a.profile,
                          capture_range=a.capture,annotate=a.capture)
        if a.devices==3000:
            assert result['cpu_score_agreement']['max_relative']<=.005,result['cpu_score_agreement']
        result['experiment']=dict(variant=a.variant,backend=a.backend,config=list(config),
            warmup_cycles=a.warmup_cycles,torch=torch.__version__,
            source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                           ['gpu/optimization_kernels.py','gpu/optimization_study.py','gpu/fused_ffn.py','gpu/gpu_suite.py']})
        result['summary_ms']=stats([c['wall_s']*1000 for c in result['cycles']])
        save(a.output/'result.json',result)
        print('RESULT',json.dumps(result['summary_ms']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['reference','micro','cell'])
    p.add_argument('--output',type=Path,default=Path('artifacts/optimizations'))
    p.add_argument('--variant',choices=['baseline','fixed','norm','norm_residual'],default='baseline')
    p.add_argument('--backend',choices=['auto','flash','cudnn','efficient'],default='auto')
    p.add_argument('--config',default='64,128,8')
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--devices',type=int,default=512)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--warmup-cycles',type=int,default=3)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--capture',action='store_true')
    a=p.parse_args()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[key]='2'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING']='0'
    a.output.mkdir(parents=True,exist_ok=True)
    if a.phase=='reference':make_cpu_reference(a.output)
    elif a.phase=='micro':micro(a.output)
    else:cell(a)


if __name__=='__main__':main()
