"""One fresh-process fusion/autotuning measurement cell."""
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING']='0'
import argparse
p=argparse.ArgumentParser();p.add_argument('--variant',default='fused');p.add_argument('--batch',type=int,default=64)
p.add_argument('--devices',type=int,default=512);p.add_argument('--cycles',type=int,default=3)
p.add_argument('--profile',action='store_true');p.add_argument('--unit',action='store_true')
p.add_argument('--label',default='')
a=p.parse_args()
import time,json,gc,traceback
from pathlib import Path
import numpy as np
import torch
from gpu import gpu_suite as g
from autoencoder.model import load_model
from autoencoder.deep_models import stock_transformer_copy
from gpu.fused_ffn import replace_ffn,linear_gelu,_gemm_bias_gelu,full_ffn,_full_ffn

A=Path('artifacts/gpu_fusion')/a.label;A.mkdir(parents=True,exist_ok=True);g.A=A
kind=a.variant+'_bf16'

class TunedScore(torch.nn.Module):
    def __init__(self,s,mode):
        super().__init__();self.compiled=torch.compile(s,mode=mode,dynamic=False,fullgraph=True)
    def forward(self,x):
        torch.compiler.cudagraph_mark_step_begin()
        return self.compiled(x).clone()

def make(kind):
    model,c=load_model()
    if a.variant=='stock_auto':model=stock_transformer_copy(model)
    model=model.cuda().bfloat16()
    if a.variant.startswith('fused'):model=replace_ffn(model,full='full' in a.variant)
    s=g.Score(model,torch.bfloat16).eval()
    if a.variant!='fused_eager':
        s=TunedScore(s,'max-autotune' if a.variant=='auto' or a.variant.endswith('_auto') else 'reduce-overhead')
    return s,c
g.scorer=make

def unit():
    from torch.nn import functional as F
    m,c=load_model();m=m.cuda().bfloat16();block=m.encoder[0]
    x=torch.from_numpy(np.load('artifacts/validation_reference_720.npz')['input']).cuda().bfloat16()
    x=m.input(x)+m.pe
    q,k,v=block.qkv(block.norm1(x)).reshape(64,720,3,4,32).permute(2,0,3,1,4).unbind(0)
    x=x+block.out(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(64,720,128))
    x=block.norm2(x);packed=block.ff1.weight.T.contiguous()
    is_full='full' in a.variant
    packed2=block.ff2.weight.T.contiguous()
    reference=block.ff2(F.gelu(block.ff1(x),approximate='none')) if is_full else F.gelu(block.ff1(x),approximate='none')
    fn=lambda:full_ffn(x,packed,block.ff1.bias,packed2,block.ff2.bias) if is_full else linear_gelu(x,packed,block.ff1.bias)
    begin=time.perf_counter();actual=fn();torch.cuda.synchronize()
    diff=(actual.float()-reference.float()).abs()
    result=dict(tune_s=time.perf_counter()-begin,max_abs=diff.max().item(),rmse=diff.square().mean().sqrt().item(),
                normalized_rmse=(diff.square().mean().sqrt()/reference.float().square().mean().sqrt()).item(),
                best_config=str((_full_ffn if is_full else _gemm_bias_gelu).best_config),shape=list(x.shape))
    assert result['normalized_rmse']<.01,result
    eager=lambda:block.ff2(F.gelu(block.ff1(x),approximate='none')) if is_full else F.gelu(block.ff1(x),approximate='none')
    for name,fn in [('torch_linear_gelu',eager),('fused_linear_gelu',fn)]:
        for _ in range(5):fn()
        torch.cuda.synchronize();times=[]
        for _ in range(100):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();fn();end.record();end.synchronize();times.append(start.elapsed_time(end))
        result[name]=dict(cuda_ms=g.distribution(times),samples_ms=times)
        result[name]['operators']=g.capture_profile(fn,('unit_full_' if is_full else 'unit_')+name)
    g.save('unit_full' if is_full else 'unit',result);print(json.dumps({k:v for k,v in result.items() if k not in ('torch_linear_gelu','fused_linear_gelu')}),flush=True)

g.PAUSE.write_text(str(os.getpid()))
try:
    while __import__('subprocess').run(['pgrep','-f','python3.*burn.py'],capture_output=True).returncode==0:time.sleep(1)
    with torch.inference_mode():
        if a.unit:unit()
        else:
            # Verify held-out model scores before running any performance cell.
            s,c=make(kind);validation=g.validation(s,kind)
            g.save(f'validation_{kind}_b{a.batch}',validation);del s;gc.collect()
            record=g.pipeline(kind,a.batch,'gpu_normalize',a.devices,a.cycles,profile=a.profile)
            g.save(f'result_{kind}_b{a.batch}_n{a.devices}',record)
finally:g.PAUSE.unlink(missing_ok=True)
