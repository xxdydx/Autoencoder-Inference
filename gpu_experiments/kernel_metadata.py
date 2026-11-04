"""Resource metadata for the selected custom kernels; not a timing run."""
from pathlib import Path
import os,time,json,re,subprocess
A=Path('artifacts/gpu_fusion')
assert (A/'PAIRED_COMPLETE').exists()
pause=Path('keepalive/measurement.pause');pause.write_text(str(os.getpid()))
try:
    while subprocess.run(['pgrep','-f','^python3 .*/burn[.]py'],capture_output=True).returncode==0:time.sleep(1)
    import torch,triton
    from model import load_model
    from fused_ffn import _full_ffn,_gemm_bias_gelu
    torch.set_num_threads(2)
    with torch.inference_mode():
        m,c=load_model();b=m.cuda().bfloat16().encoder[0];x=torch.zeros(64*720,128,device='cuda',dtype=torch.bfloat16)
        w1=b.ff1.weight.T.contiguous();w2=b.ff2.weight.T.contiguous();metadata={}
        for name,kernel in [('unit',_gemm_bias_gelu),('unit_full',_full_ffn)]:
            desc=json.loads((A/(name+'.json')).read_text())['best_config']
            cfg={k:int(v) for k,v in re.findall(r'(BM|BN|BK|GROUP|STEP|num_warps|num_stages): ([0-9]+)',desc)}
            warps=cfg.pop('num_warps');stages=cfg.pop('num_stages')
            if name=='unit':
                out=torch.empty(64*720,512,device='cuda',dtype=torch.bfloat16)
                compiled=kernel.fn[(triton.cdiv(len(x),cfg['BM'])*triton.cdiv(512,cfg['BN']),)](x,w1,b.ff1.bias,out,M=len(x),N=512,K=128,**cfg,num_warps=warps,num_stages=stages)
            else:
                out=torch.empty_like(x)
                compiled=kernel.fn[(triton.cdiv(len(x),cfg['BM']),)](x,w1,b.ff1.bias,w2,b.ff2.bias,out,M=len(x),**cfg,num_warps=warps,num_stages=stages)
            torch.cuda.synchronize()
            metadata[name]=dict(config=desc,registers_per_thread=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),shared_bytes=getattr(compiled.metadata,'shared',None),global_scratch_bytes=getattr(compiled.metadata,'global_scratch_size',None))
        (A/'kernel_metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))
finally:pause.unlink(missing_ok=True)
