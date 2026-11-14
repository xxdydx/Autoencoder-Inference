"""Explicit aten attention controls for the pinned PyTorch experiment version.

These internal operators are used to verify actual backend selection when the
compiled high-level SDPA graph reuses a FlashAttention cache entry.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['all','flash','cudnn','efficient'],default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',type=int,default=3000)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    import os
    if a.backend=='all':
        import fcntl,random,subprocess,sys,time
        from gpu.optimization_study import save
        a.output.mkdir(parents=True,exist_ok=True)
        lock=open('/tmp/autoencoder_optimizations.lock','w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        flags=[Path('keepalive/measurement.pause'),Path('/tmp/flowmesh_c_running')]
        for flag in flags:
            if flag.exists():raise RuntimeError(f'Existing measurement flag: {flag}')
        for flag in flags:
            flag.parent.mkdir(parents=True,exist_ok=True);flag.write_text(str(os.getpid()))
        jobs=[]
        try:
            order=['auto','flash','cudnn','efficient'];random.Random(20261105).shuffle(order)
            for backend in order:
                directory=a.output/backend
                if backend=='auto':
                    cmd=[sys.executable,'-u','-m','gpu.optimization_study','cell','--output',str(directory),
                         '--devices',str(a.devices),'--cycles',str(a.cycles),'--warmup-cycles','5']
                else:
                    cmd=[sys.executable,'-u','-m','gpu.attention_controls','--backend',backend,'--output',str(directory),
                         '--devices',str(a.devices),'--cycles',str(a.cycles)]
                if a.profile:cmd.append('--profile')
                row=dict(backend=backend,command=cmd,started_epoch=time.time());jobs.append(row)
                save(a.output/'status.json',dict(complete=False,jobs=jobs))
                print('START explicit attention',backend,flush=True)
                with (a.output/f'{backend}.log').open('w') as log:
                    try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900);row['returncode']=r.returncode
                    except subprocess.TimeoutExpired:row['returncode']=124
                row['elapsed_s']=time.time()-row['started_epoch']
                save(a.output/'status.json',dict(complete=False,jobs=jobs))
                print('DONE explicit attention',backend,row['returncode'],flush=True)
            save(a.output/'status.json',dict(complete=True,jobs=jobs))
        finally:
            for flag in flags:
                if flag.exists() and flag.read_text().strip()==str(os.getpid()):flag.unlink()
        return
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
    import torch
    from gpu import fused_ffn
    from gpu.optimization_study import cell,save

    class ExplicitBlock(fused_ffn.FusedBlock):
        def forward(self,x):
            block=self.original;b,t,_=x.shape
            q,k,v=block.qkv(block.norm1(x)).reshape(b,t,3,4,32).permute(2,0,3,1,4).unbind(0)
            if a.backend=='cudnn':
                attention=torch.ops.aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False,0.,False,False)[0]
            elif a.backend=='efficient':
                attention=torch.ops.aten._scaled_dot_product_efficient_attention.default(q,k,v,None,False,0.,False)[0]
            else:
                attention=torch.ops.aten._scaled_dot_product_flash_attention.default(q,k,v,0.,False,False)[0]
            x=x+block.out(attention.transpose(1,2).reshape(b,t,128))
            return x+fused_ffn.full_ffn(block.norm2(x),self.packed_ff1,block.ff1.bias,self.packed_ff2,block.ff2.bias)

    def replace(model,full=True):
        for stack in [model.encoder,model.decoder]:
            for index,block in enumerate(stack):stack[index]=ExplicitBlock(block,full=True)
        return model
    fused_ffn.replace_ffn=replace
    a.phase='cell';a.variant='baseline';a.batch=64;a.config='64,128,8'
    a.warmup_cycles=5;a.capture=False
    cell(a)
    path=a.output/'result.json';r=json.loads(path.read_text())
    r['explicit_attention_operator']=a.backend
    import hashlib
    r['experiment']['source_sha256']['gpu/attention_controls.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(path,r)


if __name__=='__main__':main()
