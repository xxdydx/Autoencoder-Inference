"""Deferred, serial CPU mechanism/streaming experiments and blog evidence."""
import os
from pathlib import Path
import json
import subprocess
import time
import itertools

root=Path(__file__).resolve().parent;os.chdir(root)
while not Path('artifacts/REMOTE_VALIDATED').exists():time.sleep(15)
Path('artifacts/deep').mkdir(exist_ok=True);Path('reports/deep').mkdir(exist_ok=True)
python=str(root/'.venv-modern/bin/python')
env=os.environ.copy();env['CPU_AE_CPUS']='0-7'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='1'
env['TORCHINDUCTOR_COMPILE_THREADS']='1'

def run(args,key,expected=None,timeout=600):
    if expected and Path(expected).exists():return json.loads(Path(expected).read_text())
    print('START '+key,flush=True);start=time.time()
    try:
        p=subprocess.run([python]+args,env=env,capture_output=True,text=True,timeout=timeout)
        record=dict(command=args,key=key,returncode=p.returncode,elapsed_s=time.time()-start,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired as error:
        record=dict(command=args,key=key,error='timeout',elapsed_s=time.time()-start,
                    stdout=str(error.stdout),stderr=str(error.stderr))
    Path(f'artifacts/deep/log_{key}.json').write_text(json.dumps(record,indent=2))
    with Path('artifacts/deep/manifest.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in record.items() if k not in ('stdout','stderr')})+'\n')
    if 'error' in record or record.get('returncode'):
        print('FAILED '+key,flush=True);return None
    print('DONE '+key,flush=True)
    return json.loads(Path(expected).read_text()) if expected and Path(expected).exists() else None

def attention(threads,batch,length,dtype='fp32',profile=False):
    key=f'attention_{dtype}_t{threads}_b{batch}_s{length}'
    args=['deep_attention.py','--threads',str(threads),'--batch',str(batch),'--length',str(length),'--dtype',dtype]
    if profile:args+=['--profile']
    return run(args,key,f'artifacts/deep/{key}.json')

def pipeline(label,workers,threads,batch,devices=512,cycles=2,backend='sdpa',mode='reusable',pin=False):
    tag=f'{label}_pipeline_{backend}_w{workers}_t{threads}_b{batch}'+('_pinned' if pin else '')+f'_n{devices}'
    args=['pipeline_deep.py','--label','deep/'+label,'--backend',backend,'--workers',str(workers),
          '--threads',str(threads),'--batch',str(batch),'--devices',str(devices),'--cycles',str(cycles),
          '--buffer-mode',mode,'--parallel-warmup']
    if pin:args+=['--pin-workers']
    return run(args,tag,f'artifacts/deep/{tag}.json',timeout=600)

for threads in (1,2,4,8):attention(threads,1,720,profile=threads in (1,8))
for threads,batch in itertools.product((1,8),(8,)):
    attention(threads,batch,720)
for threads,length in itertools.product((1,8),(120,360,1440)):
    attention(threads,1,length)
for threads in (1,8):attention(threads,1,720,'bf16',profile=True)
for threads,batch in ((1,1),(8,1),(8,8)):
    run(['deep_model_controls.py','--threads',str(threads),'--batch',str(batch)],f'model_controls_t{threads}_b{batch}',
        f'artifacts/deep/model_controls_t{threads}_b{batch}.json')
run(['deep_attention.py','--threads','8','--mkl-diagnostic'],'mkl_diagnostic')
run(['deep_memory.py'],'memory_ring','artifacts/deep/memory_ring.json')
for mask in ('0-7','0-31'):
    run(['deep_quota.py','--mask',mask,'--workers','32','--requests','32'],'quota_m'+mask,
        f'artifacts/deep/quota_w32_m{mask.replace("-","to")}.json')

grid=[]
for workers,batch in itertools.product((1,2,4,8,16),(1,2,4,8,16,32)):
    r=pipeline('grid_fp32',workers,max(1,8//workers),batch)
    if r:grid.append(r)
for workers,batch in itertools.product((1,4,8),(1,8,32)):
    pipeline('grid_bf16_score32',workers,max(1,8//workers),batch,devices=256,backend='bf16_score32')
for mode in ('reusable','fused_slices','mirrored','folded_mirrored'):
    pipeline('memory_'+mode,8,1,1,cycles=3,mode=mode,pin=True)

best=max(grid,key=lambda r:r['windows_per_second'])['config']
Path('artifacts/deep/grid_selected.json').write_text(json.dumps(best,indent=2))
cases=[('strong_stock',8,1,1,'stock','reusable',True),
       ('strong_sdpa',8,1,1,'sdpa','reusable',True),
       ('joint_grid',best['workers'],best['threads'],best['batch'],'sdpa','reusable',False),
       ('folded_mirrored',8,1,1,'sdpa','folded_mirrored',True)]
for round_index in range(3):
    for label,workers,threads,batch,backend,mode,pin in (cases if round_index%2==0 else list(reversed(cases))):
        pipeline(label+f'_r{round_index}',workers,threads,batch,devices=3000,cycles=1,backend=backend,mode=mode,pin=pin)
Path('artifacts/DEEP_COMPLETE').write_text('CPU mechanism, joint grid, circular buffers, quota traces and strong baselines complete\n')
print('DEEP_COMPLETE',flush=True)
