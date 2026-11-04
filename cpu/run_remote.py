"""All performance work executes sequentially on the remote CPU.

Resume by keeping the remote artifacts. Every successful microbenchmark and
pipeline is addressed by runtime/configuration; no Mac outputs are input data.
"""
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ['CPU_AE_CPUS']='0-7'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
ENVS={'base':str(ROOT/'.venv-base/bin/python'),'modern':str(ROOT/'.venv-modern/bin/python')}
MANIFEST=Path('artifacts/remote_manifest.jsonl')

def run(label,args,key,expected=None,timeout=600):
    if expected and Path(expected).exists():
        value=json.loads(Path(expected).read_text())
        if not isinstance(value,dict) or 'error' not in value:
            return value
    start=time.time()
    print('START '+key,flush=True)
    try:
        completed=subprocess.run([ENVS[label]]+args,capture_output=True,text=True,timeout=timeout)
        result=dict(key=key,runtime=label,command=args,returncode=completed.returncode,
                    elapsed_s=time.time()-start,stdout=completed.stdout,stderr=completed.stderr)
    except subprocess.TimeoutExpired as error:
        result=dict(key=key,runtime=label,command=args,error=f'timeout after {timeout}s',elapsed_s=time.time()-start,
                    stdout=(error.stdout or b'').decode() if isinstance(error.stdout,bytes) else error.stdout,
                    stderr=(error.stderr or b'').decode() if isinstance(error.stderr,bytes) else error.stderr)
    Path(f'artifacts/log_{key}.json').write_text(json.dumps(result,indent=2))
    with MANIFEST.open('a') as f:f.write(json.dumps({k:v for k,v in result.items() if k not in ('stdout','stderr')})+'\n')
    if 'error' in result or result.get('returncode'):
        if expected:Path(expected).write_text(json.dumps(dict(error=result.get('error') or result['stderr'],command=args),indent=2))
        print('FAILED '+key,flush=True)
        return None
    print('DONE '+key,flush=True)
    if expected and Path(expected).exists():return json.loads(Path(expected).read_text())

def micro(label,backend,threads,batch,length=720,profile=False):
    key=f'{label}_{backend}_t{threads}_b{batch}_s{length}'
    args=['-m','cpu.benchmark','--backend',backend,'--threads',str(threads),'--batch',str(batch),
          '--length',str(length),'--label',label,'--repeats','20','--min-seconds','1']
    if profile:args+=['--profile']
    return run(label,args,key,f'artifacts/{key}.json',timeout=240)

def pipeline(label,backend,workers,threads,batch,devices=512,cycles=3,pin=False,suffix=''):
    lab=label+suffix
    key=f'{lab}_pipeline_{backend}_w{workers}_t{threads}_b{batch}'+('_pinned' if pin else '')+f'_n{devices}'
    args=['-m','cpu.pipeline','--backend',backend,'--workers',str(workers),'--threads',str(threads),
          '--batch',str(batch),'--devices',str(devices),'--cycles',str(cycles),'--label',lab]
    if pin:args+=['--pin-workers']
    return run(label,args,key,f'artifacts/{key}.json',timeout=900)

if __name__=='__main__':
    Path('artifacts').mkdir(exist_ok=True);Path('reports').mkdir(exist_ok=True)
    for label in ENVS:run(label,['-m','autoencoder.environment',label],label+'_environment',f'artifacts/{label}_environment.json')
    run('base',['-m','autoencoder.train'],'training','artifacts/training.json')
    if not Path('artifacts/model.pt').exists():raise RuntimeError('Training failed')
    run('base',['-m','autoencoder.export'],'export','artifacts/export.json')
    if not Path('artifacts/model.onnx').exists():raise RuntimeError('Export failed')
    # Run comparable versions next to each other, in alternating order.
    cases=list(itertools.product(('manual','sdpa','ort'),(1,2,4,8,16,32),(1,8,32)))
    random.Random(29).shuffle(cases)
    for i,(backend,threads,batch) in enumerate(cases):
        for label in (('base','modern') if i%2==0 else ('modern','base')):
            micro(label,backend,threads,batch,profile=(threads==1 and batch==1))
    for label in ENVS:
        for backend in ('manual','sdpa','ort'):
            for batch in (1,8,32):micro(label,backend,1,batch,length=120)
        for backend in ('compile','int8','ort_int8','fp16','bf16','tanh','jit','contiguous'):
            for threads in (1,8):
                for batch in (1,8):micro(label,backend,threads,batch,profile=(threads==1 and batch==1))
        for threads in (1,8):
            run(label,['-m','cpu.paired','--threads',str(threads),'--label',label],f'{label}_paired_t{threads}',
                f'artifacts/{label}_paired_t{threads}.json')
    run('base',['-m','cpu.preprocessing'],'preprocessing','artifacts/preprocessing.json')
    run('base',['-m','autoencoder.validate'],'validation','artifacts/validation.json')
    # Cross-version checks use only reference tensors produced on this CPU.
    run('modern',['-m','autoencoder.cross_version'],'cross_version','artifacts/cross_version_validation.json')
    layouts=[(1,1,1),(1,8,1),(2,4,1),(4,2,1),(8,1,1),(8,1,8),
             (1,8,8),(2,4,8),(4,2,8),(32,1,1),(16,2,1),(8,4,1),(4,8,1)]
    for label in ENVS:
        for workers,threads,batch in layouts:pipeline(label,'sdpa',workers,threads,batch)
        # Top non-oversubscribed microbenchmark shape for every runtime/backend.
        for backend in ('ort','compile','int8','ort_int8','fp16','bf16','jit','contiguous'):
            candidates=[json.loads(p.read_text()) for p in Path('artifacts').glob(f'{label}_{backend}_t*_b*_s720.json')]
            candidates=[r for r in candidates if 'error' not in r and r['threads']<=8]
            if not candidates:continue
            best=max(candidates,key=lambda r:r['windows_per_second'])
            pipeline(label,backend,max(1,8//best['threads']),best['threads'],best['batch'])
        for workers,threads in ((8,1),(4,2),(2,4),(1,8)):
            pipeline(label,'sdpa',workers,threads,1,pin=True)
    probes=[json.loads(p.read_text()) for p in Path('artifacts').glob('*pipeline*_n512.json')]
    probes=[r for r in probes if 'windows_per_second' in r]
    exact=[r for r in probes if r['config']['backend'] in ('sdpa','ort','compile','jit','contiguous')]
    approximate=[r for r in probes if r['config']['backend'] in ('bf16','fp16','int8','ort_int8')]
    exact_best=max(exact,key=lambda r:r['windows_per_second'])
    approx_best=max(approximate,key=lambda r:r['windows_per_second']) if approximate else None
    selected=dict(exact=exact_best,approximate=approx_best)
    Path('artifacts/selected_remote.json').write_text(json.dumps(selected,indent=2))
    # Matched baseline in the winner's runtime/worker layout, plus original-runtime
    # baseline. Alternate full bursts to reduce separate-time load bias.
    c=exact_best['config']
    full=[(c['label'],'manual',c),(c['label'],c['backend'],c)]
    if c['label']!='base':full.append(('base','manual',c))
    if approx_best:
        d=approx_best['config'];full.append((d['label'],d['backend'],d))
    for round_index in range(5):
        order=full[:] if round_index%2==0 else list(reversed(full))
        for label,backend,config in order:
            pipeline(label,backend,config['workers'],config['threads'],config['batch'],
                     devices=3000,cycles=2,pin=config['pin_workers'],suffix=f'_r{round_index}')
    Path('artifacts/REMOTE_COMPLETE').write_text('All requested remote experiment phases finished\n')
    print('REMOTE_COMPLETE',flush=True)
