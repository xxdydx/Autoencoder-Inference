"""Run profiler-led optimization experiments sequentially, with no keepalive."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from gpu.optimization_study import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/optimizations'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/autoencoder_optimizations.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Both known keepalive conventions are paused for the entire study, including
    # import/compilation gaps between its sequential child jobs.
    flags=[Path('keepalive/measurement.pause'),Path('/tmp/flowmesh_c_running')]
    for flag in flags:
        if flag.exists():
            owner=flag.read_text().strip()
            if not owner.isdigit():raise RuntimeError(f'Existing unrecognized measurement flag: {flag}')
            try:os.kill(int(owner),0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Another measurement owner is alive: {owner}')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('GPU already has compute jobs: '+apps)
    for flag in flags:
        flag.parent.mkdir(parents=True,exist_ok=True);flag.write_text(str(os.getpid()))
    status_path=a.output/'suite_status.json'
    status=json.loads(status_path.read_text()) if a.resume and status_path.exists() else dict(jobs=[])
    status.update(pid=os.getpid(),started_epoch=time.time(),complete=False)
    status['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path('gpu/optimization_kernels.py'),Path('gpu/optimization_study.py'),Path(__file__)]}
    save(status_path,status)

    def run(name,args,required=False,timeout=1800):
        previous=next((r for r in reversed(status['jobs']) if r['name']==name and r.get('returncode')==0),None)
        if a.resume and previous:
            print('REUSE',name,flush=True);return True
        command=[sys.executable,'-u','-m','gpu.optimization_study',*args]
        row=dict(name=name,command=command,started_epoch=time.time())
        status['jobs'].append(row);save(status_path,status)
        print('START',name,flush=True)
        with (a.output/(name.replace('/','_')+'.log')).open('w') as log:
            try:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                row['returncode']=result.returncode
            except subprocess.TimeoutExpired:
                row['returncode']=124
        row['elapsed_s']=time.time()-row['started_epoch'];save(status_path,status)
        print('DONE',name,row['returncode'],f"{row['elapsed_s']:.1f}s",flush=True)
        if row['returncode'] and required:raise RuntimeError(f'Required job failed: {name}')
        return row['returncode']==0

    def cell(name,variant,backend,batch,config,devices=512,cycles=3,warmups=3,profile=False):
        args=['cell','--output',str(a.output/name),'--variant',variant,'--backend',backend,
              '--batch',str(batch),'--config',','.join(map(str,config)),
              '--devices',str(devices),'--cycles',str(cycles),'--warmup-cycles',str(warmups)]
        if profile:args.append('--profile')
        okay=run(name,args)
        path=a.output/name/'result.json'
        return json.loads(path.read_text()) if okay and path.exists() else None

    try:
        run('reference',['reference','--output',str(a.output)],required=True)
        run('micro',['micro','--output',str(a.output)],required=True)
        micro=json.loads((a.output/'micro.json').read_text())
        configs={}
        for variant in ['fixed','norm','norm_residual']:
            for batch in [32,48,64,80,96]:
                candidates=[r for r in micro if r['variant']==variant and r['batch']==batch and 'error' not in r]
                if candidates:configs[(variant,batch)]=min(candidates,key=lambda r:r['cuda_ms']['median'])['config']
        probes=[]
        baseline_config=[64,128,8]
        for batch in [64,32,48,80,96]:
            r=cell(f'probes/baseline_auto_b{batch}','baseline','auto',batch,baseline_config)
            if r:probes.append(r)
        for backend in ['flash','cudnn','efficient']:
            if any(r['variant']=='attention' and r['backend']==backend and r['batch']==64 and 'error' not in r for r in micro):
                r=cell(f'probes/baseline_{backend}_b64','baseline',backend,64,baseline_config)
                if r:probes.append(r)
        baseline64=[r for r in probes if r['batch']==64]
        if not baseline64:raise RuntimeError('No valid baseline full-model control')
        backend=min(baseline64,key=lambda r:r['summary_ms']['mean'])['experiment']['backend']
        if backend!='auto':
            for batch in [32,48,80,96]:
                r=cell(f'probes/baseline_{backend}_b{batch}','baseline',backend,batch,baseline_config)
                if r:probes.append(r)
        choices=[(variant,batch) for variant in ['fixed','norm','norm_residual'] for batch in [32,48,64,80,96]
                 if (variant,batch) in configs]
        random.Random(20261104).shuffle(choices)
        for variant,batch in choices:
            r=cell(f'probes/{variant}_{backend}_b{batch}',variant,backend,batch,configs[(variant,batch)])
            if r:probes.append(r)
        save(a.output/'probes.json',probes)
        # Retain the fastest candidate at every batch plus every optimization
        # family's best candidate, and always the original baseline B64 control.
        finalists=[]
        for batch in [32,48,64,80,96]:
            candidates=[r for r in probes if r['batch']==batch]
            if candidates:finalists.append(min(candidates,key=lambda r:r['summary_ms']['mean']))
        for variant in ['baseline','fixed','norm','norm_residual']:
            candidates=[r for r in probes if r['experiment']['variant']==variant]
            if candidates:finalists.append(min(candidates,key=lambda r:r['summary_ms']['mean']))
        original=[r for r in probes if r['batch']==64 and r['experiment']['variant']=='baseline' and r['experiment']['backend']=='auto']
        finalists.extend(original)
        unique={}
        for r in finalists:
            e=r['experiment'];key=f"{e['variant']}_{e['backend']}_b{r['batch']}"
            unique[key]=dict(name=key,variant=e['variant'],backend=e['backend'],batch=r['batch'],config=e['config'])
        finalists=list(unique.values());save(a.output/'finalists.json',finalists)
        for round_id in range(5):
            order=finalists.copy();random.Random(29100+round_id).shuffle(order)
            for c in order:
                r=cell(f"paired/round{round_id}/{c['name']}",c['variant'],c['backend'],c['batch'],c['config'],
                       devices=3000,cycles=3,warmups=5)
                if r is None:raise RuntimeError('Paired finalist failed: '+c['name'])
        # Separate instrumented captures of each finalist, outside timing tables.
        for c in finalists:
            cell(f"profiles/{c['name']}",c['variant'],c['backend'],c['batch'],c['config'],
                 devices=3000,cycles=1,warmups=3,profile=True)
        status.update(complete=True,finished_epoch=time.time());save(status_path,status)
    finally:
        for flag in flags:
            if flag.exists() and flag.read_text().strip()==str(os.getpid()):flag.unlink()


if __name__=='__main__':main()
