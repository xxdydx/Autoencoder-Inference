"""Sequential best-tested TensorRT search and matched randomized confirmations."""
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
    p.add_argument('--output',type=Path,default=Path('artifacts/trt_benchmark'))
    p.add_argument('--python',default='.venv-trt/bin/python')
    p.add_argument('--install',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/autoencoder_optimizations.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    flags=[Path('keepalive/measurement.pause'),Path('/tmp/flowmesh_c_running')]
    for flag in flags:
        if flag.exists():raise RuntimeError(f'Existing measurement flag: {flag}')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU has other compute jobs: '+active)
    for flag in flags:
        flag.parent.mkdir(parents=True,exist_ok=True);flag.write_text(str(os.getpid()))
    status_path=a.output/'suite_status.json'
    status=json.loads(status_path.read_text()) if a.resume and status_path.exists() else dict(jobs=[])
    status.update(complete=False,pid=os.getpid(),started_epoch=time.time())
    status['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        ['gpu/trt_comparison.py','gpu/run_trt_comparison.py','gpu/tensorrt_backend.py','gpu/optimization_kernels.py','gpu/gpu_suite.py']}
    save(status_path,status)
    def command(name,cmd,required=False,timeout=1800):
        if a.resume and any(r['name']==name and r.get('returncode')==0 for r in status['jobs']):
            print('REUSE',name,flush=True);return True
        row=dict(name=name,command=cmd,started_epoch=time.time());status['jobs'].append(row);save(status_path,status)
        print('START',name,flush=True)
        with (a.output/(name.replace('/','_')+'.log')).open('w') as log:
            try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout);row['returncode']=r.returncode
            except subprocess.TimeoutExpired:row['returncode']=124
        row['elapsed_s']=time.time()-row['started_epoch'];save(status_path,status)
        print('DONE',name,row['returncode'],f"{row['elapsed_s']:.1f}s",flush=True)
        if required and row['returncode']:raise RuntimeError('Failed required job: '+name)
        return row['returncode']==0
    def tag(c):
        if c['family']=='custom':return f"custom_{c['variant']}_b{c['batch']}"
        return f"trt_{c['dtype']}_b{c['batch']}_{c['shape_mode']}_{c['execution']}"
    def cell(name,c,execution=None,profile=False):
        args=[a.python,'-u','-m','gpu.trt_comparison','--output',str(a.output/name),
              '--engine-dir',str(a.output/'engines'),'--family',c['family'],
              '--batch',str(c['batch']),'--cycles','3','--warmup-cycles','5']
        if c['family']=='trt':args+=['--dtype',c['dtype'],'--shape-mode',c['shape_mode'],'--execution',execution or c['execution']]
        else:args+=['--variant',c['variant']]
        if profile:args+=['--profile','--cycles','1']
        okay=command(name,args)
        path=a.output/name/'result.json'
        return json.loads(path.read_text()) if okay and path.exists() else {}
    try:
        if a.install:
            if not Path(a.python).exists():
                command('venv',['uv','venv','--system-site-packages',str(Path(a.python).parent.parent)],required=True)
            command('install',['uv','pip','install','--python',a.python,'tensorrt-cu13==11.4.0.106','onnx==1.18.0'],required=True)
        configurations=[dict(family='trt',dtype=d,batch=b,shape_mode=s,execution='both')
                        for d in ['bf16','fp16'] for b in [8,16,32,48,64,80,96,128] for s in ['dynamic','static_pad']]
        random.Random(20261105).shuffle(configurations)
        known=dict(family='trt',dtype='bf16',batch=32,shape_mode='dynamic',execution='both')
        configurations.remove(known);configurations.insert(0,known)
        screens=[]
        for c in configurations:
            name=f"screen/trt_{c['dtype']}_b{c['batch']}_{c['shape_mode']}"
            result=cell(name,c)
            for execution,r in result.items():
                candidate=dict(c,execution=execution)
                screens.append(dict(candidate=candidate,name=tag(candidate),summary_ms=r['summary_ms'],source=str(a.output/name/execution/'result.json')))
            save(a.output/'screen.json',screens)
        if not screens:raise RuntimeError('No valid TensorRT configurations')
        custom=[]
        for b in [32,48,64,80,96]:
            c=dict(family='custom',variant='new',batch=b)
            result=cell('screen/'+tag(c),c)
            if result:
                r=result['graph'];custom.append(dict(candidate=c,name=tag(c),summary_ms=r['summary_ms']))
        save(a.output/'custom_screen.json',custom)
        if not custom:raise RuntimeError('No valid custom configuration')
        # Preserve both strongest screen candidates plus the best candidate of
        # each precision. Independent confirmation determines their ranking.
        selected=sorted(screens,key=lambda r:r['summary_ms']['mean'])[:2]
        for dtype in ['bf16','fp16']:
            valid=[r for r in screens if r['candidate']['dtype']==dtype]
            if valid:selected.append(min(valid,key=lambda r:r['summary_ms']['mean']))
        selected+=sorted(custom,key=lambda r:r['summary_ms']['mean'])[:2]
        candidates={r['name']:r['candidate'] for r in selected}
        for variant,batch in [('old',64),('old',32),('fixed',32),('norm',32)]:
            c=dict(family='custom',variant=variant,batch=batch);candidates[tag(c)]=c
        save(a.output/'finalists.json',candidates)
        for round_id in range(5):
            order=list(candidates.items());random.Random(43000+round_id).shuffle(order)
            for name,c in order:
                result=cell(f'paired/round{round_id}/{name}',c)
                if not result:raise RuntimeError('Paired finalist failed: '+name)
        for name,c in candidates.items():cell('profiles/'+name,c,profile=True)
        status.update(complete=True,finished_epoch=time.time());save(status_path,status)
    finally:
        for flag in flags:
            if flag.exists() and flag.read_text().strip()==str(os.getpid()):flag.unlink()


if __name__=='__main__':main()
