"""Diagnostic request timelines under narrow/wide affinity, same CPU quota.

The in-cgroup sampler can itself be descheduled while throttled. Gaps between
observations are preserved; they are not presented as exact throttle boundaries.
"""
import argparse
import os
p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);p.add_argument('--mask',default='0-7')
p.add_argument('--requests',type=int,default=16)
a=p.parse_args()
os.environ['CPU_AE_CPUS']=a.mask
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from autoencoder.measurement import configure_affinity,snapshot,delta,read_text
configure_affinity()
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from autoencoder.model import load_model,Scorer,synthetic

torch.set_num_threads(1);torch.set_num_interop_threads(1)
m,c=load_model();scorer=Scorer(m).eval()
x=(synthetic(1,720,789)-c['mean'])/c['std']
ready_clock={};samples=[];started=threading.Event();stop=threading.Event();warm=threading.Lock()

def start():
    ready_clock['before']=snapshot();ready_clock['start']=time.perf_counter();started.set()
ready=threading.Barrier(a.workers+1,action=start,timeout=120)
done=threading.Barrier(a.workers+1,timeout=120)

def sample():
    started.wait()
    while not stop.is_set():
        text=read_text('/sys/fs/cgroup/cpu.stat') or ''
        stat={k:int(v) for k,v in (line.split() for line in text.splitlines())}
        samples.append(dict(t_s=time.perf_counter()-ready_clock['start'],**stat))
        stop.wait(.005)

def worker(index):
    events=[]
    try:
        with torch.inference_mode():
            with warm:
                for _ in range(3):scorer(x)
            ready.wait()
            for i in range(a.requests):
                begin=time.perf_counter();scorer(x);end=time.perf_counter()
                events.append(dict(worker=index,request=i,start_s=begin-ready_clock['start'],end_s=end-ready_clock['start'],latency_ms=(end-begin)*1000))
            done.wait()
        return events
    except BaseException:
        ready.abort();done.abort();raise

sampler=threading.Thread(target=sample,daemon=True);sampler.start()
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    jobs=[pool.submit(worker,i) for i in range(a.workers)]
    ready.wait();done.wait();wall=time.perf_counter()-ready_clock['start']
    after=snapshot();stop.set();sampler.join()
    events=[event for job in jobs for event in job.result()]
latencies=[e['latency_ms'] for e in events]
result=dict(workers=a.workers,threads_per_worker=1,mask=a.mask,requests=len(events),wall_s=wall,
            windows_per_second=len(events)/wall,p50_ms=float(np.median(latencies)),
            p95_ms=float(np.percentile(latencies,95)),p99_ms=float(np.percentile(latencies,99)),
            resource_before=ready_clock['before'],resource_after=after,resource_delta=delta(ready_clock['before'],after),
            events=events,cpu_samples=samples,sampler_interval_target_s=.005,
            sampler_note='Sampler shares the workload cgroup and may be delayed by quota throttling; sample gaps are retained.')
Path(f'artifacts/deep/quota_w{a.workers}_m{a.mask.replace("-","to")}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('events','cpu_samples')}),flush=True)
