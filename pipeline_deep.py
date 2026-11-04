"""Warm concurrent CPU workers and real in-memory 3,000-device burst replay.

Each worker owns reusable packing buffers; weights and read-only ring arrays are
shared within one process. No affinity is claimed on macOS. Publication is local
JSON encoding only, not an external service or network acknowledgement.
"""
import argparse
import os
import json
import time
import resource
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock, BrokenBarrierError
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--workers',type=int,default=1)
p.add_argument('--threads',type=int,default=1)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--devices',type=int,default=512)
p.add_argument('--cycles',type=int,default=3)
p.add_argument('--backend',default='sdpa')
p.add_argument('--label',default='')
p.add_argument('--pin-workers',action='store_true')
p.add_argument('--buffer-mode',choices=['reusable','fused_slices','mirrored','folded_mirrored'],default='reusable')
p.add_argument('--parallel-warmup',action='store_true')
a=p.parse_args()
from runtime_setup import configure_openmp
configure_openmp(a.backend=='compile')
for key in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]=str(a.threads)
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
from measurement import configure_affinity, peak_rss_bytes, snapshot, delta, quantization_engine
configure_affinity()
import numpy as np
import torch
import psutil
from contextlib import nullcontext
from deep_models import FloatScore,FoldedScore,stock_transformer_copy
from model import Scorer, load_model

torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(123)
rng=np.random.default_rng(567)
ring=rng.standard_normal((a.devices,720,32),dtype=np.float32)
m,c=load_model('manual' if a.backend=='manual' else 'sdpa')
if a.backend=='stock':m=stock_transformer_copy(m)
dtype=torch.float32
if a.backend in ('fp16','bf16','bf16_score32'):
    dtype=torch.float16 if a.backend=='fp16' else torch.bfloat16
    m=m.to(dtype)
if a.backend=='int8':
    torch.backends.quantized.engine=quantization_engine()
    m=torch.ao.quantization.quantize_dynamic(m,{torch.nn.Linear},dtype=torch.qint8)
common_scorer=Scorer(m).eval()
if a.backend=='bf16_score32':
    common_scorer=FloatScore(m).eval()
    dtype=torch.float32  # FloatScore handles casting; retain FP32 scoring input.
if a.buffer_mode=='folded_mirrored':
    if a.backend not in ('sdpa','contiguous'):
        raise ValueError('Folded ring experiment supports FP32 SDPA/contiguous only')
    common_scorer=FoldedScore(m,c['mean'],c['std']).eval()
if 'mirrored' in a.buffer_mode:
    ring=np.concatenate((ring,ring),axis=1)
if a.backend=='compile':
    common_scorer=torch.compile(common_scorer,fullgraph=True,dynamic=False)
if a.backend=='contiguous':
    for block in list(m.encoder)+list(m.decoder):block.contiguous_qkv=True
if a.backend=='tanh':
    for block in list(m.encoder)+list(m.decoder):block.gelu_approx='tanh'
mean,scale=c['mean'].numpy(),1/c['std'].numpy()
offset=-mean*scale
splits=np.array_split(np.arange(a.devices),a.workers)
clock={}
ready=Barrier(a.workers+1,action=lambda:clock.update(start=time.perf_counter()),timeout=300)
done=Barrier(a.workers+1,timeout=300)
warmup_lock=Lock()
shared_scores=np.empty(a.devices,dtype=np.float32)

def worker_body(ids,index):
    if a.pin_workers:
        available=sorted(os.sched_getaffinity(0))
        # Each application thread and newly-created internal worker threads inherit
        # this worker's mask. Never claim isolation from other host tenants.
        group=[available[(index*a.threads+j)%len(available)] for j in range(a.threads)]
        os.sched_setaffinity(0,group)
    buffer=np.empty((a.batch,720,32),dtype=np.float32)
    scorer=common_scorer
    if a.backend.startswith('ort'):
        import onnxruntime as ort
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=a.threads
        opts.inter_op_num_threads=1
        opts.add_session_config_entry('session.intra_op.allow_spinning','0')
        name='model_int8' if a.backend=='ort_int8' else 'model'
        session=ort.InferenceSession(f'artifacts/{name}.onnx',opts,providers=['CPUExecutionProvider'])
        run=lambda data:session.run(None,{'input':data})[0]
    else:
        run=lambda data:scorer(torch.from_numpy(data).to(dtype)).float().numpy()
    out=np.empty(len(ids),dtype=np.float32)
    records=[]
    with torch.inference_mode():
        buffer.fill(0)
        # Keep first-use compiler/cache initialization out of concurrent workers.
        with (nullcontext() if a.parallel_warmup and a.backend not in ('jit','compile') else warmup_lock):
            if a.backend=='jit':
                scorer=torch.jit.freeze(torch.jit.trace(scorer,torch.from_numpy(buffer)))
            for _ in range(3):run(buffer)
            if len(ids)%a.batch:
                for _ in range(3):run(buffer[:len(ids)%a.batch])
        for cycle in range(a.cycles):
            ready.wait()
            start=time.perf_counter()
            prep=execution=0.
            batch_events=[]
            # A non-zero ring head forces real chronological packing via slices.
            head=(137+cycle)%720
            for i in range(0,len(ids),a.batch):
                count=min(a.batch,len(ids)-i)
                left=int(ids[i]); right=left+count
                t=time.perf_counter()
                data=buffer[:count]
                if a.buffer_mode=='folded_mirrored':
                    data=ring[left:right,head:head+720]
                elif a.buffer_mode=='mirrored':
                    np.multiply(ring[left:right,head:head+720],scale,out=data)
                    np.add(data,offset,out=data)
                elif a.buffer_mode=='fused_slices':
                    np.multiply(ring[left:right,head:720],scale,out=data[:,:720-head])
                    np.multiply(ring[left:right,:head],scale,out=data[:,720-head:])
                    np.add(data,offset,out=data)
                else:
                    np.copyto(data[:,:720-head],ring[left:right,head:])
                    np.copyto(data[:,720-head:],ring[left:right,:head])
                    np.multiply(data,scale,out=data)
                    np.add(data,offset,out=data)
                prep_dt=time.perf_counter()-t
                prep+=prep_dt
                t=time.perf_counter()
                out[i:i+count]=run(data)
                execution_dt=time.perf_counter()-t
                execution+=execution_dt
                batch_events.append(dict(start_s=t-clock['start'],execution_s=execution_dt,
                                         preparation_s=prep_dt,windows=count))
            records.append(dict(worker_total_s=time.perf_counter()-start,
                                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                                batch_events=batch_events,
                                preparation_s=prep,execution_s=execution))
            shared_scores[ids]=out
            done.wait()
    return out,records

def worker(ids,index):
    try:
        return worker_body(ids,index)
    except BaseException:
        ready.abort()
        done.abort()
        raise

def wait_or_raise(barrier,futures):
    try:
        barrier.wait()
    except BrokenBarrierError:
        for future in futures:
            if future.done() and future.exception() is not None:
                raise future.exception()
        raise

process=psutil.Process()
samples=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    futures=[pool.submit(worker,ids,index) for index,ids in enumerate(splits)]
    for cycle in range(a.cycles):
        wait_or_raise(ready,futures)
        start=clock['start']
        cpu0=process.cpu_times(); ctx0=process.num_ctx_switches()
        resource0=snapshot()
        wait_or_raise(done,futures)
        execution_wall=time.perf_counter()-start
        pub_start=time.perf_counter()
        payload=json.dumps({'scores':shared_scores.tolist()})
        publication_s=time.perf_counter()-pub_start
        elapsed=time.perf_counter()-start
        cpu1=process.cpu_times(); ctx1=process.num_ctx_switches()
        resource1=snapshot()
        samples.append(dict(wall_s=elapsed,packing_and_execution_wall_s=execution_wall,
                            local_json_publication_s=publication_s,
                            process_threads=process.num_threads(),
                            resource_before=resource0,resource_after=resource1,resource_delta=delta(resource0,resource1),
                            cpu_percent=100*(cpu1.user+cpu1.system-cpu0.user-cpu0.system)/elapsed,
                            context_switches=ctx1.voluntary+ctx1.involuntary-ctx0.voluntary-ctx0.involuntary))
    outputs=[future.result() for future in futures]
scores=np.concatenate([x[0] for x in outputs])
for i,s in enumerate(samples):
    s['workers']=[o[1][i] for o in outputs]
result=dict(config=vars(a),cycles=samples,torch_version=torch.__version__,
            ring_storage_bytes=ring.nbytes,
            single_openmp_runtime_requested=bool(os.environ.get('CPU_AE_OMP_CONFIGURED')),
            payload_bytes=len(payload.encode()),peak_rss_bytes=peak_rss_bytes(),
            score_checksum=float(scores.sum()),
            windows_per_second=a.devices/float(np.mean([s['wall_s'] for s in samples])))
tag=f'pipeline_{a.backend}_w{a.workers}_t{a.threads}_b{a.batch}_n{a.devices}'
if a.pin_workers:
    tag=tag.replace('_n', '_pinned_n')
if a.label:
    tag=a.label+'_'+tag
np.save(f'artifacts/{tag}_scores.npy',scores)
Path(f'artifacts/{tag}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
