"""CPU attention ablations; no CUDA execution or GPU memory-model assumptions."""
import argparse
import os
from pathlib import Path
import time
import json
import gc
import random
import ctypes
from contextlib import nullcontext

p=argparse.ArgumentParser()
p.add_argument('--threads',type=int,default=1)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--length',type=int,default=720)
p.add_argument('--dtype',choices=['fp32','bf16'],default='fp32')
p.add_argument('--profile',action='store_true')
p.add_argument('--mkl-diagnostic',action='store_true')
p.add_argument('--only',default='')
p.add_argument('--suffix',default='')
a=p.parse_args()
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(a.threads)
if a.mkl_diagnostic:os.environ['MKL_VERBOSE']='1'
from autoencoder.measurement import configure_affinity,snapshot,delta
configure_affinity()
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from autoencoder.model import load_model,synthetic

torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
torch.manual_seed(123)
class Mallinfo(ctypes.Structure):
    _fields_=[(name,ctypes.c_size_t) for name in ('arena','ordblks','smblks','hblks','hblkhd','usmblks','fsmblks','uordblks','fordblks','keepcost')]
def heap_stats():
    try:
        fn=ctypes.CDLL(None).mallinfo2;fn.restype=Mallinfo
        data=fn();return {name:int(getattr(data,name)) for name,_ in Mallinfo._fields_}
    except (AttributeError,OSError):return {}
Path('artifacts/deep').mkdir(parents=True,exist_ok=True)
Path('reports/deep').mkdir(parents=True,exist_ok=True)
model,c=load_model()
with torch.inference_mode():
    length=min(a.length,720)
    x=(synthetic(a.batch,length,789)-c['mean'])/c['std']
    h=model.input(x)+model.pe[:length]
    joint=model.encoder[0].qkv(model.encoder[0].norm1(h)).reshape(a.batch,length,3,4,32)
    if a.length>720:joint=joint.repeat(1,(a.length+719)//720,1,1,1)[:,:a.length].contiguous()
    dtype=torch.float32 if a.dtype=='fp32' else torch.bfloat16
    joint=joint.to(dtype)
    q,k,v=joint.permute(2,0,3,1,4).unbind(0)
    reference=((q.float()@k.float().transpose(-2,-1))*(32**-.5)).softmax(-1)@v.float()

def make(kind):
    scale=32**-.5;owned=0;layout=0
    ctx=nullcontext()
    if kind.startswith('sdpa'):
        if kind=='sdpa_math':ctx=sdpa_kernel(SDPBackend.MATH)
        if kind=='sdpa_flash':ctx=sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if kind=='sdpa_contiguous_runtime':
            fn=lambda:F.scaled_dot_product_attention(q.contiguous(),k.contiguous(),v.contiguous(),dropout_p=0.)
        else:fn=lambda:F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    elif kind=='manual':
        fn=lambda:((q@k.transpose(-2,-1))*scale).softmax(-1)@v
    elif kind=='manual_query_scale':
        fn=lambda:((q*scale)@k.transpose(-2,-1)).softmax(-1)@v
    elif kind=='manual_script':
        module=torch.jit.trace(lambda q,k,v:((q@k.transpose(-2,-1))*scale).softmax(-1)@v,(q,k,v))
        fn=lambda:module(q,k,v)
    elif kind in ('manual_prealloc','manual_prealloc_contiguous','manual_alpha_prealloc'):
        score=torch.empty(a.batch,4,a.length,a.length,dtype=dtype)
        prob=torch.empty_like(score);out=torch.empty_like(q, memory_format=torch.contiguous_format)
        owned=score.numel()*score.element_size()+prob.numel()*prob.element_size()+out.numel()*out.element_size()
        if kind=='manual_prealloc':qq,kk,vv=q,k,v
        else:
            qq,kk,vv=q.contiguous(),k.contiguous(),v.contiguous()
            layout=sum(t.numel()*t.element_size() for t in (qq,kk,vv));owned+=layout
        if kind=='manual_alpha_prealloc':
            qf,kf,vf=[t.reshape(-1,a.length,32) for t in (qq,kk,vv)]
            sf=score.reshape(-1,a.length,a.length);pf=prob.reshape_as(sf);of=out.reshape(-1,a.length,32)
            def fn():
                torch.baddbmm(sf,qf,kf.transpose(-2,-1),beta=0,alpha=scale,out=sf)
                torch.softmax(sf,-1,out=pf)
                torch.bmm(pf,vf,out=of)
                return out
        else:
            def fn():
                torch.matmul(qq,kk.transpose(-2,-1),out=score)
                score.mul_(scale)
                torch.softmax(score,-1,out=prob)
                torch.matmul(prob,vv,out=out)
                return out
    else:raise ValueError(kind)
    return fn,ctx,owned,layout

def memory_stats(trace):
    events=[e for e in trace['traceEvents'] if e.get('name')=='[memory]' and 'Bytes' in e.get('args',{})]
    positive=sum(max(0,e['args']['Bytes']) for e in events)
    total=[e for e in events if 'Total Allocated' in e['args']]
    if total:
        start=total[0]['args']['Total Allocated']-total[0]['args']['Bytes']
        peak=max(e['args']['Total Allocated'] for e in total)-start
    else:
        current=peak=0
        for e in sorted(events,key=lambda e:e['ts']):
            current+=e['args']['Bytes'];peak=max(peak,current)
    return dict(profiled_calls=5,gross_positive_allocated_bytes=positive,
                transient_peak_extra_bytes=max(0,peak),memory_event_count=len(events))

kinds=['manual','manual_query_scale','manual_prealloc','manual_prealloc_contiguous',
       'manual_alpha_prealloc','manual_script','sdpa_math','sdpa_flash','sdpa_default','sdpa_contiguous_runtime']
if a.dtype=='bf16':kinds=['manual','manual_prealloc','sdpa_math','sdpa_flash','sdpa_default']
if a.mkl_diagnostic:kinds=['manual','sdpa_math','sdpa_flash']
if a.only:kinds=[a.only]
random.Random(117).shuffle(kinds)
rows=[]
tag=f'attention_{a.dtype}_t{a.threads}_b{a.batch}_s{a.length}'
if a.suffix:tag+='_'+a.suffix
with torch.inference_mode():
    for kind in kinds:
        row=dict(kind=kind,threads=a.threads,batch=a.batch,length=a.length,dtype=a.dtype,
                 q_stride=list(q.stride()),torch_version=torch.__version__,git_version=torch.version.git_version)
        row['allocator_env']={k:os.environ.get(k) for k in ('MALLOC_MMAP_THRESHOLD_','MALLOC_TRIM_THRESHOLD_','MALLOC_ARENA_MAX','GLIBC_TUNABLES')}
        try:
            setup=time.perf_counter();fn,ctx,owned,layout=make(kind)
            with ctx:
                if a.mkl_diagnostic:
                    print('DIAGNOSTIC_CASE '+kind,flush=True)
                    fn()
                    print('DIAGNOSTIC_END '+kind,flush=True)
                    continue
                for _ in range(3):y=fn()
                row['setup_warmup_s']=time.perf_counter()-setup
                row['persistent_scratch_bytes']=owned;row['prepared_layout_bytes']=layout
                diff=(y.float()-reference).abs()
                row['max_abs_error']=diff.max().item();row['rmse']=(y.float()-reference).square().mean().sqrt().item()
                if a.dtype=='fp32':assert torch.allclose(y.float(),reference,rtol=1e-4,atol=1e-5)
                row['heap_before']=heap_stats()
                before=snapshot();start=time.perf_counter();times=[]
                while len(times)<20 or time.perf_counter()-start<.5:
                    t=time.perf_counter_ns();fn();times.append((time.perf_counter_ns()-t)/1e6)
                after=snapshot()
                row['heap_after']=heap_stats()
                row.update(samples_ms=times,p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                           mean_ms=float(np.mean(times)),resource_before=before,resource_after=after,resource_delta=delta(before,after))
                if a.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                            record_shapes=True,profile_memory=True,with_stack=True) as prof:
                        for i in range(5):
                            with torch.profiler.record_function(f'attention_call_{i}'):
                                fn()
                    path=Path(f'artifacts/deep/{tag}_{kind}_trace.json')
                    prof.export_chrome_trace(str(path))
                    stats=memory_stats(json.loads(path.read_text()))
                    row.update(memory=stats,trace=str(path))
                    row['owned_plus_transient_peak_bytes']=owned+stats['transient_peak_extra_bytes']
                    events=prof.key_averages(group_by_input_shape=True)
                    row['operators']=[dict(key=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,
                                           cpu_us=e.cpu_time_total,shape=e.input_shapes,
                                           self_memory_bytes=e.self_cpu_memory_usage) for e in events]
                    row['aten_events_per_call']=sum(e.count for e in events if e.key.startswith('aten::'))/5
                    Path(f'reports/deep/{tag}_{kind}_operators.txt').write_text(events.table(sort_by='self_cpu_time_total',row_limit=30))
        except Exception as error:
            import traceback
            row['error']=repr(error);row['traceback']=traceback.format_exc()
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('operators','samples_ms','traceback')}),flush=True)
        gc.collect()
if not a.mkl_diagnostic:
    Path(f'artifacts/deep/{tag}.json').write_text(json.dumps(rows,indent=2))
