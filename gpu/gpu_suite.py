"""Remote-only CUDA measurements of the unchanged CPU-trained autoencoder."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
from pathlib import Path
import time,json,subprocess,resource,traceback,random,gc,contextlib,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
from autoencoder.model import load_model
from autoencoder.deep_models import stock_transformer_copy

A=Path('artifacts/gpu');A.mkdir(parents=True,exist_ok=True)
PAUSE=Path('keepalive/measurement.pause')
PAUSE.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
torch.manual_seed(123)

def save(name,data):
    p=A/(name+'.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def gpu_state():
    r=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem','--format=csv,noheader'],capture_output=True,text=True)
    return r.stdout.strip()

def snapshot():
    ru=resource.getrusage(resource.RUSAGE_SELF)
    def read(p):
        try:return Path(p).read_text().strip()
        except OSError:return None
    return dict(time=time.time(),cpu_stat=read('/sys/fs/cgroup/cpu.stat'),pressure=read('/sys/fs/cgroup/cpu.pressure'),
                peak_rss_bytes=ru.ru_maxrss*1024,minor_faults=ru.ru_minflt,
                cpu_user_s=ru.ru_utime,cpu_system_s=ru.ru_stime)

def distribution(samples):
    x=np.asarray(samples,dtype=float)
    return dict(mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)))

class Score(nn.Module):
    def __init__(self,model,dtype):
        super().__init__();self.model=model.to('cuda',dtype=dtype).eval();self.dtype=dtype
    def forward(self,x):return (self.model(x.to(self.dtype)).float()-x).square().mean((1,2))

class CompiledScore(nn.Module):
    def __init__(self,s):
        super().__init__();self.compiled=torch.compile(s,mode='reduce-overhead',dynamic=False,fullgraph=True)
    def forward(self,x):
        torch.compiler.cudagraph_mark_step_begin()
        return self.compiled(x).clone()

def scorer(kind):
    attention='manual' if kind.startswith('manual') else 'sdpa'
    m,c=load_model(attention)
    dtype=torch.bfloat16 if 'bf16' in kind else torch.float16 if 'fp16' in kind else torch.float32
    if kind.startswith('stock'):m=stock_transformer_copy(m)
    s=Score(m,dtype).eval()
    if kind.startswith('compile'):s=CompiledScore(s)
    return s,c

def backend_context(kind):
    return sdpa_kernel(SDPBackend.MATH) if kind.startswith('math') else contextlib.nullcontext()

def validation(s,kind):
    rows=[]
    for length in (120,720):
        d=np.load(f'artifacts/validation_reference_{length}.npz')
        x=torch.from_numpy(d['input']).cuda();reference=d['scores'];threshold=float(d['threshold'])
        outputs=torch.cat([s(part).clone() for part in x.split(8)]).cpu().numpy()
        diff=np.abs(outputs-reference)
        row=dict(length=length,windows=len(outputs),max_abs=float(diff.max()),max_relative=float((diff/np.maximum(np.abs(reference),1e-12)).max()),
                 alert_agreement=float(np.mean((outputs>threshold)==(reference>threshold))),threshold=threshold)
        if kind.endswith('fp32'):assert np.allclose(outputs,reference,rtol=1e-4,atol=1e-5),row
        rows.append(row)
    return rows

def capture_profile(fn,name,calls=3):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
        for i in range(calls):
            with torch.profiler.record_function(f'measured_call_{i}'):fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(A/(name+'_trace.json')))
    rows=[]
    for e in prof.key_averages():
        rows.append(dict(key=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,cpu_us=e.cpu_time_total,
                         self_cuda_us=getattr(e,'self_device_time_total',0),cuda_us=getattr(e,'device_time_total',0),
                         self_cpu_memory_bytes=e.self_cpu_memory_usage,self_cuda_memory_bytes=getattr(e,'self_device_memory_usage',0)))
    (A/(name+'_operators.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
    return rows

def sweep():
    kinds=['manual_fp32','sdpa_fp32','sdpa_bf16','sdpa_fp16','math_bf16','stock_fp32','stock_bf16']
    reference=np.load('artifacts/validation_reference_720.npz')['input']
    x=torch.from_numpy(np.concatenate([reference,reference],axis=0)).cuda()
    rows=[]
    random.Random(2026).shuffle(kinds)
    for kind in kinds:
        s,c=scorer(kind)
        with backend_context(kind):
            try:v=validation(s,kind)
            except Exception as e:
                rows.append(dict(kind=kind,error=repr(e),traceback=traceback.format_exc()));save('micro',rows);continue
            for batch in (1,8,16,32,64,128):
                row=dict(kind=kind,batch=batch,validation=v)
                try:
                    data=x[:batch];begin=time.perf_counter()
                    for _ in range(5):s(data)
                    torch.cuda.synchronize();row['setup_warmup_s']=time.perf_counter()-begin
                    torch.cuda.reset_peak_memory_stats();before=snapshot();start=time.perf_counter();times=[];event_times=[]
                    while len(times)<30 or time.perf_counter()-start<.6:
                        e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
                        t=time.perf_counter();e0.record();s(data);e1.record();e1.synchronize()
                        times.append(1000*(time.perf_counter()-t));event_times.append(e0.elapsed_time(e1))
                    row.update(host_ms=distribution(times),cuda_ms=distribution(event_times),samples_ms=times,cuda_samples_ms=event_times,
                               windows_per_second=1000*batch/np.mean(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(),resource_before=before,resource_after=snapshot(),gpu=gpu_state())
                    if batch==32:
                        row['operators']=capture_profile(lambda:s(data),'micro_'+kind)
                    rows.append(row);save('micro',rows)
                    print('MICRO',kind,batch,row['host_ms']['p50'],flush=True)
                except Exception as e:
                    row.update(error=repr(e),traceback=traceback.format_exc());rows.append(row);save('micro',rows)
        del s;gc.collect();torch.cuda.empty_cache()
    return rows

class GraphScore:
    def __init__(self,s,shapes):
        # Captured kernels retain raw weight addresses: keep the module alive.
        self.scorer=s
        self.entries={}
        for batch in sorted(set(shapes)):
            if batch==0:continue
            x=torch.zeros(batch,720,32,device='cuda');stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):s(x)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):y=s(x)
            self.entries[batch]=(x,y,graph)
    def __call__(self,data):
        x,y,graph=self.entries[len(data)];x.copy_(data);graph.replay();return y

def pipeline(kind,batch,mode,devices,cycles,profile=False,*,warmup_cycles=0,capture_range=False,annotate=False):
    """Real rings, CPU normalization, H2D, scoring, D2H and JSON in the timer.

    pageable_sync: every batch transfers and returns scores synchronously.
    pinned_serial: reusable pinned input, queued scores, one final D2H.
    pinned_overlap: two input slots; a copy stream overlaps H2D with execution.
    gpu_normalize: same double-buffer path, normalization performed on GPU.
    resident: raw rings already on GPU; no H2D included, a distinct scenario.
    """
    name=f'{kind}_{mode}_b{batch}_n{devices}'
    setup=time.perf_counter();s,c=scorer(kind.replace('graph_','sdpa_'))
    rng=np.random.default_rng(567);ring=rng.standard_normal((devices,720,32),dtype=np.float32)
    scale=(1/c['std'].numpy()).astype(np.float32);offset=(-c['mean'].numpy()*scale).astype(np.float32)
    gs=torch.from_numpy(scale).cuda();go=torch.from_numpy(offset).cuda()
    sizes=sorted(set([min(batch,devices),devices%batch])-{0})
    for size in sizes:
        z=torch.zeros(size,720,32,device='cuda')
        for _ in range(3):s(z)
    torch.cuda.synchronize()
    if kind.startswith('graph_'):s=GraphScore(s,sizes)
    pinned=mode!='pageable_sync';slots=2 if mode in ('pinned_overlap','gpu_normalize') else 1
    host=[torch.empty((batch,720,32),pin_memory=pinned) for _ in range(slots)]
    host_arrays=[h.numpy() for h in host]
    device=[torch.empty(batch,720,32,device='cuda') for _ in range(slots)]
    copy_stream=torch.cuda.Stream();done=[None]*slots
    gpu_ring=torch.from_numpy(ring).cuda() if mode=='resident' else None
    scores=torch.empty(devices,device='cuda',dtype=torch.float32)
    # GPU normalization and transfer paths are warmed, too, before the deadline.
    for j in range(slots):
        device[j].zero_();s(device[j][:sizes[0]])
    if isinstance(s,GraphScore):
        for size in sizes:
            check=torch.randn(size,720,32,device='cuda')
            actual=s(check).clone();expected=s.scorer(check)
            assert torch.allclose(actual,expected,rtol=1e-4,atol=1e-5),'Captured scorer differs from eager'
    torch.cuda.synchronize();setup_s=time.perf_counter()-setup
    samples=[];first_scores=None
    def region(name):
        return torch.cuda.nvtx.range(name) if annotate else contextlib.nullcontext()
    def cycle(index):
        nonlocal first_scores
        head=(137+index)%720;before=snapshot();torch.cuda.synchronize();start=time.perf_counter();prep=0.;gpu_events=[]
        for chunk,left in enumerate(range(0,devices,batch)):
            n=min(batch,devices-left);j=chunk%slots
            if mode=='resident':
                raw=gpu_ring[left:left+n]
                data=torch.cat((raw[:,head:],raw[:,:head]),dim=1)*gs+go
            else:
                if done[j] is not None:done[j].synchronize()
                t=time.perf_counter();buf=host_arrays[j][:n]
                with region(f'pack_batch_{chunk}'):
                    np.copyto(buf[:,:720-head],ring[left:left+n,head:]);np.copyto(buf[:,720-head:],ring[left:left+n,:head])
                    if mode!='gpu_normalize':np.multiply(buf,scale,out=buf);np.add(buf,offset,out=buf)
                prep+=time.perf_counter()-t
                if mode in ('pinned_overlap','gpu_normalize'):
                    with region(f'h2d_batch_{chunk}'),torch.cuda.stream(copy_stream):
                        device[j][:n].copy_(host[j][:n],non_blocking=True)
                        copied=torch.cuda.Event();copied.record()
                    torch.cuda.current_stream().wait_event(copied)
                else:
                    with region(f'h2d_batch_{chunk}'):device[j][:n].copy_(host[j][:n],non_blocking=pinned)
                data=device[j][:n]
                if mode=='gpu_normalize':data.mul_(gs).add_(go)
            e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
            with region(f'compute_batch_{chunk}'):
                e0.record();output=s(data);scores[left:left+n].copy_(output);e1.record();gpu_events.append((e0,e1))
            done[j]=e1
            if mode=='pageable_sync':scores[left:left+n].cpu()
        out=scores.cpu().numpy()
        inference_end=time.perf_counter()
        payload=json.dumps({'scores':out.tolist()})
        wall=time.perf_counter()-start
        if index==0:first_scores=out.copy()
        return dict(wall_s=wall,pack_cpu_s=prep,packing_to_scores_wall_s=inference_end-start,
                    local_json_s=time.perf_counter()-inference_end,batch_cuda_ms=[a.elapsed_time(b) for a,b in gpu_events],
                    resource_before=before,resource_after=snapshot(),payload_bytes=len(payload),head=head)
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup_cycles):cycle(-i-1)
    for i in range(cycles):
        if capture_range and i==0:torch.cuda.profiler.start()
        try:
            with region(f'cycle_{i}_{kind}_{mode}_b{batch}'):row=cycle(i)
        finally:
            if capture_range and i==0:
                torch.cuda.synchronize();torch.cuda.profiler.stop()
        samples.append(row)
        record=dict(kind=kind,batch=batch,mode=mode,devices=devices,cycles=samples,setup_warmup_s=setup_s,
                    latency_s=distribution([v['wall_s'] for v in samples]),windows_per_second=devices/np.mean([v['wall_s'] for v in samples]),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=gpu_state())
        if warmup_cycles or capture_range or annotate:
            record['profiling_protocol']=dict(full_cycle_warmups=warmup_cycles,capture_range=capture_range,nvtx=annotate,
                                               scorer_warmups_per_shape=3,slot_warmups=slots)
        if first_scores is not None:
            np.save(A/(name+'_scores_head137.npy'),first_scores)
            if devices==3000:
                reference=np.load('artifacts/cpu_scores_head137.npy');diff=np.abs(first_scores-reference)
                record['cpu_score_agreement']=dict(max_abs=float(diff.max()),max_relative=float((diff/np.abs(reference)).max()),
                    fp32_tolerance_pass=bool(np.allclose(first_scores,reference,rtol=1e-4,atol=1e-5)))
                if kind.endswith('fp32'):assert record['cpu_score_agreement']['fp32_tolerance_pass'],record['cpu_score_agreement']
        save('pipeline_'+name,record)
        print('PIPELINE',name,i,row['wall_s'],flush=True)
    if profile:
        record['operators']=capture_profile(lambda:cycle(0),'pipeline_'+name,calls=1)
        save('pipeline_'+name,record)
    del s,ring,gpu_ring,host,device,scores;gc.collect();torch.cuda.empty_cache()
    return record

def run():
    metadata=dict(torch=torch.__version__,torch_git=torch.version.git_version,numpy=np.__version__,cuda=torch.version.cuda,
                  device=torch.cuda.get_device_properties(0).__repr__(),gpu=gpu_state(),affinity=sorted(os.sched_getaffinity(0)),
                  cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  checkpoint_sha256=hashlib.sha256(Path('artifacts/model.pt').read_bytes()).hexdigest())
    save('environment',metadata)
    micro=sweep()
    probes=[]
    for kind in ('sdpa_fp32','sdpa_bf16','sdpa_fp16'):
        for mode in ('pageable_sync','pinned_serial','pinned_overlap','gpu_normalize','resident'):
            for batch in (8,32,64,128):
                try:probes.append(pipeline(kind,batch,mode,512,3))
                except Exception as e:save(f'error_{kind}_{mode}_b{batch}',dict(error=repr(e),traceback=traceback.format_exc()))
    choices={}
    for kind in ('sdpa_fp32','sdpa_bf16','sdpa_fp16'):
        for scenario in ('host','resident'):
            candidates=[r for r in probes if r['kind']==kind and (r['mode']=='resident')==(scenario=='resident')]
            if candidates:choices[kind+'_'+scenario]=min(candidates,key=lambda r:r['latency_s']['mean'])
    save('selected',choices)
    confirmations=[]
    for name,choice in choices.items():
        try:confirmations.append(pipeline(choice['kind'],choice['batch'],choice['mode'],3000,5,profile=True))
        except Exception as e:save('error_full_'+name,dict(error=repr(e),traceback=traceback.format_exc()))
    for key in ('sdpa_fp32_host','sdpa_bf16_host'):
        if key not in choices:continue
        c=choices[key];dtype=c['kind'].split('_')[-1]
        for kind in ('manual_'+dtype,'stock_'+dtype,'graph_'+dtype,'compile_'+dtype):
            try:
                if kind.startswith(('graph','compile')):
                    candidate,_=scorer(kind.replace('graph','sdpa'));save('validation_'+kind,validation(candidate,kind));del candidate
                confirmations.append(pipeline(kind,c['batch'],c['mode'],3000,5,profile=True))
            except Exception as e:save('error_extra_'+kind,dict(error=repr(e),traceback=traceback.format_exc()))
    save('confirmations',confirmations)
    (A/'COMPLETE').write_text('GPU sweep, pipeline probes and full confirmations complete\n')

if __name__=='__main__':
    PAUSE.write_text(str(os.getpid()))
    # A short keepalive burst already in progress must finish before measurements.
    while subprocess.run(['pgrep','-f','python3.*burn.py'],capture_output=True).returncode==0:time.sleep(1)
    try:
        with torch.inference_mode():run()
    except Exception as e:
        save('FATAL',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:PAUSE.unlink(missing_ok=True)
