import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from autoencoder.measurement import configure_affinity
configure_affinity()
import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
import torch
from autoencoder.model import load_model,Scorer,synthetic
from autoencoder.deep_models import FoldedScore,FloatScore
import copy

torch.set_num_threads(1);torch.set_num_interop_threads(1)
# Independent wraparound correctness test, over more than three revolutions.
n,t,f=3,7,4
initial=np.arange(n*t*f,dtype=np.float32).reshape(n,t,f)
expected=initial.copy();circular=initial.copy();mirrored=np.concatenate((initial,initial),axis=1);head=0
for step in range(3*t+3):
    incoming=np.full((n,f),1000+step,dtype=np.float32)
    expected=np.roll(expected,-1,axis=1);expected[:,-1]=incoming
    circular[:,head]=incoming;mirrored[:,head]=incoming;mirrored[:,head+t]=incoming
    head=(head+1)%t
    np.testing.assert_array_equal(np.concatenate((circular[:,head:],circular[:,:head]),axis=1),expected)
    np.testing.assert_array_equal(mirrored[:,head:head+t],expected)

m,c=load_model();folded=FoldedScore(m,c['mean'],c['std']).eval()
native_half=Scorer(copy.deepcopy(m).to(torch.bfloat16)).eval()
float_score_half=FloatScore(copy.deepcopy(m)).eval()
correctness=[]
precision=[]
with torch.inference_mode():
    for length in (120,720):
        raw=synthetic(64,length,901);raw[32:,-10:,:4]+=4
        normalized=(raw-c['mean'])/c['std']
        ref=torch.cat([Scorer(m)(v) for v in normalized.split(4)])
        actual=torch.cat([folded(v) for v in raw.split(4)])
        assert torch.allclose(actual,ref,rtol=1e-4,atol=1e-5)
        threshold=torch.quantile(ref,.95)
        correctness.append(dict(length=length,windows=64,score_max_abs=float((actual-ref).abs().max()),
            score_max_relative=float(((actual-ref).abs()/ref.abs()).max()),
            threshold_agreement=float(((actual>threshold)==(ref>threshold)).float().mean())))
        for name,scorer,data in [('bf16_native_score',native_half,normalized.to(torch.bfloat16)),
                                 ('bf16_fp32_score',float_score_half,normalized)]:
            scores=torch.cat([scorer(v).float() for v in data.split(4)])
            precision.append(dict(kind=name,length=length,windows=64,
                score_max_abs=float((scores-ref).abs().max()),
                score_max_relative=float(((scores-ref).abs()/ref.abs()).max()),
                threshold_agreement=float(((scores>threshold)==(ref>threshold)).float().mean())))

rng=np.random.default_rng(123);n,t,f=3000,720,32
raw=rng.standard_normal((n,t,f),dtype=np.float32)
new=rng.standard_normal((n,f),dtype=np.float32)
shifted=raw.copy();ring=raw.copy();mirror=np.concatenate((raw,raw),axis=1)
state={'ring_head':0,'mirror_head':0}

def roll_update():
    global shifted
    shifted=np.roll(shifted,-1,axis=1);shifted[:,-1]=new
def ring_update():
    h=state['ring_head'];ring[:,h]=new;state['ring_head']=(h+1)%720
def mirror_update():
    h=state['mirror_head'];mirror[:,h]=new;mirror[:,h+720]=new;state['mirror_head']=(h+1)%720

def measure(name,fn,storage,useful_writes,repeats=10):
    fn();times=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();fn();times.append((time.perf_counter_ns()-start)/1e6)
    tracemalloc.start();tracemalloc.reset_peak();fn();current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(method=name,samples_ms=times,p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                persistent_storage_bytes=storage,useful_written_bytes=useful_writes,traced_new_peak_bytes=peak,
                traced_current_bytes=current)

updates=[measure('roll_and_append',roll_update,raw.nbytes,raw.nbytes+n*f*4),
         measure('circular_append',ring_update,ring.nbytes,n*f*4),
         measure('mirrored_append',mirror_update,mirror.nbytes,2*n*f*4)]
# Separate packing test: all methods read an identical source state/head.
ring=raw.copy();mirror=np.concatenate((raw,raw),axis=1);head=137
scale=1/c['std'].numpy();offset=-c['mean'].numpy()*scale
buffer=np.empty_like(raw)
reference=(np.concatenate((ring[:,head:],ring[:,:head]),axis=1)*scale)+offset
def allocate_pack():return np.concatenate((ring[:,head:],ring[:,:head]),axis=1)*scale+offset
def reuse_pack():
    np.copyto(buffer[:,:720-head],ring[:,head:]);np.copyto(buffer[:,720-head:],ring[:,:head])
    np.multiply(buffer,scale,out=buffer);np.add(buffer,offset,out=buffer);return buffer
def fused_pack():
    np.multiply(ring[:,head:],scale,out=buffer[:,:720-head])
    np.multiply(ring[:,:head],scale,out=buffer[:,720-head:]);np.add(buffer,offset,out=buffer);return buffer
def mirror_pack():
    np.multiply(mirror[:,head:head+720],scale,out=buffer);np.add(buffer,offset,out=buffer);return buffer
def folded_view():return mirror[:,head:head+720]
packing=[]
for name,fn,storage,writes in [
        ('allocate_concat_normalize',allocate_pack,ring.nbytes,3*ring.nbytes),
        ('reuse_copy_normalize',reuse_pack,ring.nbytes+buffer.nbytes,3*ring.nbytes),
        ('fused_slice_normalize',fused_pack,ring.nbytes+buffer.nbytes,2*ring.nbytes),
        ('mirrored_view_normalize',mirror_pack,mirror.nbytes+buffer.nbytes,2*ring.nbytes),
        ('folded_mirrored_view_only',folded_view,mirror.nbytes,0)]:
    output=fn()
    if name!='folded_mirrored_view_only':np.testing.assert_allclose(output,reference,rtol=1e-6,atol=1e-6)
    else:
        np.testing.assert_array_equal(output,np.concatenate((ring[:,head:],ring[:,:head]),axis=1))
    row=measure(name,fn,storage,writes,repeats=5)
    row['shares_source_memory']=bool(np.shares_memory(output,mirror) or np.shares_memory(output,ring))
    row['output_c_contiguous']=bool(output.flags.c_contiguous)
    packing.append(row)
result=dict(wraparound_test=dict(devices=3,length=7,features=4,updates=24,passed=True),
            folded_score_validation=correctness,updates=updates,packing=packing,
            precision_score_validation=precision,
            note='Written bytes are algorithmic useful array writes, not measured DRAM traffic. Tracemalloc peak is scoped new tracked allocation, not total RSS. Folded view excludes model execution and scoring; its view is noncontiguous across a multi-device batch.')
Path('artifacts/deep/memory_ring.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
