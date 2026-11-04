import json
import time
from pathlib import Path
from measurement import configure_affinity
configure_affinity()
import numpy as np

rng=np.random.default_rng(123)
raw=rng.standard_normal((3000,720,32),dtype=np.float32)
mean=np.linspace(-1,1,32,dtype=np.float32)
std=np.linspace(.5,2,32,dtype=np.float32)
scale=1/std
offset=-mean*scale
head=137

def scalar(x):
    out=np.empty_like(x)
    for b in range(len(x)):
        for t in range(720):
            for f in range(32):
                out[b,t,f]=(x[b,(t+head)%720,f]-mean[f])/std[f]
    return out

def per_device(x):
    out=np.empty_like(x)
    for b in range(len(x)):
        chronological=np.concatenate((x[b,head:],x[b,:head]))
        out[b]=(chronological-mean)/std
    return out

buffer=np.empty_like(raw)
def reusable(x):
    out=buffer[:len(x)]
    np.copyto(out[:,:720-head],x[:,head:])
    np.copyto(out[:,720-head:],x[:,:head])
    np.multiply(out,scale,out=out)
    np.add(out,offset,out=out)
    return out

rows=[]
for n,methods in ((32,(scalar,per_device,reusable)),(3000,(per_device,reusable))):
    reference=per_device(raw[:n])
    for method in methods:
        error=float(np.max(np.abs(reference-method(raw[:n]))))
        values=[]
        for _ in range(5):
            start=time.perf_counter()
            method(raw[:n])
            values.append((time.perf_counter()-start)*1000)
        rows.append(dict(method=method.__name__,devices=n,samples_ms=values,
                         p50_ms=float(np.median(values)),max_abs_error=error))
Path('artifacts/preprocessing.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows),flush=True)
