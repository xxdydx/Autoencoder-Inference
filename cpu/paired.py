"""Interleave candidates in seeded random order to reduce temporal-load bias."""
import argparse
import os
p=argparse.ArgumentParser()
p.add_argument('--threads',type=int,default=1)
p.add_argument('--label',default='original')
a=p.parse_args()
from autoencoder.runtime_setup import configure_openmp
configure_openmp(True)
for key in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]=str(a.threads)
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
from autoencoder.measurement import configure_affinity, snapshot, delta
configure_affinity()
import time
import random
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from autoencoder.model import Scorer, load_model, synthetic
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
m,c=load_model()
x=(synthetic(1,720,789)-c['mean'])/c['std']
runs={}
for backend in ('manual','sdpa','contiguous','tanh'):
    m,_=load_model('manual' if backend=='manual' else 'sdpa')
    for b in list(m.encoder)+list(m.decoder):
        b.contiguous_qkv=(backend=='contiguous')
        b.gelu_approx='tanh' if backend=='tanh' else 'none'
    scorer=Scorer(m).eval()
    runs[backend]=lambda scorer=scorer:scorer(x)
for backend in ('ort','ort_int8'):
    options=ort.SessionOptions()
    options.intra_op_num_threads=a.threads;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    path='artifacts/model_int8.onnx' if backend=='ort_int8' else 'artifacts/model.onnx'
    session=ort.InferenceSession(path,options,providers=['CPUExecutionProvider'])
    data=x.numpy()
    runs[backend]=lambda session=session:session.run(None,{'input':data})[0]
m,_=load_model()
compiled=torch.compile(Scorer(m).eval(),fullgraph=True,dynamic=False)
runs['compile']=lambda:compiled(x)
samples={k:[] for k in runs}
rounds=[]
random.seed(10)
with torch.inference_mode():
    for fn in runs.values():
        for _ in range(5):fn()
    resource0=snapshot()
    for iteration in range(10):
        order=list(runs);random.shuffle(order)
        record={}
        for name in order:
            values=[]
            for _ in range(3):
                start=time.perf_counter_ns();runs[name]()
                values.append((time.perf_counter_ns()-start)/1e6)
            samples[name].extend(values)
            record[name]=float(np.median(values))
        rounds.append(record)
    resource1=snapshot()
result=dict(torch_version=torch.__version__,threads=a.threads,label=a.label,
            resource_before=resource0,resource_after=resource1,resource_delta=delta(resource0,resource1),
            shape=[1,720,32],rounds=rounds,samples_ms=samples,
            summary={k:dict(p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),
                            mean_ms=float(np.mean(v)),windows_per_second=1000/float(np.mean(v)),
                            median_paired_speedup_vs_manual=float(np.median([r['manual']/r[k] for r in rounds])))
                     for k,v in samples.items()})
Path(f'artifacts/{a.label}_paired_t{a.threads}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['summary']),flush=True)
