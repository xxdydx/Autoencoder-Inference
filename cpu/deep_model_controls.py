import argparse
import os
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=8);p.add_argument('--batch',type=int,default=1)
a=p.parse_args()
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]=str(a.threads)
from autoencoder.measurement import configure_affinity,snapshot,delta
configure_affinity()
import time
import json
from pathlib import Path
import numpy as np
import torch
from autoencoder.model import load_model,Scorer,synthetic
from autoencoder.deep_models import stock_transformer_copy,FloatScore

torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
caps={name:(getattr(torch.cpu,name)() if callable(getattr(torch.cpu,name,None)) else None)
      for name in ('_is_avx512_supported','_is_avx512_bf16_supported','_is_amx_tile_supported','_is_amx_fp16_supported')}
Path(f'artifacts/deep/model_controls_t{a.threads}_b{a.batch}_metadata.json').write_text(json.dumps(dict(
    torch_version=torch.__version__,git_version=torch.version.git_version,cpu_capability=torch.backends.cpu.get_cpu_capability(),
    cpu_features=caps,parallel_info=torch.__config__.parallel_info(),mkldnn_enabled=torch.backends.mkldnn.enabled),indent=2))
m,c=load_model();x=(synthetic(a.batch,720,789)-c['mean'])/c['std']
with torch.inference_mode():reference=Scorer(m)(x)
runs={}
manual,_=load_model('manual');runs['explicit_reference']=Scorer(manual).eval()
sdpa,_=load_model();runs['eager_sdpa']=Scorer(sdpa).eval()
stock=stock_transformer_copy(sdpa)
assert sum(p.numel() for p in stock.parameters())==1003952
runs['stock_encoder_fastpath']=Scorer(stock).eval()
half,_=load_model();runs['bf16_fp32_score']=FloatScore(half).eval()
native_half,_=load_model();runs['bf16_native_score']=Scorer(native_half.to(torch.bfloat16)).eval()
rows=[]
with torch.inference_mode():
    for name,scorer in runs.items():
        data=x.to(torch.bfloat16) if name=='bf16_native_score' else x
        row=dict(kind=name,batch=a.batch,threads=a.threads,torch_version=torch.__version__,
                 native_fastpath_enabled=torch.backends.mha.get_fastpath_enabled())
        try:
            for _ in range(3):y=scorer(data)
            row['score_max_abs_error']=(y.float()-reference).abs().max().item()
            row['score_max_relative_error']=((y.float()-reference).abs()/reference.abs()).max().item()
            if not name.startswith('bf16'):assert torch.allclose(y.float(),reference,rtol=1e-4,atol=1e-5)
            before=snapshot();start=time.perf_counter();values=[]
            while len(values)<20 or time.perf_counter()-start<1:
                t=time.perf_counter_ns();scorer(data);values.append((time.perf_counter_ns()-t)/1e6)
            after=snapshot()
            row.update(samples_ms=values,p50_ms=float(np.median(values)),mean_ms=float(np.mean(values)),
                       p95_ms=float(np.percentile(values,95)),resource_before=before,resource_after=after,
                       resource_delta=delta(before,after))
            if a.batch==1:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,
                        profile_memory=True,with_stack=True) as prof:
                    for i in range(3):
                        with torch.profiler.record_function(f'model_call_{i}'):scorer(data)
                tag=f'model_control_{name}_t{a.threads}_b{a.batch}'
                path=Path('artifacts/deep',tag+'_trace.json');prof.export_chrome_trace(str(path))
                row['trace']=str(path)
                row['operators']=[dict(key=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,cpu_us=e.cpu_time_total)
                                  for e in prof.key_averages()]
                row['native_fastpath_observed']=any('transformer_encoder_layer_fwd' in e['key'] for e in row['operators'])
        except Exception as error:
            import traceback
            row['error']=repr(error);row['traceback']=traceback.format_exc()
        rows.append(row)
Path(f'artifacts/deep/model_controls_t{a.threads}_b{a.batch}.json').write_text(json.dumps(rows,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ('samples_ms','operators','traceback')} for r in rows]),flush=True)
