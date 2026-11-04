"""Sequential TensorRT tuning and randomized full-burst backend comparisons."""
from pathlib import Path
import json,os,random,subprocess,sys,time

root=Path(__file__).resolve().parent
os.chdir(root)
A=root/'artifacts/tensorrt';A.mkdir(parents=True,exist_ok=True)
status={'started_epoch':time.time(),'jobs':[]}
def run(variant,batch,devices,cycles,label,dtype='bf16',profile=False,required=True):
    directory=A/label;directory.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'tensorrt_case.py','--variant',variant,'--dtype',dtype,'--batch',str(batch),
         '--devices',str(devices),'--cycles',str(cycles),'--label',label]
    if profile:cmd.append('--profile')
    row=dict(command=cmd,label=label,started_epoch=time.time())
    status['jobs'].append(row);(A/'suite_status.json').write_text(json.dumps(status,indent=2))
    with (directory/'command.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    row.update(returncode=result.returncode,elapsed_s=time.time()-row['started_epoch'])
    (A/'suite_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(row),flush=True)
    if result.returncode:
        if required:raise RuntimeError('Failed '+label)
        return None
    return json.loads((directory/'result.json').read_text())

probes=[]
for batch in [8,32,64,128]:
    probes.append(run('trt_graph',batch,512,3,f'probes/trt_graph_bf16_b{batch}'))
# Measure enqueue without CUDA Graphs as well, using the same optimized engine.
best_graph=min(probes,key=lambda r:r['latency_s']['mean'])
probes.append(run('trt',best_graph['batch'],512,3,f'probes/trt_bf16_b{best_graph["batch"]}'))
best=min(probes,key=lambda r:r['latency_s']['mean'])
choices=[(best['kind'].removesuffix('_bf16'),best['batch'],'bf16'),
         ('auto',32,'bf16'),('auto',64,'bf16'),('fused_full_auto',64,'bf16')]
# Retain a same-batch TRT control if its selected batch differs from custom B64.
if best['batch']!=64:choices.append(('trt_graph',64,'bf16'))
choices=list(dict.fromkeys(choices))
(A/'selected.json').write_text(json.dumps({'probes':probes,'choices':choices},indent=2))
# Additional precision controls; failures are retained and do not invalidate BF16.
extra=[]
for dtype,batch in [('fp16',64),('fp32',8)]:
    r=run('trt_graph',batch,512,3,f'probes/trt_graph_{dtype}_b{batch}',dtype=dtype,required=False)
    if r:extra.append(('trt_graph',batch,dtype))
# Five independently shuffled rounds, three complete cycles per candidate.
for round_id in range(5):
    order=choices.copy();random.Random(8080+round_id).shuffle(order)
    for variant,batch,dtype in order:
        run(variant,batch,3000,3,f'paired/round{round_id}/{variant}_{dtype}_b{batch}',dtype=dtype)
for variant,batch,dtype in extra:
    run(variant,batch,3000,5,f'precision/{variant}_{dtype}_b{batch}',dtype=dtype,profile=True,required=False)
# Separate instrumented captures; do not include them in paired latency tables.
for variant,batch,dtype in choices:
    run(variant,batch,3000,1,f'profiles/{variant}_{dtype}_b{batch}',dtype=dtype,profile=True)
status['complete']=True;status['finished_epoch']=time.time()
(A/'suite_status.json').write_text(json.dumps(status,indent=2))
(A/'COMPLETE').write_text('Completed tuning, precision controls, five paired rounds and separate profiles.\n')
