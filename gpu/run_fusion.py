"""Sequential, resumable, fresh-process fusion experiment driver."""
from pathlib import Path
import subprocess,sys,os,json,time
root=Path(__file__).resolve().parent.parent;os.chdir(root)
A=Path('artifacts/gpu_fusion');A.mkdir(exist_ok=True)

def run(variant,batch=64,devices=512,cycles=3,unit=False,profile=False):
    tag='unit' if unit else f'{variant}_b{batch}_n{devices}'
    expected=A/('unit.json' if unit else f'result_{variant}_bf16_b{batch}_n{devices}.json')
    if expected.exists():return True
    cmd=[sys.executable,'-m','gpu.fusion_case','--variant',variant,'--batch',str(batch),'--devices',str(devices),'--cycles',str(cycles)]
    if unit:cmd+=['--unit']
    if profile:cmd+=['--profile']
    print('START',tag,flush=True);start=time.time()
    try:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
        data=dict(command=cmd,returncode=result.returncode,elapsed_s=time.time()-start,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired as e:data=dict(command=cmd,error='timeout',elapsed_s=time.time()-start,stdout=str(e.stdout),stderr=str(e.stderr))
    (A/f'log_{tag}.json').write_text(json.dumps(data,indent=2))
    print('DONE',tag,data.get('returncode',data.get('error')),flush=True)
    return expected.exists()

run('fused_eager',unit=True)
for variant,batches in [('baseline',[32,64,128]),('auto',[32,64,128]),('fused',[32,64,128]),('fused_auto',[64]),('stock_auto',[64])]:
    for batch in batches:run(variant,batch)
probes=[]
for p in A.glob('result_*_n512.json'):
    r=json.loads(p.read_text());probes.append(r)
choices=[]
for family in ('baseline_bf16','auto_bf16','fused_bf16','fused_auto_bf16','stock_auto_bf16'):
    candidates=[r for r in probes if r['kind']==family]
    if candidates:choices.append(min(candidates,key=lambda r:r['latency_s']['mean']))
(A/'selected.json').write_text(json.dumps(choices,indent=2))
for c in choices:run(c['kind'].removesuffix('_bf16'),c['batch'],3000,5,profile=True)
(A/'COMPLETE').write_text('Fusion/autotune probes and full confirmations completed; errors retained\n')
