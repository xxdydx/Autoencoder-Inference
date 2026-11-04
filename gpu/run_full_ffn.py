"""Extend the completed first-stage study with register-resident full FFN fusion."""
from pathlib import Path
import time,subprocess,json,sys
root=Path(__file__).resolve().parent.parent
A=root/'artifacts/gpu_fusion'
while not (A/'COMPLETE').exists():time.sleep(5)

def execute(variant,batch=64,devices=512,unit=False):
    cmd=[sys.executable,'-m','gpu.fusion_case','--variant',variant,'--batch',str(batch),'--devices',str(devices),'--cycles','5' if devices==3000 else '3']
    if unit:cmd+=['--unit']
    if devices==3000:cmd+=['--profile']
    name=f'{variant}_b{batch}_n{devices}' if not unit else 'unit_full'
    print('START',name,flush=True);start=time.time()
    p=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=900)
    (A/f'log_{name}.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,elapsed_s=time.time()-start,stdout=p.stdout,stderr=p.stderr),indent=2))
    print('DONE',name,p.returncode,flush=True)
    return p.returncode==0

if execute('fused_full',unit=True):
    unit=json.loads((A/'unit_full.json').read_text())
    if unit['fused_linear_gelu']['cuda_ms']['p50']<unit['torch_linear_gelu']['cuda_ms']['p50']:
        for kind,batches in [('fused_full',[32,64,128]),('fused_full_auto',[64])]:
            for b in batches:execute(kind,b)
            probes=[]
            for p in A.glob(f'result_{kind}_bf16*_n512.json'):probes.append(json.loads(p.read_text()))
            if probes:
                best=min(probes,key=lambda r:r['latency_s']['mean']);execute(kind,best['batch'],3000)
(A/'FULL_FFN_COMPLETE').write_text('Full FFN unit and eligible whole-model controls complete\n')
