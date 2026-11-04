"""Independent randomized confirmations against compiler-only and matched controls."""
from pathlib import Path
import subprocess,sys,json,time,random
root=Path(__file__).resolve().parent.parent;A=root/'artifacts/gpu_fusion'
while not (A/'FULL_FFN_COMPLETE').exists():time.sleep(5)
rows=[json.loads(p.read_text()) for p in A.glob('result*_n3000.json')]
custom=[r for r in rows if r['kind'].startswith('fused')]
if not custom:raise RuntimeError('No valid custom-kernel full result')
winner=min(custom,key=lambda r:r['latency_s']['mean'])
auto=min([r for r in rows if r['kind']=='auto_bf16'],key=lambda r:r['latency_s']['mean'])
choices=[('baseline',64),(auto['kind'].removesuffix('_bf16'),auto['batch']),
         (winner['kind'].removesuffix('_bf16'),winner['batch'])]
if winner['batch']!=64:choices.append(('baseline',winner['batch']))
choices=list(dict.fromkeys(choices))
P=A/'paired';P.mkdir(exist_ok=True);(P/'selected.json').write_text(json.dumps(choices,indent=2))
for round in range(5):
    order=choices.copy();random.Random(740+round).shuffle(order)
    for variant,batch in order:
        label=f'paired/round{round}/{variant}_b{batch}'
        directory=A/label;directory.mkdir(parents=True,exist_ok=True)
        expected=directory/f'result_{variant}_bf16_b{batch}_n3000.json'
        if expected.exists():continue
        cmd=[sys.executable,'-m','gpu.fusion_case','--variant',variant,'--batch',str(batch),'--devices','3000','--cycles','3','--label',label]
        print('START',label,flush=True);start=time.time()
        p=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=900)
        (directory/'command_log.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,elapsed_s=time.time()-start,stdout=p.stdout,stderr=p.stderr),indent=2))
        if p.returncode:raise RuntimeError('Paired confirmation failed: '+label)
        print('DONE',label,flush=True)
(A/'PAIRED_COMPLETE').write_text('Five randomized rounds; three complete cycles per candidate per round\n')
