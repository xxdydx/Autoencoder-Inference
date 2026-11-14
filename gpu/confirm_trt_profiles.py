"""Confirm leading TensorRT precisions/profile types in fresh randomized rounds."""
import argparse
import fcntl,json,os,pathlib,random,subprocess,time
from gpu.optimization_study import save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="artifacts/trt_benchmark")
    a=parser.parse_args()
    root=pathlib.Path(a.input);out=root/'stability';out.mkdir(exist_ok=True)
    lock=open('/tmp/autoencoder_optimizations.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    flags=[pathlib.Path('keepalive/measurement.pause'),pathlib.Path('/tmp/flowmesh_c_running')]
    assert not any(p.exists() for p in flags)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    for p in flags:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(str(os.getpid()))
    screens=json.loads((root/'screen.json').read_text());selected=sorted(screens,key=lambda r:r['summary_ms']['mean'])[:2]
    for dtype in ['bf16','fp16']:
     for shape in ['dynamic','static_pad']:
      selected.append(min([r for r in screens if r['candidate']['dtype']==dtype and r['candidate']['shape_mode']==shape],key=lambda r:r['summary_ms']['mean']))
    candidates={r['name']:r['candidate'] for r in selected};candidates['custom_new_b32']=dict(family='custom',variant='new',batch=32)
    status=dict(complete=False,pid=os.getpid(),reason='Confirm the best candidate of each precision/profile type because screening and confirmation GPU temperature differed.',candidates=candidates,jobs=[])
    save(out/'suite_status.json',status)
    try:
     for rnd in range(5):
      order=list(candidates.items());random.Random(54000+rnd).shuffle(order)
      for name,c in order:
       folder=out/f'round{rnd}'/name
       cmd=['.venv-trt/bin/python','-u','-m','gpu.trt_comparison','--output',str(folder),'--engine-dir',str(root/'engines'),'--family',c['family'],'--batch',str(c['batch']),'--cycles','3','--warmup-cycles','5']
       if c['family']=='trt':cmd+=['--dtype',c['dtype'],'--shape-mode',c['shape_mode'],'--execution',c['execution']]
       else:cmd+=['--variant','new']
       row=dict(name=f'round{rnd}/{name}',command=cmd,started_epoch=time.time());status['jobs'].append(row);save(out/'suite_status.json',status)
       print('START',row['name'],flush=True)
       with (out/f'round{rnd}_{name}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
       row.update(returncode=r.returncode,elapsed_s=time.time()-row['started_epoch']);save(out/'suite_status.json',status);print('DONE',row['name'],r.returncode,flush=True)
       assert r.returncode==0,row
     status.update(complete=True,finished_epoch=time.time());save(out/'suite_status.json',status)
    finally:
     for p in flags:
      if p.exists() and p.read_text()==str(os.getpid()):p.unlink()

if __name__=="__main__":main()
