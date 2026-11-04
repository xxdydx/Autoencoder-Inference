"""Fresh-process allocation/page-fault controls, after the deep suite finishes."""
import os
import time
import subprocess
from pathlib import Path
import json
import shutil

root=Path(__file__).resolve().parent;os.chdir(root)
while not Path('artifacts/DEEP_COMPLETE').exists():time.sleep(15)
python=str(root/'.venv-modern/bin/python')
modes={'default':{},'arena32':{'MALLOC_MMAP_THRESHOLD_':'33554432'},
       'retained_arena':{'MALLOC_MMAP_THRESHOLD_':'33554432','MALLOC_TRIM_THRESHOLD_':'67108864'}}
for threads in (1,8):
    for kind in ('manual','manual_prealloc','sdpa_flash'):
        for mode,settings in modes.items():
            for repeat in (0,1):
                suffix=f'alloc_{mode}_{kind}_r{repeat}'
                expected=Path(f'artifacts/deep/attention_fp32_t{threads}_b1_s720_{suffix}.json')
                if expected.exists():continue
                env=os.environ.copy();env['CPU_AE_CPUS']='0-7'
                for key in ('MALLOC_MMAP_THRESHOLD_','MALLOC_TRIM_THRESHOLD_','MALLOC_ARENA_MAX','GLIBC_TUNABLES'):env.pop(key,None)
                env.update(settings)
                args=[python,'deep_attention.py','--threads',str(threads),'--only',kind,'--suffix',suffix]
                if threads==8 and repeat==0:args+=['--profile']
                print('START '+suffix+f'_t{threads}',flush=True)
                start=time.time()
                p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=180)
                Path(f'artifacts/deep/log_{suffix}_t{threads}.json').write_text(json.dumps(dict(
                    command=args,environment=settings,returncode=p.returncode,elapsed_s=time.time()-start,
                    stdout=p.stdout,stderr=p.stderr),indent=2))
                print('DONE '+suffix+f'_t{threads}',flush=True)
# Re-run only the updated memory/numerical validation script; preserve initial data.
shutil.copy2('artifacts/deep/memory_ring.json','artifacts/deep/memory_ring_initial.json')
env=os.environ.copy();env['CPU_AE_CPUS']='0-7'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='1'
p=subprocess.run([python,'deep_memory.py'],env=env,capture_output=True,text=True,timeout=300)
Path('artifacts/deep/log_memory_ring_final.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2))
if p.returncode:raise RuntimeError('Final memory/numerical validation failed')
Path('artifacts/ALLOCATOR_COMPLETE').write_text('Allocator, fault and updated numerical checks complete\n')
print('ALLOCATOR_COMPLETE',flush=True)
