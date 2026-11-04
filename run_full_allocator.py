"""Matched full-cycle allocator controls; process-local environment only."""
from pathlib import Path
import os
import subprocess
import json
import time

root=Path(__file__).resolve().parent;os.chdir(root)
while not Path('artifacts/ALLOCATOR_COMPLETE').exists():time.sleep(10)
python=str(root/'.venv-modern/bin/python')
cases=[('default','manual'),('retained','manual'),('default','sdpa'),('retained','sdpa')]
for round_index in range(2):
    for mode,backend in (cases if round_index==0 else list(reversed(cases))):
        label=f'allocator_full_{mode}_{backend}_r{round_index}'
        expected=Path(f'artifacts/deep/{label}_pipeline_{backend}_w8_t1_b1_pinned_n3000.json')
        if expected.exists():continue
        env=os.environ.copy();env['CPU_AE_CPUS']='0-7'
        for key in ('MALLOC_MMAP_THRESHOLD_','MALLOC_TRIM_THRESHOLD_','MALLOC_ARENA_MAX','GLIBC_TUNABLES'):env.pop(key,None)
        if mode=='retained':env.update(MALLOC_MMAP_THRESHOLD_='33554432',MALLOC_TRIM_THRESHOLD_='67108864')
        args=[python,'pipeline_deep.py','--backend',backend,'--workers','8','--threads','1','--batch','1',
              '--devices','3000','--cycles','1','--pin-workers','--parallel-warmup','--label','deep/'+label]
        print('START '+label,flush=True)
        p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=300)
        Path(f'artifacts/deep/log_{label}.json').write_text(json.dumps(dict(command=args,returncode=p.returncode,
             allocator=mode,stdout=p.stdout,stderr=p.stderr),indent=2))
        if p.returncode:raise RuntimeError('Failed '+label)
        print('DONE '+label,flush=True)
Path('artifacts/FULL_ALLOCATOR_COMPLETE').write_text('Matched end-to-end allocator controls complete\n')
