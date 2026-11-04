"""After the main suite, test whether retaining larger heaps fixes B8 faults."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parent.parent
os.chdir(root)
out=root/'artifacts/native_path'
while True:
    try:
        status=json.loads((out/'suite_status.json').read_text())
        if status.get('complete'):break
        if any(j.get('returncode',0) for j in status['jobs']):raise SystemExit('Main suite failed')
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(2)
status={'jobs':[]}
for label in ['default_repeat','retained_large']:
    env=dict(os.environ,CPU_AE_CPUS='0-1')
    for key in ['MALLOC_MMAP_THRESHOLD_','MALLOC_TRIM_THRESHOLD_']:env.pop(key,None)
    if label=='retained_large':
        env.update(MALLOC_MMAP_THRESHOLD_='268435456',MALLOC_TRIM_THRESHOLD_='536870912')
    name=f'micro_{label}_t1_b8'
    cmd=[sys.executable,'-m','cpu.native_path_study','--label',label,'--threads','1','--batch','8',
         '--rounds','5','--calls','3']
    row={'name':name,'command':cmd,'started_epoch':time.time()}
    status['jobs'].append(row)
    (out/'allocator_followup_status.json').write_text(json.dumps(status,indent=2))
    with (out/f'{name}.log').open('w') as log:
        r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    row.update(returncode=r.returncode,finished_epoch=time.time())
    (out/'allocator_followup_status.json').write_text(json.dumps(status,indent=2))
    if r.returncode:raise SystemExit(r.returncode)
status['complete']=True
(out/'allocator_followup_status.json').write_text(json.dumps(status,indent=2))
