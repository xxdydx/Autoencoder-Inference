"""Run fresh-process CPU controls sequentially; retain every command and log."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent.parent
os.chdir(root)
out = root / 'artifacts/native_path'
out.mkdir(parents=True, exist_ok=True)
jobs = [
    ('default', 1, 1, 'micro', 7),
    ('default', 2, 1, 'micro', 7),
    ('default', 1, 8, 'micro', 7),
    ('retained', 1, 1, 'micro', 7),
    ('retained', 2, 1, 'micro', 7),
    ('default', 1, 1, 'burst', 3),
]
status = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'jobs': []}
for index, (label, threads, batch, stage, rounds) in enumerate(jobs):
    env = dict(os.environ, CPU_AE_CPUS='0-1')
    for key in ['MALLOC_MMAP_THRESHOLD_', 'MALLOC_TRIM_THRESHOLD_']:
        env.pop(key, None)
    if label == 'retained':
        env.update(MALLOC_MMAP_THRESHOLD_='33554432', MALLOC_TRIM_THRESHOLD_='67108864')
    cmd = [sys.executable, '-m','cpu.native_path_study', '--label', label, '--threads', str(threads),
           '--batch', str(batch), '--stage', stage, '--rounds', str(rounds)]
    name = f'{stage}_{label}_t{threads}_b{batch}'
    row = {'index': index, 'name': name, 'command': cmd, 'started_epoch': time.time()}
    status['jobs'].append(row)
    (out/'suite_status.json').write_text(json.dumps(status,indent=2))
    with (out/f'{name}.log').open('w') as log:
        completed = subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    row.update(returncode=completed.returncode, finished_epoch=time.time())
    (out/'suite_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(row),flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
status['complete'] = True
(out/'suite_status.json').write_text(json.dumps(status,indent=2))
