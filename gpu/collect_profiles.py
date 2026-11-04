"""Sequential blog evidence collection; run on an otherwise idle NVIDIA runner.

Nsight Compute needs host-authorized GPU counter access. Its failure is recorded
without preventing Systems captures and uninstrumented timing measurements.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nsys', required=True)
    p.add_argument('--ncu', required=True)
    p.add_argument('--output', type=Path, default=Path('artifacts/profiler'))
    p.add_argument('--skip-ncu', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    def run(name, command, required=True):
        row = dict(name=name, command=command, started_epoch=time.time())
        manifest.append(row)
        with (a.output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        row.update(returncode=result.returncode, elapsed_s=time.time()-row['started_epoch'])
        row['counter_access_denied'] = 'ERR_NVGPUCTRPERM' in (a.output / (name + '.log')).read_text()
        (a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if required and (result.returncode or row['counter_access_denied']):
            raise RuntimeError(f'{name} failed; see log')
    # Captures are separate processes from the latency study.
    for mode, batch in [('pageable_sync',64), ('pinned_serial',64), ('pinned_overlap',64), ('pinned_overlap',128)]:
        name = f'{mode}_b{batch}'
        run(name, [a.nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=true',
                   '-o', str(a.output/name), sys.executable, '-m', 'gpu.profile_pipeline',
                   '--mode', mode, '--batch', str(batch), '--capture', '--output', str(a.output/name)])
        run(name+'_export', [a.nsys, 'export', '--type=sqlite', '--force-overwrite=true',
                            '--output', str(a.output/(name+'.sqlite')), str(a.output/(name+'.nsys-rep'))])
    for variant, batch in [('eager',64), ('compiled',64), ('fused',64), ('eager',128), ('fused',128)]:
        name = f'ffn_{variant}_b{batch}'
        target = [sys.executable, '-m', 'gpu.profile_ffn', '--variant', variant, '--batch', str(batch)]
        run(name, [a.nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=true',
                   '-o', str(a.output/name), *target, '--capture', '--output', str(a.output/(name+'.json'))])
        run(name+'_export', [a.nsys, 'export', '--type=sqlite', '--force-overwrite=true',
                            '--output', str(a.output/(name+'.sqlite')), str(a.output/(name+'.nsys-rep'))])
    # Randomized order, all timings without an attached profiler.
    configs = [('eager',64,64,128,8), ('compiled',64,64,128,8), ('fused',64,64,128,8),
               ('eager',128,64,128,8), ('fused',128,64,128,8),
               ('fused',64,16,128,4), ('fused',64,32,128,4), ('fused',64,64,128,4), ('fused',64,32,64,4)]
    random.Random(20261104).shuffle(configs)
    for variant, batch, bm, step, warps in configs:
        name = f'timing_{variant}_b{batch}_bm{bm}_step{step}_w{warps}'
        run(name, [sys.executable, '-m', 'gpu.profile_ffn', '--variant', variant, '--batch', str(batch),
                   '--bm', str(bm), '--step', str(step), '--warps', str(warps), '--samples', '1000',
                   '--output', str(a.output/(name+'.json'))])
    # A warm, matched serial control separates buffering from overlap.
    run('pipeline_timings', [sys.executable, '-m', 'gpu.profile_pipeline', '--timing-study',
                             '--output', str(a.output/'timings')])
    if not a.skip_ncu:
        for variant in ['eager', 'compiled', 'fused']:
            name = f'ncu_{variant}_b64'
            run(name, [a.ncu, '--profile-from-start', 'off', '--replay-mode', 'application',
                       '--clock-control', 'none', '--cache-control', 'none',
                       '--section', 'SpeedOfLight', '--section', 'LaunchStats', '--section', 'Occupancy',
                       '--section', 'MemoryWorkloadAnalysis', '--section', 'SpeedOfLight_RooflineChart',
                       '--section', 'SpeedOfLight_HierarchicalTensorRooflineChart',
                       '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum',
                       '-o', str(a.output/name), sys.executable, '-m', 'gpu.profile_ffn',
                       '--variant', variant, '--capture', '--output', str(a.output/(name+'.json'))], required=False)


if __name__ == '__main__':
    main()
