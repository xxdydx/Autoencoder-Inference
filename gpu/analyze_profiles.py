"""Render measured blog figures from Nsight Systems SQLite and timing JSON.

Requires numpy and matplotlib, but no GPU. Run from the repository root.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3

import numpy as np


def distribution(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x)>1 else None,
                median=float(np.median(x)), p99=float(np.percentile(x, 99)))


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def intersection_ns(left, right):
    a, b = merge(left), merge(right)
    total = i = j = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def category(name):
    name = name.lower()
    if 'flash_fwd' in name: return 'Attention'
    if 'layer_norm' in name: return 'Layer norm'
    if 'gelu' in name: return 'GELU'
    if '_full_ffn' in name: return 'Full FFN'
    if 'gemm' in name: return 'GEMM'
    return 'Other'


def read_trace(path):
    with sqlite3.connect(path) as c:
        c.row_factory = sqlite3.Row
        kernels = [dict(r) for r in c.execute('''
            SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start''')]
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies = [dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start')] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
        packs = [dict(r) for r in c.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'pack_batch_%' ORDER BY start")] if 'NVTX_EVENTS' in tables else []
        sm = c.execute('SELECT numMultiprocessors FROM TARGET_INFO_CUDA_DEVICE LIMIT 1').fetchone()[0]
    busy = merge((r['start'], r['end']) for r in kernels)
    h2d = [r for r in copies if r['copyKind'] == 1]
    copy_busy = merge((r['start'], r['end']) for r in h2d)
    copy_ns = sum(e-s for s,e in copy_busy)
    overlap = intersection_ns(busy, copy_busy)
    gaps_us = [(busy[i+1][0]-busy[i][1])/1000 for i in range(len(busy)-1)]
    summary = dict(source=str(path), sm_count=sm, kernel_count=len(kernels),
                   h2d_count=len(h2d), h2d_bytes=sum(r['bytes'] for r in h2d),
                   h2d_busy_ms=copy_ns/1e6, h2d_overlapping_kernels_ms=overlap/1e6,
                   h2d_overlap_percent=100*overlap/copy_ns if copy_ns else None,
                   kernel_busy_ms=sum(e-s for s,e in busy)/1e6,
                   kernel_span_ms=(busy[-1][1]-busy[0][0])/1e6,
                   kernel_gap_us=distribution(gaps_us) if gaps_us else None,
                   kernel_gaps_total_ms=sum(gaps_us)/1000,
                   kernel_gaps_over_10us=sum(v>10 for v in gaps_us))
    groups = {}
    for k in kernels:
        blocks = k['gridX']*k['gridY']*k['gridZ']
        threads = k['blockX']*k['blockY']*k['blockZ']
        shared = k['staticSharedMemory']+k['dynamicSharedMemory']
        key = (k['name'], blocks, threads, k['registersPerThread'], shared)
        groups.setdefault(key, []).append((k['end']-k['start'])/1000)
    rows = []
    for (name, blocks, threads, registers, shared), times in groups.items():
        rows.append(dict(trace=path.stem, kernel=name, category=category(name),
                         calls=len(times), total_ms=sum(times)/1000, mean_us=float(np.mean(times)),
                         blocks=blocks, blocks_per_sm=blocks/sm, threads_per_block=threads,
                         registers_per_thread=registers, shared_bytes_per_block=shared))
    return summary, sorted(rows, key=lambda r:-r['total_ms']), kernels, copies, packs


def historical(root):
    output = {}
    for study in ['gpu_fusion', 'tensorrt']:
        groups = {}
        for p in sorted((root/study/'paired').glob('round*/*/pipeline*.json')):
            d = json.loads(p.read_text())
            key = f"{d['kind']}_b{d['batch']}"
            groups.setdefault(key, dict(values=[], sources=[]))
            groups[key]['values'].extend(c['wall_s']*1000 for c in d['cycles'])
            groups[key]['sources'].append(str(p))
        output[study] = {k:dict(latency_ms=distribution(v['values']), sources=v['sources']) for k,v in groups.items()}
    allocators = {}
    for mode, kind in [('default','manual'), ('retained_arena','manual'), ('default','manual_prealloc')]:
        rows=[]
        for repeat in range(2):
            p=root/'deep'/f'attention_fp32_t8_b1_s720_alloc_{mode}_{kind}_r{repeat}.json'
            if not p.exists(): continue
            d=json.loads(p.read_text())[0]
            rows.append(dict(source=str(p), n=len(d['samples_ms']), median_ms=d['p50_ms'],
                             minor_faults=d['resource_delta']['minor_faults'],
                             minor_faults_per_call=d['resource_delta']['minor_faults']/len(d['samples_ms'])))
        allocators[f'{mode}_{kind}']=rows
    output['cpu_allocator_getrusage']=allocators
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('artifacts/profiler'))
    p.add_argument('--output',type=Path,default=Path('reports/profiler_evidence'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','figure.facecolor':'white'})
    def save(fig,name):
        fig.savefig(a.output/(name+'.svg'),bbox_inches='tight')
        fig.savefig(a.output/(name+'.png'),bbox_inches='tight',dpi=200)
        plt.close(fig)
    traces={};rows=[];raw={}
    for name in ['pageable_sync_b64','pinned_serial_b64','pinned_overlap_b64','pinned_overlap_b128',
                 'ffn_eager_b64','ffn_compiled_b64','ffn_fused_b64','ffn_eager_b128','ffn_fused_b128']:
        path=a.input/(name+'.sqlite')
        if path.exists():
            summary, kernels, *data=read_trace(path)
            traces[name]=summary;rows.extend(kernels);raw[name]=data
    if rows:
        with (a.output/'kernel_measurements.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Timing JSONs are from runs with no profiler attached.
    ffn={p.stem:json.loads(p.read_text()) for p in sorted(a.input.glob('timing_*.json'))}
    pipeline={}
    score_checks=[]
    for folder in sorted((a.input/'timings').glob('round*')):
        for file in sorted(folder.glob('pipeline*.json')):
            d=json.loads(file.read_text());key=f"{d['mode']}_b{d['batch']}"
            pipeline.setdefault(key,dict(wall_ms=[],batch_cuda_ms=[],sources=[]))
            pipeline[key]['wall_ms'].extend(c['wall_s']*1000 for c in d['cycles'])
            pipeline[key]['batch_cuda_ms'].extend(t for c in d['cycles'] for t in c['batch_cuda_ms'])
            pipeline[key]['sources'].append(str(file))
        paths=list(folder.glob('*scores_head137.npy'))
        if paths:
            reference=np.load(paths[0])
            for file in paths[1:]:
                actual=np.load(file);diff=np.abs(actual-reference)
                check=dict(reference=str(paths[0]),actual=str(file),max_abs=float(diff.max()),
                           within_tolerance=bool(np.allclose(actual,reference,rtol=1e-4,atol=1e-5)))
                if not check['within_tolerance']:raise RuntimeError(check)
                score_checks.append(check)
    pipeline={k:dict(latency_ms=distribution(v['wall_ms']),batch_cuda_ms=distribution(v['batch_cuda_ms']),sources=v['sources']) for k,v in pipeline.items()}
    result=dict(traces=traces,ffn_timings=ffn,pipeline_timings=pipeline,score_checks=score_checks,
                historical=historical(a.input.parent),
                physical_dram_bytes=None,achieved_bandwidth_percent=None,achieved_compute_percent=None,
                achieved_occupancy=None,roofline=None,
                counter_limitation='ERR_NVGPUCTRPERM; host driver restricts access; no CAP_SYS_ADMIN/CAP_PERFMON in container',
                algorithmic_intermediate_bytes=64*720*512*2,
                algorithmic_intermediate_read_write_bytes=4*64*720*512*2)
    (a.output/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    # Timeline drawn directly from Nsight Systems timestamps, not a schematic.
    if all(k in raw for k in ['pinned_serial_b64','pinned_overlap_b64']):
        for zoom in [False, True]:
            fig,axes=plt.subplots(2,1,figsize=(12,5.8),sharex=True,layout='constrained')
            for ax,name,title in zip(axes,['pinned_serial_b64','pinned_overlap_b64'],['Pinned serial (one input slot)','Pinned overlap (two input slots + copy stream)']):
                kernels,copies,packs=raw[name]
                origin=min(r['start'] for r in copies if r['copyKind']==1)
                for items,y,color in [(packs,2,'#a78bfa'),([r for r in copies if r['copyKind']==1],1,'#e99b31'),(kernels,0,'#148f8c')]:
                    bars=[((r['start']-origin)/1e6,(r['end']-r['start'])/1e6) for r in items]
                    ax.broken_barh(bars,(y-.3,.6),facecolors=color,linewidth=0)
                ax.set_yticks([0,1,2],['CUDA kernels','H2D copies','CPU packing'])
                ax.set_title(f"{title} — {traces[name]['h2d_overlap_percent']:.1f}% of H2D time overlaps kernels",loc='left',fontsize=11)
                ax.grid(axis='x',alpha=.2);ax.set_ylim(-.6,2.7)
            axes[-1].set_xlim(-.7,18 if zoom else max(traces[k]['kernel_span_ms'] for k in ['pinned_serial_b64','pinned_overlap_b64'])+2)
            axes[-1].set_xlabel('Milliseconds from first H2D copy in each capture')
            fig.suptitle('Measured copy/compute overlap · BF16, B64, 512 windows',fontsize=14)
            save(fig,'nsight_timeline_zoom' if zoom else 'nsight_timeline')
    # Per-kernel resource and duration evidence for the two batch sizes.
    wanted=['pinned_overlap_b64','pinned_overlap_b128']
    if all(k in traces for k in wanted):
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        bottoms=np.zeros(2)
        for cat,color in [('GEMM','#528dcf'),('Attention','#a678db'),('GELU','#e99b31'),('Layer norm','#148f8c'),('Other','#adb5bd')]:
            values=[sum(r['total_ms'] for r in rows if r['trace']==name and r['category']==cat) for name in wanted]
            ax.bar(['B64 · 8 batches','B128 · 4 batches'],values,bottom=bottoms,label=cat,color=color)
            bottoms+=values
        for i,v in enumerate(bottoms):ax.text(i,v+.7,f'{v:.2f} ms',ha='center')
        ax.set_ylim(0,max(bottoms)*1.17);ax.set_ylabel('Summed kernel duration per 512-window capture (ms)')
        ax.set_title('Larger batches spend more total time in GPU kernels')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False)
        save(fig,'batch_kernel_breakdown')
    if pipeline:
        names=['pageable_sync_b64','pinned_serial_b64','pinned_overlap_b64','pinned_overlap_b128']
        names=[n for n in names if n in pipeline]
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        means=[pipeline[n]['latency_ms']['mean'] for n in names]
        std=[pipeline[n]['latency_ms']['std'] for n in names]
        ax.bar(range(len(names)),means,yerr=std,color=['#aab5c5','#7295ba','#148f8c','#a678db'][:len(names)],capsize=4)
        ax.set_xticks(range(len(names)),[n.replace('_b','\nB').replace('_',' ') for n in names])
        for i,v in enumerate(means):ax.text(i,v+std[i]+1,f'{v:.2f} ms',ha='center')
        ax.set_ylim(0,max(means)*1.22);ax.set_ylabel('512-window wall latency (ms), mean ± sample SD')
        ax.set_title('Uninstrumented timing · five randomized rounds × three cycles')
        save(fig,'pipeline_latency')
    selected=[(n,d) for n,d in ffn.items() if d['arguments']['variant']=='fused' and d['arguments']['batch']==64]
    if selected:
        selected.sort(key=lambda pair:(pair[1]['arguments']['bm'],pair[1]['arguments']['step'],pair[1]['arguments']['warps']))
        fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
        labels=[f"BM={d['arguments']['bm']}\nH={d['arguments']['step']}, W={d['arguments']['warps']}" for _,d in selected]
        for ax,values,title in zip(axes,
            [[d['cuda_ms']['median'] for _,d in selected],[d['kernel']['registers_per_thread'] for _,d in selected],[d['kernel']['shared_bytes']/1024 for _,d in selected]],
            ['Median FFN device latency (ms)','Registers per thread','Shared memory per block (KiB)']):
            ax.bar(range(len(selected)),values,color='#148f8c')
            ax.set_xticks(range(len(selected)),labels,fontsize=8)
            ax.set_title(title,fontsize=11)
            for i,v in enumerate(values):ax.text(i,v,f'{v:.3f}' if title.startswith('Median') else f'{v:g}',ha='center',va='bottom',fontsize=9)
            ax.set_ylim(0,max(values)*1.16)
        fig.suptitle('Measured chunk/tile tradeoffs · B64, 20 warmups, 1,000 timing samples')
        save(fig,'ffn_resource_tradeoff')
    print(json.dumps(dict(traces=len(traces),kernel_groups=len(rows),timing_configs=len(ffn),pipeline_configs=len(pipeline),score_checks=len(score_checks))))


if __name__=='__main__':main()
