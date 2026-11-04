"""Portable resource units and Linux quota/pressure evidence for CPU runs."""
import os
from pathlib import Path
import platform
import resource
import time


def configure_affinity():
    value=os.environ.get('CPU_AE_CPUS')
    if value and hasattr(os,'sched_setaffinity'):
        cpus=set()
        for part in value.split(','):
            bounds=list(map(int,part.split('-')))
            cpus.update(range(bounds[0],bounds[-1]+1))
        os.sched_setaffinity(0,cpus)


def peak_rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system()=='Darwin' else value*1024


def read_text(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def snapshot():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    out={'monotonic_s':time.perf_counter(),'loadavg':os.getloadavg(),
         'minor_faults':usage.ru_minflt,'major_faults':usage.ru_majflt,
         'process_user_s':usage.ru_utime,'process_system_s':usage.ru_stime,
         'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}
    stat=read_text('/sys/fs/cgroup/cpu.stat')
    out['cgroup_cpu_stat']={k:int(v) for k,v in (line.split() for line in stat.splitlines())} if stat else {}
    out['cpu_max']=read_text('/sys/fs/cgroup/cpu.max')
    out['memory_max']=read_text('/sys/fs/cgroup/memory.max')
    out['memory_current']=read_text('/sys/fs/cgroup/memory.current')
    out['cpu_pressure']=read_text('/sys/fs/cgroup/cpu.pressure')
    # Linux process status counts only the main thread. Sum all live tasks too.
    counts={'voluntary_ctxt_switches':0,'nonvoluntary_ctxt_switches':0}
    for path in Path('/proc/self/task').glob('*/status'):
        text=read_text(path)
        if not text:continue
        for line in text.splitlines():
            key,_,value=line.partition(':')
            if key in counts:counts[key]+=int(value.strip())
    out['all_thread_context_switches']=counts
    return out


def delta(before,after):
    a,b=before['cgroup_cpu_stat'],after['cgroup_cpu_stat']
    return {'wall_s':after['monotonic_s']-before['monotonic_s'],
            'minor_faults':after.get('minor_faults',0)-before.get('minor_faults',0),
            'major_faults':after.get('major_faults',0)-before.get('major_faults',0),
            'process_user_s':after.get('process_user_s',0)-before.get('process_user_s',0),
            'process_system_s':after.get('process_system_s',0)-before.get('process_system_s',0),
            'cgroup_cpu_stat':{k:b[k]-a.get(k,0) for k in b},
            'all_thread_context_switches':{k:after['all_thread_context_switches'][k]-v
                                          for k,v in before['all_thread_context_switches'].items()}}


def quantization_engine():
    import torch
    return 'x86' if 'x86' in torch.backends.quantized.supported_engines else (
        'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack')
