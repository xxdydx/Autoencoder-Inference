import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from autoencoder.measurement import snapshot
import torch

label=sys.argv[1]
def output(args):
    try:return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True).strip()
    except Exception as e:return repr(e)
result=dict(label=label,hostname=platform.node(),platform=platform.platform(),machine=platform.machine(),
            python=sys.version,executable=sys.executable,
            packages={p:importlib.metadata.version(p) for p in ('torch','numpy','onnx','onnxruntime','psutil','matplotlib')},
            torch_build=torch.__config__.show(),quantization_engines=torch.backends.quantized.supported_engines,
            resources=snapshot(),lscpu=output(['lscpu','--json']),topology=output(['lscpu','-e=CPU,CORE,SOCKET,NODE,CACHE']),
            compiler=output(['gcc','--version']),perf_path=shutil.which('perf'),
            perf_event_paranoid=Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),
            proc_self_cgroup=Path('/proc/self/cgroup').read_text(),
            torch_default_threads=torch.get_num_threads(),torch_default_interop_threads=torch.get_num_interop_threads())
Path(f'artifacts/{label}_environment.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ('label','hostname','packages','quantization_engines','perf_path')}),flush=True)
