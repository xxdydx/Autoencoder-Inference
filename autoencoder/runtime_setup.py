"""Keep PyTorch 2.4's generated CPU code on its bundled OpenMP runtime on macOS.

Re-exec before numerical imports: setting a dyld variable after startup is too
late, and shell shims can discard it. No global environment/files are changed.
"""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys


def configure_openmp(compiling):
    if (not compiling or sys.platform!='darwin' or
            not importlib.metadata.version('torch').startswith('2.4.') or
            os.environ.get('CPU_AE_OMP_CONFIGURED')):
        return
    package=Path(importlib.util.find_spec('torch').origin).parent
    env=os.environ.copy()
    env['DYLD_LIBRARY_PATH']=str(package/'lib')
    env['CPU_AE_OMP_CONFIGURED']='1'
    spec = getattr(sys.modules['__main__'], '__spec__', None)
    args = ['-m', spec.name, *sys.argv[1:]] if spec else sys.argv
    os.execve(sys.executable, [sys.executable, *args], env)
