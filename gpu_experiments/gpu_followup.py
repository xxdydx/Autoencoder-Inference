"""Correct graph weight/output ownership and independently confirm controls."""
from pathlib import Path
import shutil,os,traceback
import torch
from gpu_suite import A,PAUSE,scorer,validation,pipeline,save

diagnostics=A/'diagnostics';diagnostics.mkdir(exist_ok=True)
for p in list(A.glob('*graph*'))+list(A.glob('*compile*')):
    if p.is_file():shutil.copy2(p,diagnostics/p.name)
PAUSE.write_text(str(os.getpid()))
try:
    with torch.inference_mode():
        for dtype,batch,mode in [('fp32',8,'pinned_overlap'),('bf16',64,'gpu_normalize')]:
            for prefix in ('graph','compile'):
                kind=prefix+'_'+dtype
                try:
                    s,c=scorer(kind.replace('graph','sdpa'))
                    save('validation_'+kind,validation(s,kind));del s
                    result=pipeline(kind,batch,mode,3000,5,profile=True)
                    if prefix=='graph':
                        import numpy as np
                        stem=f'{mode}_b{batch}_n3000_scores_head137.npy'
                        actual=np.load(A/f'graph_{dtype}_{stem}');expected=np.load(A/f'sdpa_{dtype}_{stem}')
                        assert np.allclose(actual,expected,rtol=1e-4,atol=1e-5)
                    (A/('error_extra_'+kind+'.json')).unlink(missing_ok=True)
                    print('REPAIRED',kind,result['latency_s'],flush=True)
                except Exception as e:
                    save('error_extra_'+kind,dict(error=repr(e),traceback=traceback.format_exc()))
        (A/'FOLLOWUP_COMPLETE').write_text('Graph ownership and compiled-score controls rerun\n')
finally:PAUSE.unlink(missing_ok=True)
