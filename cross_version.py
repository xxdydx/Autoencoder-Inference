import json
from pathlib import Path
from measurement import configure_affinity
configure_affinity()
import numpy as np
import torch
from model import load_model
torch.set_num_threads(1)
rows=[]
with torch.inference_mode():
    for length in (120,720):
        data=np.load(f'artifacts/validation_reference_{length}.npz')
        x=torch.from_numpy(data['input'])
        ref=torch.from_numpy(data['reconstruction'])
        scores=torch.from_numpy(data['scores'])
        model,_=load_model()
        y=torch.cat([model(chunk) for chunk in x.split(4)])
        s=(y-x).square().mean((1,2))
        assert torch.allclose(s,scores,rtol=1e-4,atol=1e-5)
        rows.append(dict(torch_version=torch.__version__,length=length,windows=len(x),
                         reconstruction_max_abs=float((y-ref).abs().max()),
                         reconstruction_rmse=float((y-ref).square().mean().sqrt()),
                         score_max_abs=float((s-scores).abs().max()),
                         score_max_relative=float(((s-scores).abs()/scores.abs()).max()),
                         alert_agreement=float(((s>float(data['threshold']))==(scores>float(data['threshold']))).float().mean())))
Path('artifacts/cross_version_validation.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows),flush=True)
