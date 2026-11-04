"""Numerical checks on held-out synthetic windows, including injected perturbations.

Alert agreement here is numerical smoke testing, not measured detector quality.
"""
import json
import argparse
from pathlib import Path
import torch
import numpy as np
import onnxruntime as ort
from measurement import configure_affinity, quantization_engine
configure_affinity()
from model import load_model, synthetic

parser=argparse.ArgumentParser()
parser.add_argument('--label',default='')
args=parser.parse_args()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rows=[]
for length in (120,720):
    reference,c=load_model('manual')
    # Calibrate on a distinct seed; evaluate normal/perturbed held-out windows.
    calibration=(synthetic(64,length,900)-c['mean'])/c['std']
    x=(synthetic(64,length,901)-c['mean'])/c['std']
    x[32:,-10:,:4]+=4
    with torch.inference_mode():
        def reconstruct(model,data):
            return torch.cat([model(chunk) for chunk in data.split(4)])
        cal=(reconstruct(reference,calibration)-calibration).square().mean((1,2))
        threshold=float(torch.quantile(cal,.95))
        ref=reconstruct(reference,x)
        scores=(ref-x).square().mean((1,2))
        if not args.label:
            np.savez(f'artifacts/validation_reference_{length}.npz',input=x.numpy(),
                     reconstruction=ref.numpy(),scores=scores.numpy(),threshold=threshold)
        for name in ('sdpa','tanh','int8','fp16','bf16','ort','ort_int8'):
            m,_=load_model()
            recon=None
            if name.startswith('ort'):
                opts=ort.SessionOptions(); opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
                opts.add_session_config_entry('session.intra_op.allow_spinning','0')
                path='artifacts/model_int8.onnx' if name=='ort_int8' else 'artifacts/model.onnx'
                session=ort.InferenceSession(path,opts,providers=['CPUExecutionProvider'])
                s=torch.from_numpy(np.concatenate([session.run(None,{'input':v.numpy()})[0] for v in x.split(4)]))
            else:
                if name=='tanh':
                    for block in list(m.encoder)+list(m.decoder):block.gelu_approx='tanh'
                if name=='int8':
                    torch.backends.quantized.engine=quantization_engine()
                    m=torch.ao.quantization.quantize_dynamic(m,{torch.nn.Linear},dtype=torch.qint8)
                data=x
                if name in ('fp16','bf16'):
                    dtype=torch.float16 if name=='fp16' else torch.bfloat16
                    m=m.to(dtype);data=x.to(dtype)
                recon=reconstruct(m,data).float()
                # Quantify the actual low-precision scoring path separately from FP32 reference.
                s=(recon.to(data.dtype)-data).square().mean((1,2)).float()
            row=dict(backend=name,torch_version=torch.__version__,length=length,windows=64,threshold=threshold,
                     max_score_abs_error=float((s-scores).abs().max()),
                     max_score_relative_error=float(((s-scores).abs()/scores.abs()).max()),
                     alert_agreement=float(((s>threshold)==(scores>threshold)).float().mean()),
                     reference_alert_count=int((scores>threshold).sum()),candidate_alert_count=int((s>threshold).sum()))
            if recon is not None:
                row['reconstruction_max_abs_error']=float((recon-ref).abs().max())
                row['reconstruction_rmse']=float((recon-ref).square().mean().sqrt())
            if name in ('sdpa','ort'):
                assert torch.allclose(s,scores,rtol=1e-4,atol=1e-5), row
            rows.append(row)
name=(args.label+'_validation.json') if args.label else 'validation.json'
Path('artifacts',name).write_text(json.dumps(rows,indent=2))
print(json.dumps(rows),flush=True)
