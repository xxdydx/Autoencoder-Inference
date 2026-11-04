import json
import time
from measurement import configure_affinity
configure_affinity()
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from model import load_model, Scorer

torch.set_num_threads(1)
model, _ = load_model('manual')
with torch.inference_mode():
    torch.onnx.export(Scorer(model), torch.randn(1,720,32), 'artifacts/model.onnx',
                      input_names=['input'], output_names=['score'], opset_version=17,
                      dynamo=False,
                      dynamic_axes={'input': {0:'batch',1:'time'}, 'score':{0:'batch'}})
onnx.checker.check_model(onnx.load('artifacts/model.onnx'))
quantize_dynamic('artifacts/model.onnx','artifacts/model_int8.onnx',
                 weight_type=QuantType.QInt8, per_channel=True)
from collections import Counter
out = {}
for name in ('model','model_int8'):
    graph=onnx.load(f'artifacts/{name}.onnx')
    out[name]=dict(nodes=len(graph.graph.node), operators=dict(Counter(n.op_type for n in graph.graph.node)))
open('artifacts/export.json','w').write(json.dumps(out,indent=2))
print(json.dumps(out))
