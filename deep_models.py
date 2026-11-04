"""Semantics-preserving controls and normalization folding for CPU experiments."""
import copy
import torch
from torch import nn


def stock_transformer_copy(model):
    """Use stock TransformerEncoderLayer, with exactly the existing parameters."""
    model=copy.deepcopy(model)
    for stack in (model.encoder,model.decoder):
        for index,old in enumerate(stack):
            layer=nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',
                                             batch_first=True,norm_first=True)
            layer.self_attn.in_proj_weight=old.qkv.weight
            layer.self_attn.in_proj_bias=old.qkv.bias
            layer.self_attn.out_proj=old.out
            layer.linear1=old.ff1;layer.linear2=old.ff2
            layer.norm1=old.norm1;layer.norm2=old.norm2
            stack[index]=layer
    return model.eval()


class FloatScore(nn.Module):
    """Reduced-precision model, FP32 error against the original FP32 input."""
    def __init__(self,model,dtype=torch.bfloat16):
        super().__init__();self.model=model.to(dtype);self.dtype=dtype

    def forward(self,x):
        return (self.model(x.to(self.dtype)).float()-x.float()).square().mean((1,2))


class FoldedScore(nn.Module):
    def __init__(self,model,mean,std):
        super().__init__()
        self.model=copy.deepcopy(model)
        self.register_buffer('scale',1/std)
        with torch.no_grad():
            weight=self.model.input.weight.clone()
            self.model.input.weight.copy_(weight*self.scale)
            self.model.input.bias.add_(weight@(-mean*self.scale))
            self.model.output.weight.mul_(std[:,None])
            self.model.output.bias.mul_(std).add_(mean)

    def forward(self,raw):
        # Output is in raw feature units; weighted error equals normalized MSE.
        return ((self.model(raw)-raw)*self.scale).square().mean((1,2))
