"""Experimental fixed-tile and LayerNorm/FFN kernels; baseline stays unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from gpu.fused_ffn import _full_ffn


@triton.jit
def _norm_ffn(A, Gamma, Beta, W1, B1, W2, B2, C,
              M: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr,
              STEP: tl.constexpr, RESIDUAL: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    features = tl.arange(0, 128)
    hidden = tl.arange(0, STEP)
    raw = tl.load(A + rows[:, None] * 128 + features[None, :],
                  mask=rows[:, None] < M, other=0).to(tl.float32)
    mean = tl.sum(raw, 1) / 128
    centered = raw - mean[:, None]
    var = tl.sum(centered * centered, 1) / 128
    gamma = tl.load(Gamma + features).to(tl.float32)
    beta = tl.load(Beta + features).to(tl.float32)
    # Match LayerNorm's BF16 output boundary before the first GEMM.
    x = ((centered * tl.rsqrt(var[:, None] + EPS)) * gamma[None, :] + beta[None, :]).to(tl.bfloat16)
    output = tl.zeros((BM, 128), tl.float32)
    for block in range(512 // STEP):
        channels = block * STEP + hidden
        w1 = tl.load(W1 + features[:, None] * 512 + channels[None, :])
        b1 = tl.load(B1 + channels).to(tl.float32)
        z = (tl.dot(x, w1) + b1[None, :]).to(tl.bfloat16).to(tl.float32)
        activation = (0.5 * z * (1. + libdevice.erf(z * 0.7071067811865476))).to(tl.bfloat16)
        w2 = tl.load(W2 + channels[:, None] * 128 + features[None, :])
        output = tl.dot(activation, w2, output)
    output += tl.load(B2 + features).to(tl.float32)[None, :]
    # Preserve the separate BF16 FFN-output boundary before residual addition.
    rounded = output.to(tl.bfloat16)
    if RESIDUAL:
        rounded = (rounded.to(tl.float32) + raw).to(tl.bfloat16)
    tl.store(C + rows[:, None] * 128 + features[None, :], rounded,
             mask=rows[:, None] < M)


@torch.library.triton_op('cpu_ae::fixed_ffn_bf16', mutates_args={})
def fixed_ffn(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
              w2: torch.Tensor, b2: torch.Tensor,
              bm: int, step: int, warps: int) -> torch.Tensor:
    flat = x.reshape(-1, 128)
    out = torch.empty_like(flat)
    torch.library.wrap_triton(_full_ffn.fn)[(triton.cdiv(flat.shape[0], bm),)](
        flat, w1, b1, w2, b2, out, M=flat.shape[0], BM=bm, STEP=step,
        num_warps=warps, num_stages=1)
    return out.reshape(x.shape)


@torch.library.triton_op('cpu_ae::norm_ffn_bf16', mutates_args={})
def norm_ffn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
             w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor,
             eps: float, bm: int, step: int, warps: int, residual: bool) -> torch.Tensor:
    flat = x.reshape(-1, 128)
    out = torch.empty_like(flat)
    torch.library.wrap_triton(_norm_ffn)[(triton.cdiv(flat.shape[0], bm),)](
        flat, gamma, beta, w1, b1, w2, b2, out, M=flat.shape[0], EPS=eps,
        BM=bm, STEP=step, RESIDUAL=residual, num_warps=warps, num_stages=1,
        enable_fp_fusion=False)
    return out.reshape(x.shape)


class ExperimentalBlock(nn.Module):
    def __init__(self, block, variant, config):
        super().__init__()
        self.original = block
        self.variant = variant
        self.bm, self.step, self.warps = config
        self.register_buffer('packed_ff1', block.ff1.weight.detach().T.contiguous())
        self.register_buffer('packed_ff2', block.ff2.weight.detach().T.contiguous())

    def forward(self, x):
        block = self.original
        b, t, _ = x.shape
        q, k, v = block.qkv(block.norm1(x)).reshape(b, t, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        x = x + block.out(attention.transpose(1, 2).reshape(b, t, 128))
        if self.variant == 'fixed':
            return x + fixed_ffn(block.norm2(x), self.packed_ff1, block.ff1.bias,
                                 self.packed_ff2, block.ff2.bias, self.bm, self.step, self.warps)
        result = norm_ffn(x, block.norm2.weight, block.norm2.bias, self.packed_ff1,
                          block.ff1.bias, self.packed_ff2, block.ff2.bias, block.norm2.eps,
                          self.bm, self.step, self.warps, self.variant == 'norm_residual')
        return result if self.variant == 'norm_residual' else x + result


def replace_experimental(model, variant, config):
    for stack in (model.encoder, model.decoder):
        for index, block in enumerate(stack):
            stack[index] = ExperimentalBlock(block, variant, config)
    return model
