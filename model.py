"""Exact report architecture; explicit attention enables controlled kernel ablations."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, attention="sdpa"):
        super().__init__()
        self.attention = attention
        self.gelu_approx = 'none'
        self.contiguous_qkv = False
        self.norm1 = nn.LayerNorm(128)
        self.qkv = nn.Linear(128, 384)
        self.out = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.ff1 = nn.Linear(128, 512)
        self.ff2 = nn.Linear(512, 128)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind(0)
        if self.contiguous_qkv:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if self.attention == "manual":
            a = ((q @ k.transpose(-2, -1)) * (32 ** -0.5)).softmax(-1) @ v
        else:
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.out(a.transpose(1, 2).reshape(b, t, 128))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate=self.gelu_approx))


class Autoencoder(nn.Module):
    def __init__(self, attention="sdpa"):
        super().__init__()
        self.input = nn.Linear(32, 128)
        self.encoder = nn.Sequential(*(Block(attention) for _ in range(3)))
        self.bottleneck = nn.Linear(128, 16)
        self.decoder_input = nn.Linear(16, 128)
        self.decoder = nn.Sequential(*(Block(attention) for _ in range(2)))
        self.output = nn.Linear(128, 32)
        position = torch.arange(720).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000) / 128))
        pe = torch.zeros(720, 128)
        pe[:, 0::2], pe[:, 1::2] = (position * frequency).sin(), (position * frequency).cos()
        self.register_buffer("pe", pe)

    def forward(self, x):
        h = self.encoder(self.input(x) + self.pe[:x.shape[1]])
        h = self.decoder(self.decoder_input(self.bottleneck(h)) + self.pe[:x.shape[1]])
        return self.output(h)


class Scorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return (self.model(x) - x).square().mean(dim=(1, 2))


def synthetic(n, t, seed=123):
    """Correlated smooth signals with feature-dependent scales; no detection claim."""
    g = torch.Generator().manual_seed(seed)
    time = torch.arange(t).view(1, t, 1)
    freq = torch.rand(n, 1, 32, generator=g) * 0.08 + 0.01
    phase = torch.rand(n, 1, 32, generator=g) * 2 * math.pi
    signal = (time * freq + phase).sin() + 0.2 * torch.randn(n, t, 32, generator=g)
    return signal * torch.linspace(0.5, 2, 32) + torch.linspace(-1, 1, 32)


def load_model(attention="sdpa"):
    checkpoint = torch.load("artifacts/model.pt", map_location="cpu", weights_only=True)
    model = Autoencoder(attention)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
