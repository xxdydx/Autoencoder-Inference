"""Inference-only BF16 GEMM + bias + exact-erf GELU in one Triton kernel."""
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

CONFIGS=[triton.Config(dict(BM=m,BN=n,BK=k,GROUP=4),num_warps=w,num_stages=s)
         for m,n,k,w,s in [(32,64,32,4,2),(32,128,32,4,2),(64,64,32,4,2),
                           (64,128,32,4,2),(128,64,32,4,2),(64,64,64,4,2),
                           (64,128,64,4,2),(128,64,64,4,2),(32,128,128,4,1),
                           (64,64,128,4,1),(64,128,128,8,1),(128,64,128,8,1)]]

@triton.autotune(configs=CONFIGS,key=['M','N','K'])
@triton.jit
def _gemm_bias_gelu(A,B,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP:tl.constexpr):
    pid=tl.program_id(0)
    grid_m=tl.cdiv(M,BM);grid_n=tl.cdiv(N,BN)
    group_id=pid//(GROUP*grid_n);first_m=group_id*GROUP
    group_m=tl.minimum(grid_m-first_m,GROUP)
    pm=first_m+(pid%(GROUP*grid_n))%group_m
    pn=(pid%(GROUP*grid_n))//group_m
    rows=pm*BM+tl.arange(0,BM);cols=pn*BN+tl.arange(0,BN)
    ks=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+ks
        av=tl.load(A+rows[:,None]*K+kk[None,:],mask=(rows[:,None]<M)&(kk[None,:]<K),other=0)
        bv=tl.load(B+kk[:,None]*N+cols[None,:],mask=(kk[:,None]<K)&(cols[None,:]<N),other=0)
        acc=tl.dot(av,bv,acc)
    bias=tl.load(Bias+cols,mask=cols<N,other=0).to(tl.float32)
    # Match the BF16 Linear -> GELU boundary without a global-memory round trip.
    pre=(acc+bias[None,:]).to(tl.bfloat16).to(tl.float32)
    value=0.5*pre*(1.0+libdevice.erf(pre*0.7071067811865476))
    tl.store(C+rows[:,None]*N+cols[None,:],value.to(tl.bfloat16),mask=(rows[:,None]<M)&(cols[None,:]<N))

@torch.library.triton_op('cpu_ae::linear_gelu_bf16',mutates_args={})
def linear_gelu(x:torch.Tensor,packed:torch.Tensor,bias:torch.Tensor)->torch.Tensor:
    flat=x.reshape(-1,128)
    out=torch.empty((flat.shape[0],512),device=x.device,dtype=x.dtype)
    grid=lambda meta:(triton.cdiv(flat.shape[0],meta['BM'])*triton.cdiv(512,meta['BN']),)
    torch.library.wrap_triton(_gemm_bias_gelu)[grid](flat,packed,bias,out,M=flat.shape[0],N=512,K=128)
    return out.reshape(*x.shape[:-1],512)

@triton.autotune(configs=[triton.Config(dict(BM=m,STEP=h),num_warps=w,num_stages=1)
                         for m,h,w in [(16,128,4),(32,128,4),(64,128,4),(64,128,8),(32,64,4)]],key=['M'])
@triton.jit
def _full_ffn(A,W1,B1,W2,B2,C,M:tl.constexpr,BM:tl.constexpr,STEP:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    features=tl.arange(0,128);hidden=tl.arange(0,STEP)
    x=tl.load(A+rows[:,None]*128+features[None,:],mask=rows[:,None]<M,other=0)
    output=tl.zeros((BM,128),tl.float32)
    for block in range(512//STEP):
        channels=block*STEP+hidden
        w1=tl.load(W1+features[:,None]*512+channels[None,:])
        b1=tl.load(B1+channels).to(tl.float32)
        z=(tl.dot(x,w1)+b1[None,:]).to(tl.bfloat16).to(tl.float32)
        activation=(0.5*z*(1.+libdevice.erf(z*0.7071067811865476))).to(tl.bfloat16)
        w2=tl.load(W2+channels[:,None]*128+features[None,:])
        output=tl.dot(activation,w2,output)
    output+=tl.load(B2+features).to(tl.float32)[None,:]
    tl.store(C+rows[:,None]*128+features[None,:],output.to(tl.bfloat16),mask=rows[:,None]<M)

@torch.library.triton_op('cpu_ae::full_ffn_bf16',mutates_args={})
def full_ffn(x:torch.Tensor,w1:torch.Tensor,b1:torch.Tensor,w2:torch.Tensor,b2:torch.Tensor)->torch.Tensor:
    flat=x.reshape(-1,128);out=torch.empty_like(flat)
    grid=lambda meta:(triton.cdiv(flat.shape[0],meta['BM']),)
    torch.library.wrap_triton(_full_ffn)[grid](flat,w1,b1,w2,b2,out,M=flat.shape[0])
    return out.reshape(x.shape)

class FusedBlock(nn.Module):
    def __init__(self,block,full=False):
        super().__init__();self.original=block
        self.full=full
        self.register_buffer('packed_ff1',block.ff1.weight.detach().T.contiguous())
        if full:self.register_buffer('packed_ff2',block.ff2.weight.detach().T.contiguous())
    def forward(self,x):
        block=self.original;b,t,_=x.shape
        q,k,v=block.qkv(block.norm1(x)).reshape(b,t,3,4,32).permute(2,0,3,1,4).unbind(0)
        attention=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        x=x+block.out(attention.transpose(1,2).reshape(b,t,128))
        if self.full:return x+full_ffn(block.norm2(x),self.packed_ff1,block.ff1.bias,self.packed_ff2,block.ff2.bias)
        hidden=linear_gelu(block.norm2(x),self.packed_ff1,block.ff1.bias)
        return x+block.ff2(hidden)

def replace_ffn(model,full=False):
    for stack in (model.encoder,model.decoder):
        for index,block in enumerate(stack):stack[index]=FusedBlock(block,full=full)
    assert sum(p.numel() for p in model.parameters())==1003952
    return model
