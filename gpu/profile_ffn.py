"""Warmed FFN capture target for Nsight Compute; timings run without profilers.

Run from the repository root: python -m gpu.profile_ffn --help.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['eager', 'compiled', 'fused'], default='fused')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--bm', type=int, choices=[16, 32, 64], default=64)
    p.add_argument('--step', type=int, choices=[64, 128], default=128)
    p.add_argument('--warps', type=int, choices=[4, 8], default=8)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--capture', action='store_true', help='One call inside cudaProfilerStart/Stop; excludes warmup')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.warmup < 1 or a.samples < 2 or a.batch < 1:
        p.error('warmup/batch must be positive; samples must be at least two')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    import numpy as np
    import torch
    import triton
    from torch.nn import functional as F
    from autoencoder.model import load_model
    from gpu.fused_ffn import _full_ffn

    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        model, _ = load_model()
        model = model.cuda().bfloat16()
        block = model.encoder[0]
        original = torch.from_numpy(np.load('artifacts/validation_reference_720.npz')['input'])
        data = original.repeat(((a.batch + len(original) - 1) // len(original), 1, 1))[:a.batch].cuda().bfloat16()
        x = model.input(data) + model.pe
        q, k, v = block.qkv(block.norm1(x)).reshape(a.batch, 720, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind(0)
        x = block.norm2(x + block.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(a.batch, 720, 128)))
        eager = lambda: block.ff2(F.gelu(block.ff1(x), approximate='none'))
        w1 = block.ff1.weight.T.contiguous()
        w2 = block.ff2.weight.T.contiguous()
        output = torch.empty_like(x)
        compiled_kernel = None
        def fused():
            nonlocal compiled_kernel
            compiled_kernel = _full_ffn.fn[(triton.cdiv(a.batch * 720, a.bm),)](
                x, w1, block.ff1.bias, w2, block.ff2.bias, output,
                M=a.batch * 720, BM=a.bm, STEP=a.step, num_warps=a.warps, num_stages=1)
            return output
        # Fixed configurations make chunk-size comparisons reproducible and keep
        # autotuning kernels out of the capture. Output allocation is outside all
        # fused calls; eager/compiled retain their ordinary output allocation.
        fn = eager if a.variant == 'eager' else fused
        if a.variant == 'compiled':
            fn = torch.compile(eager, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        expected = eager()
        for _ in range(a.warmup):
            actual = fn()
        torch.cuda.synchronize()
        diff = (actual.float() - expected.float()).abs()
        nrmse = (diff.square().mean().sqrt() / expected.float().square().mean().sqrt()).item()
        if nrmse >= .01:
            raise RuntimeError(f'FFN normalized RMSE {nrmse} exceeds 0.01')
        samples = []
        if a.capture:
            torch.cuda.profiler.start()
            with torch.cuda.nvtx.range(f'ffn_{a.variant}_b{a.batch}'):
                fn()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        else:
            for _ in range(a.samples):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
        props = torch.cuda.get_device_properties(0)
        metadata = None
        if compiled_kernel is not None:
            metadata = dict(registers_per_thread=compiled_kernel.n_regs,
                            spills=compiled_kernel.n_spills,
                            shared_bytes=compiled_kernel.metadata.shared,
                            grid_blocks=triton.cdiv(a.batch * 720, a.bm),
                            threads_per_block=a.warps * 32)
        record = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                      torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                      gpu=str(props), sm_count=props.multi_processor_count,
                      checkpoint_sha256=hashlib.sha256(Path('artifacts/model.pt').read_bytes()).hexdigest(),
                      source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                     for name in ['gpu/profile_ffn.py', 'gpu/fused_ffn.py', 'autoencoder/model.py']},
                      warmup_calls=a.warmup, normalized_rmse=nrmse, max_abs_error=diff.max().item(),
                      kernel=metadata, cuda_samples_ms=samples,
                      gpu_state=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,power.limit,clocks.sm,clocks.mem', '--format=csv'], text=True))
        if samples:
            record['cuda_ms'] = dict(n=len(samples), mean=statistics.mean(samples),
                                     std=statistics.stdev(samples), median=statistics.median(samples),
                                     p99=float(np.percentile(samples, 99)))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record[k] for k in ['warmup_calls', 'normalized_rmse', 'kernel', 'cuda_ms'] if k in record}), flush=True)


if __name__ == '__main__':
    main()
