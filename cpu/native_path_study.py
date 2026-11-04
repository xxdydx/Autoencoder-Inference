"""Controlled native-transformer versus direct-SDPA CPU experiments.

Run on a remote CPU only. Timing, profiling and numerical checks are separate.
No compilation, architecture changes or checkpoint changes.
"""
import argparse
import os
p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['micro', 'burst'], default='micro')
p.add_argument('--threads', type=int, default=1)
p.add_argument('--batch', type=int, default=1)
p.add_argument('--rounds', type=int, default=7)
p.add_argument('--calls', type=int, default=5)
p.add_argument('--devices', type=int, default=3000)
p.add_argument('--label', default='default')
a = p.parse_args()
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = str(a.threads)
import copy
import json
import math
import platform
import random
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from autoencoder.deep_models import stock_transformer_copy
from autoencoder.model import Scorer, load_model, synthetic
from autoencoder.measurement import configure_affinity, snapshot, delta

configure_affinity()
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(123)
out = Path('artifacts/native_path')
out.mkdir(parents=True, exist_ok=True)
tag = f'{a.stage}_{a.label}_t{a.threads}_b{a.batch}'
base, ckpt = load_model()
x = (synthetic(a.batch, 720, 789) - ckpt['mean']) / ckpt['std']


def native_attention(block, z):
    return torch._native_multi_head_attention(
        z, z, z, 128, 4, block.qkv.weight, block.qkv.bias,
        block.out.weight, block.out.bias, None, False, True, None)[0]


def direct_attention(block, z):
    b, t, _ = z.shape
    q, k, v = block.qkv(z).reshape(b, t, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind(0)
    if block.contiguous_qkv:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    return block.out(h.transpose(1, 2).reshape(b, t, 128))


def native_forward(self, value):
    value = value + native_attention(self, self.norm1(value))
    return value + self.ff2(F.gelu(self.ff1(self.norm2(value)), approximate='none'))


models = {'direct_flash': copy.deepcopy(base), 'direct_math': copy.deepcopy(base),
          'direct_contiguous': copy.deepcopy(base), 'custom_native_attention': copy.deepcopy(base),
          'stock_auto': stock_transformer_copy(base),
          'stock_no_fast_flash': stock_transformer_copy(base),
          'stock_no_fast_math': stock_transformer_copy(base)}
for block in list(models['direct_contiguous'].encoder) + list(models['direct_contiguous'].decoder):
    block.contiguous_qkv = True
for block in list(models['custom_native_attention'].encoder) + list(models['custom_native_attention'].decoder):
    block.forward = types.MethodType(native_forward, block)
scorers = {name: Scorer(model).eval() for name, model in models.items()}


@contextmanager
def execution(name):
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(not name.startswith('stock_no_fast'))
    backend = SDPBackend.MATH if name.endswith('math') else SDPBackend.FLASH_ATTENTION
    try:
        with sdpa_kernel(backend):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(old)


result = {'stage': a.stage, 'args': vars(a), 'torch_version': torch.__version__,
          'torch_git': torch.version.git_version, 'platform': platform.platform(),
          'parallel_info': torch.__config__.parallel_info(),
          'allocator': {k: os.environ.get(k) for k in ['MALLOC_MMAP_THRESHOLD_', 'MALLOC_TRIM_THRESHOLD_']},
          'before': snapshot(), 'validation': {}, 'samples': [], 'profiles': {}}


def save():
    result['after'] = snapshot()
    path = out / f'{tag}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(path)


def time_group(fn, calls):
    before = snapshot()
    values = []
    for _ in range(calls):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return values, delta(before, snapshot())


def profile(name, fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                               record_shapes=True, profile_memory=True) as prof:
        for _ in range(3):
            fn()
    path = out / f'{tag}_{name}_trace.json'
    prof.export_chrome_trace(str(path))
    return {'trace': str(path), 'calls': 3,
            'operators': [{'key': e.key, 'calls': e.count, 'self_cpu_us': e.self_cpu_time_total,
                           'cpu_us': e.cpu_time_total, 'self_memory_bytes': e.self_cpu_memory_usage}
                          for e in prof.key_averages()]}


with torch.inference_mode():
    # All variants see the same checkpoint, input and scoring operation.
    vx = (synthetic(8, 720, 987) - ckpt['mean']) / ckpt['std']
    with execution('direct_flash'):
        reference = base(vx)
        score_ref = (reference - vx).square().mean((1, 2))
    for name, model in models.items():
        with execution(name):
            reconstruction = model(vx)
            score = (reconstruction - vx).square().mean((1, 2))
            assert torch.allclose(score, score_ref, rtol=1e-4, atol=1e-5), name
            result['validation'][name] = {
                'reconstruction_max_abs': (reconstruction-reference).abs().max().item(),
                'reconstruction_rmse': (reconstruction-reference).square().mean().sqrt().item(),
                'score_max_abs': (score-score_ref).abs().max().item(),
                'score_max_relative': ((score-score_ref).abs()/score_ref.abs().clamp_min(1e-12)).max().item()}
    save()
    if a.stage == 'micro':
        for name, scorer in scorers.items():
            with execution(name):
                for _ in range(4):
                    scorer(x)
        generator = random.Random(1122)
        for round_id in range(a.rounds):
            order = list(scorers)
            generator.shuffle(order)
            for name in order:
                with execution(name):
                    values, resources = time_group(lambda: scorers[name](x), a.calls)
                result['samples'].append({'scope': 'model', 'round': round_id, 'candidate': name,
                                          'latency_ms': values, 'resources': resources})
            save()
            print(json.dumps({'tag': tag, 'model_round': round_id}), flush=True)
        # Isolate the projected attention sublayer on actual first-block inputs.
        block = base.encoder[0]
        z = block.norm1(base.input(x) + base.pe[:720]).contiguous()
        unit = {'native': lambda: native_attention(block, z),
                'flash': lambda: direct_attention(block, z),
                'math': lambda: direct_attention(block, z)}
        with execution('flash'):
            unit_ref = unit['flash']()
        for name, fn in unit.items():
            with execution(name):
                y = fn()
                assert torch.allclose(y, unit_ref, rtol=1e-4, atol=1e-5), name
                for _ in range(4):
                    fn()
        for round_id in range(a.rounds):
            order = list(unit)
            generator.shuffle(order)
            for name in order:
                with execution(name):
                    values, resources = time_group(unit[name], a.calls*2)
                result['samples'].append({'scope': 'projected_attention', 'round': round_id,
                                          'candidate': name, 'latency_ms': values, 'resources': resources})
            save()
        for name, scorer in scorers.items():
            with execution(name):
                result['profiles'][name] = profile(name, lambda: scorer(x))
        save()
    else:
        assert a.threads == 1, 'Burst isolates two single-thread workers'
        available = sorted(os.sched_getaffinity(0))[:2]
        assert len(available) == 2
        rings = np.random.default_rng(567).standard_normal((a.devices,720,32), dtype=np.float32)
        mean = ckpt['mean'].numpy()
        scale = 1 / ckpt['std'].numpy()
        offset = -mean*scale
        scores = np.empty(a.devices, dtype=np.float32)
        splits = np.array_split(np.arange(a.devices), 2)
        candidates = ['stock_auto', 'stock_no_fast_flash', 'direct_flash', 'custom_native_attention']
        def task(name, worker_id, head):
            os.sched_setaffinity(0, {available[worker_id]})
            buf = np.empty((1,720,32), dtype=np.float32)
            with torch.inference_mode():
                for device in splits[worker_id]:
                    buf[0,:720-head] = rings[device,head:]
                    buf[0,720-head:] = rings[device,:head]
                    np.multiply(buf,scale,out=buf)
                    np.add(buf,offset,out=buf)
                    scores[device] = scorers[name](torch.from_numpy(buf)).item()
        # Warm each candidate in both persistent workers, no timed setup.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for name in candidates:
                with execution(name):
                    def warm(w):
                        os.sched_setaffinity(0,{available[w]})
                        with torch.inference_mode():
                            for _ in range(4): scorers[name](x)
                    futures=[pool.submit(warm,w) for w in range(2)]
                    for f in futures:f.result()
            generator = random.Random(6677)
            reference_scores = {}
            for round_id in range(a.rounds):
                order=list(candidates);generator.shuffle(order)
                head=(137+round_id)%720
                for name in order:
                    with execution(name):
                        before=snapshot();start=time.perf_counter_ns()
                        futures=[pool.submit(task,name,w,head) for w in range(2)]
                        for f in futures:f.result()
                        payload=json.dumps(scores.tolist())
                        elapsed=(time.perf_counter_ns()-start)/1e6
                        resources=delta(before,snapshot())
                    if round_id in reference_scores:
                        assert np.allclose(scores,reference_scores[round_id],rtol=1e-4,atol=1e-5),name
                    else:reference_scores[round_id]=scores.copy()
                    np.save(out/f'{tag}_{name}_r{round_id}_scores.npy',scores)
                    result['samples'].append({'scope':'burst','round':round_id,'candidate':name,
                                              'head':head,'latency_ms':[elapsed],
                                              'resources':resources,'json_bytes':len(payload)})
                    save()
                    print(json.dumps({'tag':tag,'round':round_id,'candidate':name,'latency_ms':elapsed}),flush=True)
    result['complete']=True
    save()
    print(json.dumps({'complete':tag}),flush=True)
