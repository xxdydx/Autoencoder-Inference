"""One isolated configuration per invocation. All timed computation runs on CPU."""
import argparse
import copy
import json
import os
import resource
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='sdpa')
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--length', type=int, default=720)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--repeats', type=int, default=20)
parser.add_argument('--label', default='')
parser.add_argument('--min-seconds', type=float, default=1.0)
args = parser.parse_args()
from autoencoder.runtime_setup import configure_openmp
configure_openmp(args.backend=='compile')
# Set before importing the numerical libraries. torch controls OpenMP; Accelerate
# has a separate control, which is held to the same count for every configuration.
os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.threads)
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
os.environ['MKL_NUM_THREADS'] = str(args.threads)
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
from autoencoder.measurement import configure_affinity, peak_rss_bytes, snapshot, delta, quantization_engine
configure_affinity()
import numpy as np
import psutil
import torch
from autoencoder.model import load_model, Scorer, synthetic

torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(123)
tag = f'{args.backend}_t{args.threads}_b{args.batch}_s{args.length}'
if args.label:
    tag = args.label+'_'+tag
result = vars(args).copy()
result['tag'] = tag
result['torch_version'] = torch.__version__
result['single_openmp_runtime_requested'] = bool(os.environ.get('CPU_AE_OMP_CONFIGURED'))
try:
    m, checkpoint = load_model('manual' if args.backend == 'manual' else 'sdpa')
    x = (synthetic(args.batch, args.length, 789) - checkpoint['mean']) / checkpoint['std']
    with torch.inference_mode():
        reference = m(x)
        reference_scores = (reference - x).square().mean((1, 2))
    setup = time.perf_counter()
    if args.backend.startswith('ort'):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.enable_profiling = False
        options.profile_file_prefix = f'artifacts/{tag}_ort'
        path = 'artifacts/model_int8.onnx' if args.backend == 'ort_int8' else 'artifacts/model.onnx'
        session = ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])
        data = x.numpy()
        run = lambda: session.run(None, {'input': data})[0]
        result['providers'] = session.get_providers()
    else:
        if args.backend == 'contiguous':
            for block in list(m.encoder)+list(m.decoder):
                block.contiguous_qkv = True
        if args.backend == 'tanh':
            for block in list(m.encoder)+list(m.decoder):
                block.gelu_approx = 'tanh'
        if args.backend == 'int8':
            torch.backends.quantized.engine = quantization_engine()
            result['quantization_engine'] = torch.backends.quantized.engine
            m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        if args.backend in ('fp16', 'bf16'):
            dtype = torch.float16 if args.backend == 'fp16' else torch.bfloat16
            m, x = m.to(dtype), x.to(dtype)
        scorer = Scorer(m).eval()
        if args.backend == 'jit':
            scorer = torch.jit.freeze(torch.jit.trace(scorer, x))
        if args.backend == 'compile':
            scorer = torch.compile(scorer, fullgraph=True, dynamic=False)
        run = lambda: scorer(x)
    with torch.inference_mode():
        for _ in range(3):
            y = run()
        result['setup_and_warmup_s'] = time.perf_counter() - setup
        candidate = torch.as_tensor(y).float()
        result['score_abs_max'] = (candidate - reference_scores).abs().max().item()
        result['score_relative_max'] = ((candidate-reference_scores).abs()/reference_scores.abs().clamp_min(1e-8)).max().item()
        if not args.backend.startswith('ort'):
            reconstruction = m(x).float()
            result['reconstruction_abs_max'] = (reconstruction-reference).abs().max().item()
            result['reconstruction_rmse'] = (reconstruction-reference).square().mean().sqrt().item()
        p = psutil.Process()
        cpu0 = p.cpu_times()
        context0 = p.num_ctx_switches()
        resource0 = snapshot()
        wall0 = time.perf_counter()
        timings = []
        while len(timings)<args.repeats or time.perf_counter()-wall0<args.min_seconds:
            start = time.perf_counter_ns()
            run()
            timings.append((time.perf_counter_ns()-start)/1e6)
        wall = time.perf_counter()-wall0
        cpu1 = p.cpu_times()
        context1 = p.num_ctx_switches()
        resource1 = snapshot()
        result.update(samples_ms=timings, p50_ms=float(np.median(timings)),
                      p95_ms=float(np.percentile(timings,95)),
                      p99_ms=float(np.percentile(timings,99)),
                      windows_per_second=1000*args.batch/float(np.mean(timings)),
                      process_cpu_percent=100*(cpu1.user+cpu1.system-cpu0.user-cpu0.system)/wall,
                      context_switches=context1.voluntary+context1.involuntary-context0.voluntary-context0.involuntary,
                      process_threads=p.num_threads(), peak_rss_bytes=peak_rss_bytes(),
                      measured_calls=len(timings),resource_before=resource0,resource_after=resource1,
                      resource_delta=delta(resource0,resource1))
        if args.profile and not args.backend.startswith('ort'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True, profile_memory=True) as prof:
                for _ in range(5):
                    run()
            prof.export_chrome_trace(f'artifacts/{tag}_trace.json')
            events = prof.key_averages(group_by_input_shape=True)
            Path(f'reports/{tag}_operators.txt').write_text(events.table(sort_by='self_cpu_time_total', row_limit=40))
            result['operators'] = [dict(key=e.key, shape=e.input_shapes, calls=e.count,
                                       self_cpu_us=e.self_cpu_time_total, cpu_us=e.cpu_time_total,
                                       self_memory_bytes=e.self_cpu_memory_usage)
                                   for e in events]
        elif args.profile:
            # A separate session keeps profiler overhead out of all latency samples.
            options.enable_profiling = True
            options.optimized_model_filepath = f'artifacts/{args.backend}_optimized.onnx'
            prof_session = ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])
            for _ in range(5):
                prof_session.run(None, {'input': data})
            result['ort_profile'] = prof_session.end_profiling()
except Exception as error:
    import traceback
    result['error'] = repr(error)
    result['traceback'] = traceback.format_exc()
Path(f'artifacts/{tag}.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('operators','samples_ms','traceback')}), flush=True)
