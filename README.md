# Optimising Transformer Autoencoder Inference

Code accompanying [Optimising Inference for Transformer Autoencoders](https://blog.arul.me/posts/optimising-autoencoder-inference).
The article is the source of truth for the results reported here.

An inference engineering study of a transformer autoencoder for time-series
scoring: 720 observations, 32 features, and bursts of 3,000 windows. The work
explores circular input storage, CPU attention backends and worker layouts,
GPU compilation and transfer overlap, and custom Triton feed-forward kernels.

## Published results

These figures come directly from the article and describe its tested hardware
and configurations. The CPU experiments used 8 CPUs and 32GB RAM; the GPU was an
RTX Pro 4000 Blackwell SFF with 24GB VRAM.

| Experiment | Result reported in the article |
| --- | --- |
| CPU scoring cycle | Explicit FP32 attention: ~45s; native PyTorch: ~20s; tuned SDPA: ~15s |
| History update | Array roll: 30.8ms; circular storage: 0.057ms |
| GPU scoring burst | Compiled FP32: 1.40s; BF16: 0.32s |
| Fused feed-forward unit | 0.56ms → 0.29ms; approximately 1.89× faster |
| Final randomised comparison | Compiled baseline: 323.56ms; selected custom path: 248.46ms; 23.2% lower latency |
| TensorRT comparison | 290ms at batch size 32 |

The final randomised comparison used 15 cycles per candidate. The rounded GPU
burst timings and the final comparison are separate experiments in the article.

## Code guide

- `model.py`, `deep_models.py`: autoencoder, scoring, and native-transformer controls.
- `benchmark.py`, `paired.py`, `pipeline*.py`: isolated CPU measurements and complete scoring pipelines.
- `deep_*.py`, `native_path_study.py`, `run_*.py`: attention, allocator, storage, worker, and CPU backend experiments.
- `gpu_experiments/gpu_suite.py`, `gpu_followup.py`: GPU precision, compilation, batching, and transfer experiments.
- `gpu_experiments/fused_ffn.py`, `fusion_case.py`, `run_fusion.py`, `run_full_ffn.py`, `pair_fusion.py`: Triton kernels and comparison drivers.
- `gpu_experiments/tensorrt_backend.py`, `tensorrt_case.py`, `run_tensorrt.py`: TensorRT export, execution, and comparisons.

## CPU quick start

From the repository root on Linux, with Python available as `python`:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-base.txt
mkdir -p artifacts reports
python train.py
python benchmark.py --backend manual
python benchmark.py --backend sdpa
```

This trains a synthetic-data checkpoint and runs isolated CPU benchmarks.
For exports and numerical checks, run `python export.py`, then `python validate.py`.
The CPU suite drivers retain the original affinity and environment assumptions;
review them before running a full sweep. Run benchmark drivers sequentially.

The GPU harnesses require CUDA-enabled PyTorch, Triton for custom kernels, and
TensorRT for its backend. They were staged alongside the root Python modules
on the GPU host and expect the same checkpoint, generated validation inputs,
and CPU reference scores. They are experiment harnesses requiring that setup;
the CPU requirements files do not install the GPU stack.

## Repository scope

Only experiment source, dependency files, and this README are included.
`.gitignore` keeps all figures, plots, generated reports, raw measurements,
model artifacts, environments, archives, and remote orchestration files local.
It also excludes the old report and plot generators, so their historical
results are not published alongside the article's results.

This is a learning project using synthetic data. Numerical agreement checks
do not establish anomaly-detection quality or production readiness.
