# Optimising Transformer Autoencoder Inference

Code accompanying [Optimising Inference for Transformer Autoencoders](https://blog.arul.me/posts/optimising-autoencoder-inference).

I optimised a transformer autoencoder for time-series scoring: 720 observations,
32 features, and bursts of 3,000 windows. The goal was to score each burst within
2 seconds. I explored circular input storage, CPU attention backends and worker
layouts, GPU compilation and transfer overlap, and custom Triton feed-forward
kernels.

The selected custom path averaged **248.46ms** per burst, compared with **290ms** for [TensorRT]((https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)) at batch size 32 in this study. Comparing against it provided a useful reference for whether the workload-specific custom kernels could improve on an automatically optimised deployment.

## Published results

The CPU experiments used 8 CPUs and 32GB RAM; the GPU was an RTX Pro 4000 Blackwell SFF with 24GB VRAM.

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

## Repository layout

```text
autoencoder/   Model, training, export, validation, and shared utilities
cpu/           CPU benchmarks, input storage, and experiment drivers
gpu/           GPU benchmarks, Triton kernels, and backend comparisons
requirements/  Pinned CPU dependency sets
```

Start with `autoencoder/model.py` for the architecture, `cpu/benchmark.py` for
CPU inference, and `gpu/fused_ffn.py` for the custom kernels.

## CPU quick start

Run commands from the repository root on Linux, with Python available as `python`:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/requirements-base.txt
mkdir -p artifacts reports
python -m autoencoder.train
python -m cpu.benchmark --backend manual
python -m cpu.benchmark --backend sdpa
```

This trains a synthetic-data checkpoint and runs isolated CPU benchmarks.
For exports and numerical checks, run `python -m autoencoder.export`, then `python -m autoencoder.validate`.
The CPU suite drivers retain the original affinity and environment assumptions;
review them before running a full sweep. Run benchmark drivers sequentially.

The GPU harnesses require CUDA-enabled PyTorch, Triton for custom kernels, and
TensorRT for its backend. Run their modules from the repository root, for example
`python -m gpu.gpu_suite`. They expect the same checkpoint, generated validation
inputs, and CPU reference scores. The CPU requirements files do not install the
GPU stack.

## Repository scope

Only experiment source, dependency files, and this README are included.
`.gitignore` keeps all figures, plots, generated reports, raw measurements,
model artifacts, environments, archives, and remote orchestration files locally on my own disk.

This is a learning project using synthetic data. Numerical agreement checks
do not establish anomaly-detection quality or production readiness. Data included in this repository are purely synthetic and do not represent any production-level data and merely for demonstration purposes.
