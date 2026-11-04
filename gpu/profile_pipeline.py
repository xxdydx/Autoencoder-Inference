"""Capture the original copy/compute pipeline with setup excluded from Nsight."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--mode', choices=['pageable_sync', 'pinned_serial', 'pinned_overlap'], default='pinned_overlap')
    p.add_argument('--devices', type=int, default=512)
    p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--warmup-cycles', type=int, default=5)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--timing-study', action='store_true', help='Five randomized rounds, three cycles per configuration')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    from gpu import gpu_suite as g
    a.output.mkdir(parents=True, exist_ok=True)
    g.A = a.output
    with torch.inference_mode():
        if a.timing_study:
            import random
            choices = [('pageable_sync', 64), ('pinned_serial', 64), ('pinned_overlap', 64), ('pinned_overlap', 128)]
            for r in range(5):
                random.Random(20261104+r).shuffle(choices)
                g.A = a.output / f'round{r}'
                g.A.mkdir(exist_ok=True)
                for mode, batch in choices:
                    g.pipeline('sdpa_bf16', batch, mode, a.devices, 3, warmup_cycles=a.warmup_cycles)
        else:
            g.pipeline('sdpa_bf16', a.batch, a.mode, a.devices, a.cycles,
                       warmup_cycles=a.warmup_cycles, capture_range=a.capture, annotate=a.capture)


if __name__ == '__main__':
    main()
