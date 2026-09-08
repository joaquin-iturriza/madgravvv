#!/usr/bin/env python
"""Find the training operating point: batch size against throughput and memory.

Batch size and horizon are NOT hyperparameters to sweep -- they are a decided operating
point, chosen once from measurement, that the searched hyperparameters then adapt to.
This measures the point.

Peak VRAM at the inherited batch of 64 is 2.6 GB of a 32 GB V100, so the card sits idle
most of the time and wall-clock is dominated by per-step overhead rather than
arithmetic. Larger batches trade that away, up to two limits: memory, and the point past
which larger batches stop helping optimisation. The first is measured here; the second
is why the recommendation is the knee rather than the maximum.

  scripts/remote.sh sbatch jobs/job_bench.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.models.cae import BaselineCAE  # noqa: E402


def bench(model, device, batch, channels, size, steps=30, warmup=5,
          amp=False, channels_last=False):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, channels, *size, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    torch.cuda.reset_peak_memory_stats(device)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            recon, _ = model(x)
            loss = torch.nn.functional.mse_loss(recon, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    return steps * batch / dt, peak, dt / steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[64, 256, 512])
    ap.add_argument("--size", type=int, nargs=2, default=[256, 128])
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--bank", type=int, default=20000,
                    help="tiles per epoch, for the epochs/hour column")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU; this measurement is meaningless on CPU", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    name = torch.cuda.get_device_name(device)
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"{name}, {total:.0f} GiB, tile {args.channels}x{args.size[0]}x{args.size[1]}")
    print(f"\n{'batch':>7}{'samples/s':>12}{'rel':>7}{'peak GiB':>10}{'ms/step':>10}"
          f"{'epochs/h':>10}")
    base = None
    for label, amp, cl in (("fp32", False, False), ("fp32+chlast", False, True),
                           ("amp fp16", True, False), ("amp+chlast", True, True)):
        print(f"  -- {label}")
        for b in args.batches:
            model = BaselineCAE(in_channels=args.channels).to(device)
            try:
                rate, peak, per_step = bench(model, device, b, args.channels,
                                             tuple(args.size), amp=amp,
                                             channels_last=cl)
            except torch.cuda.OutOfMemoryError:
                print(f"{b:>7}{'OOM':>12}")
                del model
                torch.cuda.empty_cache()
                continue
            base = base or rate
            print(f"{b:>7}{rate:>12.0f}{rate / base:>7.1f}{peak:>10.2f}"
                  f"{1000 * per_step:>10.1f}{3600 * rate / args.bank:>10.0f}")
            del model
            torch.cuda.empty_cache()

    print(f"\nepochs/h is over a {args.bank}-tile bank. The bank is the thing that "
          f"should grow: HPO_TRAIN holds ~400k distinct windows at 1 s stride and the "
          f"current bank draws 20k of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
