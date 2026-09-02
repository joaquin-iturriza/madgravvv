#!/usr/bin/env python
"""Detection efficiency at fixed false-alarm rate — the project's actual currency.

Combines the two scans: the time-slide background fixes the threshold at a chosen FAR,
and the coincident injection set says what fraction of sources clear it. Both went
through the same tiling, the same model and the same ranking statistic, which is the
only thing that makes the pairing legitimate.

  scripts/remote.sh .venv/bin/python scripts/efficiency_at_far.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.eval.far import TrialsFactor, far_of, threshold_at_far  # noqa: E402
from madgrav_ml.plotting.style import save_figure, use_style  # noqa: E402


def wilson(k: int, n: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson interval. At an efficiency of 0 or 1 the normal interval has zero width,
    which is exactly where an efficiency curve spends most of its length."""
    if n == 0:
        return (float("nan"),) * 2
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--far-curve", type=Path, default=REPO / "runs/_checks/far_curve")
    ap.add_argument("--foreground", type=Path,
                    default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/efficiency_at_far")
    ap.add_argument("--trials", type=int, default=4)
    args = ap.parse_args()

    with open(f"{args.far_curve}.json") as fh:
        cal = json.load(fh)
    bg_blob = np.load(f"{args.far_curve}_background.npz")
    background = bg_blob["net_sigma"].astype(np.float64)
    T = float(bg_blob["background_livetime_s"])
    norm = cal["sigma_norm"]

    z = np.load(args.foreground)
    # The SAME normalisation the background was reduced with. Refitting it on the
    # injections would move the foreground and background onto different scales and
    # produce an efficiency that means nothing.
    sH = (z["score_H1"].astype(np.float64) - norm["muH"]) / norm["sdH"]
    sL = (z["score_L1"].astype(np.float64) - norm["muL"]) / norm["sdL"]
    net = (sH + sL) / np.sqrt(2.0)
    snr = z["network_snr"].astype(np.float64)
    mtot = (z["mass1"] + z["mass2"]).astype(np.float64)

    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    T_yr = T / (365.25 * 86400.0)
    floor = (trials.value if hasattr(trials, "value") else trials) / T_yr
    print(f"background {T_yr:.2f} yr, {background.size} stored triggers; "
          f"floor {floor:.3g}/yr")
    print(f"foreground {net.size} coincident injections, network SNR "
          f"{snr.min():.1f}-{snr.max():.1f}\n")

    targets = [t for t in (100.0, 10.0, 1.0) if t >= floor]
    results = {}
    print(f"{'FAR [1/yr]':>11}{'net sigma':>11}{'efficiency':>12}{'68% interval':>18}")
    for t in targets:
        thr = threshold_at_far(background, T, far_target=t, trials=trials)
        k = int((net > thr).sum())
        lo, hi = wilson(k, net.size)
        results[t] = {"threshold": thr, "efficiency": k / net.size,
                      "interval": [lo, hi], "n_found": k, "n_total": int(net.size)}
        print(f"{t:>11.0f}{thr:>11.2f}{k/net.size:>12.3f}   [{lo:.3f}, {hi:.3f}]")

    edges = [6, 8, 10, 12, 15, 20, 25, 32, 40]
    print(f"\nefficiency vs network SNR")
    header = f"{'SNR bin':<10}{'n':>7}" + "".join(f"{'FAR '+format(t,'.0f'):>12}" for t in targets)
    print(header)
    curves = {t: [] for t in targets}
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (snr >= lo_e) & (snr < hi_e)
        if not m.any():
            continue
        line = f"{f'{lo_e}-{hi_e}':<10}{int(m.sum()):>7}"
        for t in targets:
            k = int((net[m] > results[t]["threshold"]).sum())
            e = k / int(m.sum())
            curves[t].append((0.5 * (lo_e + hi_e), e, int(m.sum()), k))
            line += f"{e:>12.3f}"
        print(line)

    print(f"\nnet sigma: background max {background.max():.2f}, "
          f"99.99th {np.percentile(background, 99.99):.2f}; "
          f"injections max {net.max():.2f}, median {np.median(net):.2f}")
    print(f"\nefficiency vs total mass, at FAR {targets[-1]:.0f}/yr")
    thr = results[targets[-1]]["threshold"]
    for lo_e, hi_e in ((20, 60), (60, 100), (100, 240)):
        m = (mtot >= lo_e) & (mtot < hi_e)
        if m.any():
            print(f"  Mtot {lo_e:>3}-{hi_e:<3}  n={int(m.sum()):>5}  "
                  f"eff={float((net[m] > thr).mean()):.3f}")

    use_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for t in targets:
        pts = np.array([(x, e) for x, e, _, _ in curves[t]])
        err = np.array([wilson(k, n) for _, _, n, k in curves[t]])
        # Clip: at k = 0 the Wilson bound and the point estimate are the same number
        # up to rounding, and matplotlib rejects a -1e-19 error bar.
        yerr = np.clip([pts[:, 1] - err[:, 0], err[:, 1] - pts[:, 1]], 0.0, None)
        axes[0].errorbar(pts[:, 0], pts[:, 1], yerr=yerr,
                         marker="o", ms=3, capsize=2, label=f"FAR {t:.0f}/yr")
    axes[0].set_xlabel("network SNR"); axes[0].set_ylabel("detection efficiency")
    axes[0].set_ylim(-0.02, 1.02); axes[0].legend(fontsize=8)

    grid = np.linspace(background.min(), max(background.max(), net.max()), 400)
    axes[1].plot(grid, far_of(grid, background, T, trials=trials), label="background")
    # The loudest injection, drawn on the same axis. If it sits to the LEFT of every
    # quoted threshold then no source in the campaign is detectable at that FAR, and
    # the efficiency table below is a row of honest zeros rather than a broken pairing.
    axes[1].axvline(net.max(), color="C3", ls="--",
                    label=f"loudest injection ({net.max():.1f})")
    axes[1].axhline(floor, ls=":", color="0.5", label=f"floor {floor:.2g}/yr")
    for t in targets:
        axes[1].plot(results[t]["threshold"], t, "o", ms=5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"net $\sigma$"); axes[1].set_ylabel("FAR [1/yr]")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)

    with open(f"{args.out}.json", "w") as fh:
        json.dump({"background_livetime_yr": T_yr, "trials_factor": args.trials,
                   "floor_per_yr": floor, "n_injections": int(net.size),
                   "results": {str(k): v for k, v in results.items()}}, fh, indent=2)
    print(f"\nwrote {args.out}.json / .png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
