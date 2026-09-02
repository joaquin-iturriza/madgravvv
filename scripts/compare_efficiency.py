#!/usr/bin/env python
"""Two front ends, efficiency vs network SNR at a fixed false-alarm rate.

Each model gets its own background-calibrated threshold, which is the only way to
compare two statistics that live on different scales. Everything else is identical: the
same background segments, the same injection parameters, the same vetoes, the same
slide plan.

  scripts/remote.sh .venv/bin/python scripts/compare_efficiency.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.eval import coherence as COH  # noqa: E402
from madgrav_ml.eval.far import TrialsFactor, threshold_at_far  # noqa: E402
from madgrav_ml.plotting.style import save_figure, use_style  # noqa: E402

EDGES = [6, 8, 10, 12, 15, 20, 25, 32, 40]


def load(far_curve: Path, foreground: Path):
    with open(f"{far_curve}.json") as fh:
        cal = json.load(fh)
    blob = np.load(f"{far_curve}_background.npz")
    bg = {c: blob[f"net_sigma_{c}"].astype(np.float64) for c in ("massive", "general")}
    T = float(blob["background_livetime_s"])
    z = np.load(foreground)
    n = cal["sigma_norm"]
    net = (((z["score_H1"] - n["muH"]) / n["sdH"])
           + ((z["score_L1"] - n["muL"]) / n["sdL"])) / np.sqrt(2.0)
    massive = COH.is_massive(z["coherence"], z["centroid_H1"], z["centroid_L1"])
    return bg, T, net, massive, z["network_snr"].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-far", type=Path, default=REPO / "runs/_checks/far_curve")
    ap.add_argument("--a-fore", type=Path,
                    default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--a-label", default="retrained (this work)")
    ap.add_argument("--b-far", type=Path, default=REPO / "runs/_checks/far_curve_vendored")
    ap.add_argument("--b-fore", type=Path,
                    default=REPO / "data_cache/injections/foreground_vendored.npz")
    ap.add_argument("--b-label", default="distributed weights")
    ap.add_argument("--far", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/efficiency_comparison")
    args = ap.parse_args()

    trials = TrialsFactor(2, 2)
    use_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    print(f"efficiency vs network SNR at FAR {args.far:.0f}/yr\n")
    print(f"{'SNR bin':<10}{'n':>7}" + "".join(f"{l[:20]:>22}"
                                               for l in (args.a_label, args.b_label)))
    rows = {}
    for label, far_curve, fore, colour in ((args.a_label, args.a_far, args.a_fore, "C0"),
                                           (args.b_label, args.b_far, args.b_fore, "C1")):
        bg, T, net, massive, snr = load(far_curve, fore)
        thr = {c: threshold_at_far(v, T, far_target=args.far, trials=trials)
               if v.size else np.inf for c, v in bg.items()}
        found = np.where(massive, net > thr["massive"], net > thr["general"])
        rows[label] = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (snr >= lo) & (snr < hi)
            rows[label].append((0.5 * (lo + hi), float(found[m].mean()) if m.any() else np.nan,
                                int(m.sum())))
        pts = np.array([(x, e) for x, e, _ in rows[label]])
        axes[0].plot(pts[:, 0], pts[:, 1], marker="o", ms=4, color=colour,
                     label=f"{label} ({found.mean():.3f} overall)")
        # Where each model's noise tail sits is the whole explanation for the gap.
        axes[1].hist(bg["general"], bins=np.linspace(6, 34, 120), histtype="step",
                     color=colour, label=f"{label}: max {bg['general'].max():.1f}")
        print(f"  thresholds for {label}: massive {thr['massive']:.2f}, "
              f"general {thr['general']:.2f}; overall efficiency {found.mean():.3f}")

    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        line = f"{f'{lo}-{hi}':<10}{rows[args.a_label][i][2]:>7}"
        for label in (args.a_label, args.b_label):
            line += f"{rows[label][i][1]:>22.3f}"
        print(line)

    axes[0].set_xlabel("network SNR"); axes[0].set_ylabel("detection efficiency")
    axes[0].set_ylim(-0.02, 0.5); axes[0].legend(fontsize=7)
    axes[0].set_title(f"at FAR {args.far:.0f}/yr", fontsize=9)
    axes[1].set_yscale("log"); axes[1].set_xlabel(r"background net $\sigma$")
    axes[1].set_ylabel("time-slide triggers"); axes[1].legend(fontsize=7)
    axes[1].set_title("noise tail — what sets each threshold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, args.out)
    print(f"\nwrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
