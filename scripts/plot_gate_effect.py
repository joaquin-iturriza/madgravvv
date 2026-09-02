#!/usr/bin/env python
"""Efficiency against achieved false-alarm rate, with and without the CNN glitch gate.

Both curves are the same construction: scan the massive-channel threshold, and at each
one read off the false-alarm rate the surviving background actually produces and the
fraction of injections above it. Nothing is quoted at a rate the background cannot
support -- each curve simply stops where its survivors run out, which is the honest end
of the measurement and, for the gated pipeline, the interesting one.

  scripts/remote.sh .venv/bin/python scripts/plot_gate_effect.py
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
from madgrav_ml.eval import specialists as SP  # noqa: E402
from madgrav_ml.plotting.style import save_figure, use_style  # noqa: E402

SEEDS = [("42", "", ""), ("43", "_s43", "_s43"), ("44", "_s44", "_s44")]
TRIALS = 4


def curve(far_curve: Path, foreground: Path, gated: Path | None):
    with open(f"{far_curve}.json") as fh:
        norm = json.load(fh)["sigma_norm"]
    blob = np.load(f"{far_curve}_background.npz")
    T_yr = float(blob["background_livetime_s"]) / (365.25 * 86400.0)
    bg = blob["net_sigma_massive"].astype(np.float64)

    z = np.load(foreground)
    net = (((z["score_H1"] - norm["muH"]) / norm["sdH"])
           + ((z["score_L1"] - norm["muL"]) / norm["sdL"])) / np.sqrt(2.0)
    keep = COH.is_massive(z["coherence"], z["centroid_H1"], z["centroid_L1"])
    if gated is not None:
        g = np.load(gated)
        bg = g["massive_net_sigma"][g["massive_kept"]].astype(np.float64)
        keep = keep & ~SP.is_glitch(z["cnn_hm"], z["cnn_lm"])

    order = np.sort(bg)[::-1]
    far = TRIALS * np.arange(1, len(order) + 1) / T_yr
    eff = np.array([float(((net > t) & keep).mean()) for t in order])
    return far, eff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/gate_effect")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, colour, gate in (("without the glitch gate", "C0", False),
                                ("with the glitch gate", "C1", True)):
        for i, (seed, fsuf, bsuf) in enumerate(SEEDS):
            gated = (REPO / f"data_cache/background{bsuf}_gated.npz") if gate else None
            far, eff = curve(REPO / f"runs/_checks/far_curve{fsuf}",
                             REPO / f"data_cache/injections/foreground{fsuf}.npz", gated)
            ax.step(far, eff, where="post", color=colour, alpha=0.85,
                    label=label if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("false-alarm rate achieved [1/yr]")
    ax.set_ylabel("detection efficiency")
    ax.set_title("three seeds each; each curve ends where its background runs out",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    save_figure(fig, args.out)
    print(f"wrote {args.out}.png / .pdf")
    for label, gate in (("ungated", False), ("gated", True)):
        for seed, fsuf, bsuf in SEEDS:
            gated = (REPO / f"data_cache/background{bsuf}_gated.npz") if gate else None
            far, eff = curve(REPO / f"runs/_checks/far_curve{fsuf}",
                             REPO / f"data_cache/injections/foreground{fsuf}.npz", gated)
            print(f"  {label:8s} seed {seed}: lowest FAR reached {far[0]:.3g}/yr, "
                  f"edge {far[-1]:.2f}/yr at efficiency {eff[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
