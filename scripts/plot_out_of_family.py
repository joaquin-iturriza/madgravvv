#!/usr/bin/env python
"""Efficiency against achieved false-alarm rate, tuned family versus out of family.

The table in Section 12 of `docs/results.tex` gives five rates; the argument it makes is
about the SHAPE — that the retained fraction falls as the threshold tightens rather than
holding constant — and a shape is easier to read off a curve than off five ratios.

Right panel carries the mechanism: the glitch arm's opinion and the CNN gate's retention
both degrade with the burst's central frequency, because a low-frequency burst is the
thing in this population that most resembles a heavy merger.

  scripts/remote.sh .venv/bin/python scripts/plot_out_of_family.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.plotting.style import save_figure, use_style  # noqa: E402

SEEDS = ("s42", "s43", "s44")
TRIALS = 4
YEAR = 365.25 * 86400.0


def curve(tag: str):
    bg = np.load(REPO / f"runs/_checks/{tag}_background.npz")
    fg = np.load(REPO / f"runs/_checks/{tag}_foreground.npz")
    t_yr = float(bg["background_livetime_s"]) / YEAR
    order = np.sort(bg["loglr"].astype(np.float64))[::-1]
    far = TRIALS * np.arange(1, len(order) + 1) / t_yr
    ll = fg["loglr"].astype(np.float64)
    keep = fg["keep"] if "keep" in fg.files else np.ones(len(ll), bool)
    eff = np.array([float(((ll > t) & keep).mean()) for t in order])
    return far, eff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "figures/out_of_family")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    for label, prefix, colour in (("compact binaries (tuned family)", "clean_cbc", "C0"),
                                  ("sine-Gaussian bursts", "clean_burst", "C1")):
        for i, seed in enumerate(SEEDS):
            far, eff = curve(f"{prefix}_{seed}")
            axes[0].step(far, eff, where="post", color=colour, alpha=0.85,
                         label=label if i == 0 else None)
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.3, 3e3)
    axes[0].set_xlabel("false-alarm rate achieved [1/yr]")
    axes[0].set_ylabel("detection efficiency")
    axes[0].set_title("three seeds each, same ranking statistic", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    z = np.load(REPO / "data_cache/injections/burst.npz")
    f0 = z["frequency"].astype(np.float64)
    arm = z["arm_H1"].astype(np.float64)
    gate = np.maximum(z["cnn_hm"], z["cnn_lm"]) >= 0.5
    edges = [40, 60, 80, 110, 150, 200, 260, 330, 400]
    centres, arms, keeps = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f0 >= lo) & (f0 < hi)
        if m.any():
            centres.append(np.sqrt(lo * hi))
            arms.append(arm[m].mean())
            keeps.append(gate[m].mean())

    axes[1].plot(centres, arms, marker="o", ms=4, color="C1", label="arm logit (burst)")
    cbc = np.load(REPO / "data_cache/injections/foreground.npz")
    axes[1].axhline(cbc["arm_H1"].mean(), color="C0", ls="--", lw=1,
                    label="arm logit, compact binaries")
    axes[1].axhline(-5.75, color="0.5", ls=":", lw=1, label="arm logit, background")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"burst central frequency $f_0$ [Hz]")
    axes[1].set_ylabel("glitch-arm ensemble logit")
    twin = axes[1].twinx()
    twin.plot(centres, keeps, marker="s", ms=4, color="C3", alpha=0.7)
    twin.set_ylabel("CNN gate retention", color="C3")
    twin.tick_params(axis="y", colors="C3")
    twin.set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].set_title("why: the supervised arms track the trained morphology",
                      fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)
    print(f"wrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
