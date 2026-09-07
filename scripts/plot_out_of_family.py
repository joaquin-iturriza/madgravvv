#!/usr/bin/env python
"""Efficiency against achieved false-alarm rate, tuned family versus out of family.

The table in Section 12 of `docs/results.tex` gives five rates; the argument it makes is
about the SHAPE — that the retained fraction falls as the threshold tightens rather than
holding constant — and a shape is easier to read off a curve than off five ratios.

Right panel separates the two specialists rather than plotting their maximum, because
their bands differ and the difference is the whole point. HM reads 20-140 Hz and its pass
fraction collapses past that edge, which says nothing about morphology -- it is being
shown a crop outside its own band. LM reads 50-500 Hz, so every burst in this population
lies inside its band, and it degrades anyway. Only the second is evidence that the
supervised components prefer a waveform morphology.

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

from madgrav_ml.eval.far import TrialsFactor  # noqa: E402

SEEDS = ("s42", "s43", "s44")
# Itemised, not a literal: Phase 7.1 is about reducing the arm count, and a hardcoded 4
# would keep drawing the old false-alarm rates after that lands.
TRIALS = TrialsFactor(2, 2).value
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
    from madgrav_ml.eval import specialists as SP

    hm, lm = z["cnn_hm"].astype(np.float64), z["cnn_lm"].astype(np.float64)
    # The SAME bin edges as the table in results.tex. They were different once, and the
    # figure then disagreed with the table it was supposed to make readable -- the gate
    # peak landed in a different bin. One list, used by both.
    EDGES = [40, 55, 70, 90, 115, 150, 200, 270, 400]
    centres = [np.sqrt(a * b) for a, b in zip(EDGES[:-1], EDGES[1:])]
    masks = [(f0 >= a) & (f0 < b) for a, b in zip(EDGES[:-1], EDGES[1:])]
    if any(not m.any() for m in masks):
        raise SystemExit("an f0 bin is empty; the plotted mean would be NaN and the "
                         "curve would silently break")

    # HM and LM separately, because their bands differ and the difference is the point:
    # HM reads 20-140 Hz and collapses at its own edge, which says nothing about
    # morphology; LM reads 50-500 Hz and degrades across a range it fully covers, which
    # does.
    for series, colour, label in (
            ([(hm[m] >= SP.GLITCH_THRESH).mean() for m in masks], "C0",
             "HM specialist (20-140 Hz)"),
            ([(lm[m] >= SP.GLITCH_THRESH).mean() for m in masks], "C3",
             "LM specialist (50-500 Hz)")):
        axes[1].plot(centres, series, marker="o", ms=4, color=colour, label=label)
    axes[1].axvspan(140, 400, color="0.85", zorder=0)
    axes[1].text(230, 0.93, "outside HM band", fontsize=7, ha="center", color="0.35")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r"burst central frequency $f_0$ [Hz]")
    axes[1].set_ylabel(r"fraction with $P(\mathrm{signal}) \geq 0.5$")
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].set_title("HM collapses at its band edge; LM degrades inside its own",
                      fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)
    print(f"wrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
