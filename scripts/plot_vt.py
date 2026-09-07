#!/usr/bin/env python
"""Sensitive volume against false-alarm rate, with its three error terms separated.

The argument in Section 13 of `docs/results.tex` is a comparison of error contributions
across false-alarm rate --- which one dominates changes with the rate, and the
conclusion (the volume is limited by training stochasticity, the rate LABEL by
background livetime) depends on seeing them side by side. A table of three intervals per
row states that; a figure shows it.

  scripts/remote.sh .venv/bin/python scripts/plot_vt.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.plotting.style import save_figure, use_style  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vt", type=Path, default=REPO / "runs/_checks/vt_vol.json")
    ap.add_argument("--out", type=Path, default=REPO / "figures/sensitive_volume")
    args = ap.parse_args()

    d = json.load(open(args.vt))
    rows = sorted(d["results"], key=lambda r: r["far_per_yr"])
    far = np.array([r["far_per_yr"] for r in rows])
    v = np.array([r["V_comoving_gpc3_mean"] for r in rows])
    seed = np.array([r["V_comoving_gpc3_seed_range"] for r in rows])
    bg = np.array([r["V_comoving_gpc3_from_background_count"] for r in rows])
    found = np.array([r["rel_error_found"] for r in rows])
    n_bg = [r["n_background_above_threshold"] for r in rows]

    use_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.fill_between(far, seed[:, 0], seed[:, 1], alpha=0.30, color="C0",
                    label="training stochasticity (3 seeds)")
    ax.fill_between(far, v * (1 - found), v * (1 + found), alpha=0.30, color="C2",
                    label=r"recovered count, $1/\sqrt{N_\mathrm{found}}$")
    ax.fill_between(far, bg[:, 0], bg[:, 1], alpha=0.35, color="C3",
                    label="background rank, propagated")
    ax.plot(far, v, "o-", color="0.15", ms=4, lw=1.4, label="comoving $V$")

    for x, y, n in zip(far, v, n_bg):
        ax.annotate(f"$n_\\mathrm{{bg}}$={n}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7, color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("false-alarm rate [1/yr]")
    ax.set_ylabel(r"sensitive volume [Gpc$^3$]")
    ax.set_title("the term that dominates changes with the rate", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)
    print(f"wrote {args.out}.png / .pdf")
    for r in rows:
        print(f"  FAR {r['far_per_yr']:>5.0f}/yr  V={r['V_comoving_gpc3_mean']:.3f}  "
              f"n_bg={r['n_background_above_threshold']:>4}  "
              f"seed +-{100*(r['V_comoving_gpc3_seed_range'][1]-r['V_comoving_gpc3_seed_range'][0])/2/r['V_comoving_gpc3_mean']:.0f}%  "
              f"found +-{100*r['rel_error_found']:.0f}%  "
              f"bg +-{100*(r['V_comoving_gpc3_from_background_count'][1]-r['V_comoving_gpc3_from_background_count'][0])/2/r['V_comoving_gpc3_mean']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
