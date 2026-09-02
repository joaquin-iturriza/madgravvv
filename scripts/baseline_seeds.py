#!/usr/bin/env python
"""Aggregate several seeds into one baseline, with the spread that makes it usable.

A single efficiency number cannot say whether a later change helped: training is
stochastic, so the same recipe run twice gives two different numbers. This reports the
seed-to-seed spread, which is the bar any claimed improvement has to clear. Below it, a
difference is a different random initialisation and nothing else.

Reports the full range rather than only a standard deviation. With three seeds an SD is
itself a noisy estimate, and the honest statement to a reader is "these three runs landed
between A and B".

  scripts/remote.sh .venv/bin/python scripts/baseline_seeds.py \
      --runs 42:runs/_checks/far_curve:data_cache/injections/foreground.npz ...
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

TARGETS = (100.0, 10.0, 1.0)
EDGES = [6, 8, 10, 12, 15, 20, 25, 32, 40]


def efficiency(far_curve: Path, foreground: Path, trials):
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
    snr = z["network_snr"].astype(np.float64)

    out = {"floor": (trials.value if hasattr(trials, "value") else trials)
           / (T / (365.25 * 86400.0)),
           "max_background": float(max(v.max() for v in bg.values() if v.size)),
           "efficiency": {}, "by_snr": {}}
    for t in TARGETS:
        if t < out["floor"]:
            continue
        thr = {c: threshold_at_far(v, T, far_target=t, trials=trials) if v.size else np.inf
               for c, v in bg.items()}
        found = np.where(massive, net > thr["massive"], net > thr["general"])
        out["efficiency"][t] = float(found.mean())
        if t == TARGETS[0]:
            out["by_snr"][t] = [float(found[(snr >= a) & (snr < b)].mean())
                                for a, b in zip(EDGES[:-1], EDGES[1:])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label:far_curve_prefix:foreground.npz")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/baseline_seeds.json")
    args = ap.parse_args()

    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    results = {}
    for spec in args.runs:
        label, far_curve, fore = spec.split(":", 2)
        results[label] = efficiency(Path(far_curve), Path(fore), trials)
        print(f"seed {label}: " + "  ".join(
            f"FAR {t:g}/yr {v:.4f}" for t, v in results[label]["efficiency"].items())
            + f"   (max background net sigma {results[label]['max_background']:.1f})")

    print(f"\n{'FAR [1/yr]':>11}{'mean':>9}{'min':>9}{'max':>9}{'range':>9}{'sd':>9}")
    summary = {}
    for t in TARGETS:
        vals = [r["efficiency"].get(t) for r in results.values()]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        a = np.array(vals)
        summary[str(t)] = {"mean": float(a.mean()), "min": float(a.min()),
                           "max": float(a.max()), "range": float(np.ptp(a)),
                           "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
                           "n_seeds": len(a), "values": vals}
        sd = f"{a.std(ddof=1):>9.4f}" if len(a) > 1 else f"{'-':>9}"
        print(f"{t:>11.0f}{a.mean():>9.4f}{a.min():>9.4f}{a.max():>9.4f}"
              f"{np.ptp(a):>9.4f}{sd}")

    top = TARGETS[0]
    if str(top) in summary and summary[str(top)]["n_seeds"] > 1:
        r = summary[str(top)]
        print(f"\nBASELINE at FAR {top:.0f}/yr: {r['mean']:.3f}, seeds spanning "
              f"{r['min']:.3f}-{r['max']:.3f}")
        print(f"A change must move it by more than {r['range']:.3f} to be worth "
              f"anything. Anything smaller is a different random seed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"per_seed": results, "summary": summary}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
