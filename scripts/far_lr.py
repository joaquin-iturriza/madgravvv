#!/usr/bin/env python
"""Rank on the likelihood ratio instead of net sigma, and measure what that buys.

Every earlier stage used its inputs as CUTS: coherence above a threshold, centroids
below one, `max(HM, LM) >= 0.5`. The cascade uses the same quantities as continuous
features of one statistic, so a trigger that is marginal on several of them can still be
ranked below one that is convincing on all of them --- information a chain of thresholds
throws away by construction.

The model is frozen and distributed (`data/o3a_frozen_lr_off200.npz`); we fit nothing.
That matters for interpretation: the comparison below is between two ways of COMBINING
the same measured quantities, with no new training and no new parameters, on the same
background and the same injections.

Coherence is a feature here rather than a veto, so it has to be evaluated at every grid
point of every lag rather than only on loud triggers. That is what the restricted-lag
matrix product in `eval/coherence.py` is for.

  scripts/remote.sh sbatch jobs/job_lr.sh --background data_cache/background \
      --foreground data_cache/injections/foreground.npz --out runs/_checks/lr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from far_curve import cluster  # noqa: E402
from madgrav_ml.eval import coherence as COH  # noqa: E402
from madgrav_ml.eval import likelihood as LR  # noqa: E402
from madgrav_ml.eval import specialists as SP  # noqa: E402
from madgrav_ml.eval.background import make_slide_plan  # noqa: E402
from madgrav_ml.eval.far import TrialsFactor, far_of  # noqa: E402

FROZEN = REPO / ".reference/MADGRAV/data/o3a_frozen_lr_off200.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", type=Path, default=REPO / "data_cache/background")
    ap.add_argument("--foreground", type=Path,
                    default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/lr")
    ap.add_argument("--n-lags", type=int, default=100000)
    ap.add_argument("--lag-step", type=float, default=4.0)
    ap.add_argument("--cluster-seconds", type=float, default=4.0)
    ap.add_argument("--keep-above", type=float, default=-2.0,
                    help="store slide triggers with loglr above this")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0, choices=(0, 1),
                    help="which frozen fold scores everything. Both were fitted on "
                         "upstream data disjoint from ours, so either is held out for "
                         "us; the other is the robustness check.")
    ap.add_argument("--gate", action="store_true",
                    help="also require max(HM,LM) >= 0.5 on the foreground")
    args = ap.parse_args()

    frozen = LR.load_frozen(FROZEN)
    mu, sd, beta = frozen[args.fold]
    print(f"frozen LR fold {args.fold}: beta = "
          + " ".join(f"{b:+.3f}" for b in beta))

    segs, stride, lo, nfft = [], None, None, None
    for f in sorted(args.background.glob("bg_*.npz")):
        z = np.load(f)
        if "arm_H1" not in z.files:
            print(f"{f.name} predates the arm-logit scan; re-run scan_background.py",
                  file=sys.stderr)
            return 1
        segs.append({"h": z["score_H1"].astype(np.float64),
                     "l": z["score_L1"].astype(np.float64),
                     "ch": z["coeff_H1"], "cl": z["coeff_L1"],
                     "gh": z["centroid_H1"].astype(np.float64),
                     "gl": z["centroid_L1"].astype(np.float64),
                     "ah": z["arm_H1"].astype(np.float64),
                     "al": z["arm_L1"].astype(np.float64)})
        stride, lo, nfft = float(z["stride"]), int(z["band_lo"]), int(z["band_n"])
    n_points = sum(len(s["h"]) for s in segs)
    coincident_s = n_points * stride

    all_h = np.concatenate([s["h"] for s in segs])
    all_l = np.concatenate([s["l"] for s in segs])
    norm = {"muH": float(all_h.mean()), "sdH": float(all_h.std()),
            "muL": float(all_l.mean()), "sdL": float(all_l.std())}
    for s in segs:
        s["sh"] = (s["h"] - norm["muH"]) / norm["sdH"]
        s["sl"] = (s["l"] - norm["muL"]) / norm["sdL"]

    shift = max(1, int(round(args.lag_step / stride)))
    min_n = min(len(s["sh"]) for s in segs)
    n_lags = min(args.n_lags, (min_n // shift) - 1)
    plan = make_slide_plan(coincident_s, n_lags, lag_step_s=args.lag_step)
    half = int(round(0.5 * args.cluster_seconds / stride))
    print(f"{len(segs)} segments, {n_points} points, {n_lags} lags = "
          f"{plan.background_livetime_yr:.2f} yr", flush=True)

    kept = []
    t0 = time.time()
    for li, lag in enumerate(plan.lags_s):
        k = int(round(lag / stride))
        for s in segs:
            n = len(s["sh"])
            if n <= abs(k) or n <= 2 * half:
                continue
            j = (np.arange(n) - k) % n
            coh = COH.coherence_from_coefficients(s["ch"], s["cl"][j], lo, nfft)
            f = LR.features(s["sh"], s["sl"][j], coh, s["gh"], s["gl"][j],
                            s["ah"], s["al"][j])
            ll = LR.log_likelihood_ratio(f, mu, sd, beta)
            m = cluster(ll, half) & (ll > args.keep_above)
            if m.any():
                kept.append(ll[m])
        if (li + 1) % 250 == 0:
            el = (time.time() - t0) / 60
            print(f"  {li+1}/{n_lags} lags, {sum(v.size for v in kept)} stored "
                  f"[{el:.1f} min, eta {el*(n_lags/(li+1)-1):.0f} min]", flush=True)

    background = np.concatenate(kept) if kept else np.array([])
    T = plan.background_livetime_s
    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    tv = trials.value if hasattr(trials, "value") else trials
    T_yr = T / (365.25 * 86400.0)
    print(f"\nbackground: {background.size} triggers above loglr {args.keep_above}, "
          f"max {background.max():.2f}" if background.size else "no background")

    # --- foreground ------------------------------------------------------------
    z = np.load(args.foreground)
    sH = (z["score_H1"].astype(np.float64) - norm["muH"]) / norm["sdH"]
    sL = (z["score_L1"].astype(np.float64) - norm["muL"]) / norm["sdL"]
    f = LR.features(sH, sL, z["coherence"], z["centroid_H1"], z["centroid_L1"],
                    z["arm_H1"], z["arm_L1"])
    ll = LR.log_likelihood_ratio(f, mu, sd, beta)
    keep = np.ones(len(ll), bool)
    if args.gate:
        keep = ~SP.is_glitch(z["cnn_hm"], z["cnn_lm"])
    snr = z["network_snr"].astype(np.float64)
    print(f"foreground: {len(ll)} injections, loglr median {np.median(ll):.2f}, "
          f"max {ll.max():.2f}; gate keeps {keep.mean():.3f}")

    order = np.sort(background)[::-1]
    far = tv * np.arange(1, len(order) + 1) / T_yr
    rows = []
    print(f"\n{'loglr':>9}{'FAR [1/yr]':>13}{'efficiency':>12}")
    for target in (100.0, 30.0, 10.0, 3.0, 1.0):
        idx = np.searchsorted(far, target)
        if idx >= len(order):
            print(f"{'-':>9}{target:>13.0f}   not resolvable "
                  f"({tv*len(order)/T_yr:.2f}/yr in total)")
            continue
        thr = float(order[idx])
        e = float(((ll > thr) & keep).mean())
        rows.append({"far_per_yr": target, "loglr_threshold": thr, "efficiency": e})
        print(f"{thr:>9.2f}{target:>13.0f}{e:>12.3f}")

    by_snr = {}
    if rows:
        thr = rows[-1]["loglr_threshold"]
        edges = [6, 8, 10, 12, 15, 20, 25, 32, 40]
        print(f"\nefficiency vs network SNR at FAR {rows[-1]['far_per_yr']:.0f}/yr")
        for a, b in zip(edges[:-1], edges[1:]):
            m = (snr >= a) & (snr < b)
            if m.any():
                by_snr[f"{a}-{b}"] = float(((ll[m] > thr) & keep[m]).mean())
                print(f"  {a:>2}-{b:<3} n={int(m.sum()):>5}  {by_snr[f'{a}-{b}']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"fold": args.fold, "sigma_norm": norm,
                   "background_livetime_yr": T_yr, "n_lags": n_lags,
                   "n_background": int(background.size), "gate_applied": args.gate,
                   "thresholds": rows, "efficiency_vs_snr": by_snr}, fh, indent=2)
    np.savez_compressed(f"{args.out}_background.npz",
                        loglr=background.astype(np.float32),
                        background_livetime_s=T)
    np.savez_compressed(f"{args.out}_foreground.npz", loglr=ll.astype(np.float32),
                        keep=keep, network_snr=snr.astype(np.float32))
    print(f"\nwrote {args.out}.json / _background.npz / _foreground.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
