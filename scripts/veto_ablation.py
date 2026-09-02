#!/usr/bin/env python
"""Is the morphology gate worth what it costs?

The gate discards 57% of injections. That is only a loss if it does not remove a
comparable share of background: a veto that cuts both populations equally leaves the
threshold where it was and buys nothing, while one that cuts background harder than
signal lowers the threshold and wins. The question cannot be answered from the pass
fractions alone, so this measures the end quantity -- efficiency at fixed FAR -- under
each channel definition, everything else held identical.

  scripts/remote.sh .venv/bin/python scripts/veto_ablation.py --n-lags 2000
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
from madgrav_ml.eval.background import make_slide_plan  # noqa: E402
from madgrav_ml.eval.far import TrialsFactor, far_of, threshold_at_far  # noqa: E402

from far_curve import cluster  # noqa: E402

DEFS = {
    "none (front end only)": lambda coh, ch, cl: np.ones(len(coh), bool),
    "coherence only": lambda coh, ch, cl: coh >= COH.TCOH,
    "morphology only": lambda coh, ch, cl: (ch < COH.F_CUT_HZ) & (cl < COH.F_CUT_HZ),
    "both (deployed)": lambda coh, ch, cl: COH.is_massive(coh, ch, cl),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", type=Path, default=REPO / "data_cache/background")
    ap.add_argument("--foreground", type=Path,
                    default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/veto_ablation.json")
    ap.add_argument("--n-lags", type=int, default=2000)
    ap.add_argument("--lag-step", type=float, default=4.0)
    ap.add_argument("--cluster-seconds", type=float, default=4.0)
    ap.add_argument("--keep-above", type=float, default=6.0)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()

    segs, stride, lo, nfft = [], None, None, None
    for f in sorted(args.background.glob("bg_*.npz")):
        z = np.load(f)
        segs.append({"h": z["score_H1"].astype(np.float64),
                     "l": z["score_L1"].astype(np.float64),
                     "ch": z["coeff_H1"], "cl": z["coeff_L1"],
                     "gh": z["centroid_H1"].astype(np.float64),
                     "gl": z["centroid_L1"].astype(np.float64)})
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

    # Slides wrap: `j = (arange(n) - k) % n`, so lag k and lag k+n give the IDENTICAL
    # pairing. Past that point extra lags duplicate triggers and duplicate the livetime
    # they are divided by, so the FAR is unchanged and the threshold is unchanged -- they
    # buy literally nothing while costing full compute. Cap and say so, rather than
    # quoting "63 yr of background" that is eight copies of eight.
    shift = max(1, int(round(args.lag_step / stride)))
    min_n = min(len(s["sh"]) for s in segs)
    # Distinct pairings, not distinct lag VALUES. The ladder is +s, -s, +2s, -2s, ... and
    # j = (i - k) mod n, so the positive and negative arms wrap onto each other: with
    # n = 14396 and s = 4, lag +14392 is lag -4. The set {+-js mod n} has n/s - 1
    # elements, not 2(n/s - 1). Getting this wrong does not change any FAR or threshold
    # -- duplication scales the trigger count and the livetime together -- but it doubles
    # the livetime and the survivor count that get REPORTED, which makes the statistics
    # look twice as good as they are.
    max_lags = (min_n // shift) - 1
    if args.n_lags > max_lags:
        print(f"capping {args.n_lags} lags at {max_lags}: the +/- ladder wraps onto "
              f"itself past that (segments are {min_n} points, lag step {shift})")
        args.n_lags = max_lags

    plan = make_slide_plan(coincident_s, args.n_lags, lag_step_s=args.lag_step)
    half = int(round(0.5 * args.cluster_seconds / stride))

    # One pass over the slides, storing the veto inputs alongside the statistic so every
    # channel definition below is evaluated on exactly the same trigger set.
    net_a, coh_a, ch_a, cl_a = [], [], [], []
    for li, lag in enumerate(plan.lags_s):
        k = int(round(lag / stride))
        for s in segs:
            n = len(s["sh"])
            if n <= abs(k) or n <= 2 * half:
                continue
            j = (np.arange(n) - k) % n
            net = (s["sh"] + s["sl"][j]) / np.sqrt(2.0)
            idx = np.flatnonzero(cluster(net, half) & (net > args.keep_above))
            for b in range(0, idx.size, args.batch):
                sel = idx[b:b + args.batch]
                net_a.append(net[sel])
                coh_a.append(COH.coherence_from_coefficients(
                    s["ch"][sel], s["cl"][j[sel]], lo, nfft))
                ch_a.append(s["gh"][sel]); cl_a.append(s["gl"][j[sel]])
        if (li + 1) % 500 == 0:
            print(f"  {li+1}/{len(plan.lags_s)} lags", flush=True)

    net_b = np.concatenate(net_a); coh_b = np.concatenate(coh_a)
    ch_b = np.concatenate(ch_a); cl_b = np.concatenate(cl_a)
    T = plan.background_livetime_s
    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    tv = trials.value if hasattr(trials, "value") else trials
    floor = tv / plan.background_livetime_yr

    z = np.load(args.foreground)
    sH = (z["score_H1"].astype(np.float64) - norm["muH"]) / norm["sdH"]
    sL = (z["score_L1"].astype(np.float64) - norm["muL"]) / norm["sdL"]
    net_f = (sH + sL) / np.sqrt(2.0)
    coh_f, ch_f, cl_f = z["coherence"], z["centroid_H1"], z["centroid_L1"]

    print(f"\n{plan.background_livetime_yr:.2f} yr background, {net_b.size} loud "
          f"triggers above net sigma {args.keep_above}; {net_f.size} injections; "
          f"FAR floor {floor:.3g}/yr\n")
    header = (f"{'channel definition':<24}{'bg kept':>9}{'inj kept':>9}"
              + "".join(f"{'eff@'+format(t,'.0f'):>10}" for t in (100.0, 10.0, 1.0))
              + f"{'thr@100':>9}")
    print(header)
    out = {}
    for name, fn in DEFS.items():
        mb = fn(coh_b, ch_b, cl_b)
        mf = fn(coh_f, ch_f, cl_f)
        bg = net_b[mb]
        row = f"{name:<24}{mb.mean():>9.4f}{mf.mean():>9.3f}"
        rec = {"background_fraction": float(mb.mean()),
               "injection_fraction": float(mf.mean()), "efficiency": {}}
        thr100 = np.nan
        for t in (100.0, 10.0, 1.0):
            if t < floor or bg.size == 0:
                row += f"{'-':>10}"
                continue
            thr = threshold_at_far(bg, T, far_target=t, trials=trials)
            # An injection counts only if it is IN the channel and beats its threshold.
            e = float((mf & (net_f > thr)).mean())
            rec["efficiency"][str(t)] = {"threshold": thr, "efficiency": e}
            row += f"{e:>10.3f}"
            if t == 100.0:
                thr100 = thr
        row += f"{thr100:>9.2f}"
        print(row)
        out[name] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"background_livetime_yr": plan.background_livetime_yr,
                   "floor_per_yr": floor, "n_injections": int(net_f.size),
                   "channels": out}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
