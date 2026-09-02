#!/usr/bin/env python
"""Time slides over the scanned background -> two-channel false-alarm rate curves.

Rolling L1 against H1 inside each segment destroys astrophysical coincidence while
preserving each detector's own noise character, so every lag buys another copy of the
coincident livetime.

TWO CHANNELS, because that is what the deployed pipeline does. A trigger whose centroids
are both below `f_cut` AND whose coherence clears `tcoh` is ranked against the
mass-conditioned background; everything else is ranked against the general one. The
shipped calibration stores exactly this pair, as `far_curve_cond_coh` and
`far_curve_global_coh`. Ranking every trigger against a single pooled background would
be a different, and much weaker, search -- the point of the veto is that the surviving
population is small, so the same statistic buys a lower FAR inside it.

The ranking statistic is upstream's, from `MassiveEventPipeline._fullmag`:

    sigma_d = (recon_error_d - mu_d) / sd_d ;  net = (sigma_H1 + sigma_L1) / sqrt(2)

the SUM over sqrt(2), not the quadrature sum.

  scripts/remote.sh .venv/bin/python scripts/far_curve.py --n-lags 2000
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


def cluster(net: np.ndarray, half_width: int) -> np.ndarray:
    """Keep only local maxima within +-`half_width` samples.

    A loud glitch at 1 s stride lights up every window containing it, so without this one
    excursion is counted four times over. Upstream clusters on 4 s, which is also why
    lags step by 4 s: consecutive lags then cannot return the same trigger twice.
    """
    from scipy.ndimage import maximum_filter1d

    if half_width <= 0:
        return np.ones(net.shape, dtype=bool)
    return net >= maximum_filter1d(net, size=2 * half_width + 1, mode="nearest")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", type=Path, default=REPO / "data_cache/background")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/far_curve")
    ap.add_argument("--n-lags", type=int, default=2000)
    ap.add_argument("--lag-step", type=float, default=4.0)
    ap.add_argument("--cluster-seconds", type=float, default=4.0)
    ap.add_argument("--keep-above", type=float, default=6.0,
                    help="compute vetoes and store triggers above this net sigma; the "
                         "quoted thresholds are far above it and coherence is the "
                         "expensive part")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()

    files = sorted(args.background.glob("bg_*.npz"))
    if not files:
        print(f"no background shards in {args.background}", file=sys.stderr)
        return 1

    segs, stride, lo, nfft = [], None, None, None
    for f in files:
        z = np.load(f)
        if "coeff_H1" not in z.files:
            print(f"{f.name} predates the coherence scan; re-run scan_background.py",
                  file=sys.stderr)
            return 1
        segs.append({
            "h": z["score_H1"].astype(np.float64), "l": z["score_L1"].astype(np.float64),
            "ch": z["coeff_H1"], "cl": z["coeff_L1"],
            "gh": z["centroid_H1"].astype(np.float64),
            "gl": z["centroid_L1"].astype(np.float64),
        })
        stride, lo, nfft = float(z["stride"]), int(z["band_lo"]), int(z["band_n"])
    n_points = sum(len(s["h"]) for s in segs)
    coincident_s = n_points * stride
    print(f"{len(segs)} segments, {n_points} grid points at {stride} s stride "
          f"= {coincident_s / 86400:.3f} d coincident livetime")

    all_h = np.concatenate([s["h"] for s in segs])
    all_l = np.concatenate([s["l"] for s in segs])
    norm = {"muH": float(all_h.mean()), "sdH": float(all_h.std()),
            "muL": float(all_l.mean()), "sdL": float(all_l.std())}
    print(f"sigma_norm: muH={norm['muH']:.6g} sdH={norm['sdH']:.6g} "
          f"muL={norm['muL']:.6g} sdL={norm['sdL']:.6g}")
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
    kept = {"massive": [], "general": []}
    n_clustered = 0

    for li, lag in enumerate(plan.lags_s):
        k = int(round(lag / stride))
        for s in segs:
            n = len(s["sh"])
            if n <= abs(k) or n <= 2 * half:
                continue
            # np.roll(x, k)[i] == x[i - k], so H1 at grid point i is paired with L1 at
            # (i - k) mod n. The same index map has to be used for the coefficients and
            # the centroids or the veto would be evaluated on a different pair than the
            # statistic.
            j = (np.arange(n) - k) % n
            net = (s["sh"] + s["sl"][j]) / np.sqrt(2.0)
            m = cluster(net, half)
            n_clustered += int(m.sum())
            idx = np.flatnonzero(m & (net > args.keep_above))
            if not idx.size:
                continue
            for b in range(0, idx.size, args.batch):
                sel = idx[b:b + args.batch]
                coh = COH.coherence_from_coefficients(
                    s["ch"][sel], s["cl"][j[sel]], lo, nfft)
                massive = COH.is_massive(coh, s["gh"][sel], s["gl"][j[sel]])
                kept["massive"].append(net[sel][massive])
                kept["general"].append(net[sel][~massive])
        if (li + 1) % 500 == 0:
            print(f"  {li+1}/{len(plan.lags_s)} lags, "
                  f"{sum(v.size for v in kept['massive'])} massive / "
                  f"{sum(v.size for v in kept['general'])} general stored", flush=True)

    T = plan.background_livetime_s
    channels = {c: (np.concatenate(v) if v else np.array([])) for c, v in kept.items()}
    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    tv = trials.value if hasattr(trials, "value") else trials
    floor = tv / plan.background_livetime_yr
    print(f"\nbackground: {plan.background_livetime_yr:.2f} yr over "
          f"{len(plan.lags_s)} lags; {n_clustered} clustered triggers")
    for c, v in channels.items():
        print(f"  {c:8s}: {v.size} stored above net sigma {args.keep_above}"
              + (f", max {v.max():.2f}" if v.size else ""))
    frac = channels["massive"].size / max(1, sum(v.size for v in channels.values()))
    print(f"  coherence+morphology keeps {frac:.4f} of loud background triggers")
    print(f"smallest resolvable FAR at trials={args.trials}: {floor:.3g}/yr")

    rows = {}
    for c, v in channels.items():
        rows[c] = []
        for target in (100.0, 10.0, 1.0, 0.1):
            if target < floor or v.size == 0:
                continue
            thr = threshold_at_far(v, T, far_target=target, trials=trials)
            rows[c].append({"far_per_yr": target, "net_sigma_threshold": thr})
            print(f"  [{c:8s}] FAR {target:>6}/yr  ->  net sigma >= {thr:.3f}")

    out = {
        "sigma_norm": norm, "stride_s": stride,
        "cluster_seconds": args.cluster_seconds,
        "coincident_livetime_s": coincident_s,
        "slide_plan": plan.as_dict(),
        "n_clustered_triggers": n_clustered,
        "keep_above_net_sigma": args.keep_above,
        "trials_factor": args.trials,
        "smallest_resolvable_far_per_yr": floor,
        "massive_fraction_of_loud_background": frac,
        "veto": {"tcoh": COH.TCOH, "f_cut_hz": COH.F_CUT_HZ,
                 "coherence_band_hz": list(COH.COHERENCE_BAND_HZ),
                 "lag_samples": COH.LAG_SAMPLES},
        "thresholds": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(
        f"{args.out}_background.npz", background_livetime_s=T,
        net_sigma_massive=channels["massive"].astype(np.float32),
        net_sigma_general=channels["general"].astype(np.float32))

    import matplotlib.pyplot as plt

    from madgrav_ml.plotting.style import save_figure, use_style

    use_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for c, v in channels.items():
        if not v.size:
            continue
        grid = np.linspace(v.min(), v.max(), 300)
        ax.plot(grid, far_of(grid, v, T, trials=trials), label=f"{c} channel")
    ax.axhline(floor, ls=":", color="0.5", label=f"floor {floor:.2g}/yr")
    ax.set_yscale("log")
    ax.set_xlabel(r"net $\sigma$"); ax.set_ylabel("false-alarm rate [1/yr]")
    ax.set_title(f"{plan.background_livetime_yr:.1f} yr of slides, trials={args.trials}",
                 fontsize=9)
    ax.legend(fontsize=8)
    save_figure(fig, args.out)
    print(f"\nwrote {args.out}.json / .png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
