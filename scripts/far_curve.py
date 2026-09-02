#!/usr/bin/env python
"""Time slides over the scanned background -> a false-alarm rate curve.

The scan is the expensive part and is already done; this is pure numpy. Rolling L1's
score series against H1's inside each segment destroys astrophysical coincidence while
preserving each detector's own noise character, so every lag buys another copy of the
coincident livetime. 1.4 days becomes years.

The ranking statistic is upstream's, read off `MassiveEventPipeline._fullmag`:

    sigma_d = (recon_error_d - mu_d) / sd_d          per detector
    net     = (sigma_H1 + sigma_L1) / sqrt(2)

Note it is the SUM over sqrt(2), not the quadrature sum. That matters: the sum rewards
a coherent excess in both detectors and penalises a deficit in one, while quadrature
would treat a single loud detector the same as two moderate ones.

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

from madgrav_ml.eval.background import make_slide_plan  # noqa: E402
from madgrav_ml.eval.far import TrialsFactor, far_of, threshold_at_far  # noqa: E402


def cluster(net: np.ndarray, half_width: int) -> np.ndarray:
    """Keep only local maxima within +-`half_width` samples.

    A loud glitch at 1 s stride lights up every window that contains it, so without
    this one noise excursion is counted four times over and the background is inflated
    by a factor that has nothing to do with the detector. Upstream clusters on 4 s,
    which is also why `make_slide_plan` steps lags by 4 s: consecutive lags then cannot
    return the same trigger twice.
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
    ap.add_argument("--keep-above", type=float, default=2.0,
                    help="store slide triggers above this net sigma; below it the "
                         "curve is not quotable anyway and the arrays get large")
    ap.add_argument("--trials", type=int, default=4,
                    help="upstream: 2 statistics x 2 arms")
    args = ap.parse_args()

    files = sorted(args.background.glob("bg_*.npz"))
    if not files:
        print(f"no background shards in {args.background}", file=sys.stderr)
        return 1

    segs, stride = [], None
    for f in files:
        with np.load(f) as z:
            segs.append((z["score_H1"].astype(np.float64), z["score_L1"].astype(np.float64)))
            stride = float(z["stride"])
    n_points = sum(len(h) for h, _ in segs)
    coincident_s = n_points * stride
    print(f"{len(segs)} segments, {n_points} grid points at {stride} s stride "
          f"= {coincident_s / 86400:.3f} d coincident livetime")

    # Per-detector normalisation, upstream's sigma_norm. Fitted on the background's own
    # bulk: it is a two-parameter linear rescaling over ~1e5 samples, it does not select
    # on the tail, and without it net sigma is not defined at all. Upstream fits theirs
    # on separate splits and ships the numbers; ours travel with the run.
    all_h = np.concatenate([h for h, _ in segs])
    all_l = np.concatenate([l for _, l in segs])
    norm = {"muH": float(all_h.mean()), "sdH": float(all_h.std()),
            "muL": float(all_l.mean()), "sdL": float(all_l.std())}
    print(f"sigma_norm: muH={norm['muH']:.6g} sdH={norm['sdH']:.6g} "
          f"muL={norm['muL']:.6g} sdL={norm['sdL']:.6g}")

    sig = [((h - norm["muH"]) / norm["sdH"], (l - norm["muL"]) / norm["sdL"])
           for h, l in segs]

    plan = make_slide_plan(coincident_s, args.n_lags, lag_step_s=args.lag_step)
    half = int(round(0.5 * args.cluster_seconds / stride))
    shift = int(round(args.lag_step / stride))

    kept, n_clustered = [], 0
    for i, lag in enumerate(plan.lags_s):
        k = int(round(lag / stride))
        for sh, sl in sig:
            if len(sh) <= abs(k) or len(sh) <= 2 * half:
                continue
            net = (sh + np.roll(sl, k)) / np.sqrt(2.0)
            m = cluster(net, half)
            n_clustered += int(m.sum())
            v = net[m]
            v = v[v > args.keep_above]
            if v.size:
                kept.append(v)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(plan.lags_s)} lags, {sum(len(v) for v in kept)} "
                  f"stored triggers", flush=True)

    background = np.concatenate(kept) if kept else np.array([])
    T = plan.background_livetime_s
    print(f"\nbackground: {plan.background_livetime_yr:.2f} yr over "
          f"{len(plan.lags_s)} lags; {n_clustered} clustered triggers, "
          f"{background.size} stored above net sigma {args.keep_above}")

    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    floor = (trials.value if hasattr(trials, "value") else trials) / plan.background_livetime_yr
    print(f"smallest resolvable FAR at trials={args.trials}: {floor:.3g}/yr")

    rows = []
    for target in (100.0, 10.0, 1.0, 0.1):
        if target < floor:
            print(f"FAR {target}/yr: BELOW the resolvable floor, not quoted")
            continue
        thr = threshold_at_far(background, T, far_target=target, trials=trials)
        rows.append({"far_per_yr": target, "net_sigma_threshold": thr})
        print(f"FAR {target:>6}/yr  ->  net sigma >= {thr:.3f}")

    out = {
        "sigma_norm": norm,
        "stride_s": stride,
        "cluster_seconds": args.cluster_seconds,
        "coincident_livetime_s": coincident_s,
        "slide_plan": plan.as_dict(),
        "n_clustered_triggers": n_clustered,
        "keep_above_net_sigma": args.keep_above,
        "trials_factor": args.trials,
        "smallest_resolvable_far_per_yr": floor,
        "thresholds": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(f"{args.out}_background.npz", net_sigma=background.astype(np.float32),
                        background_livetime_s=T)

    import matplotlib.pyplot as plt

    from madgrav_ml.plotting.style import save_figure, use_style

    use_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    grid = np.linspace(max(args.keep_above, background.min()), background.max(), 300)
    ax.plot(grid, far_of(grid, background, T, trials=trials))
    ax.axhline(floor, ls=":", label=f"resolvable floor {floor:.2g}/yr")
    ax.set_yscale("log")
    ax.set_xlabel(r"net $\sigma$")
    ax.set_ylabel("false-alarm rate [1/yr]")
    ax.set_title(f"{plan.background_livetime_yr:.1f} yr of time slides, "
                 f"trials={args.trials}", fontsize=9)
    ax.legend(fontsize=8)
    save_figure(fig, args.out)
    print(f"\nwrote {args.out}.json / .png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
