#!/usr/bin/env python
"""Detection efficiency at fixed false-alarm rate — the project's actual currency.

Combines the two scans: the time-slide background fixes the threshold at a chosen FAR,
and the coincident injection set says what fraction of sources clear it. Both went
through the same tiling, the same model and the same ranking statistic, which is the
only thing that makes the pairing legitimate.

  scripts/remote.sh .venv/bin/python scripts/efficiency_at_far.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.eval.far import TrialsFactor, far_of, threshold_at_far  # noqa: E402
from madgrav_ml.plotting.style import save_figure, use_style  # noqa: E402


def wilson(k: int, n: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson interval. At an efficiency of 0 or 1 the normal interval has zero width,
    which is exactly where an efficiency curve spends most of its length."""
    if n == 0:
        return (float("nan"),) * 2
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--far-curve", type=Path, default=REPO / "runs/_checks/far_curve")
    ap.add_argument("--foreground", type=Path,
                    default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/efficiency_at_far")
    ap.add_argument("--trials", type=int, default=4)
    args = ap.parse_args()

    with open(f"{args.far_curve}.json") as fh:
        cal = json.load(fh)
    bg_blob = np.load(f"{args.far_curve}_background.npz")
    channels = {c: bg_blob[f"net_sigma_{c}"].astype(np.float64)
                for c in ("massive", "general")}
    T = float(bg_blob["background_livetime_s"])
    norm = cal["sigma_norm"]

    z = np.load(args.foreground)
    # The SAME normalisation the background was reduced with. Refitting it on the
    # injections would move the foreground and background onto different scales and
    # produce an efficiency that means nothing.
    sH = (z["score_H1"].astype(np.float64) - norm["muH"]) / norm["sdH"]
    sL = (z["score_L1"].astype(np.float64) - norm["muL"]) / norm["sdL"]
    net = (sH + sL) / np.sqrt(2.0)
    snr = z["network_snr"].astype(np.float64)
    mtot = (z["mass1"] + z["mass2"]).astype(np.float64)

    # Each injection is ranked against the channel its own vetoes put it in — the same
    # branch the background triggers went through. Scoring every injection against the
    # massive background regardless of whether it passes the vetoes would be quoting a
    # threshold the source never had to clear.
    from madgrav_ml.eval import coherence as COH

    massive = COH.is_massive(z["coherence"], z["centroid_H1"], z["centroid_L1"])
    print(f"vetoes: {massive.mean():.3f} of injections reach the massive channel "
          f"(coherence >= {COH.TCOH:.4f} and both centroids < {COH.F_CUT_HZ:.1f} Hz)")
    print(f"        median injection coherence {np.median(z['coherence']):.4f}, "
          f"median centroid H1 {np.median(z['centroid_H1']):.1f} Hz")

    trials = TrialsFactor(2, 2) if args.trials == 4 else args.trials
    T_yr = T / (365.25 * 86400.0)
    floor = (trials.value if hasattr(trials, "value") else trials) / T_yr
    print(f"background {T_yr:.2f} yr; floor {floor:.3g}/yr; stored triggers: "
          + ", ".join(f"{c} {v.size}" for c, v in channels.items()))
    print(f"foreground {net.size} coincident injections, network SNR "
          f"{snr.min():.1f}-{snr.max():.1f}\n")

    targets = [t for t in (100.0, 10.0, 1.0) if t >= floor]
    results = {}
    print(f"\n{'FAR [1/yr]':>11}{'thr massive':>13}{'thr general':>13}"
          f"{'efficiency':>12}{'68% interval':>18}")
    for t in targets:
        thr = {c: threshold_at_far(v, T, far_target=t, trials=trials) if v.size else np.inf
               for c, v in channels.items()}
        # A trigger clears the search if it beats the threshold OF ITS OWN CHANNEL.
        found = np.where(massive, net > thr["massive"], net > thr["general"])
        k = int(found.sum())
        lo, hi = wilson(k, net.size)
        results[t] = {"threshold": thr, "efficiency": k / net.size,
                      "interval": [lo, hi], "n_found": k, "n_total": int(net.size)}
        print(f"{t:>11.0f}{thr['massive']:>13.2f}{thr['general']:>13.2f}"
              f"{k/net.size:>12.3f}   [{lo:.3f}, {hi:.3f}]")

    edges = [6, 8, 10, 12, 15, 20, 25, 32, 40]
    print(f"\nefficiency vs network SNR")
    header = f"{'SNR bin':<10}{'n':>7}" + "".join(f"{'FAR '+format(t,'.0f'):>12}" for t in targets)
    print(header)
    curves = {t: [] for t in targets}
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (snr >= lo_e) & (snr < hi_e)
        if not m.any():
            continue
        line = f"{f'{lo_e}-{hi_e}':<10}{int(m.sum()):>7}"
        for t in targets:
            thr = results[t]["threshold"]
            k = int(np.where(massive[m], net[m] > thr["massive"],
                             net[m] > thr["general"]).sum())
            e = k / int(m.sum())
            curves[t].append((0.5 * (lo_e + hi_e), e, int(m.sum()), k))
            line += f"{e:>12.3f}"
        print(line)

    for c, v in channels.items():
        print(f"\nnet sigma, {c} background: n={v.size} max "
              f"{v.max() if v.size else float('nan'):.2f}")
    print(f"injections: max {net.max():.2f}, median {np.median(net):.2f}; "
          f"in the massive channel max "
          f"{net[massive].max() if massive.any() else float('nan'):.2f}")
    # Efficiency at a handful of thresholds is a coarse view when it is near zero. The
    # FAR each injection actually achieves says how far away it is, and in which
    # direction, which is the difference between "needs a little more background" and
    # "the statistic does not separate these populations at all".
    inj_far = np.where(
        massive,
        far_of(net, channels["massive"], T, trials=trials) if channels["massive"].size else np.inf,
        far_of(net, channels["general"], T, trials=trials) if channels["general"].size else np.inf,
    )
    print(f"\nFAR achieved by the injections themselves [1/yr]")
    for lo_e, hi_e in ((6, 10), (10, 15), (15, 25), (25, 40)):
        m = (snr >= lo_e) & (snr < hi_e)
        if m.any():
            print(f"  SNR {lo_e:>2}-{hi_e:<3} n={int(m.sum()):>5}  "
                  f"median {np.median(inj_far[m]):>12.3g}  "
                  f"best {inj_far[m].min():>12.3g}")
    print(f"  loudest injection overall reaches FAR {inj_far.min():.3g}/yr "
          f"(floor {floor:.3g}/yr)")

    print(f"\nefficiency vs total mass, at FAR {targets[-1]:.0f}/yr")
    thr = results[targets[-1]]["threshold"]
    for lo_e, hi_e in ((20, 60), (60, 100), (100, 240)):
        m = (mtot >= lo_e) & (mtot < hi_e)
        if m.any():
            e = float(np.where(massive[m], net[m] > thr["massive"],
                               net[m] > thr["general"]).mean())
            print(f"  Mtot {lo_e:>3}-{hi_e:<3}  n={int(m.sum()):>5}  eff={e:.3f}  "
                  f"(massive channel {massive[m].mean():.2f})")

    use_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for t in targets:
        pts = np.array([(x, e) for x, e, _, _ in curves[t]])
        err = np.array([wilson(k, n) for _, _, n, k in curves[t]])
        # Clip: at k = 0 the Wilson bound and the point estimate are the same number
        # up to rounding, and matplotlib rejects a -1e-19 error bar.
        yerr = np.clip([pts[:, 1] - err[:, 0], err[:, 1] - pts[:, 1]], 0.0, None)
        axes[0].errorbar(pts[:, 0], pts[:, 1], yerr=yerr,
                         marker="o", ms=3, capsize=2, label=f"FAR {t:.0f}/yr")
    axes[0].set_xlabel("network SNR"); axes[0].set_ylabel("detection efficiency")
    axes[0].set_ylim(-0.02, 1.02); axes[0].legend(fontsize=8)

    for c, v in channels.items():
        if not v.size:
            continue
        grid = np.linspace(v.min(), max(v.max(), net.max()), 400)
        axes[1].plot(grid, far_of(grid, v, T, trials=trials), label=f"{c} background")
    # The loudest injection, drawn on the same axis. If it sits to the LEFT of every
    # quoted threshold then no source in the campaign is detectable at that FAR, and
    # the efficiency table below is a row of honest zeros rather than a broken pairing.
    axes[1].axvline(net.max(), color="C3", ls="--",
                    label=f"loudest injection ({net.max():.1f})")
    axes[1].axhline(floor, ls=":", color="0.5", label=f"floor {floor:.2g}/yr")
    for t in targets:
        axes[1].plot(results[t]["threshold"]["massive"], t, "o", ms=5, color="C0")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"net $\sigma$"); axes[1].set_ylabel("FAR [1/yr]")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)

    with open(f"{args.out}.json", "w") as fh:
        json.dump({"background_livetime_yr": T_yr, "trials_factor": args.trials,
                   "floor_per_yr": floor, "n_injections": int(net.size),
                   "results": {str(k): v for k, v in results.items()}}, fh, indent=2)
    print(f"\nwrote {args.out}.json / .png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
