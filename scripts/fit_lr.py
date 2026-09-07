#!/usr/bin/env python
"""Fit our OWN likelihood ratio with fold discipline, because the shipped one leaks here.

The distributed coefficients in `data/o3a_frozen_lr_off200.npz` were fitted on O3a
background --- `driver_search_multi.fit_lr` does it at run time --- and our background
comes from the same 56-segment O3a set. Upstream protects itself by fitting two models on
two folds and scoring a fold-g trigger with model 1-g. Reusing their frozen artifact on
our data throws that protection away: whichever fold we pick, the model has seen noise
from the segments we then measure a false-alarm rate on, and the rate comes out
optimistic by an unknown amount.

So we fit it ourselves, on our own splits, with their recipe:

  * balanced class weights, ridge on the non-intercept coefficients,
  * the coherence coefficient constrained non-negative -- a physics prior, not something
    a fit should be free to invert,
  * two models over span-disjoint folds, each trigger scored by the one that did not see
    its span.

The background used for fitting is HPO_BG, which the autoencoder never saw either, so the
whole chain stays clean: fit on HPO_TRAIN, select on HPO_VAL, fit the ranking statistic
and measure the rate on disjoint halves of HPO_BG.

  scripts/remote.sh sbatch jobs/job_lr.sh scripts/fit_lr.py --background data_cache/background
"""

from __future__ import annotations

import argparse
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
from madgrav_ml.eval.background import make_slide_plan  # noqa: E402

RIDGE = 1e-3


def fit_one(noise: np.ndarray, signal: np.ndarray):
    """Upstream's `fit_lr`: balanced logistic regression, ridge, coherence >= 0."""
    from scipy.optimize import minimize

    x = np.vstack([noise, signal])
    y = np.concatenate([np.zeros(len(noise)), np.ones(len(signal))])
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    w = np.where(y == 1, len(y) / (2 * max(1, len(signal))),
                 len(y) / (2 * max(1, len(noise))))

    def nll(b):
        p = 1.0 / (1.0 + np.exp(-(z @ b)))
        return -np.sum(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) \
            + RIDGE * np.sum(b[1:] ** 2)

    def grad(b):
        p = 1.0 / (1.0 + np.exp(-(z @ b)))
        g = z.T @ (w * (p - y))
        g[1:] += 2 * RIDGE * b[1:]
        return g

    bounds = [(None, None)] * z.shape[1]
    bounds[3] = (0, None)   # coherence
    beta = minimize(nll, np.zeros(z.shape[1]), jac=grad, method="L-BFGS-B",
                    bounds=bounds).x
    return mu, sd, beta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", type=Path, default=REPO / "data_cache/background")
    ap.add_argument("--foreground", type=Path,
                    default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/lr_model.npz")
    ap.add_argument("--fit-lags", type=int, default=60,
                    help="lags used to build the NOISE training rows. A fit needs a "
                         "sample of the noise distribution, not the whole tail; the "
                         "measurement lags are a separate, much larger set.")
    ap.add_argument("--lag-step", type=float, default=4.0)
    ap.add_argument("--cluster-seconds", type=float, default=4.0)
    ap.add_argument("--fit-floor", type=float, default=0.0,
                    help="net sigma floor for noise rows entering the fit")
    args = ap.parse_args()

    segs, stride, lo, nfft = [], None, None, None
    for f in sorted(args.background.glob("bg_*.npz")):
        z = np.load(f)
        segs.append({k: z[v].astype(np.float64) if v.startswith(("score", "centroid",
                                                                "arm")) else z[v]
                     for k, v in (("h", "score_H1"), ("l", "score_L1"),
                                  ("ch", "coeff_H1"), ("cl", "coeff_L1"),
                                  ("gh", "centroid_H1"), ("gl", "centroid_L1"),
                                  ("ah", "arm_H1"), ("al", "arm_L1"))})
        segs[-1]["span"] = np.asarray(z["span"], dtype=float)
        stride, lo, nfft = float(z["stride"]), int(z["band_lo"]), int(z["band_n"])

    # Span-disjoint folds, upstream's assignment.
    fold = np.array([i % 2 for i in range(len(segs))])
    span_start = np.array([s["span"][0] for s in segs], dtype=float)
    print(f"{len(segs)} spans, folds {fold.tolist()}")

    # PER-FOLD standardisation. Pooling the two folds to compute mu/sd would put a
    # summary of fold g into the features that fold 1-g's model is later declared held
    # out from. Two numbers per detector over ~1e5 samples is a small leak, but it is a
    # leak of exactly the kind everything else here is careful about, and the whole
    # point of the two-fold construction is that the scoring model saw none of the
    # trigger's own span.
    norms = {}
    for g in (0, 1):
        h = np.concatenate([s["h"] for i, s in enumerate(segs) if fold[i] == g])
        l = np.concatenate([s["l"] for i, s in enumerate(segs) if fold[i] == g])
        norms[g] = {"muH": float(h.mean()), "sdH": float(h.std()),
                    "muL": float(l.mean()), "sdL": float(l.std())}
        print(f"  fold {g} sigma_norm: muH={norms[g]['muH']:.6g} "
              f"sdH={norms[g]['sdH']:.6g} muL={norms[g]['muL']:.6g} "
              f"sdL={norms[g]['sdL']:.6g}")

    z = np.load(args.foreground)
    if "span_index" not in z.files:
        print("foreground has no span_index; re-run scan_injections.py", file=sys.stderr)
        return 1
    if "span_start" in z.files:
        # Join injections to spans by GPS, not by a position recorded against a
        # directory listing that may since have changed.
        idx = np.searchsorted(span_start, z["span_start"].astype(float))
        if not np.allclose(span_start[np.clip(idx, 0, len(span_start) - 1)],
                           z["span_start"].astype(float)):
            print("injection span starts do not match the background spans",
                  file=sys.stderr)
            return 1
        sig_fold = fold[np.clip(idx, 0, len(span_start) - 1)]
    else:
        print("foreground has no span_start; falling back to the positional index",
              file=sys.stderr)
        sig_fold = fold[z["span_index"].astype(int)]

    def sig_features(g):
        n = norms[g]
        return LR.features((z["score_H1"].astype(np.float64) - n["muH"]) / n["sdH"],
                           (z["score_L1"].astype(np.float64) - n["muL"]) / n["sdL"],
                           z["coherence"], z["centroid_H1"], z["centroid_L1"],
                           z["arm_H1"], z["arm_L1"])

    plan = make_slide_plan(1.0, args.fit_lags, lag_step_s=args.lag_step)
    half = int(round(0.5 * args.cluster_seconds / stride))
    t0 = time.time()
    models = {}
    for g in (0, 1):
        n = norms[g]
        for i, s in enumerate(segs):
            if fold[i] == g:
                s["sh"] = (s["h"] - n["muH"]) / n["sdH"]
                s["sl"] = (s["l"] - n["muL"]) / n["sdL"]
        rows = []
        for lag in plan.lags_s:
            k = int(round(lag / stride))
            for si, s in enumerate(segs):
                if fold[si] != g:
                    continue
                n = len(s["sh"])
                if n <= abs(k) or n <= 2 * half:
                    continue
                j = (np.arange(n) - k) % n
                coh = COH.coherence_from_coefficients(s["ch"], s["cl"][j], lo, nfft)
                net = (s["sh"] + s["sl"][j]) / np.sqrt(2.0)
                m = cluster(net, half) & (net > args.fit_floor)
                if m.any():
                    rows.append(LR.features(s["sh"][m], s["sl"][j][m], coh[m],
                                            s["gh"][m], s["gl"][j][m],
                                            s["ah"][m], s["al"][j][m]))
        noise = np.vstack(rows)
        signal = sig_features(g)[sig_fold == g]
        mu, sd, beta = fit_one(noise, signal)
        models[g] = (mu, sd, beta)
        held = 1 - g
        ho_noise = None
        print(f"fold {g}: {len(noise)} noise rows, {len(signal)} signal rows, "
              f"beta_coh={beta[3]:+.3f}  [{(time.time()-t0)/60:.1f} min]", flush=True)
        print("   beta = " + " ".join(f"{b:+.3f}" for b in beta), flush=True)

    # Sanity: each model must separate the fold it did NOT see, standardised by its own
    # fold's sigma_norm -- the same combination `far_lr.py` will use.
    for g in (0, 1):
        mu, sd, beta = models[g]
        s_other = sig_features(g)[sig_fold == (1 - g)]
        print(f"model[{g}] on held-out fold {1-g} signal: median loglr "
              f"{np.median(LR.log_likelihood_ratio(s_other, mu, sd, beta)):.2f}")

    np.savez(args.out, **{f"mu{g}": models[g][0] for g in (0, 1)},
             **{f"sd{g}": models[g][1] for g in (0, 1)},
             **{f"be{g}": models[g][2] for g in (0, 1)},
             fold=fold, span_start=span_start,
             **{f"norm{g}": np.array([norms[g]["muH"], norms[g]["sdH"],
                                      norms[g]["muL"], norms[g]["sdL"]])
                for g in (0, 1)})
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
