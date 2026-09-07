#!/usr/bin/env python
"""Sensitive volume and VT from a volumetric injection campaign.

Efficiency at fixed false-alarm rate answers "of the sources I injected, how many did I
find". Its value depends on what was injected, and the SNR-uniform draw used everywhere
else in this note is an arbitrary choice. VT removes that: sources are distributed
uniformly in comoving volume, which is physics rather than a choice, so integrating the
recovery probability over distance gives a reach in Gpc^3 yr that connects directly to
an observable --- the expected number of detections is the astrophysical rate times VT.

The campaign this reads must therefore be drawn over DISTANCE, not over SNR
(`scan_injections.py --distance-max`). An SNR-targeted campaign cannot supply a
sensitive volume at all: every source in it is placed at whatever distance makes it as
loud as the draw demanded, so the distance distribution carries no information.

TWO VOLUMES ARE REPORTED and they differ by a lot at these distances. The Euclidean one
is what the draw actually samples. The comoving one applies the volume element a source
population really follows plus the (1+z) time dilation of the observed rate, computed
from each injection's own luminosity distance. At 5 Gpc the redshift is near 0.8 and the
two disagree by more than a factor of two, so the Euclidean figure is reported only
because it is what the sampling assumed, and the comoving one is the number to quote.

  scripts/remote.sh .venv/bin/python scripts/sensitive_volume.py \
      --injections data_cache/injections/vol.npz --far-curve runs/_checks/clean_cbc_s42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.eval import likelihood as LR  # noqa: E402

YEAR = 365.25 * 86400.0
TARGETS = (100.0, 30.0, 10.0, 3.0, 1.0)


def comoving_weight(distance_mpc: np.ndarray, n_grid: int = 4000):
    """(dV_c/dV_euclid) / (1+z) at each luminosity distance, plus the redshifts.

    Two corrections in one factor. The comoving volume element differs from the
    Euclidean `4 pi d_L^2 dd_L` the campaign sampled, and an observed rate is diluted by
    (1+z) because clocks at the source run slow. Both matter at a few Gpc and both are
    routinely forgotten.

    Everything is built on a dense monotone redshift GRID and interpolated onto the
    samples, rather than evaluated per sample. Three reasons, and the first two were
    found by running it: differentiating d_L against 12000 sorted sample redshifts hits
    duplicate abscissae and returns NaN, inverting d_L per sample is a numerical
    root-find repeated 12000 times, and a grid makes both the inversion and the
    derivative exact-by-construction monotone.
    """
    import astropy.units as u
    from astropy.cosmology import Planck18

    d = np.atleast_1d(np.asarray(distance_mpc, dtype=float))
    z_grid = np.linspace(1e-6, 3.0, n_grid)
    dl_grid = Planck18.luminosity_distance(z_grid).to(u.Mpc).value
    if d.max() > dl_grid[-1]:
        raise ValueError(f"distance {d.max():.0f} Mpc is past the z=3 grid edge")

    # z(d_L) by interpolation on a strictly increasing grid -- no root-finding.
    z = np.interp(d, dl_grid, z_grid)

    dvc_dz = (Planck18.differential_comoving_volume(z_grid).to(u.Mpc ** 3 / u.sr).value
              * 4.0 * np.pi)
    ddl_dz = np.gradient(dl_grid, z_grid)          # strictly increasing: no zero spacing
    dve_dz = 4.0 * np.pi * dl_grid ** 2 * ddl_dz
    w_grid = (dvc_dz / dve_dz) / (1.0 + z_grid)
    return np.interp(d, dl_grid, w_grid), z


def one_seed(inj_path: Path, far_path: Path, model_path: Path, trials: int) -> dict:
    """Efficiency and sensitive volume for one seed's campaign."""
    m = np.load(model_path)
    own = {g: (m[f"mu{g}"], m[f"sd{g}"], m[f"be{g}"]) for g in (0, 1)}
    norms = {g: dict(zip(("muH", "sdH", "muL", "sdL"), m[f"norm{g}"])) for g in (0, 1)}
    span_fold, model_span = m["fold"], m["span_start"]

    bg = np.load(f"{far_path}_background.npz")
    background = np.sort(bg["loglr"].astype(np.float64))[::-1]
    t_yr = float(bg["background_livetime_s"]) / YEAR
    # The threshold read off this background is only meaningful for a foreground scored
    # by the same statistic. Nothing could check that before far_lr.py started recording
    # which model wrote it.
    if "model_path" in bg.files:
        wrote = str(bg["model_path"])
        if wrote != str(model_path):
            raise SystemExit(
                f"{far_path}_background.npz was produced with {wrote!r} but --model is "
                f"{str(model_path)!r}: the threshold and the foreground would come from "
                f"different statistics")
        gated = bool(bg["gate_applied"]) if "gate_applied" in bg.files else False
    else:
        raise SystemExit(f"{far_path}_background.npz predates model provenance; re-run "
                         f"far_lr.py so the background records its statistic")

    z = np.load(inj_path)
    if "distance_mpc" not in z.files:
        raise SystemExit(f"{inj_path} is not a volumetric campaign; re-run "
                         f"scan_injections.py --distance-max")

    # Same hard-fail GPS join as fit_lr.py and far_lr.py. searchsorted always returns an
    # index, so without this an injection whose span is absent from the model silently
    # takes a neighbour's fold and half of those are scored by the model fitted on their
    # own span.
    order = np.argsort(model_span)
    sorted_span = model_span[order]
    idx = np.clip(np.searchsorted(sorted_span, z["span_start"].astype(float)),
                  0, len(sorted_span) - 1)
    if not np.allclose(sorted_span[idx], z["span_start"].astype(float)):
        raise SystemExit(f"{inj_path}: injection spans are not the spans {model_path} "
                         f"was fitted on; refit or rescan")
    fold = span_fold[order][idx]

    sH = np.empty(len(fold)); sL = np.empty(len(fold))
    for g in (0, 1):
        sel = fold == g
        n = norms[1 - g]
        sH[sel] = (z["score_H1"].astype(np.float64)[sel] - n["muH"]) / n["sdH"]
        sL[sel] = (z["score_L1"].astype(np.float64)[sel] - n["muL"]) / n["sdL"]
    f = LR.features(sH, sL, z["coherence"], z["centroid_H1"], z["centroid_L1"],
                    z["arm_H1"], z["arm_L1"])
    ll = LR.score_held_out(f, fold, own)

    # The foreground must carry whatever selection the background carried.
    keep = np.ones(len(ll), bool)
    if gated:
        from madgrav_ml.eval import specialists as SP

        keep = ~SP.is_glitch(z["cnn_hm"], z["cnn_lm"])

    d = z["distance_mpc"].astype(np.float64)
    # The requested horizon, not the largest realised draw, and the attempted count, not
    # the surviving one: a dropped injection is an unrecovered injection and belongs in
    # the denominator.
    d_max = float(z["distance_max_requested_mpc"]) if \
        "distance_max_requested_mpc" in z.files else float(d.max())
    n_attempted = int(z["n_attempted"]) if "n_attempted" in z.files else len(d)
    v_euclid = (4.0 / 3.0) * np.pi * (d_max / 1000.0) ** 3
    w, zs = comoving_weight(d)

    out = {"d_max_mpc": d_max, "z_max": float(zs.max()), "n_attempted": n_attempted,
           "v_euclid_gpc3": v_euclid, "weight_range": [float(w.min()), float(w.max())],
           "by_far": {}}
    for target in TARGETS:
        k = int(np.floor(target * t_yr / trials))
        if k < 1 or k >= len(background):
            continue
        found = (ll > float(background[k])) & keep
        out["by_far"][target] = {
            "n_found": int(found.sum()), "n_attempted": n_attempted,
            "sensitive_volume_gpc3_euclid": v_euclid * found.sum() / n_attempted,
            "sensitive_volume_gpc3_comoving": v_euclid * float(w[found].sum()) / n_attempted,
        }
    print(f"{inj_path.name}: {n_attempted} attempted to {d_max:.0f} Mpc "
          f"(z<={zs.max():.2f}), comoving weight {w.min():.3f}-{w.max():.3f}"
          + (", gate applied" if gated else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--injections", type=Path, nargs="+", required=True,
                    help="one volumetric campaign per seed; VT is quoted as a mean and "
                         "a range over them, because a single seed cannot be compared "
                         "against a spread that is only known across seeds")
    ap.add_argument("--far-curve", type=Path, nargs="+", required=True,
                    help="matching lr run per seed; its _background.npz sets the "
                         "thresholds and names the model that produced it")
    ap.add_argument("--model", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not len(args.injections) == len(args.far_curve) == len(args.model):
        ap.error("--injections, --far-curve and --model must have the same length: "
                 "each is one seed's campaign, background and statistic")
    out = args.out or (REPO / f"runs/_checks/vt_{args.injections[0].stem}.json")

    from madgrav_ml.eval.far import TrialsFactor

    trials = TrialsFactor(2, 2)
    per_seed = []
    for inj_path, far_path, model_path in zip(args.injections, args.far_curve,
                                              args.model):
        per_seed.append(one_seed(inj_path, far_path, model_path, trials.value))

    print(f"\n{'FAR [1/yr]':>11}{'found':>8}{'of':>8}{'V_euclid':>11}"
          f"{'V_comoving':>12}{'68% interval':>20}")
    rows = []
    for target in TARGETS:
        vals = [r["by_far"][target] for r in per_seed if target in r["by_far"]]
        if not vals:
            print(f"{target:>11.0f}   not resolvable in every seed")
            continue
        vc = np.array([v["sensitive_volume_gpc3_comoving"] for v in vals])
        ve = np.array([v["sensitive_volume_gpc3_euclid"] for v in vals])
        found = int(np.mean([v["n_found"] for v in vals]))
        n = int(np.mean([v["n_attempted"] for v in vals]))
        # Poisson on the recovered count, which is what actually limits this: the
        # weighted estimator is unbiased but its variance is set by how few injections
        # land close enough to be found at all.
        rel = 1.0 / np.sqrt(max(found, 1))
        rows.append({"far_per_yr": target, "n_found_mean": found, "n_attempted": n,
                     "V_comoving_gpc3_mean": float(vc.mean()),
                     "V_comoving_gpc3_range": [float(vc.min()), float(vc.max())],
                     "V_euclid_gpc3_mean": float(ve.mean()),
                     "poisson_relative_error": float(rel)})
        print(f"{target:>11.0f}{found:>8}{n:>8}{ve.mean():>11.3f}{vc.mean():>12.3f}"
              f"   {vc.mean()*(1-rel):>7.3f}-{vc.mean()*(1+rel):<7.3f}")
        if found < 20:
            print(f"{'':>11}   only {found} recovered: this row is a Poisson estimate "
                  f"at +-{100*rel:.0f}%, not a measurement")

    print("\nVT for a one-year observation is the comoving column in Gpc^3 yr. The "
          "masses are DETECTOR-FRAME: the campaign draws the same component range at "
          "every distance, so `expected detections = rate x VT` holds against a "
          "detector-frame rate, not a source-frame one.")
    print("This is a NETWORK statistic. Constraint C1 makes the single-detector number "
          "primary and there is no single-detector variant here, because the ranking "
          "statistic uses coherence and has no per-detector form.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"per_seed": [{k: v for k, v in r.items() if k != "by_far"}
                                for r in per_seed],
                   "n_seeds": len(per_seed), "trials_factor": trials.value,
                   "mass_frame": "detector",
                   "single_detector_variant": None,
                   "results": rows}, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
