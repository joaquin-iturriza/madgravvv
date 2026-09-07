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
from madgrav_ml.eval.far import TrialsFactor  # noqa: E402

TRIALS = 4
YEAR = 365.25 * 86400.0
TARGETS = (100.0, 30.0, 10.0, 3.0, 1.0)


def comoving_weight(distance_mpc: np.ndarray) -> np.ndarray:
    """(dV_c/dV_euclid) / (1+z) at each luminosity distance.

    Two corrections in one factor. The comoving volume element differs from the
    Euclidean `4 pi d_L^2 dd_L` the campaign sampled, and an observed rate is diluted by
    (1+z) because clocks at the source run slow. Both matter at a few Gpc and both are
    routinely forgotten.
    """
    from astropy.cosmology import Planck18
    import astropy.units as u

    d = np.atleast_1d(np.asarray(distance_mpc, dtype=float))
    z = np.array([Planck18.z_at_value(Planck18.luminosity_distance, x * u.Mpc).value
                  for x in d])
    # dV_c/dz / (dV_euclid/dz), evaluated through the shared dz
    dvc_dz = Planck18.differential_comoving_volume(z).to(u.Mpc ** 3 / u.sr).value * 4 * np.pi
    ddl_dz = np.gradient(Planck18.luminosity_distance(np.sort(z)).to(u.Mpc).value,
                         np.sort(z))
    ddl_dz = ddl_dz[np.argsort(np.argsort(z))]
    dve_dz = 4.0 * np.pi * d ** 2 * ddl_dz
    return (dvc_dz / dve_dz) / (1.0 + z), z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--injections", type=Path, required=True)
    ap.add_argument("--far-curve", type=Path, required=True,
                    help="an lr run (`clean_cbc_s42`) whose _background.npz sets the "
                         "thresholds; its sigma_norm and model must match --model")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (REPO / f"runs/_checks/vt_{args.injections.stem}.json")

    m = np.load(args.model)
    own = {g: (m[f"mu{g}"], m[f"sd{g}"], m[f"be{g}"]) for g in (0, 1)}
    norms = {g: dict(zip(("muH", "sdH", "muL", "sdL"), m[f"norm{g}"])) for g in (0, 1)}
    span_fold, model_span = m["fold"], m["span_start"]

    bg = np.load(f"{args.far_curve}_background.npz")
    background = np.sort(bg["loglr"].astype(np.float64))[::-1]
    t_yr = float(bg["background_livetime_s"]) / YEAR

    z = np.load(args.injections)
    if "distance_mpc" not in z.files:
        print("injections are not volumetric; re-run with --distance-max",
              file=sys.stderr)
        return 1
    idx = np.searchsorted(np.sort(model_span), z["span_start"].astype(float))
    fold = span_fold[np.argsort(model_span)][np.clip(idx, 0, len(model_span) - 1)]

    sH = np.empty(len(fold)); sL = np.empty(len(fold))
    for g in (0, 1):
        sel = fold == g
        n = norms[1 - g]
        sH[sel] = (z["score_H1"].astype(np.float64)[sel] - n["muH"]) / n["sdH"]
        sL[sel] = (z["score_L1"].astype(np.float64)[sel] - n["muL"]) / n["sdL"]
    f = LR.features(sH, sL, z["coherence"], z["centroid_H1"], z["centroid_L1"],
                    z["arm_H1"], z["arm_L1"])
    ll = LR.score_held_out(f, fold, own)

    d = z["distance_mpc"].astype(np.float64)
    d_max = float(d.max())
    v_euclid_gpc3 = (4.0 / 3.0) * np.pi * (d_max / 1000.0) ** 3
    w, zs = comoving_weight(d)
    print(f"{len(d)} volumetric injections out to {d_max:.0f} Mpc "
          f"(z up to {zs.max():.2f})")
    print(f"Euclidean shell volume {v_euclid_gpc3:.1f} Gpc^3; "
          f"comoving+time-dilation weight ranges {w.min():.2f}-{w.max():.2f}")
    if "achieved_network_snr" in z.files:
        snr = z["achieved_network_snr"]
        print(f"achieved network SNR: median {np.median(snr):.1f}, "
              f"{100*np.mean(snr > 8):.0f}% above 8")

    rows = []
    print(f"\n{'FAR [1/yr]':>11}{'loglr':>9}{'found':>8}{'eff':>8}"
          f"{'V_euclid':>11}{'V_comoving':>12}")
    for target in TARGETS:
        k = int(np.floor(target * t_yr / TRIALS))
        if k < 1 or k >= len(background):
            print(f"{target:>11.0f}   not resolvable")
            continue
        thr = float(background[k])
        found = ll > thr
        eff = float(found.mean())
        # Monte Carlo over a uniform-in-Euclidean-volume draw: the mean of the weighted
        # indicator times the sampled shell volume.
        v_e = v_euclid_gpc3 * eff
        v_c = v_euclid_gpc3 * float(w[found].sum() / len(d))
        rows.append({"far_per_yr": target, "loglr_threshold": thr,
                     "n_found": int(found.sum()), "efficiency": eff,
                     "sensitive_volume_gpc3_euclid": v_e,
                     "sensitive_volume_gpc3_comoving": v_c})
        print(f"{target:>11.0f}{thr:>9.2f}{int(found.sum()):>8}{eff:>8.4f}"
              f"{v_e:>11.3f}{v_c:>12.3f}")

    print("\nVT for a one-year observation is the comoving column in Gpc^3 yr; "
          "expected detections = astrophysical rate x VT.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"n_injections": int(len(d)), "d_max_mpc": d_max,
                   "z_max": float(zs.max()),
                   "euclid_shell_volume_gpc3": v_euclid_gpc3,
                   "background_livetime_yr": t_yr, "trials_factor": TRIALS,
                   "results": rows}, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
