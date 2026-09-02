#!/usr/bin/env python
"""Apply the CNN glitch gate to the loudest time-slide triggers.

WHY ONLY THE LOUDEST. The gate costs a Q-transform per detector and a backward pass
through the glitch arm, and there are 3.7 million clustered slide triggers above net
sigma 6. Upstream does not gate them all either: it prunes to candidates and background
survivors and gates those. The threshold at any quoted false-alarm rate lies far out in
the tail, so gating the top K per channel is exact for every threshold at or above the
K-th trigger — and the script refuses to report a rate whose threshold falls below it.

WHY THE PAIRING MATTERS. A slide trigger is H1 at grid point i against L1 at j = i - k.
Both legs are cropped at the attention peak of the H1 leg, so the same L1 window reached
through a different lag is a different input. Windows are deduplicated (the loud tail is
a handful of glitches seen many times) but crops are not.

  scripts/remote.sh sbatch jobs/job_gate.sh --background data_cache/background --top 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import scipy.ndimage  # noqa: F401,E402
import scipy.signal  # noqa: F401,E402
from gwpy.frequencyseries import FrequencySeries  # noqa: F401,E402
from gwpy.timeseries import TimeSeries  # noqa: F401,E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from far_curve import cluster  # noqa: E402
from madgrav_ml.data.representation import (  # noqa: E402
    TileSpec, notch, notch_lines_for, whiten,
)
from madgrav_ml.data.strain import (  # noqa: E402
    SegmentReader, available_segments, load_reference_psd, load_segments,
)
from madgrav_ml.eval import coherence as COH  # noqa: E402
from madgrav_ml.eval import specialists as SP  # noqa: E402
from madgrav_ml.eval.background import make_slide_plan  # noqa: E402
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402
from madgrav_ml.models.arms import GlitchArm, SpecialistCNN  # noqa: E402

SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
PSD_DIR = REPO / ".reference/MADGRAV/data/o3a_search_prep"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", type=Path, default=REPO / "data_cache/background")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top", type=int, default=4000, help="triggers gated per channel")
    ap.add_argument("--n-lags", type=int, default=100000)
    ap.add_argument("--lag-step", type=float, default=4.0)
    ap.add_argument("--cluster-seconds", type=float, default=4.0)
    ap.add_argument("--keep-above", type=float, default=6.0)
    ap.add_argument("--window-seconds", type=float, default=4.0)
    args = ap.parse_args()
    out = args.out or (args.background.parent / f"{args.background.name}_gated.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(cls, rel):
        m = cls()
        m.load_state_dict(torch.load(REPO / ".reference/MADGRAV" / rel,
                                     map_location="cpu"), strict=True)
        return m.eval().to(device)

    arm = load(GlitchArm, "lr_cascade/p1v42/arm_deploy_seed0.pt")
    hm_net = load(SpecialistCNN, "search_mode/hm_native_seed0.pt")
    lm_net = load(SpecialistCNN, "search_mode/lm_native_seed0.pt")

    files = sorted(args.background.glob("bg_*.npz"))
    segs, stride, lo, nfft = [], None, None, None
    for f in files:
        z = np.load(f)
        segs.append({"h": z["score_H1"].astype(np.float64),
                     "l": z["score_L1"].astype(np.float64),
                     "ch": z["coeff_H1"], "cl": z["coeff_L1"],
                     "gh": z["centroid_H1"].astype(np.float64),
                     "gl": z["centroid_L1"].astype(np.float64),
                     "gps": z["gps"], "span": z["span"]})
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
    n_lags = min(args.n_lags, 2 * ((min_n - 1) // shift))
    plan = make_slide_plan(coincident_s, n_lags, lag_step_s=args.lag_step)
    half = int(round(0.5 * args.cluster_seconds / stride))
    print(f"{len(segs)} segments, {n_lags} lags = "
          f"{plan.background_livetime_yr:.2f} yr", flush=True)

    # --- find the loudest triggers per channel --------------------------------
    trig = {"massive": [], "general": []}
    for lag in plan.lags_s:
        k = int(round(lag / stride))
        for si, s in enumerate(segs):
            n = len(s["sh"])
            if n <= abs(k) or n <= 2 * half:
                continue
            j = (np.arange(n) - k) % n
            net = (s["sh"] + s["sl"][j]) / np.sqrt(2.0)
            idx = np.flatnonzero(cluster(net, half) & (net > args.keep_above))
            if not idx.size:
                continue
            coh = COH.coherence_from_coefficients(s["ch"][idx], s["cl"][j[idx]], lo, nfft)
            mass = COH.is_massive(coh, s["gh"][idx], s["gl"][j[idx]])
            for name, sel in (("massive", mass), ("general", ~mass)):
                if sel.any():
                    trig[name].append(np.stack([np.full(sel.sum(), si), idx[sel],
                                                j[idx][sel], net[idx][sel]], axis=1))
    for name in trig:
        trig[name] = (np.concatenate(trig[name]) if trig[name]
                      else np.zeros((0, 4)))
        order = np.argsort(-trig[name][:, 3])[:args.top]
        trig[name] = trig[name][order]
        print(f"{name}: {len(trig[name])} gated, net sigma "
              f"{trig[name][:, 3].min():.2f}-{trig[name][:, 3].max():.2f}"
              if len(trig[name]) else f"{name}: none", flush=True)

    # --- the windows those triggers need --------------------------------------
    wanted: set = set()
    for name, t in trig.items():
        for si, i, j, _ in t:
            wanted.add((int(si), "H1", int(i)))
            wanted.add((int(si), "L1", int(j)))
    print(f"{len(wanted)} distinct windows to transform", flush=True)

    guard = FoldGuard.from_segments(
        load_segments(SEGMENTS, ifo="H1") + load_segments(SEGMENTS, ifo="L1"),
        eval_fold=1, n_folds=2, audit_path=REPO / "runs/fold_audit.jsonl")
    with guard.calibration("glitch-gate"):
        have = available_segments(guard.segments(Split.HPO_BG),
                                  REPO / "data_cache/strain")
    by_span = {}
    for s in have:
        by_span.setdefault((s.start, s.end), {})[s.ifo] = s
    spans = [v for k, v in sorted(by_span.items()) if {"H1", "L1"} <= set(v)]

    spec = TileSpec()
    fs = spec.sample_rate
    psds = {i: load_reference_psd(PSD_DIR / f"reference_psd_{i}.npz") for i in ("H1", "L1")}
    lines = {i: notch_lines_for(i, "o1") for i in ("H1", "L1")}
    reader = SegmentReader(REPO / "data_cache/strain", capacity=2)
    n_samp = int(args.window_seconds * fs)

    cache: dict = {}
    t0_cache: dict = {}
    t_start = time.time()
    for w, (si, ifo, idx) in enumerate(sorted(wanted)):
        offset = int(round((segs[si]["gps"][idx] - segs[si]["span"][0]
                            - 0.5 * args.window_seconds) * fs))
        arr = reader.segment(spans[si][ifo])
        raw = arr[offset:offset + n_samp]
        wh = notch(whiten(raw, fs, reference_psd=psds[ifo]), fs, lines=lines[ifo])
        tile, mag = SP.tile_and_magnitude(wh, spec)
        cache[(si, ifo, idx)] = mag
        if ifo == "H1":
            t0_cache[(si, idx)] = int(SP.cam_t0(arm, tile[None], device)[0])
        if (w + 1) % 500 == 0:
            print(f"  {w+1}/{len(wanted)} windows "
                  f"[{(time.time()-t_start)/60:.1f} min]", flush=True)

    # --- gate ------------------------------------------------------------------
    result = {}
    for name, t in trig.items():
        hm = np.zeros(len(t), np.float32)
        lm = np.zeros(len(t), np.float32)
        for r, (si, i, j, _) in enumerate(t):
            si, i, j = int(si), int(i), int(j)
            inputs = SP.specialist_inputs(cache[(si, "H1", i)], cache[(si, "L1", j)],
                                          t0_cache[(si, i)])
            hm[r], lm[r] = SP.specialist_scores(hm_net, lm_net, inputs, device)
        keep = ~SP.is_glitch(hm, lm)
        result[name] = {"net_sigma": t[:, 3].astype(np.float32), "hm": hm, "lm": lm,
                        "kept": keep}
        print(f"{name}: gate keeps {keep.mean():.4f} "
              f"({int(keep.sum())}/{len(keep)}); floor net sigma "
              f"{t[:, 3].min():.2f}", flush=True)

    np.savez_compressed(
        out, background_livetime_s=plan.background_livetime_s,
        **{f"{c}_{k}": v[k] for c, v in result.items()
           for k in ("net_sigma", "hm", "lm", "kept")},
        **{f"{c}_floor": result[c]["net_sigma"].min() for c in result})
    with open(str(out).replace(".npz", ".json"), "w") as fh:
        json.dump({"sigma_norm": norm, "n_lags": n_lags,
                   "background_livetime_yr": plan.background_livetime_yr,
                   "top_per_channel": args.top,
                   "glitch_thresh": SP.GLITCH_THRESH,
                   "kept_fraction": {c: float(v["kept"].mean()) for c, v in result.items()},
                   "gated_floor": {c: float(v["net_sigma"].min()) for c, v in result.items()}},
                  fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
