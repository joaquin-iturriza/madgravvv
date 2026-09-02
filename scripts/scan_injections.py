#!/usr/bin/env python
"""Coincident injections through the same path as the background — the foreground.

An efficiency at fixed FAR needs both halves measured with the SAME statistic. The
background scan gives per-detector score series; this gives per-detector scores for
injected sources. What makes them comparable is that one source is projected onto both
detectors with a single common amplitude scale, the correct antenna factors, and the
geocentre time delay — so the H1/L1 relationship the ranking statistic reads is the one
a real source would produce. Two independently drawn single-detector injections would
score well on a per-detector arm and mean nothing coincident.

Injections go into HPO_BG, the same split the background comes from. That is the
standard construction: efficiency is measured by adding signals to the data the search
runs on, and HPO_BG is neither fitted to nor selected against.

Stores per-detector scores rather than a net sigma, so the normalisation can be applied
downstream from the background's own fit instead of being baked in here.

  scripts/remote.sh sbatch jobs/job_scan_injections.sh \
      --checkpoint runs/madgrav/<run>/models/model_best.pt --n-injections 4000
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import scipy.ndimage  # noqa: F401,E402
import scipy.signal  # noqa: F401,E402
from gwpy.frequencyseries import FrequencySeries  # noqa: F401,E402
from gwpy.timeseries import TimeSeries  # noqa: F401,E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import lal  # noqa: F401,E402
import lalsimulation  # noqa: F401,E402
import torch  # noqa: E402

from madgrav_ml.data.injections import ParameterSampler  # noqa: E402
from madgrav_ml.data.representation import (  # noqa: E402
    TileSpec, make_tile, notch, notch_lines_for, whiten,
)
from madgrav_ml.data.strain import (  # noqa: E402
    SegmentReader, available_segments, load_reference_psd, load_segments,
)
from madgrav_ml.data.waveforms import InjectionEngine, LALWaveformBackend  # noqa: E402
from madgrav_ml.eval import coherence as COH  # noqa: E402
from madgrav_ml.eval import specialists as SP  # noqa: E402
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402
from madgrav_ml.models.arms import GlitchArm, SpecialistCNN  # noqa: E402
from madgrav_ml.models.cae import BaselineCAE  # noqa: E402

SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
PSD_DIR = REPO / ".reference/MADGRAV/data/o3a_search_prep"
_CTX: dict = {}


def _init(psds, lines, spec, fs, engine, gate):
    """Workers own the glitch gate too.

    The gate needs a BACKWARD pass through the arm (Grad-CAM), which is small enough on
    CPU and keeps the ~2 MB native magnitudes inside the worker instead of shipping them
    to the parent — 8000 injections x 2 detectors of those would be 37 GB through a pipe.
    """
    _CTX.update(psds=psds, lines=lines, spec=spec, fs=fs, engine=engine, gate=gate)


def _pair(args):
    """One injection projected onto both detectors: tiles, vetoes, gate.

    Coherence and the specialists must be computed on the SAME whitened series the tile
    came from, and both detectors must carry the same source with its true relative
    amplitude and arrival delay — that relationship is the entire content of both vetoes.
    """
    raw_h1, raw_l1, gps, params = args
    try:
        tiles, mags, coeffs, cents = [], [], [], []
        lo = n = None
        for ifo, raw in (("H1", raw_h1), ("L1", raw_l1)):
            w = whiten(raw, _CTX["fs"], reference_psd=_CTX["psds"][ifo])
            w = notch(w, _CTX["fs"], lines=_CTX["lines"][ifo])
            w = _CTX["engine"].inject(w, params, ifo, gps)
            tile, mag = SP.tile_and_magnitude(w, _CTX["spec"])
            c, lo, n = COH.band_coefficients(w, _CTX["fs"])
            tiles.append(tile[None] if tile.ndim == 2 else tile)
            mags.append(mag)
            coeffs.append(c[0])
            cents.append(float(COH.centroid(w, _CTX["fs"])[0]))
        g = _CTX["gate"]
        (hm, lm), t0 = SP.gate_scores(g["arm"], g["hm"], g["lm"], mags[0], mags[1],
                                      tiles[0][0], g["device"])
        return (np.stack(tiles), np.stack(coeffs), np.array(cents),
                lo, n, hm, lm, t0, params)
    except Exception:
        return None


@torch.no_grad()
def score(model, tiles, device):
    x = torch.from_numpy(np.asarray(tiles)).to(device)
    return model.reconstruction_error(x, reduction="none").cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "data_cache/injections/foreground.npz")
    ap.add_argument("--n-injections", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--window-seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--snr-range", type=float, nargs=2, default=(6.0, 40.0),
                    help="wider than the training band (8, 25) on purpose: an "
                         "efficiency curve needs points where it is near 0 and near 1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = BaselineCAE()
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    spec = TileSpec()
    fs = spec.sample_rate
    psds = {i: load_reference_psd(PSD_DIR / f"reference_psd_{i}.npz") for i in ("H1", "L1")}
    lines = {i: notch_lines_for(i, "o1") for i in ("H1", "L1")}
    engine = InjectionEngine(
        backend=LALWaveformBackend(), psds=psds, notch_lines=lines, sample_rate=fs,
        window_seconds=args.window_seconds, snr_convention="network",
    )

    segs = load_segments(SEGMENTS, ifo="H1") + load_segments(SEGMENTS, ifo="L1")
    guard = FoldGuard.from_segments(segs, eval_fold=1, n_folds=2,
                                    audit_path=REPO / "runs/fold_audit.jsonl")
    with guard.calibration(f"injection-scan:{args.checkpoint.parent.parent.name}"):
        wanted = guard.segments(Split.HPO_BG)
    have = available_segments(wanted, REPO / "data_cache/strain")
    by_span: dict[tuple[float, float], dict] = {}
    for s in have:
        by_span.setdefault((s.start, s.end), {})[s.ifo] = s
    spans = [(k, v) for k, v in sorted(by_span.items()) if {"H1", "L1"} <= set(v)]
    if not spans:
        print("no coincident spans with both detectors cached", file=sys.stderr)
        return 1
    live = np.array([e - s for (s, e), _ in spans], dtype=float)
    print(f"{len(spans)} coincident spans, {live.sum()/86400:.2f} d; "
          f"{args.n_injections} injections, network SNR "
          f"{args.snr_range[0]}-{args.snr_range[1]}", flush=True)

    rng = np.random.default_rng(args.seed)
    sampler = ParameterSampler(snr_range=tuple(args.snr_range))
    reader = SegmentReader(REPO / "data_cache/strain", capacity=2)
    n_samp = int(args.window_seconds * fs)

    # Group by span so each segment is read once rather than once per injection.
    choice = rng.choice(len(spans), size=args.n_injections, p=live / live.sum())
    t0 = time.time()
    rows, sH, sL, coh, cen, hms, lms, t0s = [], [], [], [], [], [], [], []
    band_lo = band_n = None

    def load(cls, rel):
        m = cls()
        m.load_state_dict(torch.load(REPO / ".reference/MADGRAV" / rel,
                                     map_location="cpu"), strict=True)
        return m.eval()

    gate = {"arm": load(GlitchArm, "lr_cascade/p1v42/arm_deploy_seed0.pt"),
            "hm": load(SpecialistCNN, "search_mode/hm_native_seed0.pt"),
            "lm": load(SpecialistCNN, "search_mode/lm_native_seed0.pt"),
            "device": torch.device("cpu")}

    with Pool(args.workers, initializer=_init,
              initargs=(psds, lines, spec, fs, engine, gate)) as pool:
        for si, ((start, end), pair) in enumerate(spans):
            n_here = int((choice == si).sum())
            if not n_here:
                continue
            arrs = {ifo: reader.segment(pair[ifo]) for ifo in ("H1", "L1")}
            size = min(a.size for a in arrs.values())
            starts = rng.integers(0, size - n_samp, size=n_here)
            work = []
            for i in starts:
                gps = start + int(i) / fs + 0.5 * args.window_seconds
                work.append((arrs["H1"][i:i + n_samp], arrs["L1"][i:i + n_samp],
                             gps, sampler.draw(rng)))
            for out in pool.imap(_pair, work, chunksize=4):
                if out is None:
                    continue
                tiles, coeffs, cents, band_lo, band_n, hm, lm, cam_t0, params = out
                sc = score(model, tiles, device)
                sH.append(float(sc[0])); sL.append(float(sc[1]))
                coh.append(float(COH.coherence_from_coefficients(
                    coeffs[0:1], coeffs[1:2], band_lo, band_n)[0]))
                cen.append(cents)
                hms.append(hm); lms.append(lm); t0s.append(cam_t0)
                rows.append(params.as_dict())
            el = time.time() - t0
            print(f"[{si+1}/{len(spans)}] {int(start)}  {len(rows)} injections  "
                  f"[{el/60:.1f} min]", flush=True)

    keys = sorted(rows[0]) if rows else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cen = np.asarray(cen, dtype=np.float32)
    np.savez_compressed(
        args.out,
        score_H1=np.array(sH, dtype=np.float32),
        score_L1=np.array(sL, dtype=np.float32),
        coherence=np.array(coh, dtype=np.float32),
        centroid_H1=cen[:, 0], centroid_L1=cen[:, 1],
        cnn_hm=np.array(hms, dtype=np.float32),
        cnn_lm=np.array(lms, dtype=np.float32),
        cam_t0=np.array(t0s, dtype=np.int16),
        **{k: np.array([r[k] for r in rows], dtype=np.float32) for k in keys},
    )
    print(f"\ndone: {len(rows)} injections, {(time.time()-t0)/60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
