#!/usr/bin/env python
"""Score the background split on a fixed GPS grid — the instrument every FAR needs.

WHY THIS SHAPE. The expensive thing is the Q-transform, and it is expensive once. So
this writes a per-detector score *time series* on a grid both detectors share, and
nothing else. Time slides then cost an array roll rather than a rescan, which is what
makes a year of background affordable: 1.4 days of coincident livetime becomes 4 years
at a thousand lags without touching the strain again.

FOLD DISCIPLINE. Runs inside `guard.calibration()` and reads `Split.HPO_BG` only — the
slice of the training fold that is never fitted to and never selected against. Reading
the background off HPO_VAL would be quietly optimistic, because the stage-2 checkpoint
is chosen to maximise detections above a threshold set by HPO_VAL's own noise. The
evaluation fold is not touched here and stays sealed until there is a report to write.

  scripts/remote.sh sbatch --array=0-3 jobs/job_scan_background.sh \
      --checkpoint runs/madgrav/<run>/models/model_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Parent-side imports before Pool forks — see build_tile_cache.py for what happens
# otherwise (a 54-minute import storm that produced nothing).
import scipy.ndimage  # noqa: F401,E402
import scipy.signal  # noqa: F401,E402
from gwpy.frequencyseries import FrequencySeries  # noqa: F401,E402
from gwpy.timeseries import TimeSeries  # noqa: F401,E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from madgrav_ml.data.representation import (  # noqa: E402
    TileSpec, make_tile, notch, notch_lines_for, whiten,
)
from madgrav_ml.data.strain import (  # noqa: E402
    SegmentReader, available_segments, load_reference_psd, load_segments,
)
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402
from madgrav_ml.models.cae import BaselineCAE  # noqa: E402

SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
PSD_DIR = REPO / ".reference/MADGRAV/data/o3a_search_prep"
_CTX: dict = {}


def _init(psds, lines, spec, fs):
    _CTX.update(psds=psds, lines=lines, spec=spec, fs=fs)


def _tile(args):
    raw, ifo = args
    try:
        w = whiten(raw, _CTX["fs"], reference_psd=_CTX["psds"][ifo])
        w = notch(w, _CTX["fs"], lines=_CTX["lines"][ifo])
        return make_tile(w, _CTX["spec"]).astype(np.float32)
    except Exception:
        return None


def load_model(path: Path, device) -> BaselineCAE:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = BaselineCAE()
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.no_grad()
def score_batch(model, tiles: list[np.ndarray], device) -> np.ndarray:
    x = torch.from_numpy(np.stack(tiles)).to(device)
    return model.reconstruction_error(x, reduction="none").cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "data_cache/background")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stride", type=float, default=1.0,
                    help="seconds between window centres; 1.0 makes the 1 s crops "
                         "the network sees exactly non-overlapping")
    ap.add_argument("--window-seconds", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--split", default="hpo_bg", choices=("hpo_bg",))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    spec = TileSpec()
    fs = spec.sample_rate
    psds = {i: load_reference_psd(PSD_DIR / f"reference_psd_{i}.npz") for i in ("H1", "L1")}
    lines = {i: notch_lines_for(i, "o1") for i in ("H1", "L1")}

    segs = load_segments(SEGMENTS, ifo="H1") + load_segments(SEGMENTS, ifo="L1")
    guard = FoldGuard.from_segments(segs, eval_fold=1, n_folds=2,
                                    audit_path=REPO / "runs/fold_audit.jsonl")
    with guard.calibration(f"background-scan:{args.checkpoint.parent.parent.name}"):
        wanted = guard.segments(Split(args.split))
    have = available_segments(wanted, REPO / "data_cache/strain")

    # Pair the detectors by GPS span: a slide is only meaningful between two detectors
    # observing the same stretch of time.
    by_span: dict[tuple[float, float], dict[str, object]] = {}
    for s in have:
        by_span.setdefault((s.start, s.end), {})[s.ifo] = s
    pairs = [(k, v) for k, v in sorted(by_span.items()) if {"H1", "L1"} <= set(v)]
    mine = pairs[args.shard::args.n_shards]
    coincident = sum(e - s for (s, e), _ in pairs)
    print(f"{args.split}: {len(pairs)} coincident spans, "
          f"{coincident / 86400:.2f} d coincident livetime; "
          f"shard {args.shard}/{args.n_shards} takes {len(mine)}", flush=True)
    if not mine:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    reader = SegmentReader(REPO / "data_cache/strain", capacity=2)
    n_samp = int(args.window_seconds * fs)
    step = int(args.stride * fs)
    t0 = time.time()
    total = 0

    with Pool(args.workers, initializer=_init, initargs=(psds, lines, spec, fs)) as pool:
        for si, ((start, end), pair) in enumerate(mine):
            out_path = args.out / f"bg_{int(start)}.npz"
            if out_path.exists():
                print(f"[{si+1}/{len(mine)}] {int(start)} already done", flush=True)
                continue
            scores: dict[str, np.ndarray] = {}
            offsets = None
            for ifo in ("H1", "L1"):
                arr = reader.segment(pair[ifo])
                starts = np.arange(0, arr.size - n_samp, step, dtype=np.int64)
                offsets = starts
                windows = ((arr[i:i + n_samp], ifo) for i in starts)
                vals, buf = [], []
                # imap keeps the grid ordered; a permuted score series would make every
                # time slide meaningless while looking perfectly healthy.
                for tile in pool.imap(_tile, windows, chunksize=8):
                    # A failed window must not shift the grid. NaN holds its place and
                    # is dropped by name downstream, never by silently closing the gap.
                    buf.append(np.zeros((1,) + spec.size, np.float32) if tile is None
                               else tile)
                    if len(buf) >= args.batch:
                        vals.append(score_batch(model, buf, device))
                        buf = []
                if buf:
                    vals.append(score_batch(model, buf, device))
                scores[ifo] = np.concatenate(vals) if vals else np.array([])
            gps = start + offsets / fs + 0.5 * args.window_seconds
            np.savez_compressed(out_path, gps=gps.astype(np.float64),
                                score_H1=scores["H1"].astype(np.float32),
                                score_L1=scores["L1"].astype(np.float32),
                                stride=args.stride, span=np.array([start, end]))
            total += len(gps)
            el = time.time() - t0
            print(f"[{si+1}/{len(mine)}] {int(start)}  {len(gps)} grid points  "
                  f"[{el/60:.1f} min, {2*total/el:.1f} tiles/s]", flush=True)

    print(f"\ndone: {total} grid points x 2 detectors, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
