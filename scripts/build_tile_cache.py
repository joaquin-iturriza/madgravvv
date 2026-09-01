#!/usr/bin/env python3
"""Precompute a bank of noise tiles from the cached strain.

WHY THIS EXISTS, since Phase 2 of the plan asks for on-the-fly generation. Measured on
this cluster: one tile costs ~4.6 s, essentially all of it the Q-transform. At batch 64
that is ~5 minutes per batch on one core, so a 20k-step run would be years. Upstream hits
the same wall and solves it the same way — `improved_pipeline.compute_qt_images` fans the
Q-transform over a 16-process pool, and the search precomputes tile caches rather than
transforming during inference.

So the representation is precomputed and training reads a bank. That does bound Phase 2:
"effectively infinite non-repeating data" is affordable for the *injection* half (drawing
new waveforms is cheap) but not for the noise half at this Q-transform cost. Note it in
the run record rather than claiming a generator that is really a bank.

Segments are processed one at a time and many windows drawn from each, rather than
sampling segments at random: a segment is 236 MB and a random draw over 56 of them misses
the reader's cache almost every time, which measured 1.7 s per window in loading alone.

    scripts/remote.sh sbatch jobs/job_build_tiles.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.data.representation import (  # noqa: E402
    TileSpec, make_tile, notch_and_highpass, whiten,
)
from madgrav_ml.data.strain import (  # noqa: E402
    SegmentReader, available_segments, load_reference_psd, load_segments,
)
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402

SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
PSD_DIR = REPO / ".reference/MADGRAV/data/o3a_search_prep"

_CTX: dict = {}


def _init(psds, spec, fs, lines):
    _CTX.update(psds=psds, spec=spec, fs=fs, lines=lines)


def _one(args):
    """Whiten, notch, transform. Runs in a worker; returns the tile or None."""
    raw, ifo = args
    try:
        w = whiten(raw, _CTX["fs"], reference_psd=_CTX["psds"][ifo])
        w = notch_and_highpass(w, _CTX["fs"], lines=_CTX["lines"][ifo])
        return make_tile(w, _CTX["spec"]).astype(np.float32)
    except Exception:
        # A single bad window must not kill a two-hour build. Dropped tiles are counted
        # and reported; a silent zero tile would be far worse than a missing one.
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data_cache/tiles/train")
    ap.add_argument("--n-tiles", type=int, default=20000)
    ap.add_argument("--split", choices=("hpo_train", "hpo_val", "train"), default="hpo_train")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--window-seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    segs = load_segments(SEGMENTS, ifo="H1") + load_segments(SEGMENTS, ifo="L1")
    guard = FoldGuard.from_segments(segs, eval_fold=1, n_folds=2)
    with guard.training(f"build-tiles:{args.split}"):
        wanted = guard.segments(Split(args.split))
    have = available_segments(wanted, REPO / "data_cache/strain")
    if not have:
        print(f"no cached strain for split {args.split}", file=sys.stderr)
        return 1

    spec = TileSpec()
    fs = spec.sample_rate
    psds = {i: load_reference_psd(PSD_DIR / f"reference_psd_{i}.npz") for i in ("H1", "L1")}
    lines = {
        "H1": (15.1, 15.6, 16.4, 16.7, 17.1, 17.6, 35.9, 36.7, 331.9, 410.3,
               60.0, 120.0, 180.0, 240.0, 300.0),
        "L1": (15.1, 15.7, 16.3, 16.9, 30.8, 31.4, 32.0, 32.6, 33.2, 33.8,
               60.0, 120.0, 180.0, 240.0, 300.0),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    reader = SegmentReader(REPO / "data_cache/strain", capacity=1)

    # Tiles per segment, in proportion to livetime, so the bank samples the fold the way
    # the fold actually is rather than the way its segment count suggests.
    live = np.array([s.duration for s in have], dtype=float)
    per_seg = np.maximum(1, np.round(args.n_tiles * live / live.sum()).astype(int))
    print(f"split={args.split}  {len(have)} segments, {live.sum()/86400:.2f} d")
    print(f"target {args.n_tiles} tiles -> {per_seg.sum()} planned, "
          f"{args.workers} workers, shards of {args.shard_size}", flush=True)

    buf: list[np.ndarray] = []
    shard = 0
    dropped = 0
    made = 0
    t0 = time.time()

    with Pool(args.workers, initializer=_init, initargs=(psds, spec, fs, lines)) as pool:
        for si, (seg, n_here) in enumerate(zip(have, per_seg)):
            arr = reader.segment(seg)                       # one 236 MB load per segment
            n_samp = int(args.window_seconds * fs)
            starts = rng.integers(0, arr.size - n_samp, size=int(n_here))
            windows = [(arr[i:i + n_samp].copy(), seg.ifo) for i in starts]
            for tile in pool.imap_unordered(_one, windows, chunksize=4):
                if tile is None:
                    dropped += 1
                    continue
                buf.append(tile)
                made += 1
                if len(buf) >= args.shard_size:
                    path = args.out / f"tiles_{shard:04d}.npz"
                    np.savez_compressed(path, x=np.stack(buf),
                                        y=np.zeros(len(buf), dtype=np.float32),
                                        source=np.array([args.split] * len(buf)))
                    print(f"  wrote {path.name}  ({made} tiles, "
                          f"{(time.time()-t0)/60:.1f} min)", flush=True)
                    buf, shard = [], shard + 1
            el = time.time() - t0
            print(f"[{si+1}/{len(have)}] {seg.ifo} {int(seg.start)}  {made} tiles  "
                  f"[{el/60:.1f} min, {made/el:.2f} tiles/s]", flush=True)

    if buf:
        path = args.out / f"tiles_{shard:04d}.npz"
        np.savez_compressed(path, x=np.stack(buf), y=np.zeros(len(buf), dtype=np.float32),
                            source=np.array([args.split] * len(buf)))
        print(f"  wrote {path.name}  ({made} tiles)", flush=True)

    print(f"\ndone: {made} tiles, {dropped} dropped, {(time.time()-t0)/60:.1f} min")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
