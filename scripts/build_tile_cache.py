#!/usr/bin/env python3
"""Precompute a bank of noise tiles from the cached strain.

WHY THIS EXISTS. Properly profiled (three timed calls after a warm-up, not one cold
call), a tile costs about 277 ms:

    whiten                     0.5 ms
    highpass + 15 notches     18   ms
    Q-transform              252   ms      <- gwpy, upstream settings
    resize to 256x128          7   ms

An earlier version of this docstring claimed 4.6 s and concluded that on-the-fly noise
generation was infeasible. That was a cold-call artifact — one untimed-warm-up
measurement per stage, dominated by lazy `import scipy` / `import gwpy` inside the
functions. The true cost is 17x smaller and on-the-fly IS feasible: batch 64 is 17.7
core-seconds, so ~0.55 s/batch at 32 workers.

The bank is still the right default, for a different and weaker reason: a multi-epoch run
recomputes the identical transform for the identical window on every epoch, so
precomputing is roughly a 50x saving across a run and makes iteration interactive. It is
an efficiency choice, not a feasibility one, and Phase 2 on-the-fly generation remains
open — `data.source: generated` is a supported path, not a blocked one.

Building 20k tiles at 16 workers takes about six minutes.

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

# Import the heavy, lazily-imported dependencies HERE, in the parent, before any worker
# is forked. On Linux `Pool` forks, so the children inherit these already-loaded modules.
# Without this each worker imports scipy and gwpy on its first tile, from a venv on the
# shared /sps filesystem — 32 processes doing a metadata-heavy import storm at once. That
# is not a slowdown: a 32-worker build sat for 54 minutes and produced zero tiles, with
# every worker stuck inside `scipy.linalg.blas`.
import scipy.ndimage  # noqa: F401,E402
import scipy.signal  # noqa: F401,E402
from gwpy.frequencyseries import FrequencySeries  # noqa: F401,E402
from gwpy.timeseries import TimeSeries  # noqa: F401,E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.data.representation import (  # noqa: E402
    TileSpec, make_tile, notch, notch_lines_for, whiten,
)
from madgrav_ml.data.injections import ParameterSampler  # noqa: E402
from madgrav_ml.data.strain import (  # noqa: E402
    SegmentReader, available_segments, load_reference_psd, load_segments,
)
from madgrav_ml.data.waveforms import InjectionEngine, LALWaveformBackend  # noqa: E402
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402

SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
PSD_DIR = REPO / ".reference/MADGRAV/data/o3a_search_prep"

_CTX: dict = {}


def _init(psds, spec, fs, lines, engine):
    _CTX.update(psds=psds, spec=spec, fs=fs, lines=lines, engine=engine)


def _one(args):
    """Whiten, notch, optionally inject, transform. Returns (tile, meta) or None.

    The injection is added to the *whitened* series, which is exact rather than
    approximate: whitening and the notch chain are LTI, so whitening noise and signal
    separately and summing equals whitening their sum. See data/waveforms.py.
    """
    raw, ifo, gps, params = args
    try:
        w = whiten(raw, _CTX["fs"], reference_psd=_CTX["psds"][ifo])
        w = notch(w, _CTX["fs"], lines=_CTX["lines"][ifo])
        if params is not None:
            w = _CTX["engine"].inject(w, params, ifo, gps)
        return make_tile(w, _CTX["spec"]).astype(np.float32), params
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
    ap.add_argument("--workers", type=int, default=16)  # see the import note above
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--window-seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    # "o1" is what the deployed search uses even on O3a -- see representation.notch_lines_for.
    ap.add_argument("--line-configuration", choices=("o1", "o3a"), default="o1")
    # Build the SIGNAL half of the stage-2 pair. Windows are drawn from a separate RNG
    # stream from the injections, so `--inject` and a plain run with the same --seed
    # draw the IDENTICAL noise windows in the identical order. That is what makes the
    # two banks a matched pair rather than two samples of the fold, and it matches
    # upstream, which injects into the noise split it also trains on.
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--snr-convention", choices=("network", "detector"), default="network")
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
    # One source of truth. This used to be a hand-copied, silently truncated version of
    # the O3a list; the deployed search notches the O1 list instead.
    lines = {i: notch_lines_for(i, args.line_configuration) for i in psds}

    engine = None
    sampler = None
    inj_rng = None
    if args.inject:
        # Load LAL in the PARENT, before Pool forks. Same lesson as scipy/gwpy above: a
        # 16-worker first-tile import storm against a venv on shared /sps once sat for
        # 54 minutes and produced nothing.
        import lal  # noqa: F401
        import lalsimulation  # noqa: F401

        engine = InjectionEngine(
            backend=LALWaveformBackend(),
            psds=psds,
            notch_lines=lines,
            sample_rate=fs,
            window_seconds=args.window_seconds,
            snr_convention=args.snr_convention,
        )
        # Upstream draws the target uniformly in (8, 25) and rescales so the NETWORK
        # SNR -- sqrt(rho_H1^2 + rho_L1^2) -- hits it (maybe_rescale_projected_signal_pair).
        sampler = ParameterSampler(snr_range=(8.0, 25.0))
        inj_rng = np.random.default_rng(args.seed + 10_000)
        print(f"injecting: {engine.backend.approximant_name} from "
              f"{engine.backend.f_lower} Hz, network SNR U(8, 25), "
              f"'{args.snr_convention}' convention", flush=True)

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

    meta: list = []

    def flush(buf, meta, shard, made, t0, final=False):
        path = args.out / f"tiles_{shard:04d}.npz"
        extra = {}
        if args.inject:
            extra = {k: np.array([m[k] for m in meta], dtype=np.float32)
                     for k in ("mass1", "mass2", "network_snr", "spin1z", "spin2z",
                               "inclination", "time_shift")}
        np.savez_compressed(
            path, x=np.stack(buf),
            y=np.full(len(buf), 1.0 if args.inject else 0.0, dtype=np.float32),
            source=np.array([args.split] * len(buf)), **extra)
        note = "" if final else f", {(time.time()-t0)/60:.1f} min"
        print(f"  wrote {path.name}  ({made} tiles{note})", flush=True)

    with Pool(args.workers, initializer=_init,
              initargs=(psds, spec, fs, lines, engine)) as pool:
        for si, (seg, n_here) in enumerate(zip(have, per_seg)):
            arr = reader.segment(seg)                       # one 236 MB load per segment
            n_samp = int(args.window_seconds * fs)
            starts = rng.integers(0, arr.size - n_samp, size=int(n_here))
            windows = []
            for i in starts:
                # GPS of the window CENTRE, where the coalescence sits: the antenna
                # pattern and the geocentre delay are both functions of sidereal time.
                gps = seg.start + (int(i) / fs) + 0.5 * args.window_seconds
                params = sampler.draw(inj_rng) if args.inject else None
                windows.append((arr[i:i + n_samp].copy(), seg.ifo, gps, params))
            # imap, not imap_unordered: the noise and signal banks must stay index
            # aligned so a pair is one window with and without its injection.
            for out in pool.imap(_one, windows, chunksize=4):
                if out is None:
                    dropped += 1
                    continue
                tile, params = out
                buf.append(tile)
                if args.inject:
                    meta.append(params.as_dict())
                made += 1
                if len(buf) >= args.shard_size:
                    flush(buf, meta, shard, made, t0)
                    buf, meta, shard = [], [], shard + 1
            el = time.time() - t0
            print(f"[{si+1}/{len(have)}] {seg.ifo} {int(seg.start)}  {made} tiles  "
                  f"[{el/60:.1f} min, {made/el:.2f} tiles/s]", flush=True)

    if buf:
        flush(buf, meta, shard, made, t0, final=True)

    print(f"\ndone: {made} tiles, {dropped} dropped, {(time.time()-t0)/60:.1f} min")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
