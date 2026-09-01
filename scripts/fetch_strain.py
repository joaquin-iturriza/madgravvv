#!/usr/bin/env python3
"""Warm the strain cache from GWOSC, for the training fold only.

Run it as a SLURM job on the `htc` CPU partition:

    scripts/remote.sh sbatch jobs/job_fetch_strain.sh

NOT on a login node, despite it being a download. A full fetch is an hour or more of
wall clock, and a login node killed ours partway through with no message -- the log
simply stops mid-run. Login nodes are for `--dry-run` and for one-segment checks. And
NOT as one job per segment either: the point is a single, polite, resumable filler.

    .venv/bin/python scripts/fetch_strain.py --dry-run          # what it would fetch (login node, fine)

WHY TRAINING FOLD ONLY, BY DEFAULT. Constraint C4 says the evaluation fold is read once,
at the end, for the quoted number. Not downloading it is the cheapest possible
enforcement of that: `FoldGuard` can be worked around by a determined mistake, an absent
file cannot. It also halves the bytes. `--fold eval` exists for the day the final report
is actually run, and it says so loudly.

The segments are fetched whole and cached per (ifo, start, end) as float32 `.npz`. They
are NOT whitened here — whitening is against the run-averaged reference PSD and belongs
to the representation, which is an ablation axis (R2), so caching whitened data would
bake one choice of it into the cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.data.strain import cache_path, fetch_strain, load_segments  # noqa: E402
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402

DEFAULT_SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
DEFAULT_CACHE = REPO / "data_cache/strain"
IFOS = ("H1", "L1")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--fold", choices=("train", "eval"), default="train")
    ap.add_argument("--sample-rate", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch at most N segment-detector pairs (a smaller slice still)")
    ap.add_argument("--jobs", type=int, default=2,
                    help="concurrent fetches. Network-bound, so threads. Two, not four: "
                         "at four we were rate-limited by GWOSC and 29 of 30 segments "
                         "failed. GWOSC is a shared public service and we are not its "
                         "only user.")
    ap.add_argument("--chunk-seconds", type=float, default=None,
                    help="split each segment into fetches of this length. Off by "
                         "default -- gwpy issues an API query per call, so chunking "
                         "multiplies the request rate and is what got us rate-limited.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.segments.exists():
        print(f"no segment list at {args.segments}\n"
              f"vendor the upstream repo first: bash scripts/vendor_reference.sh",
              file=sys.stderr)
        return 1

    segs = []
    for ifo in IFOS:
        segs.extend(load_segments(args.segments, ifo=ifo))
    guard = FoldGuard.from_segments(segs, eval_fold=1, n_folds=2)

    if args.fold == "eval":
        print("!" * 76)
        print("! Fetching the EVALUATION fold. This is the data C4 quarantines: it is read")
        print("! once, at the end, for the quoted number. Having it on disk removes the")
        print("! cheapest protection against reading it early. Do this only when the final")
        print("! report is actually being run, and record why in the run record.")
        print("!" * 76)
        with guard.final_report("fetch-eval-fold"):
            wanted = guard.segments(Split.EVAL)
    else:
        with guard.training("fetch-train-fold"):
            wanted = guard.segments(Split.TRAIN)

    if args.limit:
        wanted = wanted[: args.limit]

    live = sum(s.duration for s in wanted)
    est = live * args.sample_rate * 4
    print(f"segment list : {args.segments}")
    print(f"cache        : {args.cache}")
    print(f"fold         : {args.fold}")
    print(f"to fetch     : {len(wanted)} segment-detector pairs, {live / 86400:.2f} d")
    print(f"estimate     : {human(est)} uncompressed float32 at {args.sample_rate} Hz")

    have = [s for s in wanted if cache_path(args.cache, s.ifo, s.start, s.end).exists()]
    print(f"already cached: {len(have)}/{len(wanted)}")
    todo = [s for s in wanted if s not in have]
    if args.dry_run:
        for s in todo[:5]:
            print(f"  would fetch {s.ifo} {int(s.start)}..{int(s.end)} ({s.duration / 3600:.1f} h)")
        if len(todo) > 5:
            print(f"  ... and {len(todo) - 5} more")
        return 0

    args.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failures = []
    done = 0

    def one(seg):
        return seg, fetch_strain(seg.ifo, seg.start, seg.end, args.cache,
                                 args.sample_rate, chunk_seconds=args.chunk_seconds)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"fetching with {args.jobs} concurrent workers\n")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(one, s): s for s in todo}
        for fut in as_completed(futures):
            seg = futures[fut]
            done += 1
            tag = f"[{done}/{len(todo)}] {seg.ifo} {int(seg.start)}"
            try:
                _, arr = fut.result()
                el = time.time() - t0
                eta = (len(todo) - done) * el / done if done else 0
                print(f"{tag}  {arr.size:,} samples  "
                      f"[{el / 60:.1f} min elapsed, ~{eta / 60:.0f} min left]", flush=True)
            except Exception as exc:
                # One bad segment must not lose the whole fetch: GWOSC drops connections,
                # and a partial cache is resumable while a crash at segment 40 of 58 is
                # just lost time. Failures are reported at the end, never swallowed.
                failures.append((seg, f"{type(exc).__name__}: {exc}"))
                print(f"{tag}  FAILED {type(exc).__name__}: {exc}", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    if failures:
        print(f"{len(failures)} segment(s) failed — re-run to retry them:")
        for s, err in failures[:10]:
            print(f"  {s.ifo} {int(s.start)}..{int(s.end)}  {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
