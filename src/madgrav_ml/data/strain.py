"""Strain access: analysable segments and the data behind them.

Strain is not shipped with the upstream repo (~262 GB for O3a alone); it is fetched
from GWOSC by `search_mode/fetch_locks.py` and `fetch_bg.py`. This module wraps that
so the rest of the package deals in `Segment` objects and cached arrays rather than in
network calls, and so a fold assignment can be built before a single byte is fetched.

Data-quality flags, matching upstream: DATA + CBC_CAT1 + CBC_CAT2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..eval.folds import Segment

DQ_FLAGS = ("DATA", "CBC_CAT1", "CBC_CAT2")


class StrainUnavailable(RuntimeError):
    """GWOSC publishes no strain for this detector over this span. Not retryable."""
RUNS = ("O3a", "O3b", "O4a", "O4b")


def load_segments(
    path: str | Path, ifo: str | None = None, drop_event_segments: bool = False
) -> list[Segment]:
    """Read a segment list in any of the shapes the upstream repo ships.

    Three shapes are accepted:

    * **The O3a background format** — a dict with a `"segments"` key holding
      `[start, stop, duration, event_name_or_null]` rows, as in
      `search_mode/o3a_bg_segments_56.json`. These are **coincident** segments (both
      detectors analysable), so the same list describes H1 and L1 and `ifo` only labels
      the copy you want.
    * a mapping from IFO to a list of `[start, stop]` pairs;
    * a bare list of `[start, stop]` pairs, with `ifo` supplied.

    `drop_event_segments` removes the rows whose fourth field names a catalogue event.
    Off by default: a named event occupies a couple of seconds of a four-hour segment,
    so dropping the whole segment throws away ~14 % of the O3a-56 livetime to remove a
    negligible amount of signal. Turn it on only when a run must be provably
    signal-free, and say so in the run record.
    """
    data = json.loads(Path(path).read_text())

    rows = None
    if isinstance(data, dict) and "segments" in data:
        rows = data["segments"]
    elif isinstance(data, dict):
        if ifo is None:
            if len(data) != 1:
                raise ValueError(
                    f"{path} holds segments for {sorted(data)}; pass ifo= to choose one"
                )
            ifo, rows = next(iter(data.items()))
        else:
            rows = data[ifo]
    else:
        if ifo is None:
            raise ValueError(f"{path} is a bare segment list; pass ifo= to label it")
        rows = data

    if ifo is None:
        raise ValueError(f"{path} does not name a detector; pass ifo=")

    out = []
    for row in rows:
        if drop_event_segments and len(row) > 3 and row[3]:
            continue
        out.append(Segment(ifo=ifo, start=float(row[0]), end=float(row[1])))
    if not out:
        raise ValueError(f"{path}: no segments left after filtering")
    return out


def named_events(path: str | Path) -> dict[str, float]:
    """`{event_name: segment_start}` for the rows that carry one.

    Useful as a sanity check rather than as physics: if a run's foreground never lights
    up on a segment known to contain GW190521, something upstream of the score is wrong.
    """
    data = json.loads(Path(path).read_text())
    rows = data["segments"] if isinstance(data, dict) and "segments" in data else data
    return {r[3]: float(r[0]) for r in rows if len(r) > 3 and r[3]}


def total_livetime(segments) -> float:
    return sum(s.duration for s in segments)


def load_reference_psd(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a run-averaged reference PSD from `data/<run>_search_prep/reference_psd_*.npz`.

    Whitening against the *run-averaged* reference rather than the local segment is
    what makes one frozen model transferable across observing runs. Do not silently
    substitute a locally-estimated PSD; the ASD-consistency veto at the end of the
    upstream pipeline exists precisely because the two differ.
    """
    with np.load(path) as z:
        keys = set(z.files)
        fkey = next((k for k in ("freqs", "f", "frequency") if k in keys), None)
        pkey = next((k for k in ("psd", "PSD", "power") if k in keys), None)
        if fkey is None or pkey is None:
            raise KeyError(f"{path}: expected frequency and psd arrays, found {sorted(keys)}")
        return np.asarray(z[fkey], dtype=float), np.asarray(z[pkey], dtype=float)


# Chunking is OFF by default, and that is a measured decision rather than a default.
# GWOSC serves 4096 s files, so chunking at 4096 s looks like the natural unit -- but
# gwpy issues an `event-versions` API query per fetch_open_data CALL, not per file. So
# chunking a four-hour segment turned 1 API query into 4, and at four concurrent workers
# that was 16x the request rate of the unchunked version. GWOSC rate-limited us: 58
# "Too much trials for https://gwosc.org/api/v2/event-versions?..." lines and 29 of 30
# segments failing with "failed to get data from any source".
#
# One request per segment is what actually worked (3.0 min for a 2.6 h segment). Pass
# chunk_seconds explicitly if a segment is too large to fetch in one call.
CHUNK_SECONDS = None
FETCH_RETRIES = 5
# Backoff schedule in seconds, not a multiplier: a rate limit needs minutes, not the
# ~1 s a naive exponential starts with. Index i is the wait after attempt i.
RETRY_BACKOFF_S = (15.0, 60.0, 180.0, 300.0)


def cache_path(cache_dir: str | Path, ifo: str, start: float, end: float) -> Path:
    return Path(cache_dir) / f"{ifo}_{int(start)}_{int(end)}.npz"


def fetch_strain(ifo: str, start: float, end: float, cache_dir: str | Path,
                 sample_rate: int = 4096, chunk_seconds: float | None = CHUNK_SECONDS,
                 retries: int = FETCH_RETRIES) -> np.ndarray:
    """Cached GWOSC fetch for one stretch of strain.

    Reads `<cache_dir>/<ifo>_<start>_<end>.npz` when present, otherwise downloads and
    writes it. The cache is gitignored and expected to be large; on CC-IN2P3 it belongs
    under `/sps`, never the tiny `/pbs/home`.

    Fetched in `chunk_seconds` pieces with retries, following the upstream
    `search_mode/fetch_bg_par.py`, for two reasons that a single large request does not
    give: GWOSC serves 4096 s files, so a chunk is one round trip rather than an
    open-ended stitch, and a dropped connection three hours into a four-hour segment
    costs one chunk instead of the segment.

    The write is atomic — a temporary file renamed into place. Without that, an
    interrupted fetch leaves a truncated `.npz` that the cache check treats as complete,
    which then silently feeds a short array into training.
    """
    import time

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, ifo, start, end)
    if path.exists():
        with np.load(path) as z:
            return np.asarray(z["strain"], dtype=np.float32)

    from gwpy.timeseries import TimeSeries

    step = float(chunk_seconds) if chunk_seconds else float(end) - float(start)
    chunks: list[np.ndarray] = []
    t = float(start)
    while t < end:
        te = min(t + step, float(end))
        for attempt in range(retries):
            try:
                ts = TimeSeries.fetch_open_data(ifo, t, te, sample_rate=sample_rate,
                                                cache=False)
                chunks.append(np.asarray(ts.value, dtype=np.float32))
                break
            except Exception as exc:
                # "no dataset covering" is a statement about what GWOSC publishes, not a
                # transient failure. Retrying it five times with minute-scale backoff
                # burns ~9 minutes to arrive at the same answer, which is how the two
                # unavailable O3a spans cost an extra quarter hour.
                if "Cannot find a GWOSC dataset" in str(exc):
                    raise StrainUnavailable(
                        f"GWOSC publishes no {ifo} data covering "
                        f"[{int(start)}, {int(end)}) at {sample_rate} Hz"
                    ) from exc
                if attempt == retries - 1:
                    raise
                wait = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                time.sleep(wait)
        t = te

    strain = np.concatenate(chunks)
    expected = int(round((end - start) * sample_rate))
    if abs(strain.size - expected) > sample_rate:
        raise ValueError(
            f"{ifo} {int(start)}..{int(end)}: got {strain.size} samples, expected "
            f"~{expected}. Refusing to cache a short segment — it would train silently."
        )

    # The temp name must itself end in `.npz`. `np.savez_compressed` APPENDS `.npz`
    # when the filename does not, so a `.npz.partial` temp gets written as
    # `.npz.partial.npz` and the rename then fails on a file that was never created --
    # after paying for the whole download. Caught here the expensive way.
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, strain=strain, start=start, end=end,
                        sample_rate=sample_rate, ifo=ifo)
    tmp.replace(path)
    return strain


def available_segments(segments, cache_dir: str | Path) -> list[Segment]:
    """Filter a segment list down to what is actually cached.

    This exists because of a real gap in the O3a-56 set: GWOSC publishes no L1 data
    covering [1242308162, 1242322562) and no H1 data covering [1245948743, 1245963143),
    though in both cases the *partner* detector is available. The upstream list calls
    those spans coincident; the archive disagrees.

    The consequence is asymmetric, which is why this matters. Single-detector work
    (constraint C1 makes it the primary target) is unaffected -- the partner detector's
    data is perfectly good noise. But **coincident** livetime is not: those two spans
    contribute nothing, and computing background livetime from the nominal list would
    overstate it by eight hours. An overstated T_bg *understates* the FAR, which makes
    every result look better than it is. Always take coincident livetime from this, not
    from `load_segments`.
    """
    cache_dir = Path(cache_dir)
    return [s for s in segments if cache_path(cache_dir, s.ifo, s.start, s.end).exists()]
