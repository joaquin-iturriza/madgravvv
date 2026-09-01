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


CHUNK_SECONDS = 4096.0   # GWOSC serves 4096 s files; asking for one is one round trip
FETCH_RETRIES = 4
RETRY_BACKOFF_S = 15.0


def cache_path(cache_dir: str | Path, ifo: str, start: float, end: float) -> Path:
    return Path(cache_dir) / f"{ifo}_{int(start)}_{int(end)}.npz"


def fetch_strain(ifo: str, start: float, end: float, cache_dir: str | Path,
                 sample_rate: int = 4096, chunk_seconds: float = CHUNK_SECONDS,
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

    chunks: list[np.ndarray] = []
    t = float(start)
    while t < end:
        te = min(t + chunk_seconds, float(end))
        for attempt in range(retries):
            try:
                ts = TimeSeries.fetch_open_data(ifo, t, te, sample_rate=sample_rate,
                                                cache=False)
                chunks.append(np.asarray(ts.value, dtype=np.float32))
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
        t = te

    strain = np.concatenate(chunks)
    expected = int(round((end - start) * sample_rate))
    if abs(strain.size - expected) > sample_rate:
        raise ValueError(
            f"{ifo} {int(start)}..{int(end)}: got {strain.size} samples, expected "
            f"~{expected}. Refusing to cache a short segment — it would train silently."
        )

    tmp = path.with_suffix(".npz.partial")
    np.savez_compressed(tmp, strain=strain, start=start, end=end,
                        sample_rate=sample_rate, ifo=ifo)
    tmp.replace(path)
    return strain
