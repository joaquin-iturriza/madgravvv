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


def load_segments(path: str | Path, ifo: str | None = None) -> list[Segment]:
    """Read a segment list.

    Accepts the upstream JSON shapes found in `search_mode/*.json` — either a list of
    `[start, stop]` pairs, or a mapping from IFO to such a list. `ifo` selects one
    detector from a mapping, and is required when the file holds several.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if ifo is None:
            if len(data) != 1:
                raise ValueError(
                    f"{path} holds segments for {sorted(data)}; pass ifo= to choose one"
                )
            ifo, pairs = next(iter(data.items()))
        else:
            pairs = data[ifo]
    else:
        if ifo is None:
            raise ValueError(f"{path} is a bare segment list; pass ifo= to label it")
        pairs = data
    return [Segment(ifo=ifo, start=float(a), end=float(b)) for a, b in pairs]


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


def fetch_strain(ifo: str, start: float, end: float, cache_dir: str | Path,
                 sample_rate: int = 4096) -> np.ndarray:
    """Cached GWOSC fetch for one stretch of strain.

    Reads `<cache_dir>/<ifo>_<start>_<end>.npz` when present, otherwise downloads via
    gwpy and writes it. The cache is gitignored and is expected to be large; on
    CC-IN2P3 it belongs under `/sps`, never under the tiny `/pbs/home`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ifo}_{int(start)}_{int(end)}.npz"
    if path.exists():
        with np.load(path) as z:
            return np.asarray(z["strain"], dtype=np.float32)

    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.fetch_open_data(ifo, start, end, sample_rate=sample_rate, cache=False)
    strain = np.asarray(ts.value, dtype=np.float32)
    np.savez_compressed(path, strain=strain, start=start, end=end,
                        sample_rate=sample_rate, ifo=ifo)
    return strain
