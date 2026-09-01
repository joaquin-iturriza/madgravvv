"""The strain cache: the atomic write must actually be atomic.

Written after the bug it describes. `np.savez_compressed` appends `.npz` to a filename
that lacks it, so a `.npz.partial` temp was written as `.npz.partial.npz` and the rename
failed — after paying for a three-minute download, eleven times over. The cost of the
bug was not correctness (nothing bad was cached) but an hour of wasted fetching, and
nothing in the test suite could have caught it because no test touched the write path.
"""

import numpy as np
import pytest

from madgrav_ml.data.strain import cache_path, fetch_strain


def test_cache_hit_reads_without_network(tmp_path, monkeypatch):
    """A cached segment must never reach gwpy — that is the whole point of the cache."""
    p = cache_path(tmp_path, "H1", 100, 200)
    np.savez_compressed(p, strain=np.arange(10, dtype=np.float32), start=100, end=200,
                        sample_rate=1, ifo="H1")
    out = fetch_strain("H1", 100, 200, tmp_path, sample_rate=1)
    assert out.shape == (10,)


def test_the_write_lands_on_the_exact_cache_path(tmp_path, monkeypatch):
    """The regression: the temp name must end in .npz or numpy renames it out from under us."""
    fs, start, end = 16, 0.0, 4.0
    n = int((end - start) * fs)

    class FakeTS:
        def __init__(self, size):
            self.value = np.zeros(size, dtype=np.float32)

    import madgrav_ml.data.strain as strain_mod

    class FakeTimeSeries:
        @staticmethod
        def fetch_open_data(ifo, a, b, sample_rate=None, cache=False):
            return FakeTS(int(round((b - a) * sample_rate)))

    fake_gwpy = type("m", (), {"TimeSeries": FakeTimeSeries})
    monkeypatch.setitem(__import__("sys").modules, "gwpy.timeseries", fake_gwpy)

    out = fetch_strain("H1", start, end, tmp_path, sample_rate=fs, chunk_seconds=2.0)
    assert out.size == n
    assert cache_path(tmp_path, "H1", start, end).exists()
    # and nothing left behind under any partial name
    leftovers = sorted(q.name for q in tmp_path.iterdir() if "partial" in q.name)
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_a_short_download_is_refused(tmp_path, monkeypatch):
    """A truncated fetch must not be cached: it would train silently on short data."""
    class FakeTS:
        def __init__(self, size):
            self.value = np.zeros(size, dtype=np.float32)

    class FakeTimeSeries:
        @staticmethod
        def fetch_open_data(ifo, a, b, sample_rate=None, cache=False):
            return FakeTS(1)          # far short of what was asked for

    monkeypatch.setitem(__import__("sys").modules, "gwpy.timeseries",
                        type("m", (), {"TimeSeries": FakeTimeSeries}))
    with pytest.raises(ValueError, match="Refusing to cache a short segment"):
        fetch_strain("H1", 0.0, 100.0, tmp_path, sample_rate=16, chunk_seconds=50.0)
    assert not cache_path(tmp_path, "H1", 0.0, 100.0).exists()


def test_unavailable_span_is_not_retried(tmp_path, monkeypatch):
    """A span GWOSC does not publish is a fact, not a transient failure."""
    import madgrav_ml.data.strain as sm

    calls = []

    class FakeTimeSeries:
        @staticmethod
        def fetch_open_data(ifo, a, b, sample_rate=None, cache=False):
            calls.append((a, b))
            raise ValueError(
                f"Cannot find a GWOSC dataset for {ifo} covering [{a}, {b})")

    monkeypatch.setitem(__import__("sys").modules, "gwpy.timeseries",
                        type("m", (), {"TimeSeries": FakeTimeSeries}))
    with pytest.raises(sm.StrainUnavailable, match="publishes no"):
        sm.fetch_strain("L1", 0.0, 100.0, tmp_path, sample_rate=16)
    assert len(calls) == 1, f"retried an unavailable span {len(calls)} times"


def test_available_segments_reflects_the_cache(tmp_path):
    """Coincident livetime must come from what is on disk, not the nominal list."""
    from madgrav_ml.data.strain import available_segments
    from madgrav_ml.eval.background import coincident_livetime
    from madgrav_ml.eval.folds import Segment

    nominal = [Segment("H1", 0, 100), Segment("L1", 0, 100),
               Segment("H1", 200, 300), Segment("L1", 200, 300)]
    # the L1 half of the second span was never published
    for s in nominal[:3]:
        np.savez_compressed(cache_path(tmp_path, s.ifo, s.start, s.end),
                            strain=np.zeros(4, dtype=np.float32))

    have = available_segments(nominal, tmp_path)
    assert len(have) == 3

    nominal_live = coincident_livetime([s for s in nominal if s.ifo == "H1"],
                                       [s for s in nominal if s.ifo == "L1"])
    actual_live = coincident_livetime([s for s in have if s.ifo == "H1"],
                                      [s for s in have if s.ifo == "L1"])
    assert nominal_live == 200          # what the list claims
    assert actual_live == 100           # what we can actually slide
