"""Coherence and morphology, checked against `MassiveEventPipeline`'s own methods.

The vetoes are where the search's discriminating power actually lives (Section 8 of
results.tex), so a paraphrase of them is not good enough. These call the vendored
methods directly through a stub carrying the shipped calibration's constants, rather
than reimplementing the comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from madgrav_ml.eval import coherence as C

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / ".reference/MADGRAV"
FS = 4096

pytestmark = pytest.mark.skipif(not REFERENCE.exists(),
                                reason="run scripts/vendor_reference.sh")


def stub():
    """A real `MassiveEventPipeline` with only the constants set.

    Built with `__new__` so the methods under test are the vendored ones, bound to an
    instance, without `__init__` loading a checkpoint and a GPU. Reimplementing
    `_bandlimit` in a hand-rolled stub would defeat the point of the comparison.
    """
    cls = upstream_pipeline()
    s = cls.__new__(cls)
    s.fs = FS
    s.coh_mode = C.COHERENCE_MODE
    s.coh_band = C.COHERENCE_BAND_HZ
    s.coh_win = C.COHERENCE_WINDOW_S
    s.lag = C.LAG_SAMPLES
    s.cband = C.CENTROID_BAND_HZ
    s.cwin = C.CENTROID_WINDOW_S
    return s


def upstream_pipeline():
    for rel in ("spectrogram_cascade", "improved"):
        path = str(REFERENCE / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("MADGRAV_ROOT", str(REFERENCE))
    mp = pytest.importorskip("massive_pipeline")
    return mp.MassiveEventPipeline


def pair(seed=0, n=3, samples=4 * FS, common=0.0):
    """`common` mixes a shared component in, so the pair spans incoherent to coherent."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, (n, samples))
    a = rng.normal(0, 1, (n, samples)) + common * shared
    b = rng.normal(0, 1, (n, samples)) + common * shared
    return a, b


@pytest.mark.parametrize("common", [0.0, 0.5, 3.0])
def test_coherence_matches_upstream(common):
    a, b = pair(common=common)
    mine = C.coherence(a, b, FS)
    theirs = stub()._coherence(a, b)
    assert np.allclose(mine, theirs, rtol=1e-6, atol=1e-9), (mine, theirs)


def test_centroid_matches_upstream():
    a, _ = pair(common=0.0)
    assert np.allclose(C.centroid(a, FS), stub()._centroid(a), rtol=1e-6)


def test_bandlimit_matches_upstream():
    a, _ = pair()
    assert np.allclose(C.bandlimit(C._centre(a, FS, 1.0), FS),
                       stub()._bandlimit(C._centre(a, FS, 1.0)), rtol=1e-9)


# --- properties that make it a veto rather than a number ----------------------


def test_identical_series_score_one_and_a_common_component_passes():
    rng = np.random.default_rng(2)
    shared = rng.normal(0, 1, (16, 4 * FS))
    assert C.coherence(shared, shared.copy(), FS).min() > 0.99
    a = rng.normal(0, 1, (64, 4 * FS)) + shared[0]
    b = rng.normal(0, 1, (64, 4 * FS)) + shared[0]
    assert (C.coherence(a, b, FS) >= C.TCOH).mean() == 1.0


def test_the_threshold_sits_inside_the_incoherent_distribution():
    """tcoh is a ~6x rejection on random coincidences, not a hard gate.

    Independent white noise scores a median of ~0.106 against a threshold of 0.128, and
    ~16% of pairs clear it. Worth pinning: it is easy to assume a veto named "coherence"
    removes accidental coincidences outright, and the measured efficiency of the whole
    search depends on it not doing that.
    """
    rng = np.random.default_rng(0)
    inc = C.coherence(rng.normal(0, 1, (200, 4 * FS)),
                      rng.normal(0, 1, (200, 4 * FS)), FS)
    assert 0.08 < np.median(inc) < 0.13
    assert 0.05 < (inc >= C.TCOH).mean() < 0.30


def test_maximising_over_lags_is_most_of_the_incoherent_floor():
    """The +-45-sample scan is a look-elsewhere effect inside the statistic.

    The lag range has to cover the 10 ms H1-L1 light travel time, but the plain maximum
    over 91 lags is taken with no trials correction, and that alone raises the
    incoherent median from ~0.028 to ~0.106 — nearly a factor of four, which is most of
    the distance to the threshold. A lag-trials correction is a candidate improvement
    that costs no parameters.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (200, 4 * FS))
    b = rng.normal(0, 1, (200, 4 * FS))
    scanned = np.median(C.coherence(a, b, FS))
    single = np.median(C.coherence(a, b, FS, lag_samples=0))
    assert single < 0.5 * scanned
    assert (C.coherence(a, b, FS, lag_samples=0) >= C.TCOH).mean() < 0.05


def test_symmetric_norm_punishes_an_amplitude_mismatch():
    """This is why it is not a Pearson correlation.

    A loud glitch in one detector against quiet data in the other is the dominant
    background for a coincident search. Pearson normalises each series to unit norm and
    so is blind to the mismatch; `2<a,b>/(|a|^2+|b|^2)` is maximised only when the two
    have comparable energy, so it rejects exactly that population.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (1, 4 * FS))
    same = C.coherence(x, x.copy(), FS)
    louder = C.coherence(100.0 * x, x.copy(), FS)
    assert same[0] > 0.5
    assert louder[0] < 0.1 * same[0]


def test_stored_coefficients_reproduce_the_direct_computation():
    """The scan stores 481 complex numbers per detector instead of 4096 real samples,
    and the lag scan becomes one transform. It must be the same number."""
    a, b = pair(seed=4, n=6, common=1.0)
    direct = C.coherence(a, b, FS)
    ca, lo, n = C.band_coefficients(a, FS)
    cb, _, _ = C.band_coefficients(b, FS)
    fast = C.coherence_from_coefficients(ca, cb, lo, n)
    assert np.allclose(direct, fast, rtol=1e-4, atol=1e-6), (direct, fast)


def test_is_massive_needs_both_morphology_and_coherence():
    low, high = 100.0, 300.0
    assert C.is_massive(np.array([0.5]), np.array([low]), np.array([low]))[0]
    assert not C.is_massive(np.array([0.5]), np.array([high]), np.array([low]))[0]
    assert not C.is_massive(np.array([0.01]), np.array([low]), np.array([low]))[0]
