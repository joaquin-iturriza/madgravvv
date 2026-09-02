"""The representation, checked against upstream's own code rather than against the paper.

WHY THIS FILE EXISTS. There was no test here, and that is how the pipeline ran for a
day on strain that was never whitened. `improved/utilities.py::whiten` divides by
`sqrt(psd + 1e-40)`; an O3a reference PSD is 3e-51 to 7e-40 and sits at ~3e-47 at
100 Hz, so that absolute epsilon is ~1e6 times the PSD across the entire sensitive band
and the division degenerates into a constant rescale. It was also the wrong function:
the deployed search calls `MassiveEventPipeline._whiten` -> `whiten_batch_gwpy_o1`,
which is gwpy's FIR whitening against a *relatively* floored ASD.

The tiles that came out correlated 0.28 with correctly whitened ones and carried 19% of
their band power below 100 Hz where the real thing carries 3%.

So the rule these tests encode: every step of the representation is checked against the
upstream function the shipped weights were actually built with, on real reference PSDs,
not against a description of it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from madgrav_ml.data import representation as R

REPO = Path(__file__).resolve().parents[1]
PREP = REPO / ".reference/MADGRAV/data/o3a_search_prep"
FS = R.FS

pytestmark = pytest.mark.skipif(
    not PREP.exists(),
    reason="vendored upstream missing; run scripts/vendor_reference.sh",
)


def upstream():
    """Import the vendored `improved_pipeline` — the code the weights were built with."""
    path = str(REPO / ".reference/MADGRAV/improved")
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("MADGRAV_ROOT", str(REPO / ".reference/MADGRAV"))
    return pytest.importorskip("improved_pipeline")


def psd(ifo: str):
    from madgrav_ml.data.strain import load_reference_psd

    return load_reference_psd(PREP / f"reference_psd_{ifo}.npz")


@pytest.mark.parametrize("ifo", ["H1", "L1"])
def test_whiten_reproduces_the_deployed_search(ifo):
    """Bit-level fidelity against `MassiveEventPipeline._whiten`.

    This is the whole point of the file. Anything less than agreement here means the
    front end sees a different representation from the one the shipped weights were
    trained on, and every comparison against them is meaningless.
    """
    ip = upstream()
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1e-21, 4 * FS)

    asd = ip.load_detector_asd_o1(str(PREP), ifo)
    reference = ip.whiten_batch_gwpy_o1(
        x[None, :].astype(np.float32), [ifo], {ifo: asd}, True, "o1"
    )[0]
    mine = R.notch(R.whiten(x, FS, reference_psd=psd(ifo)), FS,
                   lines=R.notch_lines_for(ifo, "o1"))

    # Compare on the central 2 s: the FIR corrupts fduration/2 = 1 s at each edge, and
    # the pipeline discards exactly that before the Q-transform.
    c, half = len(x) // 2, FS
    a, b = reference[c - half:c + half].astype(np.float64), mine[c - half:c + half]
    rel = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2))
    assert rel < 1e-6, f"relative rms difference {rel:.3e}"


@pytest.mark.parametrize("ifo", ["H1", "L1"])
def test_notch_lines_match_the_deployed_configuration(ifo):
    """`_whiten` hard-codes `line_configuration="o1"`, even for the O3a prep directory."""
    ip = upstream()
    assert R.notch_lines_for(ifo, "o1") == tuple(
        ip.detector_line_frequencies(ifo, "o1")
    )
    assert R.notch_lines_for(ifo, "o3a") == tuple(
        ip.detector_line_frequencies(ifo, "o3a")
    )
    # And the two really are different, so the choice is not cosmetic.
    assert R.notch_lines_for(ifo, "o1") != R.notch_lines_for(ifo, "o3a")


def test_asd_floor_is_relative_not_absolute():
    """The regression test for the bug itself.

    An O3a PSD sits ~six orders of magnitude below 1e-40 in band. A floor that does not
    scale with the data destroys the whitening while leaving every downstream shape
    check — finite values, sensible tile means, a decreasing loss — perfectly happy.
    """
    f, p = psd("H1")
    assert p.min() < 1e-48 and p.max() > 1e-41, "PSD span assumed by this test changed"
    asd = R.reference_asd((f, p))
    floored = np.asarray(asd.value) ** 2
    in_band = (f > 50) & (f < 500)
    # In band the floor must not touch a single bin.
    assert np.allclose(floored[in_band], p[in_band], rtol=1e-12)
    # An absolute 1e-40 floor would have flattened all of it.
    assert np.median(p[in_band]) < 1e-40


def test_whiten_flattens_real_detector_noise():
    """Whitened noise drawn from the reference PSD is white and unit-variance.

    The direct behavioural statement of what whitening is for. The broken version failed
    this by a factor of ~45 in scale and left the spectrum steeply red.
    """
    p = psd("H1")
    f, s = p
    n = 4 * FS
    rng = np.random.default_rng(1)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    sigma = np.sqrt(np.interp(freqs, f, s) * FS * n / 4.0)
    raw = np.fft.irfft(rng.normal(0, sigma) + 1j * rng.normal(0, sigma), n=n)

    w = R.whiten(raw, FS, reference_psd=p)[n // 4:3 * n // 4]
    assert 0.8 < np.std(w) < 1.25, np.std(w)

    # Flat in band: compare power in two octaves that the raw PSD separates by >100x.
    power = np.abs(np.fft.rfft(w)) ** 2
    fw = np.fft.rfftfreq(len(w), 1.0 / FS)
    low = power[(fw > 40) & (fw < 80)].mean()
    high = power[(fw > 320) & (fw < 640)].mean()
    assert 0.5 < low / high < 2.0, f"not flat: {low / high:.2f}"


def test_whiten_refuses_to_guess_a_psd():
    """A segment-local estimate is a different operation and not what the weights saw."""
    with pytest.raises(ValueError, match="reference PSD"):
        R.whiten(np.zeros(4 * FS), FS)


def test_spectral_variant_is_shaped_like_the_deployed_one():
    """`whiten_spectral` is an ablation, so it must differ only in conditioning.

    Same spectral shape (they divide by the same ASD), different normalisation and
    different edge behaviour. Pinned so that "does the FIR conditioning matter?" is a
    question about conditioning and not about a second bug.
    """
    p = psd("H1")
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1e-21, 4 * FS)
    c, half = len(x) // 2, FS
    fir = R.whiten(x, FS, reference_psd=p)[c - half:c + half]
    spec = R.whiten_spectral(x, FS, reference_psd=p)[c - half:c + half]
    # Highpass the spectral variant to match: the FIR whitening includes the 20 Hz cut.
    from scipy.signal import butter, sosfiltfilt

    spec = sosfiltfilt(butter(8, 20.0, btype="highpass", fs=FS, output="sos"), spec)
    assert np.corrcoef(fir, spec)[0, 1] > 0.98
