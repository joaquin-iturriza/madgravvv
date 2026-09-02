"""The injection path.

The load-bearing test here is `test_recovered_snr_matches_request`: it matched-filters
the injection back out of the whitened window and checks the SNR is the one that was
asked for. Everything upstream of it (the epoch convention, the projection, the
whitening, the rescaling) has to be right for that number to come out, and every one of
those is a mistake that produces a training set which looks fine and is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from madgrav_ml.data.injections import InjectionParameters, ParameterSampler
from madgrav_ml.data.representation import notch, whiten, whiten_spectral
from madgrav_ml.data import waveforms as W

lal = pytest.importorskip("lal")
pytest.importorskip("lalsimulation")

FS = 4096
WINDOW = 4.0


REFERENCE_PSD_DIR = Path(__file__).resolve().parents[1] / ".reference/MADGRAV/data/o3a_search_prep"


def flat_psd(level: float = 1e-46):
    """A white reference PSD. Makes the whitened-domain identities exact and checkable."""
    f = np.linspace(0.0, FS / 2.0, 4097)
    return f, np.full_like(f, level)


def real_psd(ifo: str = "H1"):
    """The run-averaged O3a reference the search actually whitens against.

    Worth using rather than a flat curve: it spans 1e-51 to 1e-40, and it is that span
    that turned an absolute `+1e-40` floor into a whitening that did nothing.
    """
    path = REFERENCE_PSD_DIR / f"reference_psd_{ifo}.npz"
    if not path.exists():
        pytest.skip("vendored reference PSDs not present; run scripts/vendor_reference.sh")
    from madgrav_ml.data.strain import load_reference_psd

    return load_reference_psd(path)


def coloured_noise(psd, n: int, rng) -> np.ndarray:
    """Gaussian noise with one-sided PSD `psd`, in the convention E|X_k|^2 = S_k fs N / 2."""
    f, s = psd
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    s_i = np.interp(freqs, f, s)
    sigma = np.sqrt(s_i * FS * n / 4.0)  # per real/imag part
    spectrum = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)
    return np.fft.irfft(spectrum, n=n)


def params(**kw) -> InjectionParameters:
    base = dict(
        mass1=35.0, mass2=30.0, spin1z=0.0, spin2z=0.0, network_snr=12.0,
        ra=1.2, dec=-0.4, psi=0.7, inclination=0.3, phase=0.0, time_shift=0.0,
    )
    base.update(kw)
    return InjectionParameters(**base)


def engine(convention: str = "detector", ifos=("H1",)) -> W.InjectionEngine:
    return W.InjectionEngine(
        backend=W.LALWaveformBackend(),
        psds={ifo: flat_psd() for ifo in ifos},
        notch_lines={ifo: () for ifo in ifos},
        sample_rate=FS,
        window_seconds=WINDOW,
        snr_convention=convention,
    )


# --- generation ---------------------------------------------------------------


def test_coalescence_sits_at_the_window_centre():
    """A mis-read epoch shifts every injection by the same amount, which no loss curve
    would ever show. The merger is the loudest sample, so peak position pins it."""
    w = W.LALWaveformBackend().generate_window(params(), FS, WINDOW)
    n = len(w.plus)
    assert n == int(WINDOW * FS)
    amplitude = np.hypot(w.plus, w.cross)
    peak = int(np.argmax(amplitude))
    # Ringdown follows the peak by a few ms; the merger should land within 20 ms of
    # centre, i.e. far tighter than anything the 2 ms tile resolution could hide.
    assert abs(peak - n // 2) < 0.02 * FS, f"peak at {peak}, centre {n // 2}"
    # And nothing after the ringdown.
    assert np.all(amplitude[n // 2 + int(0.2 * FS):] == 0.0)


def test_long_inspiral_reports_what_it_dropped():
    """A 10+10 source from 20 Hz outruns a 1 s window. The lost inspiral must be
    reported, not swallowed: the rescaling that follows would otherwise compensate for
    it and hide SNR that is in the real signal and absent from ours."""
    w = W.LALWaveformBackend().generate_window(params(mass1=10.0, mass2=10.0), FS, 1.0)
    assert w.truncated_seconds > 0.0
    longer = W.LALWaveformBackend().generate_window(
        params(mass1=10.0, mass2=10.0), FS, 16.0
    )
    assert longer.truncated_seconds < w.truncated_seconds


# --- projection ---------------------------------------------------------------


def test_antenna_response_is_bounded_and_time_dependent():
    fp, fc = W.antenna_response("H1", 1.2, -0.4, 0.7, W.REFERENCE_GPS)
    assert abs(fp) <= 1.0 and abs(fc) <= 1.0
    # Six hours of Earth rotation must change the pattern; a GPS time that did nothing
    # would mean the sidereal dependence had been dropped somewhere.
    fp2, fc2 = W.antenna_response("H1", 1.2, -0.4, 0.7, W.REFERENCE_GPS + 6 * 3600)
    assert not np.isclose(fp, fp2, atol=1e-3) or not np.isclose(fc, fc2, atol=1e-3)


def test_geocentre_delay_is_within_the_light_travel_time():
    for ifo in ("H1", "L1"):
        dt = W.earth_centre_delay(ifo, 1.2, -0.4, W.REFERENCE_GPS)
        assert abs(dt) < 0.0213, f"{ifo} delay {dt} exceeds an Earth radius of light time"
    h = W.earth_centre_delay("H1", 1.2, -0.4, W.REFERENCE_GPS)
    l = W.earth_centre_delay("L1", 1.2, -0.4, W.REFERENCE_GPS)
    assert abs(h - l) < 0.0101, "H1-L1 separation is 10 ms of light travel"


def test_fractional_shift_moves_the_peak():
    x = np.zeros(1024)
    x[512] = 1.0
    y = W._fractional_shift(x, 10.0)
    assert int(np.argmax(y)) == 522
    half = W._fractional_shift(x, 0.5)
    # A half-sample delay splits the impulse between the two neighbouring samples.
    assert np.isclose(half[512], half[513], atol=1e-6)


def test_time_shift_too_large_is_refused_not_wrapped():
    e = engine()
    w = e.backend.generate_window(params(), FS, WINDOW)
    with pytest.raises(ValueError, match="wrap"):
        e.project(w, "H1", params(time_shift=2.0), W.REFERENCE_GPS)


# --- SNR ----------------------------------------------------------------------


def test_optimal_snr_matches_the_whitened_norm():
    """Cross-check of the frequency-domain SNR against an independent identity.

    For this code's whitening convention, rho^2 = (2 / fs) * sum(w^2) over the whitened
    waveform. Two derivations agreeing is what makes the SNR number trustworthy; a
    single formula with a factor of 2 wrong is not detectable from inside the pipeline.
    """
    psd = flat_psd()
    w = W.LALWaveformBackend().generate_window(params(), FS, WINDOW)
    h = 0.7 * w.plus + 0.3 * w.cross
    rho_freq = W.optimal_snr(h, FS, psd, f_low=0.0)
    # The identity is a property of a plain spectral divide. `whiten` is the deployed
    # FIR path, which normalises differently; the two are compared in
    # test_recovered_snr_matches_request, where the normalisation cancels.
    whitened = whiten_spectral(h, FS, reference_psd=psd)
    rho_time = np.sqrt(2.0 * np.sum(whitened ** 2) / FS)
    assert np.isclose(rho_freq, rho_time, rtol=1e-6), (rho_freq, rho_time)


def test_optimal_snr_scales_linearly_with_amplitude():
    psd = flat_psd()
    w = W.LALWaveformBackend().generate_window(params(), FS, WINDOW)
    assert np.isclose(W.optimal_snr(3.0 * w.plus, FS, psd),
                      3.0 * W.optimal_snr(w.plus, FS, psd), rtol=1e-9)


def test_snr2_fraction_in_crop_is_a_fraction():
    """It must never exceed 1, which the first implementation did.

    That version zeroed the tails and re-integrated |h~|^2/S in the frequency domain.
    Truncating in time adds broadband leakage, the leakage lands where the ASD is small,
    and dividing by a tiny PSD returned a "fraction" of 1.084 on a real O3a curve.
    """
    e = engine()
    p = params(mass1=15.0, mass2=12.0, network_snr=10.0)
    h = e.whitened_signal(p, "H1", W.REFERENCE_GPS)
    whole = W.snr2_fraction_in_crop(h, FS, WINDOW)
    crop = W.snr2_fraction_in_crop(h, FS, 1.0)
    assert np.isclose(whole, 1.0, atol=1e-12)
    assert 0.0 < crop <= 1.0
    assert crop <= whole


# --- the whole path -----------------------------------------------------------


def matched_filter_snr(data: np.ndarray, template: np.ndarray, noise_sigma: float,
                       keep_seconds: float = 2.0) -> float:
    """Peak matched-filter SNR of `template` in whitened `data`.

    Both are whitened, so the noise-weighted inner product is a plain correlation. The
    normalisation is `||h|| * sigma_n`, which makes the statistic unit-variance under
    noise alone and equal to the optimal SNR when the signal is present.

    Only the central `keep_seconds` are used: the whitening FIR corrupts fduration/2 =
    1 s at each edge of the window, which is exactly why the pipeline whitens 4 s and
    keeps 2 s of context.
    """
    n = len(data)
    keep = int(keep_seconds * FS)
    lo = (n - keep) // 2
    d = data[lo:lo + keep]
    t = template[lo:lo + keep]
    norm = np.sqrt(np.sum(t ** 2))
    corr = np.fft.irfft(np.fft.rfft(d) * np.conj(np.fft.rfft(t)), n=keep)
    return float(np.max(np.abs(corr)) / (norm * noise_sigma))


def test_recovered_snr_matches_request():
    """Inject at a known SNR into real-PSD noise and matched-filter it back out.

    This is the load-bearing test. It would catch a wrong LAL epoch, a dropped antenna
    factor, a signal whitened differently from the noise it is added to, or a rescaling
    against the wrong PSD — every one of which yields a stage-2 training set that trains
    happily and means nothing. It is also the test that caught the whitening bug: with
    the old `sqrt(psd + 1e-40)` divide the recovered SNR was off by orders of magnitude,
    because that epsilon is ~1e6 times an O3a PSD across the whole sensitive band.
    """
    psd = real_psd("H1")
    e = W.InjectionEngine(
        backend=W.LALWaveformBackend(), psds={"H1": psd}, notch_lines={"H1": ()},
        sample_rate=FS, window_seconds=WINDOW, snr_convention="detector",
    )
    rng = np.random.default_rng(0)
    n = int(WINDOW * FS)
    raw_noise = coloured_noise(psd, n, rng)
    white_noise = whiten(raw_noise, FS, reference_psd=psd)
    centre = white_noise[n // 4:3 * n // 4]
    sigma = float(np.std(centre))
    # The whitening is calibrated against this PSD, so noise drawn from it comes out at
    # unit variance. A sigma far from 1 would mean the two disagree.
    assert 0.8 < sigma < 1.25, sigma

    p = params(network_snr=20.0)
    h = e.whitened_signal(p, "H1", W.REFERENCE_GPS)
    assert np.isclose(matched_filter_snr(h, h, sigma), 20.0, rtol=0.10), \
        matched_filter_snr(h, h, sigma)
    recovered = matched_filter_snr(white_noise + h, h, sigma)
    assert 20.0 - 5.0 < recovered < 20.0 + 5.0, recovered


def test_whitened_injection_is_linear():
    """Adding a whitened waveform to whitened noise == whitening noise + waveform.

    Trivially true today because `whiten` and the filter chain are linear. The test
    exists so that a future normalised or segment-local whitening cannot break the
    equivalence silently — the injections would then be quietly mis-scaled.
    """
    psd = real_psd("H1")
    rng = np.random.default_rng(1)
    n = int(WINDOW * FS)
    noise = coloured_noise(psd, n, rng)
    w = W.LALWaveformBackend().generate_window(params(), FS, WINDOW)
    h = 1e-21 * w.plus
    lines = (60.0, 120.0)

    def chain(x):
        return notch(whiten(x, FS, reference_psd=psd), FS, lines=lines)

    together = chain(noise + h)
    apart = chain(noise) + chain(h)
    assert np.allclose(together, apart, rtol=1e-8, atol=1e-8 * np.abs(together).max())


def test_network_convention_is_quieter_per_detector_than_detector_convention():
    """The two SNR conventions are not interchangeable, and the difference is large.

    A tile drawn at rho = 12 on the `detector` convention holds a louder source than one
    drawn at rho = 12 on `network`. Quoting an efficiency curve without saying which was
    used is meaningless, so the distinction is pinned by a test rather than a comment.
    """
    p = params(network_snr=12.0)
    per_det = engine("detector", ("H1", "L1")).whitened_signal(p, "H1", W.REFERENCE_GPS)
    network = engine("network", ("H1", "L1")).whitened_signal(p, "H1", W.REFERENCE_GPS)
    assert np.sum(network ** 2) < np.sum(per_det ** 2)


def test_injection_preserves_relative_detector_amplitudes():
    """The coherence statistic reads the H1/L1 amplitude ratio, so the network-SNR
    rescaling must be a single common factor, not one factor per detector."""
    e = engine("network", ("H1", "L1"))
    p = params()
    w = e.backend.generate_window(p, FS, WINDOW)
    raw = {ifo: e.project(w, ifo, p, W.REFERENCE_GPS) for ifo in ("H1", "L1")}
    scaled = {ifo: e.whitened_signal(p, ifo, W.REFERENCE_GPS) for ifo in ("H1", "L1")}
    raw_ratio = W.optimal_snr(raw["H1"], FS, flat_psd()) / W.optimal_snr(raw["L1"], FS, flat_psd())
    scaled_ratio = (np.sqrt(np.sum(scaled["H1"] ** 2))
                    / np.sqrt(np.sum(scaled["L1"] ** 2)))
    assert np.isclose(raw_ratio, scaled_ratio, rtol=1e-4)


def test_network_snr_sums_in_quadrature_over_detectors():
    """A source drawn at network SNR 15 must actually have network SNR 15."""
    e = engine("network", ("H1", "L1"))
    p = params(network_snr=15.0)
    wave = e.backend.generate_window(p, FS, WINDOW)
    factor = e.scale_factor(wave, "H1", p, W.REFERENCE_GPS)
    total = 0.0
    for ifo in ("H1", "L1"):
        h = factor * e.project(wave, ifo, p, W.REFERENCE_GPS)
        total += W.optimal_snr(h, FS, e.psds[ifo], e.f_low) ** 2
    assert np.isclose(np.sqrt(total), 15.0, rtol=1e-6)


def test_length_mismatch_is_refused():
    e = engine()
    with pytest.raises(ValueError, match="window_seconds"):
        e.inject(np.zeros(100), params(), "H1", W.REFERENCE_GPS)


def test_sampler_with_an_external_rng_is_reproducible_and_independent():
    """Two workers with different seeds must not draw the same injections."""
    s = ParameterSampler(seed=0)
    a = s.draw(np.random.default_rng([7, 0]))
    b = ParameterSampler(seed=0).draw(np.random.default_rng([7, 0]))
    c = s.draw(np.random.default_rng([7, 1]))
    assert a.as_dict() == b.as_dict()
    assert a.as_dict() != c.as_dict()
