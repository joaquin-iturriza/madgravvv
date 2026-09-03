"""Coherence and morphology — the vetoes that do the actual separating.

Section 8 of `docs/results.tex` measured the front end alone at a fixed false-alarm rate
and got zero efficiency everywhere, because the loudest injection reaches net sigma 19.3
and the glitch background reaches 32.4. Upstream never claims the raw statistic is a
search: its shipped FAR curves are named `far_curve_cond_coh` and
`far_curve_global_coh`, both conditioned on what is in this module.

Ported from `spectrogram_cascade/massive_pipeline.py`, constants from
`massive_calibration_BA.json`. Two quantities and one branch:

  coherence  band-limited symmetric-norm cross-correlation between detectors,
             maximised over +-45 samples (11 ms at 4096 Hz, which covers the 10 ms
             H1-L1 light travel time). `2<a,b> / (|a|^2 + |b|^2)`, NOT a Pearson
             correlation: the symmetric norm penalises an amplitude mismatch between
             detectors, where Pearson would normalise it away. A glitch in one detector
             paired with quiet data in the other scores near zero here and can score
             high under Pearson.

  centroid   energy-weighted mean frequency of the central 0.5 s in 20-400 Hz. A
             high-mass merger puts its energy low; many glitch classes do not.

  branch     `massive = (centroid_H1 < f_cut) and (centroid_L1 < f_cut) and
             (coherence >= tcoh)`. Massive candidates are ranked against one FAR curve
             and everything else against another. That is a two-channel search, and it
             is where the trials factor comes from.
"""

from __future__ import annotations

import numpy as np

# massive_calibration_BA.json, the BA (band-limited + symmetric-norm) calibration.
COHERENCE_MODE = "band_symnorm"
COHERENCE_BAND_HZ = (20.0, 500.0)
COHERENCE_WINDOW_S = 1.0
LAG_SAMPLES = 45
TCOH = 0.12767678245902062
CENTROID_BAND_HZ = (20.0, 400.0)
CENTROID_WINDOW_S = 0.5
F_CUT_HZ = 190.78961461069167


def _centre(x: np.ndarray, fs: int, seconds: float) -> np.ndarray:
    """The central `seconds` of each row. The 4 s window's edges are whitening-corrupted."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    cc = x.shape[1] // 2
    half = int(seconds * fs / 2)
    return x[:, cc - half:cc + half]


def bandlimit(x: np.ndarray, fs: int, band=COHERENCE_BAND_HZ) -> np.ndarray:
    """Hanning window, then zero every FFT bin outside `band`.

    The window comes first and is not undone, exactly as upstream does it. It is part of
    the statistic rather than a preprocessing nicety: it tapers the ends of the 1 s
    window so that a glitch sitting at the edge cannot dominate the cross-correlation.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    spec = np.fft.rfft(x * np.hanning(x.shape[1])[None, :], axis=1)
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    spec[:, (freqs < band[0]) | (freqs > band[1])] = 0
    return np.fft.irfft(spec, n=x.shape[1], axis=1)


def coherence(h1: np.ndarray, l1: np.ndarray, fs: int = 4096,
              band=COHERENCE_BAND_HZ, window_s: float = COHERENCE_WINDOW_S,
              lag_samples: int = LAG_SAMPLES) -> np.ndarray:
    """`max_lag |2<a, roll(b, lag)>| / (|a|^2 + |b|^2)` on band-limited central windows."""
    a = bandlimit(_centre(h1, fs, window_s), fs, band)
    b = bandlimit(_centre(l1, fs, window_s), fs, band)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    ea = (a * a).sum(1)
    out = np.zeros(len(a), np.float32)
    for lag in range(-lag_samples, lag_samples + 1):
        bs = np.roll(b, lag, axis=1)
        eb = (bs * bs).sum(1)
        out = np.maximum(out, (np.abs(2.0 * (a * bs).sum(1)) / (ea + eb + 1e-30)).astype(np.float32))
    return out


def centroid(x: np.ndarray, fs: int = 4096, band=CENTROID_BAND_HZ,
             window_s: float = CENTROID_WINDOW_S) -> np.ndarray:
    """Energy-weighted mean frequency of the central `window_s`, inside `band`."""
    w = _centre(x, fs, window_s)
    w = w - w.mean(1, keepdims=True)
    power = np.abs(np.fft.rfft(w * np.hanning(w.shape[1])[None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(w.shape[1], 1.0 / fs)
    m = (freqs >= band[0]) & (freqs <= band[1])
    return (power[:, m] * freqs[m]).sum(1) / (power[:, m].sum(1) + 1e-30)


def is_massive(coh: np.ndarray, centroid_h1: np.ndarray, centroid_l1: np.ndarray,
               tcoh: float = TCOH, f_cut: float = F_CUT_HZ) -> np.ndarray:
    """The channel branch: which FAR curve a trigger is ranked against."""
    morphology = (np.asarray(centroid_h1) < f_cut) & (np.asarray(centroid_l1) < f_cut)
    return morphology & (np.asarray(coh) >= tcoh)


# --- the storable form -------------------------------------------------------
#
# A background scan cannot keep 100k whitened 1 s series per detector and then
# cross-correlate every one against every time slide: that is 2e8 length-4096 transforms.
# But the band-limited series is *entirely* described by its in-band Fourier
# coefficients — 481 complex numbers for 20-500 Hz over a 1 s window against 4096 real
# samples — and the whole lag scan is one inverse transform of the cross-spectrum, since
# a circular roll does not change the energy in the denominator.


def band_coefficients(x: np.ndarray, fs: int = 4096, band=COHERENCE_BAND_HZ,
                      window_s: float = COHERENCE_WINDOW_S) -> tuple[np.ndarray, int, int]:
    """In-band rfft coefficients of the windowed central slice, plus the band's bin range.

    Lossless for everything `coherence` reads: the out-of-band bins it would zero are
    exactly the ones not stored.
    """
    w = _centre(x, fs, window_s)
    spec = np.fft.rfft(w * np.hanning(w.shape[1])[None, :], axis=1)
    freqs = np.fft.rfftfreq(w.shape[1], 1.0 / fs)
    m = (freqs >= band[0]) & (freqs <= band[1])
    lo = int(np.argmax(m))
    return spec[:, m].astype(np.complex64), lo, w.shape[1]


def _lag_matrix(lo: int, n_band: int, n: int, lag_samples: int) -> np.ndarray:
    """exp(2*pi*i*f*tau/n) for the in-band bins and the +-lag_samples window.

    The full inverse transform produces 4096 lags to use 91 of them. Evaluating only the
    lags that are wanted turns the lag scan into one small matrix product, which is what
    makes it affordable to rank EVERY grid point rather than only the loud ones -- the
    likelihood ratio needs coherence as an input, not as a veto applied afterwards.
    """
    f = np.arange(lo, lo + n_band)[:, None]
    tau = np.arange(-lag_samples, lag_samples + 1)[None, :]
    return np.exp(2j * np.pi * f * tau / n)


_LAG_CACHE: dict = {}


def coherence_from_coefficients(a: np.ndarray, b: np.ndarray, lo: int, n: int,
                                lag_samples: int = LAG_SAMPLES) -> np.ndarray:
    """`coherence` computed from stored band coefficients, for arbitrary pairings.

    The cross-correlation at every lag at once is `irfft(A conj(B))`, and the energies
    come from Parseval, so a lag scan costs one transform instead of 91 dot products.
    """
    a = np.atleast_2d(np.asarray(a))
    b = np.atleast_2d(np.asarray(b))

    # Parseval for numpy's rfft on a real signal of even length: the DC and Nyquist bins
    # count once, every other bin twice. Both are outside a 20-500 Hz band, so the
    # doubling applies to all stored coefficients.
    ea = 2.0 * (np.abs(a) ** 2).sum(1) / n
    eb = 2.0 * (np.abs(b) ** 2).sum(1) / n

    key = (lo, a.shape[1], n, lag_samples)
    if key not in _LAG_CACHE:
        _LAG_CACHE[key] = _lag_matrix(*key)
    # irfft of the cross-spectrum, evaluated only at the lags that matter. The 1/n and
    # the doubling of the one-sided bins are the same conventions as `ea`/`eb` above.
    cross = (a * np.conj(b)) @ _LAG_CACHE[key]
    best = np.abs(2.0 * (2.0 * cross.real) / n).max(axis=1)
    return (best / (ea + eb + 1e-30)).astype(np.float32)
