"""Strain -> network input: whitening, Q-transform, tiling, normalisation.

This module reproduces the upstream representation exactly at its defaults, and makes
each of its three lossy steps a switchable knob, because those switches *are* Phase 3
of the improvement plan (experiments R1-R4).

Upstream defaults, read off `improved/improved_pipeline.py` rather than from the paper:

    sample rate           4096 Hz  (noise stored at 8192 Hz, downsampled)
    context               2.0 s fed to the Q-transform, centre-cropped to 1.0 s
    Q-transform           frange (10, 1291) Hz, qrange (4, 64),
                          tres 0.002 s, fres 0.5 Hz, norm="median", whiten=False
    magnitude             log1p(|Q|)
    resize                bilinear (scipy.ndimage.zoom order=1) to 256 x 128 (f x t)
    normalisation         per-tile min-max to [0, 1]

Note for anyone reading the plan alongside this: the amplitude information is not
thrown away by a missing log — `log1p` is already applied. It is thrown away by the
**per-tile min-max**, which rescales every tile to the same [0,1] range and so removes
any comparison of one tile's loudness against another. That is what `amplitude="asd"`
and `amplitude="log"` below exist to test.

Constraint C5: this file deliberately implements whitening with numpy/scipy and gwpy
only. Do not add `ml4gw` — its whitening changes the coherence statistic and therefore
the results, and the upstream README says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# --- upstream constants, kept together so a change is visible in one diff -----
FS = 4096
SPEC_SIZE = (256, 128)          # (frequency bins, time bins)
QTRANSFORM_FRANGE = (10.0, 1291.0)
QTRANSFORM_QRANGE = (4.0, 64.0)
QTRANSFORM_TRES = 0.002
QTRANSFORM_FRES = 0.5
CONTEXT_SECONDS = 2.0
CENTER_CROP_SECONDS = 1.0
HIGHPASS_HZ = 20.0
WHITEN_FDURATION = 2.0          # gwpy FIR length; corrupts 1 s at each window edge
NOTCH_Q = 40.0
ASD_FLOOR_RELATIVE = 1e-10      # PSD floor as a fraction of the PSD's own median
POWERLINE_BASE_HZ = 60.0
POWERLINE_HARMONICS = tuple(POWERLINE_BASE_HZ * n for n in range(1, 9))
CAL_LINES_O1 = {
    "H1": (35.9, 36.7, 37.3, 331.9),
    "L1": (33.7, 34.7, 35.3, 331.3),
}
L1_O1_DITHER_LINES = (600.1, 625.1, 650.1, 675.1)
CAL_LINES_O3A = {
    "H1": (15.1, 15.6, 16.4, 16.7, 17.1, 17.6, 35.9, 36.7, 331.9, 410.3,
           1001.3, 1083.7, 1153.1, 1501.3),
    "L1": (15.1, 15.7, 16.3, 16.9, 30.8, 31.4, 32.0, 32.6, 33.2, 33.8,
           434.9, 451.2, 451.8, 1083.1, 1153.1, 1503.1, 1653.1),
}
# Which of the two the DEPLOYED search uses is not a matter of taste -- see
# `notch_lines_for` below. "o1" is the default because it is what the shipped weights
# were built against, not because it is the physically right choice for O3a.
LINE_CONFIGURATION = "o1"


@dataclass
class TileSpec:
    """Everything that turns a stretch of strain into a network input.

    Every field here is an ablation axis. Constructing one of these, stamping it into
    the run config and carrying it into `summary.json` is what makes an R1-R4 result
    reproducible; do not thread loose keyword arguments through the pipeline instead.
    """

    sample_rate: int = FS
    context_seconds: float = CONTEXT_SECONDS
    crop_seconds: float = CENTER_CROP_SECONDS
    frange: tuple[float, float] = QTRANSFORM_FRANGE
    qrange: tuple[float, float] = QTRANSFORM_QRANGE
    tres: float = QTRANSFORM_TRES
    fres: float = QTRANSFORM_FRES
    size: tuple[int, int] = SPEC_SIZE

    # R1: keep the Q-transform phase as a second channel instead of discarding it.
    # Phase coherence is exactly what matched filtering exploits. Cost is a few hundred
    # parameters in the first conv (1->32 becomes 2->32), which C2 must account for.
    phase_channel: bool = False

    # R2: what replaces per-tile min-max. "minmax" is upstream.
    #   "log"    - log1p magnitude, then a fixed global scale (tile-independent)
    #   "asd"    - standardise against the reference noise floor, so relative
    #              amplitude survives across tiles
    amplitude: str = "minmax"

    # Fixed statistics for the tile-independent modes. Fit on the training fold only.
    noise_mean: float | None = None
    noise_std: float | None = None

    def __post_init__(self) -> None:
        if self.amplitude not in ("minmax", "log", "asd"):
            raise ValueError(f"unknown amplitude mode {self.amplitude!r}")
        if self.amplitude == "asd" and (self.noise_mean is None or self.noise_std is None):
            raise ValueError(
                "amplitude='asd' needs noise_mean/noise_std measured on the training "
                "fold; leaving them unset would fit the scale on whatever data is at "
                "hand, which is exactly the leak the fold discipline exists to stop"
            )

    @property
    def n_channels(self) -> int:
        return 2 if self.phase_channel else 1

    def as_dict(self) -> dict:
        return asdict(self)


# --- whitening ----------------------------------------------------------------

def notch_lines_for(ifo: str, configuration: str = LINE_CONFIGURATION) -> tuple[float, ...]:
    """The line list the pipeline notches for `ifo`.

    Upstream keeps two lists and a `infer_line_configuration()` that would pick "o3a"
    for an O3a prep directory — but `MassiveEventPipeline._whiten` calls
    `whiten_batch_gwpy_o1(..., "o1")` with the string hard-coded, so the O3a search
    actually notches the **O1** calibration lines and the O1 L1 dither lines. The
    shipped weights were therefore built against `configuration="o1"`, which is why
    that is the default here: matching the deployed model matters more than notching
    the physically correct lines. `configuration="o3a"` is an R-series ablation, and a
    plausible upstream bug worth reporting separately.
    """
    lines = list(POWERLINE_HARMONICS)
    if configuration == "o3a":
        lines.extend(CAL_LINES_O3A[ifo])
    elif configuration == "o1":
        lines.extend(CAL_LINES_O1[ifo])
        if ifo == "L1":
            lines.extend(L1_O1_DITHER_LINES)
    else:
        raise ValueError(f"unknown line configuration {configuration!r}")
    return tuple(lines)


def reference_asd(reference_psd: tuple, floor_relative: float = ASD_FLOOR_RELATIVE):
    """`(freqs, psd)` -> a gwpy ASD `FrequencySeries`, floored the way upstream floors it.

    The floor is RELATIVE — `median(psd) * 1e-10` — and that detail is load-bearing.
    An O3a reference PSD runs from 3e-51 to 7e-40, so an *absolute* floor anywhere near
    1e-40 swamps the entire sensitive band and the "whitening" degenerates into a
    constant rescale. See `whiten_spectral`.
    """
    from gwpy.frequencyseries import FrequencySeries

    f, psd = reference_psd
    psd = np.asarray(psd, dtype=np.float64)
    positive = psd[np.isfinite(psd) & (psd > 0.0)]
    if positive.size == 0:
        raise ValueError("reference PSD has no positive finite bins")
    floored = np.maximum(psd, float(np.median(positive)) * floor_relative)
    f = np.asarray(f, dtype=np.float64)
    return FrequencySeries(np.sqrt(floored), f0=float(f[0]), df=float(f[1] - f[0]))


def whiten(x: np.ndarray, fs: int = FS, reference_psd: tuple | None = None,
           fduration: float = WHITEN_FDURATION,
           highpass_hz: float = HIGHPASS_HZ,
           floor_relative: float = ASD_FLOOR_RELATIVE) -> np.ndarray:
    """The whitening the deployed search performs, ported exactly.

    `MassiveEventPipeline._whiten` -> `whiten_batch_gwpy_o1`:

        ts  = TimeSeries(x - mean(x), sample_rate=fs)
        tsw = ts.whiten(asd=<floored reference ASD>, fduration=2.0, highpass=20)

    which is a time-domain FIR whitening filter, not a spectral division. The 20 Hz
    high-pass is part of it, so there is no separate high-pass step afterwards. The FIR
    corrupts `fduration / 2` = 1 s at each edge of the window, which is exactly why the
    pipeline whitens 4 s and keeps only the central 2 s of context.

    `reference_psd` is `(freqs, psd)`, the *run-averaged* reference that ships in
    `data/<run>_search_prep/reference_psd_{H1,L1}.npz`. Whitening against the run
    average rather than the local segment is what makes one frozen model transferable
    across observing runs; the ASD-consistency veto at the end of the upstream pipeline
    exists precisely because the two differ.

    WHY THIS REPLACED A SPECTRAL DIVISION. The first version of this function ported
    `improved/utilities.py::whiten`, which divides by `sqrt(psd + 1e-40)`. That is the
    training-data helper, not the search path, and its absolute epsilon is ~1e6 times
    the real PSD across the whole sensitive band — so it whitened nothing. Every tile
    built before this fix is invalid.
    """
    from gwpy.timeseries import TimeSeries

    x = np.asarray(x, dtype=np.float64)
    if reference_psd is None:
        raise ValueError(
            "whiten() needs the run-averaged reference PSD; a segment-local estimate "
            "is a different operation and is not what the shipped weights saw"
        )
    ts = TimeSeries(x - x.mean(), sample_rate=fs)
    asd = reference_asd(reference_psd, floor_relative)
    tsw = ts.whiten(asd=asd, fduration=fduration, highpass=highpass_hz)
    return np.asarray(tsw.value, dtype=np.float64)


def whiten_spectral(x: np.ndarray, fs: int = FS, reference_psd: tuple | None = None,
                    nperseg: int = 1024,
                    floor_relative: float = ASD_FLOOR_RELATIVE) -> np.ndarray:
    """Plain spectral division, `X / sqrt(PSD)`. An ablation, not the default.

    This is upstream's `improved/utilities.py::whiten` with its absolute `+1e-40`
    replaced by the same relative floor `reference_asd` uses. Kept because "does the
    FIR conditioning matter, or would a spectral divide do?" is a real question for the
    R-series, and because the two differ mainly at the window edges — which the centre
    crop discards anyway.
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=np.float64)
    if reference_psd is not None:
        f, psd = reference_psd
    else:
        f, psd = welch(x, fs=fs, nperseg=nperseg)
    psd = np.asarray(psd, dtype=np.float64)
    positive = psd[np.isfinite(psd) & (psd > 0.0)]
    floor = float(np.median(positive)) * floor_relative if positive.size else 0.0
    psd = np.maximum(psd, floor)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return np.fft.irfft(X / np.sqrt(np.interp(freqs, f, psd)), n=len(x))


def notch(x: np.ndarray, fs: int = FS, lines: tuple[float, ...] = (),
          q: float = NOTCH_Q) -> np.ndarray:
    """Notch the calibration and mains lines, Q=40, `filtfilt` — as `apply_o1_notches`.

    No high-pass here: `whiten` already applied it inside the FIR, and running a second
    Butterworth over the same corner would be a filter the shipped weights never saw.
    """
    from scipy.signal import filtfilt, iirnotch

    y = np.asarray(x, dtype=np.float64).copy()
    for f0 in lines:
        if 0 < f0 < fs / 2:
            b, a = iirnotch(w0=f0, Q=q, fs=fs)
            y = filtfilt(b, a, y)
    return y


# --- time-frequency -----------------------------------------------------------

def centre_crop(x: np.ndarray, fs: int, seconds: float | None) -> np.ndarray:
    """Central `seconds` of a 1-D or (N, T) array."""
    x = np.asarray(x)
    if seconds is None or seconds <= 0:
        return x
    total = x.shape[-1]
    target = int(round(seconds * fs))
    if target <= 0 or target >= total:
        return x
    start = max(0, (total - target) // 2)
    return x[..., start:start + target]


def _crop_bounds(n_cols: int, total_seconds: float, crop_seconds: float | None):
    if crop_seconds is None or crop_seconds <= 0 or crop_seconds >= total_seconds:
        return 0, n_cols
    r = crop_seconds / float(total_seconds)
    start = int(np.floor(0.5 * (1.0 - r) * n_cols))
    stop = int(np.floor(0.5 * (1.0 + r) * n_cols))
    start = max(0, min(start, n_cols))
    stop = max(start + 1, min(stop, n_cols))
    return start, stop


def qtransform(waveform: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Complex Q-transform of one whitened waveform, as (frequency, time).

    Returns the complex array so the caller can choose magnitude, phase, or both.
    Upstream throws the phase away here; R1 is the experiment that keeps it.
    """
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries(np.asarray(waveform, dtype=np.float64), sample_rate=spec.sample_rate)
    qgram = ts.q_transform(
        qrange=tuple(spec.qrange),
        frange=tuple(spec.frange),
        tres=spec.tres,
        fres=spec.fres,
        norm="median",
        whiten=False,
    )
    return np.asarray(qgram).T


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize to (frequency, time), matching upstream's `zoom(..., order=1)`."""
    from scipy.ndimage import zoom

    zf = size[0] / image.shape[0]
    zt = size[1] / image.shape[1]
    return zoom(image, (zf, zt), order=1).astype(np.float32)


# --- normalisation ------------------------------------------------------------

def min_max_norm(x: np.ndarray) -> np.ndarray:
    """Per-tile min-max to [0,1] — the upstream default, and the amplitude-killer.

    Deliberate upstream choice: it stops the model learning "loud = signal", which
    would otherwise be dominated by glitches. Any replacement has to keep that
    property, which is why the alternatives below are *fixed, tile-independent*
    transforms rather than a different per-tile rescaling.
    """
    x = np.asarray(x, dtype=np.float32)
    lo = x.min(axis=(-2, -1), keepdims=True)
    hi = x.max(axis=(-2, -1), keepdims=True)
    return (x - lo) / (hi - lo + 1e-12)


def noise_referenced_norm(x: np.ndarray, noise_mean: float, noise_std: float) -> np.ndarray:
    """Standardise against the noise-pool statistics, preserving relative amplitude.

    A faint injection stays near zero and a loud one goes well above it, which is the
    discriminant per-tile min-max removes. The statistics must come from the training
    fold; passing anything else silently leaks.
    """
    return ((np.asarray(x, dtype=np.float32) - noise_mean) / (noise_std + 1e-12)).astype(np.float32)


# --- the whole path -----------------------------------------------------------

def make_tile(waveform: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Whitened strain -> (channels, frequency, time) network input.

    `waveform` is expected already whitened and filtered; this function owns only the
    time-frequency half of the path so the whitening can be cached across ablations.
    """
    qi = centre_crop(waveform, spec.sample_rate, spec.context_seconds)
    total_seconds = len(qi) / float(spec.sample_rate)
    q = qtransform(qi, spec)

    start, stop = _crop_bounds(q.shape[1], total_seconds, spec.crop_seconds)
    q = q[:, start:stop]

    mag = np.log1p(np.abs(q)).astype(np.float32)
    mag = resize(mag, spec.size)

    if spec.amplitude == "minmax":
        mag = min_max_norm(mag)
    elif spec.amplitude == "asd":
        mag = noise_referenced_norm(mag, spec.noise_mean, spec.noise_std)
    # "log" leaves log1p as-is: a fixed, tile-independent transform

    channels = [mag]
    if spec.phase_channel:
        # Phase is circular, so feed it as-is only where the magnitude supports it;
        # a raw angle in [-pi, pi] has a discontinuity a convolution cannot cross.
        # sin/cos would cost a third channel, so the wrapped angle is scaled to
        # [-1, 1] here and the sin/cos variant is left as an explicit R1 sub-ablation.
        ph = resize((np.angle(q) / np.pi).astype(np.float32), spec.size)
        channels.append(ph)

    return np.stack(channels, axis=0).astype(np.float32)
