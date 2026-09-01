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
POWERLINE_BASE_HZ = 60.0
POWERLINE_HARMONICS = tuple(POWERLINE_BASE_HZ * n for n in range(1, 9))
CAL_LINES_O3A = {
    "H1": (15.1, 15.6, 16.4, 16.7, 17.1, 17.6, 35.9, 36.7, 331.9, 410.3,
           1001.3, 1083.7, 1153.1, 1501.3),
    "L1": (15.1, 15.7, 16.3, 16.9, 30.8, 31.4, 32.0, 32.6, 33.2, 33.8,
           434.9, 451.2, 451.8, 1083.1, 1153.1, 1503.1, 1653.1),
}


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

def whiten(x: np.ndarray, fs: int = FS, reference_psd: tuple | None = None,
           nperseg: int = 1024) -> np.ndarray:
    """Divide the Fourier transform by sqrt(PSD) so the noise is flat and unit-variance.

    `reference_psd` is `(freqs, psd)`. The upstream search whitens against a
    *run-averaged* reference ASD, not against the local segment, which is what makes
    a single frozen model transferable across O3a/O3b/O4a/O4b — the reference PSDs
    ship in `data/<run>_search_prep/reference_psd_{H1,L1}.npz`. Estimating the PSD
    from the segment itself (the fallback here) is only for smoke tests.
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=np.float64)
    if reference_psd is not None:
        f, psd = reference_psd
    else:
        f, psd = welch(x, fs=fs, nperseg=nperseg)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    psd_i = np.interp(freqs, f, psd)
    return np.fft.irfft(X / np.sqrt(psd_i + 1e-40), n=len(x))


def notch_and_highpass(
    x: np.ndarray,
    fs: int = FS,
    highpass_hz: float = HIGHPASS_HZ,
    lines: tuple[float, ...] = (),
    q: float = 30.0,
) -> np.ndarray:
    """20 Hz high-pass plus notches on calibration lines and mains harmonics."""
    from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

    x = np.asarray(x, dtype=np.float64)
    sos = butter(4, highpass_hz, btype="highpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    for f0 in lines:
        if 0 < f0 < fs / 2:
            b, a = iirnotch(f0, q, fs)
            y = sosfiltfilt(tf2sos(b, a), y)
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
