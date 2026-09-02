"""The waveform backend: parameters in, whitened detector strain out.

This is what `stage2_margin.py::_injector` refuses to run without. A stage-2 run whose
"signal" tiles are pure noise trains to a perfectly plausible loss curve and is
worthless, so the backend is a hard dependency rather than an optional extra.

WHERE THE INJECTION HAPPENS. The noise provider hands out strain that is already
whitened and notched, so this module adds a *whitened* waveform to it. That is not an
approximation: whitening and the notch/high-pass chain are both linear and
time-invariant, so

    filter(whiten(n + h))  ==  filter(whiten(n)) + filter(whiten(h))

exactly. `tests/test_waveforms.py::test_whitened_injection_is_linear` asserts it
numerically, which is what would catch a future normalised whitening that broke the
identity silently.

WHAT SETS THE AMPLITUDE. The distance passed to LAL is arbitrary and cancels: every
waveform is rescaled to a drawn target SNR. The convention for that target is a config
switch, and it matters more than it looks --

  * `network`  (default) -- scale so that sqrt(sum_ifo rho_ifo^2) over every detector
    with a reference PSD equals the target, then inject the scaled waveform into the
    one detector this tile belongs to. The single-detector front end (C1) sees one
    detector at a time, but the efficiency curve this project quotes is against
    *network* SNR, and computing it needs both PSDs -- which we have.
  * `detector` -- scale so this detector alone sees the target. This is the cheaper
    convention and is probably what upstream's per-detector training banks use, so it
    exists to make that comparison possible. A tile drawn at rho=8 under `detector`
    holds a substantially louder source than one drawn at rho=8 under `network`;
    quoting an efficiency curve without saying which was used is meaningless.

WHAT THE NETWORK ACTUALLY SEES. The target SNR is the standard one, integrated over
the whole analysis window. The tile is a Q-transform of the central `crop_seconds` of
the central `context_seconds`, so a long low-mass inspiral puts part of its SNR outside
the tile. `snr_in_window_fraction` measures that; it is a diagnostic for the injection
campaign, not something the training loop needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from madgrav_ml.data.injections import InjectionParameters

# Only used when a noise window carries no GPS time of its own. Earth rotation makes the
# antenna pattern a function of GPS time, but we draw right ascension uniformly, so the
# *marginal* distribution of (F+, Fx) is the same whichever time is used -- the real GPS
# time matters for a coherent two-detector injection, not for a single-detector draw.
REFERENCE_GPS = 1_240_000_000.0  # mid-O3a

DEFAULT_APPROXIMANT = "IMRPhenomPv2"
DEFAULT_F_LOWER = 20.0            # matches the 20 Hz high-pass in the filter chain
DEFAULT_DISTANCE_MPC = 1000.0     # cancels under SNR rescaling; only sets the raw scale


def _lal():
    """Import lal/lalsimulation lazily.

    Deliberate: importing lalsimulation costs ~2 s and pulls a large shared library, and
    the same cold-import cost is what once made a tile look like it took 4.6 s when it
    takes 277 ms. Nothing that only builds noise tiles should pay it.
    """
    import lal
    import lalsimulation

    return lal, lalsimulation


def detector(ifo: str):
    """The LAL detector record for `ifo`."""
    lal, _ = _lal()
    index = {
        "H1": lal.LALDetectorIndexLHODIFF,
        "L1": lal.LALDetectorIndexLLODIFF,
        "V1": lal.LALDetectorIndexVIRGODIFF,
    }
    if ifo not in index:
        raise KeyError(f"unknown detector {ifo!r}; known: {sorted(index)}")
    return lal.CachedDetectors[index[ifo]]


def antenna_response(ifo: str, ra: float, dec: float, psi: float,
                     gps: float) -> tuple[float, float]:
    """(F+, Fx) for `ifo` toward (ra, dec) with polarisation `psi` at GPS time `gps`."""
    lal, _ = _lal()
    gmst = lal.GreenwichMeanSiderealTime(gps)
    return lal.ComputeDetAMResponse(detector(ifo).response, ra, dec, psi, gmst)


def earth_centre_delay(ifo: str, ra: float, dec: float, gps: float) -> float:
    """Arrival time at `ifo` minus arrival time at the geocentre, in seconds.

    Up to ~10 ms between H1 and L1. Irrelevant to a single-detector tile (it is
    degenerate with the drawn coalescence offset) and essential to the coherence
    statistic, which is why it is applied rather than dropped.
    """
    lal, _ = _lal()
    return float(lal.TimeDelayFromEarthCenter(detector(ifo).location, ra, dec, gps))


def _fractional_shift(x: np.ndarray, samples: float) -> np.ndarray:
    """Delay `x` by `samples` (may be fractional) via a Fourier phase ramp.

    The shift is circular. That is safe here only because the waveform is compactly
    supported well away from both window edges, which `_centre` guarantees by placing
    the coalescence at the centre and zero-padding; the caller asserts the shift is
    small compared with the padding.
    """
    n = len(x)
    k = np.fft.rfftfreq(n)  # cycles per sample
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * k * samples), n=n)


@dataclass
class GeneratedWaveform:
    """A polarisation pair placed in a fixed-length window, coalescence at the centre."""

    plus: np.ndarray
    cross: np.ndarray
    sample_rate: int
    truncated_seconds: float  # early inspiral dropped because it predates the window

    @property
    def duration(self) -> float:
        return len(self.plus) / float(self.sample_rate)


class LALWaveformBackend:
    """IMRPhenomPv2 (baseline) / IMRPhenomXPHM (the upstream banks) behind one call.

    `generate` returns both polarisations already placed in a window of the requested
    duration with the coalescence at the centre sample, so no caller has to reason about
    LAL's epoch convention. That is the whole reason this class exists rather than a
    bare function: getting the epoch wrong shifts every injection by the same amount,
    which does not look like a bug in any loss curve.
    """

    def __init__(
        self,
        approximant: str = DEFAULT_APPROXIMANT,
        f_lower: float = DEFAULT_F_LOWER,
        f_ref: float | None = None,
        distance_mpc: float = DEFAULT_DISTANCE_MPC,
    ):
        self.approximant_name = approximant
        self.f_lower = float(f_lower)
        self.f_ref = float(f_ref) if f_ref is not None else float(f_lower)
        self.distance_mpc = float(distance_mpc)

    def generate(
        self, params: InjectionParameters, sample_rate: int, duration: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Satisfies the `WaveformBackend` protocol: (h+, hx) at `sample_rate`."""
        w = self.generate_window(params, sample_rate, duration)
        return w.plus, w.cross

    def generate_window(
        self, params: InjectionParameters, sample_rate: int, duration: float
    ) -> GeneratedWaveform:
        lal, lalsim = self._lalsim()
        approx = lalsim.GetApproximantFromString(self.approximant_name)
        hp, hc = lalsim.SimInspiralTD(
            params.mass1 * lal.MSUN_SI,
            params.mass2 * lal.MSUN_SI,
            0.0, 0.0, float(params.spin1z),
            0.0, 0.0, float(params.spin2z),
            self.distance_mpc * 1.0e6 * lal.PC_SI,
            float(params.inclination),
            float(params.phase),
            0.0,   # longAscNodes
            0.0,   # eccentricity
            0.0,   # meanPerAno
            1.0 / float(sample_rate),
            self.f_lower,
            self.f_ref,
            None,  # LALparams
            approx,
        )
        n = int(round(duration * sample_rate))
        plus, dropped = _centre(np.asarray(hp.data.data, dtype=float),
                               float(hp.epoch), sample_rate, n)
        cross, _ = _centre(np.asarray(hc.data.data, dtype=float),
                           float(hc.epoch), sample_rate, n)
        return GeneratedWaveform(plus, cross, int(sample_rate), dropped)

    def _lalsim(self):
        return _lal()


def _centre(data: np.ndarray, epoch: float, sample_rate: int,
            n: int) -> tuple[np.ndarray, float]:
    """Place a LAL time series in an `n`-sample window with coalescence at `n // 2`.

    LAL sets the epoch so that t = 0 is the coalescence, i.e. `epoch` is negative and
    `-epoch` is the length of inspiral before merger. A waveform longer than the window
    loses its earliest inspiral; the amount is returned rather than swallowed, because
    it is SNR that is present in the physical signal and absent from ours, and the
    rescaling that follows would otherwise hide it by compensating.
    """
    out = np.zeros(n, dtype=float)
    coalescence = int(round(-epoch * sample_rate))
    target = n // 2
    src_lo = max(0, coalescence - target)
    src_hi = min(len(data), coalescence - target + n)
    if src_hi <= src_lo:
        return out, 0.0
    dst_lo = src_lo - coalescence + target
    out[dst_lo:dst_lo + (src_hi - src_lo)] = data[src_lo:src_hi]
    return out, src_lo / float(sample_rate)


def optimal_snr(
    h: np.ndarray,
    sample_rate: int,
    psd: tuple[np.ndarray, np.ndarray],
    f_low: float = DEFAULT_F_LOWER,
    f_high: float | None = None,
) -> float:
    """rho = sqrt(4 int |h~(f)|^2 / S(f) df) over [f_low, f_high].

    `psd` is `(freqs, psd)` as `strain.load_reference_psd` returns it: the *run-averaged*
    reference, the same one the whitening uses. Using a locally-estimated PSD here and
    the reference for whitening would make the quoted SNR and the injected amplitude
    disagree by whatever the two curves differ by, which is exactly the discrepancy the
    upstream ASD-consistency veto exists to catch.
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    # rfft is a discrete sum; dividing by the sample rate converts it to the
    # continuous-time Fourier transform the SNR integral is defined against.
    spectrum = np.fft.rfft(h) / float(sample_rate)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    df = float(sample_rate) / n
    f, s = psd
    s_interp = np.interp(freqs, f, s)
    hi = freqs[-1] if f_high is None else float(f_high)
    band = (freqs >= float(f_low)) & (freqs <= hi) & (s_interp > 0)
    if not band.any():
        return 0.0
    return float(np.sqrt(4.0 * np.sum(np.abs(spectrum[band]) ** 2 / s_interp[band]) * df))


def snr_in_window_fraction(
    h: np.ndarray,
    sample_rate: int,
    psd: tuple[np.ndarray, np.ndarray],
    seconds: float,
    f_low: float = DEFAULT_F_LOWER,
) -> float:
    """Fraction of rho^2 carried by the central `seconds` of `h`.

    The tile is a Q-transform of a central crop, so a long inspiral deposits SNR the
    network never sees. This is the number that says how much.
    """
    n = len(h)
    keep = int(round(seconds * sample_rate))
    lo = max(0, (n - keep) // 2)
    cropped = np.zeros_like(np.asarray(h, dtype=float))
    cropped[lo:lo + keep] = np.asarray(h, dtype=float)[lo:lo + keep]
    total = optimal_snr(h, sample_rate, psd, f_low=f_low)
    if total <= 0:
        return 0.0
    return float((optimal_snr(cropped, sample_rate, psd, f_low=f_low) / total) ** 2)


class InjectionEngine:
    """Draws a waveform, projects it onto a detector, whitens it, scales it, adds it.

    Holds the reference PSDs and notch lists so that a whitened injection goes through
    byte-for-byte the same filter chain as the noise it is added to. Two different
    whitenings in one pipeline is the sort of bug that shows up only as a stage-2 model
    that works on injections and not on real events.
    """

    def __init__(
        self,
        backend: LALWaveformBackend,
        psds: dict[str, tuple[np.ndarray, np.ndarray]],
        notch_lines: dict[str, tuple[float, ...]],
        sample_rate: int,
        window_seconds: float,
        f_low: float = DEFAULT_F_LOWER,
        snr_convention: str = "network",
        reference_gps: float = REFERENCE_GPS,
    ):
        if snr_convention not in ("network", "detector"):
            raise ValueError(
                f"snr_convention must be 'network' or 'detector', got {snr_convention!r}"
            )
        self.backend = backend
        self.psds = dict(psds)
        self.notch_lines = {k: tuple(v) for k, v in notch_lines.items()}
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.f_low = float(f_low)
        self.snr_convention = snr_convention
        self.reference_gps = float(reference_gps)

    # -- pieces ---------------------------------------------------------------

    def project(self, wave: GeneratedWaveform, ifo: str, params: InjectionParameters,
                gps: float) -> np.ndarray:
        """F+ h+ + Fx hx, delayed by the geocentre offset and the drawn time shift."""
        fp, fc = antenna_response(ifo, params.ra, params.dec, params.psi, gps)
        h = fp * wave.plus + fc * wave.cross
        shift = params.time_shift + earth_centre_delay(ifo, params.ra, params.dec, gps)
        samples = shift * self.sample_rate
        # The circular shift is only safe while the waveform stays clear of the edges.
        if abs(samples) > 0.25 * len(h):
            raise ValueError(
                f"time shift {shift:.3f} s is too large for a {wave.duration:.1f} s "
                "window; the circular Fourier shift would wrap the waveform"
            )
        return _fractional_shift(h, samples)

    def whiten_like_noise(self, h: np.ndarray, ifo: str) -> np.ndarray:
        """The same whitening and notch/high-pass chain the noise provider applies."""
        from madgrav_ml.data.representation import notch, whiten

        w = whiten(h, self.sample_rate, reference_psd=self.psds[ifo])
        return notch(w, self.sample_rate, lines=self.notch_lines.get(ifo, ()))

    def scale_factor(self, wave: GeneratedWaveform, ifo: str,
                     params: InjectionParameters, gps: float) -> float:
        """Multiplier that puts the drawn SNR on the requested convention.

        Under `network` the sum runs over every detector we hold a reference PSD for,
        using the *same* source: that is what makes a tile drawn at rho = 8 mean the
        same thing as a network trigger at rho = 8.
        """
        if self.snr_convention == "detector":
            rho2 = optimal_snr(self.project(wave, ifo, params, gps),
                               self.sample_rate, self.psds[ifo], self.f_low) ** 2
        else:
            rho2 = 0.0
            for other in self.psds:
                rho2 += optimal_snr(self.project(wave, other, params, gps),
                                    self.sample_rate, self.psds[other],
                                    self.f_low) ** 2
        if rho2 <= 0.0:
            raise ValueError(
                "drawn source has zero SNR in every detector (an antenna-pattern null, "
                "or a waveform entirely outside the band) -- redraw the parameters"
            )
        return float(params.network_snr) / float(np.sqrt(rho2))

    # -- the whole path -------------------------------------------------------

    def whitened_signal(self, params: InjectionParameters, ifo: str,
                        gps: float | None = None) -> np.ndarray:
        """A whitened, filtered, SNR-scaled detector strain, ready to be added."""
        gps = self.reference_gps if gps is None else float(gps)
        wave = self.backend.generate_window(params, self.sample_rate, self.window_seconds)
        h = self.project(wave, ifo, params, gps)
        # Scaling commutes with whitening and with the linear filter chain, so the
        # factor is computed on the raw projection against the physical PSD -- which is
        # where "SNR" is defined -- and applied after.
        return self.scale_factor(wave, ifo, params, gps) * self.whiten_like_noise(h, ifo)

    def inject(self, strain: np.ndarray, params: InjectionParameters, ifo: str,
               gps: float | None = None) -> np.ndarray:
        """Add a drawn source to an already-whitened noise window."""
        h = self.whitened_signal(params, ifo, gps)
        strain = np.asarray(strain, dtype=float)
        if len(h) != len(strain):
            raise ValueError(
                f"waveform window is {len(h)} samples and the noise window is "
                f"{len(strain)}; window_seconds/sample_rate disagree between the "
                "injection engine and the noise provider"
            )
        return strain + h


def build_engine(cfg, sample_rate: int) -> InjectionEngine:
    """Assemble an `InjectionEngine` from the `data` config node.

    Reads the same `reference_psd` and `notch_lines` keys the noise provider reads, so
    the two cannot be pointed at different files by editing one of them.
    """
    from madgrav_ml.data.strain import load_reference_psd

    backend = LALWaveformBackend(
        approximant=cfg.get("approximant", DEFAULT_APPROXIMANT),
        f_lower=float(cfg.get("injection_f_lower", DEFAULT_F_LOWER)),
        distance_mpc=float(cfg.get("injection_distance_mpc", DEFAULT_DISTANCE_MPC)),
    )
    from madgrav_ml.data.representation import notch_lines_for

    psds = {ifo: load_reference_psd(p) for ifo, p in cfg.reference_psd.items()}
    config = str(cfg.get("line_configuration", "o1"))
    return InjectionEngine(
        backend=backend,
        psds=psds,
        notch_lines={ifo: notch_lines_for(ifo, config) for ifo in psds},
        sample_rate=sample_rate,
        window_seconds=float(cfg.window_seconds),
        f_low=float(cfg.get("injection_f_lower", DEFAULT_F_LOWER)),
        snr_convention=str(cfg.get("snr_convention", "network")),
    )
