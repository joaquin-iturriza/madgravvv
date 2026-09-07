"""On-the-fly training-data generation — Phase 2, the highest-leverage change.

The supervised arms currently train on a fixed ~10k glitch tiles and ~1k signal tiles:
an 11k dataset with a 10:1 imbalance, which the author's own slides note "can probably
be enlarged". Everything needed to make it effectively infinite already exists in the
ecosystem, so the work is plumbing rather than research, and the plan expects it to
outperform every architecture and activation change combined at zero parameter cost.

What is resampled every epoch: component masses, mass ratio, spins, network SNR, sky
position and polarisation (which set the antenna-pattern projection onto each
detector), and the coalescence-time offset within the tile.

Two modes, both required:
  * `seed=None`  — non-repeating stream; this is the point of the module.
  * `seed=<int>` — deterministic, so any single reported run is reproducible.

The waveform backend is behind `WaveformBackend` so IMRPhenomPv2 (the baseline) and
IMRPhenomXPHM (used in the upstream injection banks) are a config switch rather than a
rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol

import numpy as np

# Upstream training-injection population (improved_pipeline.py + the paper).
MASS_COMPONENT_RANGE = (10.0, 120.0)     # solar masses
MASS_TOTAL_RANGE = (20.0, 240.0)
MASS_RATIO_MAX = 6.0
NETWORK_SNR_RANGE = (8.0, 25.0)
COALESCENCE_SHIFT_RANGE = (-0.5, 0.5)    # seconds within the tile


@dataclass
class InjectionParameters:
    """One drawn source. Carried into the run record so a campaign is reconstructible.

    The in-plane spin components default to zero, which is the aligned-spin case the
    baseline population uses; a precessing draw fills them in. `distance_mpc` is None for
    an SNR-targeted draw (the amplitude is rescaled afterwards) and set for a volumetric
    one, where the distance IS the amplitude and no rescaling happens -- that distinction
    is what makes a sensitive-volume measurement possible.
    """

    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    network_snr: float | None
    ra: float
    dec: float
    psi: float
    inclination: float
    phase: float
    time_shift: float
    spin1x: float = 0.0
    spin1y: float = 0.0
    spin2x: float = 0.0
    spin2y: float = 0.0
    distance_mpc: float | None = None

    @property
    def total_mass(self) -> float:
        return self.mass1 + self.mass2

    @property
    def mass_ratio(self) -> float:
        return max(self.mass1, self.mass2) / min(self.mass1, self.mass2)

    def as_dict(self) -> dict:
        return asdict(self)


# Sine-Gaussian burst population, for the out-of-family test. Central frequencies span
# the band the search is sensitive in; the quality factor sets how many cycles the burst
# lasts, from a near-impulse to a long ringing tone.
BURST_F0_RANGE = (40.0, 400.0)
BURST_Q_RANGE = (3.0, 30.0)


@dataclass
class BurstParameters:
    """A sine-Gaussian burst. Deliberately NOT a compact binary.

    Carries the same sky, polarisation, time and SNR fields as `InjectionParameters` so
    the projection and SNR-rescaling machinery is shared verbatim — the only thing that
    differs is the waveform, which is the point. An anomaly search is supposed to be
    sensitive to things it was not tuned on, and every number this project has quoted so
    far was measured against the same IMRPhenomPv2 population the ranking statistic was
    fitted to.
    """

    frequency: float
    quality: float
    network_snr: float
    ra: float
    dec: float
    psi: float
    inclination: float
    phase: float
    time_shift: float

    def as_dict(self) -> dict:
        return asdict(self)


class BurstSampler:
    """Draws sine-Gaussian bursts. Sky and time distributions match `ParameterSampler`
    exactly, so a difference in measured efficiency is attributable to the waveform and
    not to where or when the sources were placed."""

    def __init__(
        self,
        seed: int | None = None,
        f0_range: tuple[float, float] = BURST_F0_RANGE,
        q_range: tuple[float, float] = BURST_Q_RANGE,
        snr_range: tuple[float, float] = NETWORK_SNR_RANGE,
        time_shift_range: tuple[float, float] = COALESCENCE_SHIFT_RANGE,
    ):
        self.rng = np.random.default_rng(seed)
        self.f0_range = f0_range
        self.q_range = q_range
        self.snr_range = snr_range
        self.time_shift_range = time_shift_range

    def draw(self, rng=None) -> BurstParameters:
        r = self.rng if rng is None else rng
        return BurstParameters(
            # log-uniform in frequency: the band spans a decade and a linear draw would
            # put most bursts above 200 Hz, where the detectors are least sensitive
            frequency=float(np.exp(r.uniform(*np.log(self.f0_range)))),
            quality=float(r.uniform(*self.q_range)),
            network_snr=float(r.uniform(*self.snr_range)),
            ra=float(r.uniform(0.0, 2.0 * np.pi)),
            dec=float(np.arcsin(r.uniform(-1.0, 1.0))),
            psi=float(r.uniform(0.0, np.pi)),
            inclination=float(np.arccos(r.uniform(-1.0, 1.0))),
            phase=float(r.uniform(0.0, 2.0 * np.pi)),
            time_shift=float(r.uniform(*self.time_shift_range)),
        )

    def draw_many(self, n: int, rng=None) -> list[BurstParameters]:
        return [self.draw(rng) for _ in range(n)]


class WaveformBackend(Protocol):
    """Anything that can turn parameters into an h+, hx pair at a sample rate."""

    def generate(
        self, params: InjectionParameters, sample_rate: int, duration: float
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class ParameterSampler:
    """Draws the injection population. Pure numpy, no waveform dependency.

    Rejection-samples the mass pair so that the component, total and ratio bounds all
    hold simultaneously — drawing the components independently and clipping would
    distort the population at the corners, which is where the sensitivity curve is
    least well measured and most argued about.
    """

    def __init__(
        self,
        seed: int | None = None,
        component_range: tuple[float, float] = MASS_COMPONENT_RANGE,
        total_range: tuple[float, float] = MASS_TOTAL_RANGE,
        q_max: float = MASS_RATIO_MAX,
        snr_range: tuple[float, float] = NETWORK_SNR_RANGE,
        spin_max: float = 0.99,
        time_shift_range: tuple[float, float] = COALESCENCE_SHIFT_RANGE,
        precessing: bool = False,
        distance_max_mpc: float | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        # Isotropic spin directions rather than aligned ones. A precessing binary still
        # chirps, so unlike the burst probe it stays inside the broad class the search
        # targets -- it is the "unusual compact binary" case, which is where a realistic
        # unmodelled source sits.
        self.precessing = precessing
        # When set, distance is drawn uniform in Euclidean volume and the waveform keeps
        # its physical amplitude instead of being rescaled to a target SNR. That is what
        # a sensitive volume needs: efficiency as a function of DISTANCE, integrated
        # against the volume element, rather than efficiency at a chosen SNR.
        self.distance_max_mpc = distance_max_mpc
        self.component_range = component_range
        self.total_range = total_range
        self.q_max = q_max
        self.snr_range = snr_range
        self.spin_max = spin_max
        self.time_shift_range = time_shift_range

    def _masses(self, rng=None) -> tuple[float, float]:
        rng = self.rng if rng is None else rng
        lo, hi = self.component_range
        for _ in range(10_000):
            m1, m2 = rng.uniform(lo, hi, size=2)
            m1, m2 = max(m1, m2), min(m1, m2)
            if not (self.total_range[0] <= m1 + m2 <= self.total_range[1]):
                continue
            if m1 / m2 > self.q_max:
                continue
            return float(m1), float(m2)
        raise RuntimeError(
            "mass rejection sampling failed; the component/total/ratio bounds "
            f"({self.component_range}, {self.total_range}, q<={self.q_max}) may be "
            "mutually unsatisfiable"
        )

    def _spins(self, r) -> dict:
        """Aligned by default; isotropic in direction when `precessing`."""
        if not self.precessing:
            return {"spin1z": float(r.uniform(-self.spin_max, self.spin_max)),
                    "spin2z": float(r.uniform(-self.spin_max, self.spin_max))}
        out = {}
        for i in (1, 2):
            # magnitude uniform in [0, spin_max], direction isotropic on the sphere
            a = float(r.uniform(0.0, self.spin_max))
            cos_t = float(r.uniform(-1.0, 1.0))
            phi = float(r.uniform(0.0, 2.0 * np.pi))
            sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t ** 2)))
            out[f"spin{i}x"] = a * sin_t * np.cos(phi)
            out[f"spin{i}y"] = a * sin_t * np.sin(phi)
            out[f"spin{i}z"] = a * cos_t
        return out

    def draw(self, rng=None) -> InjectionParameters:
        """Draw one source. Pass `rng` to use a caller-owned stream.

        Multi-worker generated datasets MUST pass their own rng. The sampler's internal
        stream is seeded once at construction, and a DataLoader forks the whole dataset
        into every worker, so eight workers sharing this sampler would draw eight
        identical injection sequences — a fifth of the "effectively infinite" training
        set Phase 2 is for, and invisible in any loss curve.
        """
        m1, m2 = self._masses(rng)
        r = self.rng if rng is None else rng
        volumetric = self.distance_max_mpc is not None
        return InjectionParameters(
            mass1=m1,
            mass2=m2,
            **self._spins(r),
            # p(d) ~ d^2 out to d_max, i.e. uniform in Euclidean volume. The cube root of
            # a uniform draw is the inverse CDF.
            distance_mpc=(float(self.distance_max_mpc * r.random() ** (1.0 / 3.0))
                          if volumetric else None),
            network_snr=None if volumetric else float(r.uniform(*self.snr_range)),
            ra=float(r.uniform(0.0, 2.0 * np.pi)),
            # isotropic on the sphere, not uniform in declination
            dec=float(np.arcsin(r.uniform(-1.0, 1.0))),
            psi=float(r.uniform(0.0, np.pi)),
            inclination=float(np.arccos(r.uniform(-1.0, 1.0))),
            phase=float(r.uniform(0.0, 2.0 * np.pi)),
            time_shift=float(r.uniform(*self.time_shift_range)),
        )

    def draw_many(self, n: int, rng=None) -> list[InjectionParameters]:
        return [self.draw(rng) for _ in range(n)]


def rescale_to_network_snr(
    strains: dict[str, np.ndarray], current_snr: float, target_snr: float
) -> dict[str, np.ndarray]:
    """Uniform rescaling of every detector's projected strain to a target network SNR.

    Rescaling after projection keeps the *relative* amplitudes between detectors,
    which is what the coherence statistic reads. Rescaling per detector would destroy
    it, and the failure would not show up in any single-detector metric.
    """
    if current_snr <= 0:
        raise ValueError("current network SNR must be positive")
    factor = target_snr / current_snr
    return {ifo: h * factor for ifo, h in strains.items()}
