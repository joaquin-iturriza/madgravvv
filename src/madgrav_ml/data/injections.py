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
    """One drawn source. Carried into the run record so a campaign is reconstructible."""

    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    network_snr: float
    ra: float
    dec: float
    psi: float
    inclination: float
    phase: float
    time_shift: float

    @property
    def total_mass(self) -> float:
        return self.mass1 + self.mass2

    @property
    def mass_ratio(self) -> float:
        return max(self.mass1, self.mass2) / min(self.mass1, self.mass2)

    def as_dict(self) -> dict:
        return asdict(self)


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
    ):
        self.rng = np.random.default_rng(seed)
        self.component_range = component_range
        self.total_range = total_range
        self.q_max = q_max
        self.snr_range = snr_range
        self.spin_max = spin_max
        self.time_shift_range = time_shift_range

    def _masses(self) -> tuple[float, float]:
        lo, hi = self.component_range
        for _ in range(10_000):
            m1, m2 = self.rng.uniform(lo, hi, size=2)
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

    def draw(self) -> InjectionParameters:
        m1, m2 = self._masses()
        return InjectionParameters(
            mass1=m1,
            mass2=m2,
            spin1z=float(self.rng.uniform(-self.spin_max, self.spin_max)),
            spin2z=float(self.rng.uniform(-self.spin_max, self.spin_max)),
            network_snr=float(self.rng.uniform(*self.snr_range)),
            ra=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            # isotropic on the sphere, not uniform in declination
            dec=float(np.arcsin(self.rng.uniform(-1.0, 1.0))),
            psi=float(self.rng.uniform(0.0, np.pi)),
            inclination=float(np.arccos(self.rng.uniform(-1.0, 1.0))),
            phase=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            time_shift=float(self.rng.uniform(*self.time_shift_range)),
        )

    def draw_many(self, n: int) -> list[InjectionParameters]:
        return [self.draw() for _ in range(n)]


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
