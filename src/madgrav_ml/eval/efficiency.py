"""Detection efficiency at fixed FAR — the headline metric.

The plan is blunt about this: do not report AUC or ROC on injections as a headline
result. It will not persuade anyone in gravitational-wave physics and it should not.
What counts is the fraction of an injection campaign recovered above the threshold
that a fixed false-alarm rate buys, and the sensitive volume that implies.

Because of constraint C1 (single-detector front end), every metric here has a
single-detector variant: per-detector efficiency at a fixed *single-detector*
false-alarm rate, reported separately from the network number.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .far import TrialsFactor, threshold_at_far


@dataclass(frozen=True)
class Efficiency:
    """Efficiency with a binomial interval, at a stated FAR and threshold."""

    far_target: float
    threshold: float
    n_injections: int
    n_found: int
    efficiency: float
    lo: float
    hi: float
    label: str = ""

    def __str__(self) -> str:
        tag = f"{self.label}: " if self.label else ""
        return (
            f"{tag}eff = {self.efficiency:.3f} [{self.lo:.3f}, {self.hi:.3f}] "
            f"({self.n_found}/{self.n_injections}) at FAR <= {self.far_target}/yr "
            f"(threshold {self.threshold:.3f})"
        )

    def as_dict(self) -> dict:
        return asdict(self)


def wilson_interval(k: int, n: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson score interval for a binomial fraction.

    Wilson rather than the normal approximation because efficiency is routinely
    measured near 0 or 1, where the normal interval runs outside [0,1] and gives a
    zero-width interval at k=0 — which then reads as a measurement with no
    uncertainty. Default z=1 (~68%), matching the usual GW convention.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def efficiency_at_far(
    injection_statistic: np.ndarray,
    background: np.ndarray,
    background_livetime_s: float,
    far_target: float = 1.0,
    trials: TrialsFactor | int = 4,
    weights: np.ndarray | None = None,
    label: str = "",
) -> Efficiency:
    """Fraction of the injection campaign recovered at FAR <= `far_target`.

    `background` must be the time-slide statistic **after the same selection the
    injections went through**. Re-score the slides whenever the gate changes.

    `weights` carries the injection campaign's importance weights when the population
    was drawn from a different distribution than the astrophysical one you want to
    quote against; the point estimate becomes the weighted mean, and the count-based
    interval is then indicative rather than exact.
    """
    thr = threshold_at_far(background, background_livetime_s, far_target, trials)
    stat = np.asarray(injection_statistic, dtype=float)
    found = stat >= thr
    n = stat.size
    if weights is None:
        k = int(found.sum())
        eff = k / n if n else 0.0
        lo, hi = wilson_interval(k, n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != stat.shape:
            raise ValueError(f"weights {w.shape} do not match statistics {stat.shape}")
        eff = float((w * found).sum() / w.sum())
        k = int(found.sum())
        # effective sample size for the interval (Kish), so importance weighting does
        # not silently claim more precision than the campaign supports
        n_eff = int(round(w.sum() ** 2 / (w**2).sum()))
        lo, hi = wilson_interval(int(round(eff * n_eff)), n_eff)
    return Efficiency(
        far_target=float(far_target),
        threshold=float(thr),
        n_injections=int(n),
        n_found=k,
        efficiency=float(eff),
        lo=float(lo),
        hi=float(hi),
        label=label,
    )


def efficiency_vs(
    parameter: np.ndarray,
    injection_statistic: np.ndarray,
    background: np.ndarray,
    background_livetime_s: float,
    bins: np.ndarray,
    far_target: float = 1.0,
    trials: TrialsFactor | int = 4,
    label: str = "",
) -> list[Efficiency]:
    """Efficiency in bins of an injection parameter (total mass, SNR, distance, ...).

    The threshold is computed once from the background and applied to every bin: it is
    a property of the search, not of the injection population. Per-bin thresholds
    would make the curve incomparable across bins.
    """
    thr = threshold_at_far(background, background_livetime_s, far_target, trials)
    par = np.asarray(parameter, dtype=float)
    stat = np.asarray(injection_statistic, dtype=float)
    idx = np.digitize(par, bins) - 1
    out: list[Efficiency] = []
    for b in range(len(bins) - 1):
        sel = idx == b
        n = int(sel.sum())
        k = int((stat[sel] >= thr).sum())
        lo, hi = wilson_interval(k, n)
        out.append(
            Efficiency(
                far_target=float(far_target),
                threshold=float(thr),
                n_injections=n,
                n_found=k,
                efficiency=(k / n if n else float("nan")),
                lo=lo,
                hi=hi,
                label=f"{label}[{bins[b]:g},{bins[b + 1]:g})",
            )
        )
    return out
