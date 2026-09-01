"""Sensitive volume-time — the reach metric a GW search is judged on.

VT is estimated by Monte Carlo over an injection campaign: inject a population over a
generated volume V_gen, push every injection through the *whole* pipeline including
the selection the foreground uses, and count what survives the FAR threshold.

    V_sens = V_gen * sum_i(w_i * found_i) / sum_i(w_i)
    VT     = V_sens * T_analysed

The upstream repo already computes these (`lr_cascade/vt_absolute.py`,
`vt_vs_mass.py`, `vt_vs_far_panels.py`); reuse those where they fit and treat this
module as the harness-side interface, not a competing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .far import TrialsFactor, threshold_at_far


@dataclass(frozen=True)
class SensitiveVolume:
    far_target: float
    threshold: float
    v_sens_gpc3: float
    v_sens_err_gpc3: float
    horizon_mpc: float
    vt_gpc3_yr: float | None
    n_injections: int
    n_found: int
    label: str = ""

    def __str__(self) -> str:
        tag = f"{self.label}: " if self.label else ""
        vt = f", VT = {self.vt_gpc3_yr:.4g} Gpc^3 yr" if self.vt_gpc3_yr is not None else ""
        return (
            f"{tag}V_sens = {self.v_sens_gpc3:.4g} +/- {self.v_sens_err_gpc3:.2g} Gpc^3 "
            f"(d_sens = {self.horizon_mpc:.0f} Mpc){vt} at FAR <= {self.far_target}/yr"
        )

    def as_dict(self) -> dict:
        return asdict(self)


def sensitive_distance_mpc(v_sens_gpc3: float) -> float:
    """Radius of the Euclidean sphere with the same volume, in Mpc.

    A single number that is easier to compare across searches than a volume, and the
    form the upstream `vt_absolute.py` reports.
    """
    v_mpc3 = v_sens_gpc3 * 1e9
    return float((3.0 * v_mpc3 / (4.0 * np.pi)) ** (1.0 / 3.0))


def sensitive_volume(
    injection_statistic: np.ndarray,
    background: np.ndarray,
    background_livetime_s: float,
    v_generated_gpc3: float,
    analysis_time_yr: float | None = None,
    far_target: float = 1.0,
    trials: TrialsFactor | int = 4,
    weights: np.ndarray | None = None,
    label: str = "",
) -> SensitiveVolume:
    """Monte-Carlo V_sens (and VT, when `analysis_time_yr` is given).

    The uncertainty is the Monte-Carlo error on the recovered fraction only. It does
    not include the systematic from the injection population itself, which is usually
    the larger term when comparing across searches — quote V_sens *ratios* between
    two of our own configurations rather than absolutes wherever possible, since the
    population cancels there.
    """
    thr = threshold_at_far(background, background_livetime_s, far_target, trials)
    stat = np.asarray(injection_statistic, dtype=float)
    found = (stat >= thr).astype(float)
    n = stat.size
    if n == 0:
        raise ValueError("empty injection campaign")

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != stat.shape:
            raise ValueError(f"weights {w.shape} do not match statistics {stat.shape}")

    wsum = w.sum()
    frac = float((w * found).sum() / wsum)
    # variance of a weighted mean of a 0/1 variable
    var = float((w**2 * (found - frac) ** 2).sum()) / wsum**2
    v_sens = v_generated_gpc3 * frac
    v_err = v_generated_gpc3 * float(np.sqrt(max(var, 0.0)))

    return SensitiveVolume(
        far_target=float(far_target),
        threshold=float(thr),
        v_sens_gpc3=v_sens,
        v_sens_err_gpc3=v_err,
        horizon_mpc=sensitive_distance_mpc(v_sens) if v_sens > 0 else 0.0,
        vt_gpc3_yr=(v_sens * analysis_time_yr if analysis_time_yr is not None else None),
        n_injections=int(n),
        n_found=int(found.sum()),
        label=label,
    )


def vt_ratio(candidate: SensitiveVolume, baseline: SensitiveVolume) -> tuple[float, float]:
    """(ratio, 1-sigma error) of two VT measurements on the *same* injection campaign.

    This is the number to quote for a proposed change. On a shared campaign the
    population systematic cancels, so the ratio is far better determined than either
    absolute — and it is the form the question is actually asked in ("how much more
    volume does this buy?").
    """
    if baseline.v_sens_gpc3 <= 0:
        raise ValueError("baseline sensitive volume is zero")
    r = candidate.v_sens_gpc3 / baseline.v_sens_gpc3
    rel = np.hypot(
        candidate.v_sens_err_gpc3 / candidate.v_sens_gpc3 if candidate.v_sens_gpc3 else 0.0,
        baseline.v_sens_err_gpc3 / baseline.v_sens_gpc3,
    )
    return float(r), float(r * rel)
