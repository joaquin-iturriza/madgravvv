"""False-alarm rate from time-slide background, and the trials factor.

FAR is the currency. Every threshold in this project is defined by the FAR it buys,
never by a fixed value of a ranking statistic, because a fixed statistic threshold
means something different for every model.

Upstream convention: FAR = trials * N(>= x) / T_bg, with a **trials factor of 4** —
two ranking statistics (ln-Lambda and net sigma) times two arms (HM and LM). Section 9.1
of the plan proposes collapsing the two arms into a single calibrated statistic, which
lowers the trials factor and therefore improves the quoted FAR *arithmetically*, with
every model held fixed. `TrialsFactor` exists so that improvement is auditable rather
than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class TrialsFactor:
    """The multiplicity a FAR must be penalised by, itemised rather than hardcoded.

    Upstream is `n_statistics=2, n_arms=2` -> 4. A single calibrated statistic
    replacing `max(HM, LM) >= 0.5` gives `n_arms=1` -> 2, which halves the FAR at
    fixed candidate set. Report the two side by side; do not silently drop the factor.
    """

    n_statistics: int = 2
    n_arms: int = 2
    note: str = ""

    @property
    def value(self) -> int:
        return int(self.n_statistics * self.n_arms)

    def as_dict(self) -> dict:
        return {
            "n_statistics": self.n_statistics,
            "n_arms": self.n_arms,
            "trials_factor": self.value,
            "note": self.note,
        }


def far_of(
    statistic: np.ndarray | float,
    background: np.ndarray,
    background_livetime_s: float,
    trials: TrialsFactor | int = 4,
) -> np.ndarray:
    """FAR in events/year for each value of `statistic`.

    `background` is the ranking statistic of every time-slide trigger that survived
    **the same selection the foreground went through**. That is not a detail: a FAR
    measured under a different gate than the one applied to the foreground is
    meaningless, so if you change the CNN gate you must re-score the slides.

    The count is inclusive (`>=`), and a statistic louder than every background event
    gets the one-count bound `trials / T_bg` rather than zero — never quote FAR = 0.
    """
    trials_value = trials.value if isinstance(trials, TrialsFactor) else int(trials)
    if background_livetime_s <= 0:
        raise ValueError("background livetime must be positive")
    bg = np.sort(np.asarray(background, dtype=float))
    x = np.atleast_1d(np.asarray(statistic, dtype=float))

    # number of background events with stat >= x
    n_louder = bg.size - np.searchsorted(bg, x, side="left")
    n_louder = np.maximum(n_louder, 1)  # one-count upper bound, not zero
    rate = trials_value * n_louder / (background_livetime_s / SECONDS_PER_YEAR)
    return rate if np.ndim(statistic) else float(rate[0])


def threshold_at_far(
    background: np.ndarray,
    background_livetime_s: float,
    far_target: float = 1.0,
    trials: TrialsFactor | int = 4,
) -> float:
    """Ranking-statistic threshold whose FAR is `far_target` events/year.

    Raises when the background is too short to resolve the requested FAR: with
    T_bg years of slides and a trials factor k, the smallest resolvable FAR is
    k / T_bg, and quoting anything below that is extrapolation, not measurement.
    """
    trials_value = trials.value if isinstance(trials, TrialsFactor) else int(trials)
    bg = np.sort(np.asarray(background, dtype=float))
    T_yr = background_livetime_s / SECONDS_PER_YEAR
    floor = trials_value / T_yr
    if far_target < floor:
        raise ValueError(
            f"FAR <= {far_target}/yr is below what {T_yr:.0f} yr of background can "
            f"resolve at trials factor {trials_value} (floor {floor:.3g}/yr). Generate "
            f"more slides, or quote the FAR you actually measured."
        )
    # allowed number of louder background events
    n_allowed = int(np.floor(far_target * T_yr / trials_value))
    n_allowed = max(n_allowed, 1)
    if n_allowed >= bg.size:
        return float(bg[0])
    return float(bg[bg.size - n_allowed])


def ifar(*args, **kwargs) -> np.ndarray:
    """Inverse FAR in years — the axis most GW sensitivity plots are drawn against."""
    return 1.0 / far_of(*args, **kwargs)
