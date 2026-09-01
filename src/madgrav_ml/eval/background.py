"""Time-slide background estimation.

The essential property: pair H1 data at time t with L1 data at time t + delta, for
delta far larger than the 11 ms light-travel time between the detectors. That destroys
any astrophysical coincidence while preserving the noise statistics, glitches included.
Each independent lag over coincident livetime T buys another T of background, which is
how the upstream search accumulates ~3540 years from a few months of data.

Two properties this module exists to enforce:

1. **The background is scored by the same selection as the foreground.** If you change
   the CNN gate, the slides must be re-scored by your gate. A FAR measured under a
   different selection than the one applied to the foreground is meaningless. So the
   API takes a *callable* selection, not a precomputed trigger list.
2. **The background comes from the fold you are allowed to use.** Slides for tuning
   come from the training fold; slides for the quoted FAR come from the evaluation
   fold, read once. `FoldGuard` is threaded through rather than bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from .folds import Segment

# Light travel time between LIGO Hanford and Livingston, ~3000 km apart.
LIGHT_TRAVEL_H1L1_S = 0.010002567
# Lags must clear the physical window by a comfortable margin, not just exceed it:
# a lag of 12 ms would still let a real signal's tails overlap.
MIN_LAG_S = 1.0


@dataclass(frozen=True)
class SlidePlan:
    """A reproducible set of time slides and the background livetime they buy."""

    lags_s: tuple[float, ...]
    coincident_livetime_s: float
    zero_lag_included: bool

    @property
    def background_livetime_s(self) -> float:
        """Total background livetime. Zero lag is foreground and never counts."""
        n = len(self.lags_s) - (1 if self.zero_lag_included else 0)
        return n * self.coincident_livetime_s

    @property
    def background_livetime_yr(self) -> float:
        return self.background_livetime_s / (365.25 * 86400.0)

    def as_dict(self) -> dict:
        return {
            "n_lags": len(self.lags_s),
            "zero_lag_included": self.zero_lag_included,
            "min_abs_lag_s": min(abs(l) for l in self.lags_s if l != 0.0),
            "coincident_livetime_s": self.coincident_livetime_s,
            "background_livetime_yr": self.background_livetime_yr,
        }


def coincident_livetime(h1: Sequence[Segment], l1: Sequence[Segment]) -> float:
    """Seconds during which both detectors are analysable.

    Note what this number implies for constraint C1: each detector's duty cycle is
    ~70-80%, so both-up is only ~50-60% of calendar time. That is exactly why the
    single-detector arm matters and why coherence must not move upstream of it.
    """
    a = sorted((s.start, s.end) for s in h1)
    b = sorted((s.start, s.end) for s in l1)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def make_slide_plan(
    coincident_livetime_s: float,
    n_lags: int,
    lag_step_s: float = 4.0,
    include_zero_lag: bool = False,
) -> SlidePlan:
    """Symmetric lag ladder: +-step, +-2*step, ... clear of the light-travel window.

    `lag_step_s` defaults to 4 s, the upstream clustering window, so that consecutive
    lags cannot produce the same trigger twice.
    """
    if lag_step_s < MIN_LAG_S:
        raise ValueError(
            f"lag step {lag_step_s}s is inside the physical window; a slide must clear "
            f"the {LIGHT_TRAVEL_H1L1_S * 1e3:.1f} ms H1-L1 light travel time by a wide "
            f"margin (>= {MIN_LAG_S}s) or it does not destroy astrophysical coincidence"
        )
    lags: list[float] = [0.0] if include_zero_lag else []
    k = 1
    while len(lags) < n_lags + (1 if include_zero_lag else 0):
        lags.append(+k * lag_step_s)
        if len(lags) < n_lags + (1 if include_zero_lag else 0):
            lags.append(-k * lag_step_s)
        k += 1
    return SlidePlan(
        lags_s=tuple(lags),
        coincident_livetime_s=coincident_livetime_s,
        zero_lag_included=include_zero_lag,
    )


def run_slides(
    plan: SlidePlan,
    score: Callable[[float], Iterable[float]],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, SlidePlan]:
    """Collect the background ranking statistic over every lag in `plan`.

    `score(lag) -> iterable of ranking-statistic values` must apply the **complete**
    selection under test — gate, clustering, vetoes — exactly as the foreground path
    does. Passing a function that skips a stage is the single most common way a FAR
    gets quoted too optimistically, and nothing downstream can detect it.

    Returns the concatenated background statistics and the plan they came from, so a
    caller cannot pair a statistic array with the wrong livetime.
    """
    out: list[np.ndarray] = []
    lags = [l for l in plan.lags_s if l != 0.0 or plan.zero_lag_included]
    for i, lag in enumerate(lags):
        vals = np.asarray(list(score(lag)), dtype=float)
        out.append(vals)
        if progress is not None:
            progress(i + 1, len(lags))
    stats = np.concatenate(out) if out else np.array([], dtype=float)
    return stats, plan
