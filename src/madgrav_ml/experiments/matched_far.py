"""The matched-FAR evaluation: the only path that may produce a quoted number.

Everything else in `experiments/` trains models. This module takes trained models,
pushes both the injection campaign and the time-slide background through the *same*
selection, and produces detection efficiency and sensitive volume at fixed FAR — plus
the single-detector variants that constraint C1 makes primary.

It is deliberately not a `BaseExperiment` subclass: it does not train, and it is the
one place allowed to open `FoldGuard.final_report()`. Keeping it separate makes that
privilege easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from madgrav_ml.eval.background import SlidePlan, run_slides
from madgrav_ml.eval.efficiency import efficiency_at_far, efficiency_vs
from madgrav_ml.eval.far import TrialsFactor
from madgrav_ml.eval.folds import FoldGuard, Split
from madgrav_ml.eval.vt import sensitive_volume
from madgrav_ml.logger import LOGGER


@dataclass
class Selection:
    """The complete foreground selection, as one callable.

    `score(strain_pair) -> ranking statistic` must include every stage the foreground
    goes through: the per-detector anomaly score, the CNN gate, clustering, and any
    veto. The background is scored with this same object, which is the mechanical
    guarantee that the FAR and the efficiency were measured under one selection.

    `trials` itemises the multiplicity. Changing the arm structure changes it, and
    that change is itself a reportable FAR improvement (plan section 9.1).
    """

    score: Callable
    trials: TrialsFactor
    name: str = ""

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)


def run(
    guard: FoldGuard,
    selection: Selection,
    slide_plan: SlidePlan,
    background_scorer: Callable[[float], list[float]],
    injection_statistic: np.ndarray,
    injection_parameters: dict[str, np.ndarray] | None = None,
    v_generated_gpc3: float | None = None,
    far_targets: tuple[float, ...] = (1.0, 0.1),
    label: str = "",
) -> dict:
    """Produce the primary metrics for one configuration.

    Opens `guard.final_report()`, which the guard permits once per process. If this
    raises, the run has already read the evaluation fold and the numbers it would
    produce are selection-contaminated — that is the guard working, not a bug.
    """
    with guard.final_report(label or selection.name or "matched-far"):
        eval_segments = guard.segments(Split.EVAL)
    LOGGER.info(
        f"Evaluation fold: {len(eval_segments)} segments, "
        f"{sum(s.duration for s in eval_segments) / 86400:.1f} d livetime"
    )

    background, plan = run_slides(slide_plan, background_scorer)
    LOGGER.info(
        f"Background: {background.size:,} triggers over "
        f"{plan.background_livetime_yr:.0f} yr of slides"
    )

    out: dict = {
        "selection": selection.name,
        "trials": selection.trials.as_dict(),
        "background": plan.as_dict(),
        "n_background_triggers": int(background.size),
        "efficiency": {},
        "vt": {},
    }

    for far in far_targets:
        eff = efficiency_at_far(
            injection_statistic,
            background,
            plan.background_livetime_s,
            far_target=far,
            trials=selection.trials,
            label=f"{label}@{far}/yr",
        )
        LOGGER.info(str(eff))
        out["efficiency"][str(far)] = eff.as_dict()

        if v_generated_gpc3 is not None:
            vt = sensitive_volume(
                injection_statistic,
                background,
                plan.background_livetime_s,
                v_generated_gpc3=v_generated_gpc3,
                far_target=far,
                trials=selection.trials,
                label=f"{label}@{far}/yr",
            )
            LOGGER.info(str(vt))
            out["vt"][str(far)] = vt.as_dict()

    if injection_parameters and "total_mass" in injection_parameters:
        bins = np.array([20.0, 40.0, 60.0, 100.0, 160.0, 240.0])
        curve = efficiency_vs(
            injection_parameters["total_mass"],
            injection_statistic,
            background,
            plan.background_livetime_s,
            bins=bins,
            far_target=far_targets[0],
            trials=selection.trials,
            label="M_total",
        )
        out["efficiency_vs_total_mass"] = [e.as_dict() for e in curve]

    return out
