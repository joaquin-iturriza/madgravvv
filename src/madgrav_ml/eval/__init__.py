"""Evaluation harness: folds, background, FAR, efficiency, VT, calibration.

Phase 1 of the improvement plan, and blocking for everything else. No experiment may
quote a number that did not come through here.
"""

from .background import SlidePlan, coincident_livetime, make_slide_plan, run_slides
from .calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from .efficiency import Efficiency, efficiency_at_far, efficiency_vs
from .far import TrialsFactor, far_of, ifar, threshold_at_far
from .folds import FoldGuard, FoldLeakError, Phase, Segment, Split, gps_grouped_folds
from .vt import SensitiveVolume, sensitive_distance_mpc, sensitive_volume, vt_ratio

__all__ = [
    "Efficiency",
    "FoldGuard",
    "FoldLeakError",
    "Phase",
    "Segment",
    "SensitiveVolume",
    "SlidePlan",
    "Split",
    "TrialsFactor",
    "apply_temperature",
    "coincident_livetime",
    "efficiency_at_far",
    "efficiency_vs",
    "expected_calibration_error",
    "far_of",
    "fit_temperature",
    "gps_grouped_folds",
    "ifar",
    "make_slide_plan",
    "reliability_curve",
    "run_slides",
    "sensitive_distance_mpc",
    "sensitive_volume",
    "threshold_at_far",
    "vt_ratio",
]
