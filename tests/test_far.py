"""FAR arithmetic, thresholds, efficiency and VT."""

import numpy as np
import pytest

from madgrav_ml.eval.efficiency import efficiency_at_far, wilson_interval
from madgrav_ml.eval.far import SECONDS_PER_YEAR, TrialsFactor, far_of, threshold_at_far
from madgrav_ml.eval.vt import sensitive_distance_mpc, sensitive_volume, vt_ratio

YEAR = SECONDS_PER_YEAR


def test_trials_factor_is_itemised():
    assert TrialsFactor().value == 4                       # upstream: 2 stats x 2 arms
    assert TrialsFactor(n_statistics=2, n_arms=1).value == 2  # single calibrated statistic


def test_far_counts_inclusively_and_scales_with_trials():
    bg = np.arange(100.0)
    a = far_of(50.0, bg, 10 * YEAR, trials=1)
    b = far_of(50.0, bg, 10 * YEAR, trials=2)
    assert b == pytest.approx(2 * a)
    assert a == pytest.approx(50 / 10)                     # 50 louder over 10 yr


def test_far_never_returns_zero():
    """A statistic louder than every background trigger gets the one-count bound."""
    bg = np.arange(100.0)
    assert far_of(1e6, bg, 10 * YEAR, trials=4) == pytest.approx(4 / 10)


def test_threshold_refuses_to_extrapolate_below_the_resolvable_far():
    bg = np.random.default_rng(0).normal(size=1000)
    with pytest.raises(ValueError, match="below what"):
        threshold_at_far(bg, 2 * YEAR, far_target=0.01, trials=4)


def test_threshold_round_trips_through_far():
    rng = np.random.default_rng(1)
    bg = rng.normal(size=20000)
    T = 100 * YEAR
    thr = threshold_at_far(bg, T, far_target=1.0, trials=4)
    assert far_of(thr, bg, T, trials=4) <= 1.0 + 1e-9


def test_efficiency_at_far_and_its_interval():
    rng = np.random.default_rng(2)
    bg = rng.normal(size=50000)
    inj = rng.normal(loc=6.0, size=1000)
    eff = efficiency_at_far(inj, bg, 100 * YEAR, far_target=1.0, trials=4)
    assert 0.9 < eff.efficiency <= 1.0
    assert eff.lo <= eff.efficiency <= eff.hi


def test_wilson_interval_is_not_degenerate_at_the_boundary():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and hi > 0.0                          # normal approx would give 0,0


def test_sensitive_volume_and_distance():
    rng = np.random.default_rng(3)
    bg = rng.normal(size=50000)
    inj = rng.normal(loc=6.0, size=2000)
    v = sensitive_volume(inj, bg, 100 * YEAR, v_generated_gpc3=10.0,
                         analysis_time_yr=0.5, far_target=1.0, trials=4)
    assert 0 < v.v_sens_gpc3 <= 10.0
    assert v.vt_gpc3_yr == pytest.approx(0.5 * v.v_sens_gpc3)
    assert v.horizon_mpc == pytest.approx(sensitive_distance_mpc(v.v_sens_gpc3))


def test_vt_ratio_on_a_shared_campaign():
    rng = np.random.default_rng(4)
    bg = rng.normal(size=50000)
    inj = rng.normal(loc=5.0, size=2000)
    base = sensitive_volume(inj, bg, 100 * YEAR, 10.0, far_target=1.0, trials=4)
    cand = sensitive_volume(inj + 0.5, bg, 100 * YEAR, 10.0, far_target=1.0, trials=4)
    r, err = vt_ratio(cand, base)
    assert r > 1.0 and err > 0.0
