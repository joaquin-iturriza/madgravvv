"""Time slides: lags must actually destroy astrophysical coincidence."""

import pytest

from madgrav_ml.eval.background import (
    LIGHT_TRAVEL_H1L1_S,
    coincident_livetime,
    make_slide_plan,
)
from madgrav_ml.eval.folds import Segment


def test_lag_inside_the_physical_window_is_refused():
    with pytest.raises(ValueError, match="light travel"):
        make_slide_plan(1000.0, n_lags=10, lag_step_s=LIGHT_TRAVEL_H1L1_S * 1.2)


def test_background_livetime_excludes_zero_lag():
    p = make_slide_plan(1000.0, n_lags=10, lag_step_s=4.0, include_zero_lag=True)
    assert p.background_livetime_s == pytest.approx(10 * 1000.0)


def test_lags_are_symmetric_and_clear_of_zero():
    p = make_slide_plan(1000.0, n_lags=6, lag_step_s=4.0)
    assert 0.0 not in p.lags_s
    assert sorted(p.lags_s) == sorted(-l for l in p.lags_s)


def test_coincident_livetime_is_the_overlap():
    h1 = [Segment("H1", 0, 100), Segment("H1", 200, 300)]
    l1 = [Segment("L1", 50, 250)]
    assert coincident_livetime(h1, l1) == pytest.approx(50 + 50)
