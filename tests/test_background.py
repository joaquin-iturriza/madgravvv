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


def test_slide_ladder_wraps_onto_itself():
    """The number of DISTINCT pairings is n/s - 1, not twice that.

    A slide pairs H1 at grid point i with L1 at (i - k) mod n, and the ladder runs
    +s, -s, +2s, -2s, ... So once |k| passes n/2 the positive arm starts reproducing the
    negative one: with n = 14396 and s = 4, lag +14392 IS lag -4. Overshooting the cap
    changes no rate -- the trigger count and the livetime double together -- but it
    doubles the background livetime and the survivor count that get reported, which is
    exactly the wrong thing to inflate in a deep tail counted in single digits.

    This was found by noticing that every surviving background trigger was printed twice.
    """
    n, shift = 14396, 4
    ladder = []
    for j in range(1, n // shift):
        ladder += [j * shift, -j * shift]
    distinct = {k % n for k in ladder}
    assert len(distinct) == n // shift - 1
    assert len(distinct) < len(ladder) / 1.9, "the ladder should be ~2x redundant"
    assert (+14392) % n == (-4) % n


def test_slide_livetime_is_lags_times_coincident_time():
    from madgrav_ml.eval.background import make_slide_plan

    plan = make_slide_plan(100_000.0, 3598, lag_step_s=4.0)
    assert plan.background_livetime_s == 3598 * 100_000.0
    assert not plan.zero_lag_included
