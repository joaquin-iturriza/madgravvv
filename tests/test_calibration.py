"""Post-hoc calibration: temperature scaling must move ECE and not ranking."""

import numpy as np

from madgrav_ml.eval.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)


def _overconfident(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    # logits scaled up by 3: correct ordering, badly overconfident probabilities
    return 3.0 * (rng.normal(loc=np.where(y > 0, 1.0, -1.0), scale=1.0)), y


def test_temperature_scaling_reduces_ece():
    z, y = _overconfident()
    before = expected_calibration_error(1 / (1 + np.exp(-z)), y)
    t = fit_temperature(z, y)
    after = expected_calibration_error(apply_temperature(z, t), y)
    assert after < before


def test_temperature_scaling_preserves_ranking():
    z, y = _overconfident()
    t = fit_temperature(z, y)
    p = apply_temperature(z, t)
    assert np.array_equal(np.argsort(z), np.argsort(p))


def test_reliability_curve_shapes():
    z, y = _overconfident()
    pred, obs, cnt = reliability_curve(1 / (1 + np.exp(-z)), y, n_bins=10)
    assert pred.shape == obs.shape == cnt.shape == (10,)
    assert cnt.sum() == len(y)
