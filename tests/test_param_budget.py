"""C2: the parameter budget is enforced on measured numbers."""

import pytest

from madgrav_ml.models.param_budget import check_budget, load_reference


def test_within_tolerance_passes():
    v = check_budget("glitch_arm", 106_000, 105_953, tolerance=0.10)
    assert v.passed and 0.99 < v.ratio < 1.01


def test_out_of_budget_raises_by_default():
    with pytest.raises(ValueError, match="C2 forbids"):
        check_budget("glitch_arm", 300_000, 105_953)


def test_out_of_budget_can_be_measured_without_raising():
    v = check_budget("glitch_arm", 300_000, 105_953, strict=False)
    assert not v.passed and v.ratio > 2.8


def test_reference_file_holds_the_measured_counts():
    ref = load_reference("config/param_budget.yaml")
    assert ref["cae"] == 251_394
    assert ref["cae_conv"] + ref["cae_classifier"] == ref["cae"]
    assert ref["specialist_hm"] - ref["glitch_arm"] == 144   # one extra input channel
