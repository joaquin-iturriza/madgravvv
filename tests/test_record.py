"""The record refuses to serialise a claim it cannot support."""

import pytest

from madgrav_ml.report.record import ExperimentRecord, Verdict


def complete(**kw):
    base = dict(
        name="r",
        hypothesis="masked prediction separates better than autoencoding",
        change="model.objective=masked",
        parameters={"counts": {"cae": 251394}},
        folds={"eval_fold": 1},
        primary={"efficiency@1/yr": 0.51},
        seeds=[42, 43, 44],
        verdict=Verdict.KEEP,
        reasoning="+3% efficiency at FAR<=1/yr, 3 seeds",
    )
    base.update(kw)
    return ExperimentRecord(**base)


def test_complete_record_validates(tmp_path):
    complete().save(tmp_path)
    assert (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("field", ["parameters", "folds", "primary"])
def test_missing_load_bearing_field_is_rejected(field):
    with pytest.raises(ValueError):
        complete(**{field: {}}).validate()


def test_keep_verdict_needs_three_seeds():
    with pytest.raises(ValueError, match="seed"):
        complete(seeds=[42]).validate()


def test_discard_verdict_does_not_need_three_seeds():
    complete(seeds=[42], verdict=Verdict.DISCARD, reasoning="loss diverged").validate()


def test_unreasoned_verdict_is_rejected():
    with pytest.raises(ValueError, match="reasoning"):
        complete(reasoning="").validate()
