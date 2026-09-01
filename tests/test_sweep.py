"""The sweep: a trial cannot reach the evaluation fold, and every trial is logged.

This is the mechanical half of the claim the plan calls the strongest available signal
that a contributor understands the project — that hyperparameter tuning never saw the
evaluation background. It is worth testing that the claim is enforced rather than
documented.
"""

import json

import pytest

from madgrav_ml.eval.folds import FoldGuard, FoldLeakError, Segment, Split
from madgrav_ml.sweep import SEARCH_SPACES, RandomSampler, SweepRunner
from madgrav_ml.sweep.search_space import Parameter, SearchSpace


def segments(n=24, start=1_238_000_000.0, dur=4096.0, gap=512.0):
    return [Segment("H1", start + i * (dur + gap), start + i * (dur + gap) + dur)
            for i in range(n)]


def guard(tmp_path):
    return FoldGuard.from_segments(segments(), audit_path=tmp_path / "fold_audit.jsonl")


SPACE = SearchSpace("probe", (
    Parameter("a", 1e-4, 1e-1, log=True),
    Parameter("b", 0.0, 1.0),
))


def test_log_parameter_spans_its_decades():
    p = Parameter("lr", 1e-5, 1e-1, log=True)
    assert p.from_unit(0.0) == pytest.approx(1e-5)
    assert p.from_unit(1.0) == pytest.approx(1e-1)
    # the midpoint of a log parameter is the geometric mean, not the arithmetic one
    assert p.from_unit(0.5) == pytest.approx(1e-3)


def test_integer_parameter_rounds():
    p = Parameter("iterations", 1000, 50000, log=True, integer=True)
    v = p.from_unit(0.5)
    assert isinstance(v, int) and 1000 <= v <= 50000


def test_bad_bounds_are_rejected():
    with pytest.raises(ValueError):
        Parameter("x", 1.0, 1.0)
    with pytest.raises(ValueError):
        Parameter("x", 0.0, 1.0, log=True)


def test_trial_can_read_the_hpo_subsets(tmp_path):
    g = guard(tmp_path)
    seen = []

    def objective(params, gd):
        seen.append((len(gd.segments(Split.HPO_TRAIN)), len(gd.segments(Split.HPO_VAL))))
        return params["b"]

    r = SweepRunner(SPACE, g, tmp_path / "sweep")
    r.run(objective, n_trials=3)
    assert len(seen) == 3
    assert all(tr > 0 and va > 0 for tr, va in seen)


def test_a_trial_reaching_the_eval_fold_fails_and_is_recorded(tmp_path):
    g = guard(tmp_path)

    def leaky(params, gd):
        return len(gd.segments(Split.EVAL))      # C4 violation

    r = SweepRunner(SPACE, g, tmp_path / "sweep")
    with pytest.raises(RuntimeError, match="no trial completed"):
        r.run(leaky, n_trials=2)
    logged = [json.loads(l) for l in (tmp_path / "sweep" / "trials.jsonl").read_text().splitlines()]
    assert len(logged) == 2
    assert all(t["state"] == "failed" for t in logged)
    assert all("FoldLeakError" in t["error"] for t in logged)


def test_a_trial_cannot_read_the_whole_training_fold(tmp_path):
    """HPO_VAL exists to be the scoring set; reading all of TRAIN makes it decorative."""
    g = guard(tmp_path)

    def greedy(params, gd):
        return len(gd.segments(Split.TRAIN))

    r = SweepRunner(SPACE, g, tmp_path / "sweep")
    with pytest.raises(RuntimeError):
        r.run(greedy, n_trials=1)


def test_every_trial_is_logged_with_its_fold(tmp_path):
    g = guard(tmp_path)
    r = SweepRunner(SPACE, g, tmp_path / "sweep", label="probe-sweep")
    r.run(lambda p, gd: p["b"], n_trials=4)
    logged = [json.loads(l) for l in (tmp_path / "sweep" / "trials.jsonl").read_text().splitlines()]
    assert len(logged) == 4
    assert [t["index"] for t in logged] == [0, 1, 2, 3]
    assert all(t["fold"]["eval_fold"] == 1 for t in logged)
    assert all(t["fold"]["eval_fold_reads"] == 0 for t in logged)


def test_fold_audit_tags_each_access_with_its_trial(tmp_path):
    g = guard(tmp_path)
    r = SweepRunner(SPACE, g, tmp_path / "sweep", label="probe-sweep")
    r.run(lambda p, gd: len(gd.segments(Split.HPO_VAL)), n_trials=3)
    audit = [json.loads(l) for l in (tmp_path / "fold_audit.jsonl").read_text().splitlines()]
    assert {a["trial"] for a in audit} == {0, 1, 2}
    assert all(a["phase"] == "hpo" for a in audit)
    assert all(a["split"] == "hpo_val" for a in audit)


def test_best_is_the_minimum(tmp_path):
    g = guard(tmp_path)
    r = SweepRunner(SPACE, g, tmp_path / "sweep")
    best = r.run(lambda p, gd: p["b"], n_trials=6)
    assert best.objective == min(t.objective for t in r.trials if t.state == "done")


def test_summary_carries_the_search_space(tmp_path):
    g = guard(tmp_path)
    r = SweepRunner(SEARCH_SPACES["stage2"], g, tmp_path / "sweep")
    r.run(lambda p, gd: p["model.margin"], n_trials=2)
    s = r.summary()
    assert s["n_trials"] == 2 and s["n_done"] == 2
    assert {p["name"] for p in s["search_space"]["parameters"]} >= {
        "model.margin", "model.margin_weight"}
    assert s["fold"]["eval_fold_reads"] == 0


def test_sampler_is_deterministic_under_a_seed(tmp_path):
    def draw(seed):
        g = guard(tmp_path / f"s{seed}")
        r = SweepRunner(SPACE, g, tmp_path / f"sweep{seed}", sampler=RandomSampler(seed=7))
        r.run(lambda p, gd: p["b"], n_trials=4)
        return [t.params for t in r.trials]

    assert draw(1) == draw(2)
