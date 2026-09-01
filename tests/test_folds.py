"""FoldGuard: the C4 enforcement. These tests are the guard's specification."""

import pytest

from madgrav_ml.eval.folds import (
    FoldGuard,
    FoldLeakError,
    Segment,
    Split,
    gps_grouped_folds,
)


def segments(n=20, start=1_238_000_000.0, dur=4096.0, gap=512.0, ifo="H1"):
    return [
        Segment(ifo, start + i * (dur + gap), start + i * (dur + gap) + dur)
        for i in range(n)
    ]


def test_folds_are_contiguous_in_gps():
    folds = gps_grouped_folds(segments(20), n_folds=2)
    assert max(s.end for s in folds[0]) <= min(s.start for s in folds[1])


def test_folds_balance_livetime():
    folds = gps_grouped_folds(segments(20), n_folds=2)
    live = [sum(s.duration for s in f) for f in folds]
    assert abs(live[0] - live[1]) / sum(live) < 0.10


def test_too_few_segments_raises():
    with pytest.raises(ValueError):
        gps_grouped_folds(segments(1), n_folds=2)


def test_training_phase_cannot_read_eval_fold(tmp_path):
    g = FoldGuard.from_segments(segments(), audit_path=tmp_path / "audit.jsonl")
    with g.training("x"):
        g.segments(Split.TRAIN)
        with pytest.raises(FoldLeakError):
            g.segments(Split.EVAL)


def test_hpo_phase_cannot_read_the_whole_training_fold(tmp_path):
    """HPO sees only its own subsets: scoring a trial on all of TRAIN would make the
    HPO-val split decorative."""
    g = FoldGuard.from_segments(segments(), audit_path=tmp_path / "a.jsonl")
    with g.hpo("sweep", trial=3):
        g.segments(Split.HPO_TRAIN)
        g.segments(Split.HPO_VAL)
        with pytest.raises(FoldLeakError):
            g.segments(Split.TRAIN)


def test_sealed_by_default(tmp_path):
    g = FoldGuard.from_segments(segments(), audit_path=tmp_path / "a.jsonl")
    with pytest.raises(FoldLeakError):
        g.segments(Split.TRAIN)


def test_phases_do_not_nest(tmp_path):
    g = FoldGuard.from_segments(segments(), audit_path=tmp_path / "a.jsonl")
    with g.training("x"):
        with pytest.raises(FoldLeakError):
            with g.final_report("sneaky"):
                pass


def test_eval_fold_readable_once_only(tmp_path):
    g = FoldGuard.from_segments(segments(), audit_path=tmp_path / "a.jsonl")
    with g.final_report("report"):
        assert g.segments(Split.EVAL)
    with pytest.raises(FoldLeakError):
        with g.final_report("again"):
            pass


def test_repeat_final_is_opt_in(tmp_path):
    g = FoldGuard.from_segments(
        segments(), audit_path=tmp_path / "a.jsonl", allow_repeat_final=True
    )
    for _ in range(2):
        with g.final_report("frozen-rerun"):
            g.segments(Split.EVAL)


def test_hpo_val_split_is_time_grouped(tmp_path):
    g = FoldGuard.from_segments(segments(40), audit_path=tmp_path / "a.jsonl")
    with g.hpo("s"):
        tr = g.segments(Split.HPO_TRAIN)
        va = g.segments(Split.HPO_VAL)
    assert max(s.end for s in tr) <= min(s.start for s in va)
    assert not set(tr) & set(va)


def test_audit_trail_is_written_and_checkable(tmp_path):
    path = tmp_path / "audit.jsonl"
    g = FoldGuard.from_segments(segments(), audit_path=path)
    with g.training("t"):
        g.segments(Split.TRAIN)
    with g.final_report("r"):
        g.segments(Split.EVAL)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    g.assert_eval_untouched_by_tuning()


def test_summary_reports_the_fold_accounting(tmp_path):
    g = FoldGuard.from_segments(segments(), audit_path=tmp_path / "a.jsonl")
    s = g.summary()
    assert s["n_folds"] == 2 and s["eval_fold"] == 1
    assert s["train_livetime_s"] > 0 and s["eval_livetime_s"] > 0
    assert s["eval_fold_reads"] == 0


def test_audit_records_the_calling_line(tmp_path):
    """The audit trail is the evidence shown to a reviewer, so the recorded caller must
    be the line that asked for the data, not a frame inside the guard."""
    import json

    path = tmp_path / "audit.jsonl"
    g = FoldGuard.from_segments(segments(), audit_path=path)
    with g.training("t"):
        g.segments(Split.TRAIN)
    rec = json.loads(path.read_text().strip())
    assert rec["caller"].startswith("test_folds.py:")
    assert rec["phase"] == "training" and rec["label"] == "t"
