"""FoldGuard: the C4 enforcement. These tests are the guard's specification."""

import pytest

from madgrav_ml.eval.folds import (
    FoldGuard,
    FoldLeakError,
    Segment,
    Split,
    assert_disjoint_folds,
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


# --- the split must be on GPS TIME, not on the segment list -------------------
# Found the first time real data hit it: the coincident O3a lists describe both
# detectors with one row, so a flat sort put the H1 copy of a segment in fold 0 and its
# L1 copy in fold 1. Livetime balance looked perfect and the folds shared an hour of
# detector state.


def coincident(n=20, start=1_238_000_000.0, dur=14400.0, gap=3600.0):
    """Both detectors' view of the same n stretches of time."""
    out = []
    for i in range(n):
        a = start + i * (dur + gap)
        out.append(Segment("H1", a, a + dur))
        out.append(Segment("L1", a, a + dur))
    return out


def test_folds_do_not_share_gps_time():
    folds = gps_grouped_folds(coincident(), n_folds=2)
    a_end = max(s.end for s in folds[0])
    b_start = min(s.start for s in folds[1])
    assert b_start >= a_end


def test_both_detectors_of_a_span_land_in_the_same_fold():
    folds = gps_grouped_folds(coincident(), n_folds=2)
    where = {}
    for i, f in enumerate(folds):
        for s in f:
            where.setdefault((s.start, s.end), set()).add(i)
    straddling = [k for k, v in where.items() if len(v) > 1]
    assert not straddling, f"spans split across folds: {straddling}"


def test_assert_disjoint_folds_catches_an_overlap():
    a = [Segment("H1", 0, 100)]
    b = [Segment("L1", 50, 150)]
    with pytest.raises(FoldLeakError, match="overlap in GPS time"):
        assert_disjoint_folds([a, b])


def test_too_few_independent_blocks_is_reported_as_such():
    """Twenty segments that all overlap are ONE block, not twenty."""
    segs = [Segment("H1", 0, 1000) for _ in range(20)]
    with pytest.raises(ValueError, match="independent GPS-time blocks"):
        gps_grouped_folds(segs, n_folds=2)


def test_the_real_upstream_segment_list_splits_cleanly():
    from pathlib import Path

    from madgrav_ml.data.strain import load_segments

    f = Path(".reference/MADGRAV/search_mode/o3a_bg_segments_56.json")
    if not f.exists():
        pytest.skip("upstream repo not vendored; run scripts/vendor_reference.sh")
    segs = load_segments(f, ifo="H1") + load_segments(f, ifo="L1")
    folds = gps_grouped_folds(segs, n_folds=2)
    assert_disjoint_folds(folds)
    live = [sum(s.duration for s in f_) for f_ in folds]
    assert abs(live[0] - live[1]) / sum(live) < 0.10


# --- the background split ------------------------------------------------------


def _fold_guard_three_way(**kw):
    from madgrav_ml.eval.folds import FoldGuard, Segment

    segs = [Segment(ifo, 1_000_000 + 4000 * i, 1_000_000 + 4000 * i + 3600)
            for i in range(40) for ifo in ("H1", "L1")]
    return FoldGuard.from_segments(segs, eval_fold=1, n_folds=2, **kw)


def test_background_split_is_disjoint_from_fit_and_select():
    """A false-alarm rate quoted against data the model was fitted to, or SELECTED
    against, is optimistic. HPO_BG exists so the threshold has somewhere clean to come
    from that is not the quarantined evaluation fold."""
    from madgrav_ml.eval.folds import Split

    guard = _fold_guard_three_way()
    with guard.calibration("test"):
        fit = guard.segments(Split.HPO_TRAIN)
        sel = guard.segments(Split.HPO_VAL)
        bg = guard.segments(Split.HPO_BG)
    for a, b in ((fit, sel), (fit, bg), (sel, bg)):
        assert not ({(s.ifo, s.start) for s in a} & {(s.ifo, s.start) for s in b})
    assert bg and sel and fit
    # contiguous in GPS: fit, then select, then background
    assert max(s.end for s in fit) <= min(s.start for s in sel)
    assert max(s.end for s in sel) <= min(s.start for s in bg)


def test_background_split_is_unreadable_while_training_or_tuning():
    """The guard, not a convention, is what stops a background from being trained on."""
    import pytest as _pytest

    from madgrav_ml.eval.folds import FoldLeakError, Split

    guard = _fold_guard_three_way()
    for phase in ("training", "hpo"):
        with getattr(guard, phase)("test"):
            with _pytest.raises(FoldLeakError, match="hpo_bg"):
                guard.segments(Split.HPO_BG)


def test_background_split_covers_the_training_fold_exactly():
    from madgrav_ml.eval.folds import Split

    guard = _fold_guard_three_way()
    with guard.calibration("test"):
        parts = sum((guard.segments(s) for s in
                     (Split.HPO_TRAIN, Split.HPO_VAL, Split.HPO_BG)), [])
        whole = guard.segments(Split.TRAIN)
    assert sorted((s.ifo, s.start) for s in parts) == sorted((s.ifo, s.start) for s in whole)


def test_split_fractions_must_leave_something_to_train_on():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="nothing to train on"):
        _fold_guard_three_way(hpo_val_frac=0.5, hpo_bg_frac=0.5)
