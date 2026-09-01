"""Fold discipline, enforced in code rather than by convention.

Constraint C4: no FAR may ever be quoted against background that was used in fitting,
training, or hyperparameter selection. The plan asks for this as a guard object that
*raises* on evaluation-fold access outside the final report path, plus an audit trail
an external reviewer can check. That is what this module is.

The model:

* Segments are grouped into folds by **GPS time**, never randomly — adjacent strain
  segments share the same detector state, so a random split leaks.
* Exactly one fold is the evaluation fold. Everything else is the training fold.
* The training fold splits again into an HPO-train and an HPO-validation subset, and
  **all** hyperparameter search happens inside those. Hundreds of Bayesian-HPO trials
  scored on the evaluation fold is a textbook overfitting vector, and in a search that
  quotes a FAR it invalidates the result outright.
* The evaluation fold is touched **once**, at the end, inside `final_report()`.

Usage::

    guard = FoldGuard.from_segments(segments, eval_fold=1, audit_path="runs/x/folds.jsonl")

    with guard.training("stage1-pretrain"):
        segs = guard.segments(Split.TRAIN)          # ok
        bad  = guard.segments(Split.EVAL)           # FoldLeakError

    with guard.hpo("margin-sweep", trial=17):
        tr = guard.segments(Split.HPO_TRAIN)
        va = guard.segments(Split.HPO_VAL)

    with guard.final_report("efficiency-at-far"):   # allowed exactly once
        ev = guard.segments(Split.EVAL)
"""

from __future__ import annotations

import getpass
import json
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class FoldLeakError(RuntimeError):
    """Raised when code reaches for data the current phase is not allowed to see."""


class Phase(str, Enum):
    """What the code is currently allowed to do."""

    SEALED = "sealed"            # nothing granted; the default between blocks
    TRAINING = "training"        # model fitting on the training fold
    HPO = "hpo"                  # hyperparameter search, inside the training fold only
    CALIBRATION = "calibration"  # fitting the noise calibration / likelihood ratio
    FINAL_REPORT = "final"       # the one pass over the evaluation fold


class Split(str, Enum):
    """Which slice of data is being requested."""

    TRAIN = "train"          # the whole training fold
    HPO_TRAIN = "hpo_train"  # training-fold subset used to fit HPO trials
    HPO_VAL = "hpo_val"      # training-fold subset used to score HPO trials
    EVAL = "eval"            # the evaluation fold — quarantined


# What each phase may read. Anything not listed raises.
_ALLOWED: dict[Phase, frozenset[Split]] = {
    Phase.SEALED: frozenset(),
    Phase.TRAINING: frozenset({Split.TRAIN, Split.HPO_TRAIN, Split.HPO_VAL}),
    Phase.HPO: frozenset({Split.HPO_TRAIN, Split.HPO_VAL}),
    Phase.CALIBRATION: frozenset({Split.TRAIN, Split.HPO_TRAIN, Split.HPO_VAL}),
    Phase.FINAL_REPORT: frozenset({Split.EVAL, Split.TRAIN}),
}


@dataclass(frozen=True)
class Segment:
    """A contiguous stretch of analysable strain, the unit a fold is built from."""

    ifo: str
    start: float   # GPS seconds
    end: float     # GPS seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


def gps_grouped_folds(
    segments: Sequence[Segment],
    n_folds: int = 2,
) -> list[list[Segment]]:
    """Split `segments` into `n_folds` contiguous GPS-time blocks of ~equal livetime.

    Contiguous, not interleaved and not random: the failure this defends against is
    two halves of the same hour of detector state landing on opposite sides of the
    split, which makes the evaluation fold a near-copy of the training fold. Balance
    is on livetime (sum of durations), not on segment count, because segments vary in
    length by orders of magnitude.

    Returns folds in GPS order; `folds[i]` is a list of Segments.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.ifo))
    if len(ordered) < n_folds:
        raise ValueError(f"{len(ordered)} segments cannot fill {n_folds} folds")

    total = sum(s.duration for s in ordered)
    target = total / n_folds

    folds: list[list[Segment]] = [[] for _ in range(n_folds)]
    acc, idx = 0.0, 0
    for seg in ordered:
        # Move to the next fold once this one has its share, but never leave a fold
        # empty and never overflow the last one.
        if idx < n_folds - 1 and acc >= target * (idx + 1):
            idx += 1
        folds[idx].append(seg)
        acc += seg.duration

    empty = [i for i, f in enumerate(folds) if not f]
    if empty:
        raise ValueError(f"folds {empty} came out empty; too few segments for n_folds={n_folds}")
    return folds


@dataclass
class _AuditRecord:
    t: float
    phase: str
    split: str
    label: str
    trial: int | None
    n_segments: int
    livetime: float
    caller: str

    def as_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


@dataclass
class FoldGuard:
    """Gatekeeper over the fold structure. Raises rather than warns.

    Parameters
    ----------
    folds
        GPS-grouped folds, as returned by `gps_grouped_folds`.
    eval_fold
        Index of the quarantined evaluation fold.
    hpo_val_frac
        Fraction of the *training* fold livetime reserved to score HPO trials. Taken
        from the tail in GPS order, so the HPO split is itself time-grouped.
    audit_path
        JSONL file recording every granted access. Written append-only; this is the
        artifact that lets a reviewer verify tuning never saw the evaluation fold.
    allow_repeat_final
        Off by default. The evaluation fold is meant to be read once. Set it only for
        a deliberate re-run of the same frozen report, and say so in the run record.
    """

    folds: list[list[Segment]]
    eval_fold: int
    hpo_val_frac: float = 0.25
    audit_path: str | os.PathLike | None = None
    allow_repeat_final: bool = False

    phase: Phase = field(default=Phase.SEALED, init=False)
    _label: str = field(default="", init=False)
    _trial: int | None = field(default=None, init=False)
    _final_reports: int = field(default=0, init=False)
    _records: list[_AuditRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.eval_fold < len(self.folds):
            raise ValueError(f"eval_fold {self.eval_fold} out of range for {len(self.folds)} folds")
        if not 0.0 < self.hpo_val_frac < 1.0:
            raise ValueError(f"hpo_val_frac must be in (0,1), got {self.hpo_val_frac}")
        if self.audit_path is not None:
            Path(self.audit_path).parent.mkdir(parents=True, exist_ok=True)

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[Segment],
        eval_fold: int = 1,
        n_folds: int = 2,
        **kwargs,
    ) -> "FoldGuard":
        return cls(folds=gps_grouped_folds(segments, n_folds), eval_fold=eval_fold, **kwargs)

    # ---- the splits --------------------------------------------------------

    @property
    def _train_segments(self) -> list[Segment]:
        out: list[Segment] = []
        for i, f in enumerate(self.folds):
            if i != self.eval_fold:
                out.extend(f)
        return sorted(out, key=lambda s: (s.start, s.end, s.ifo))

    @property
    def _eval_segments(self) -> list[Segment]:
        return sorted(self.folds[self.eval_fold], key=lambda s: (s.start, s.end, s.ifo))

    def _hpo_split(self) -> tuple[list[Segment], list[Segment]]:
        """Time-grouped HPO train/val split of the training fold (val is the GPS tail)."""
        tr = self._train_segments
        total = sum(s.duration for s in tr)
        cut = total * (1.0 - self.hpo_val_frac)
        acc, boundary = 0.0, len(tr)
        for i, s in enumerate(tr):
            if acc >= cut:
                boundary = i
                break
            acc += s.duration
        boundary = min(max(boundary, 1), len(tr) - 1)
        return tr[:boundary], tr[boundary:]

    # ---- phase context managers -------------------------------------------

    @contextmanager
    def _enter(self, phase: Phase, label: str, trial: int | None = None):
        if self.phase is not Phase.SEALED:
            raise FoldLeakError(
                f"cannot enter {phase.value} while already in {self.phase.value} "
                f"({self._label!r}); phases do not nest"
            )
        self.phase, self._label, self._trial = phase, label, trial
        try:
            yield self
        finally:
            self.phase, self._label, self._trial = Phase.SEALED, "", None

    def training(self, label: str):
        """Model fitting on the training fold."""
        return self._enter(Phase.TRAINING, label)

    def hpo(self, label: str, trial: int | None = None):
        """A hyperparameter-search trial. Sees only the HPO subsets of the training fold."""
        return self._enter(Phase.HPO, label, trial=trial)

    def calibration(self, label: str):
        """Fitting the noise calibration / likelihood ratio. Training fold only."""
        return self._enter(Phase.CALIBRATION, label)

    def final_report(self, label: str):
        """The single pass over the evaluation fold that produces the quoted numbers."""
        if self._final_reports >= 1 and not self.allow_repeat_final:
            raise FoldLeakError(
                "the evaluation fold has already been read once in this process. "
                "Re-reading it turns the held-out fold into a selection set. If this "
                "is a deliberate re-run of an already-frozen report, construct the "
                "guard with allow_repeat_final=True and record why."
            )
        self._final_reports += 1
        return self._enter(Phase.FINAL_REPORT, label)

    # ---- the gate ----------------------------------------------------------

    def segments(self, split: Split | str) -> list[Segment]:
        """Return the segments for `split`, or raise if the current phase may not see them."""
        split = Split(split)
        allowed = _ALLOWED[self.phase]
        if split not in allowed:
            raise FoldLeakError(
                f"{split.value!r} is not readable in phase {self.phase.value!r} "
                f"(label={self._label!r}). Allowed here: "
                f"{sorted(s.value for s in allowed) or 'nothing — open a phase first'}. "
                "If you need the evaluation fold, you are writing the final report; "
                "use `with guard.final_report(...)`. If you are tuning, you are not "
                "allowed to see it at all (constraint C4)."
            )
        if split is Split.TRAIN:
            segs = self._train_segments
        elif split is Split.EVAL:
            segs = self._eval_segments
        else:
            hpo_tr, hpo_va = self._hpo_split()
            segs = hpo_tr if split is Split.HPO_TRAIN else hpo_va
        self._audit(split, segs)
        return segs

    def _audit(self, split: Split, segs: Iterable[Segment]) -> None:
        segs = list(segs)
        # Two frames up is the caller of `segments`; that is the line worth recording.
        stack = traceback.extract_stack(limit=4)
        frame = stack[0] if len(stack) < 4 else stack[-4]
        rec = _AuditRecord(
            t=time.time(),
            phase=self.phase.value,
            split=split.value,
            label=self._label,
            trial=self._trial,
            n_segments=len(segs),
            livetime=sum(s.duration for s in segs),
            caller=f"{os.path.basename(frame.filename)}:{frame.lineno}",
        )
        self._records.append(rec)
        if self.audit_path is not None:
            with open(self.audit_path, "a") as fh:
                fh.write(rec.as_json() + "\n")

    # ---- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        """Fold accounting for `summary.json` — item 4 of the per-experiment record."""
        hpo_tr, hpo_va = self._hpo_split()
        return {
            "n_folds": len(self.folds),
            "eval_fold": self.eval_fold,
            "fold_livetime_s": [sum(s.duration for s in f) for f in self.folds],
            "fold_gps_span": [
                [min(s.start for s in f), max(s.end for s in f)] for f in self.folds
            ],
            "train_livetime_s": sum(s.duration for s in self._train_segments),
            "eval_livetime_s": sum(s.duration for s in self._eval_segments),
            "hpo_train_livetime_s": sum(s.duration for s in hpo_tr),
            "hpo_val_livetime_s": sum(s.duration for s in hpo_va),
            "eval_fold_reads": self._final_reports,
            "n_audited_accesses": len(self._records),
            "audit_path": str(self.audit_path) if self.audit_path else None,
            "user": getpass.getuser(),
        }

    def assert_eval_untouched_by_tuning(self) -> None:
        """Post-hoc check over the audit trail: no eval read outside FINAL_REPORT.

        Cheap, and it is the claim you make to a reviewer, so make it mechanically.
        """
        offenders = [
            r for r in self._records
            if r.split == Split.EVAL.value and r.phase != Phase.FINAL_REPORT.value
        ]
        if offenders:
            where = ", ".join(f"{r.caller} ({r.phase}/{r.label})" for r in offenders[:5])
            raise FoldLeakError(f"evaluation fold was read outside the report path: {where}")
