"""The per-experiment record — section 13 of the improvement plan, as an object.

Every experiment produces one of these and writes it to `runs/<name>/summary.json`.
The point is that the final writeup should be assembly, not archaeology: if the
records are complete, the results table is a `glob` and a `pandas.DataFrame`.

The required fields are the plan's nine, and the object refuses to serialise while any
of the load-bearing ones is missing. A record with no fold assignment or no parameter
count is not a partial record, it is an unusable one.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class Verdict(str):
    """keep | discard | needs-more-work — with the reasoning, always."""

    KEEP = "keep"
    DISCARD = "discard"
    NEEDS_MORE_WORK = "needs-more-work"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except Exception:
        return False


@dataclass
class ExperimentRecord:
    """One row of the running results table.

    Parameters map onto the plan's numbered list:

    1. `hypothesis`  — what is expected to improve, and by what mechanism.
    2. `change`      — precise description; a diff or a config path.
    3. `parameters`  — measured counts before/after plus the C2 verdict.
    4. `folds`       — which fold trained, which validated, and confirmation that the
                       evaluation fold was untouched (from `FoldGuard.summary()`).
    5. `primary`     — single-detector efficiency at fixed single-detector FAR;
                       network efficiency and VT at FAR <= 1/yr.
    6. `secondary`   — AUC, AP, ECE. Development only; never the headline.
    7. `seeds`       — at least 3, reported as mean +/- spread.
    8. `compute`     — GPU-hours, so cost/benefit is legible.
    9. `verdict`     — keep / discard / needs-more-work, with reasoning.
    """

    name: str
    hypothesis: str
    change: str
    parameters: dict[str, Any] = field(default_factory=dict)
    folds: dict[str, Any] = field(default_factory=dict)
    primary: dict[str, Any] = field(default_factory=dict)
    secondary: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    compute_gpu_hours: float | None = None
    verdict: str = ""
    reasoning: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    git_sha: str = field(default_factory=_git_sha)
    git_dirty: bool = field(default_factory=_git_dirty)
    host: str = field(default_factory=socket.gethostname)
    python: str = field(default_factory=platform.python_version)

    def validate(self) -> None:
        """Refuse to persist a record that cannot support a claim."""
        problems = []
        if not self.hypothesis.strip():
            problems.append("hypothesis is empty — say what should improve and why")
        if not self.change.strip():
            problems.append("change is empty — point at a diff or a config")
        if not self.parameters:
            problems.append(
                "no measured parameter counts — C2 is enforced on measured numbers, "
                "not on the topology you believe you implemented"
            )
        if not self.folds:
            problems.append(
                "no fold assignment — attach FoldGuard.summary(); a number without a "
                "fold record cannot be audited and cannot be quoted"
            )
        if not self.primary:
            problems.append(
                "no primary metrics — efficiency/VT at fixed FAR. AUC is not a headline"
            )
        if len(self.seeds) < 3 and self.verdict == Verdict.KEEP:
            problems.append(
                f"only {len(self.seeds)} seed(s) for a 'keep' verdict — report mean "
                f"+/- spread over at least 3"
            )
        if self.verdict not in (Verdict.KEEP, Verdict.DISCARD, Verdict.NEEDS_MORE_WORK):
            problems.append(f"verdict {self.verdict!r} is not keep/discard/needs-more-work")
        if not self.reasoning.strip():
            problems.append("verdict has no reasoning")
        if problems:
            raise ValueError(
                f"incomplete experiment record {self.name!r}:\n  - " + "\n  - ".join(problems)
            )

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d

    def save(self, run_dir: str | os.PathLike, filename: str = "summary.json") -> Path:
        self.validate()
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str))
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ExperimentRecord":
        data = json.loads(Path(path).read_text())
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def collect(runs_dir: str | os.PathLike, pattern: str = "*/summary.json") -> list[ExperimentRecord]:
    """Every record under `runs_dir` — the running results table, assembled."""
    return [ExperimentRecord.load(p) for p in sorted(Path(runs_dir).glob(pattern))]
