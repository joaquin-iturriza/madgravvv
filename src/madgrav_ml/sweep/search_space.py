"""What is searched, over what range, and why.

FA's CLAUDE.md keeps its search ranges in one table and narrows them from evidence
rather than carrying a flat prior for every knob — its rule 3, "lr is the dominant knob;
narrow the rest, don't blindly fix them", cut the search volume by 10-100x. We have no
such evidence yet, so the ranges here are priors with their reasoning attached, and the
honest thing is to say so: they are starting points to be narrowed once trials exist,
not measured optima.

Phase 5 of the plan names the targets. The primary two are the stage-2 margin `m` and its
weight `lambda`: they directly shape the score distribution every downstream component
consumes and have no obvious a priori value. Training length is a first-class target
too, because the upstream run was 10 epochs and best at epoch 9 — meaning longer training
was never tested, not that 10 was chosen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    """One searched dimension.

    `log` marks a scale parameter, sampled uniformly in log space. Getting this wrong is
    the classic way a search wastes its budget: a uniform draw over [1e-6, 1e-2] spends
    99% of its trials in the top decade.
    """

    name: str
    low: float
    high: float
    log: bool = False
    integer: bool = False
    note: str = ""

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must exceed low, got [{self.low}, {self.high}]")
        if self.log and self.low <= 0:
            raise ValueError(f"{self.name}: a log-scaled parameter needs a positive low bound")

    def from_unit(self, u: float) -> float | int:
        """Map u in [0,1] to a value in range."""
        if not 0.0 <= u <= 1.0:
            raise ValueError(f"{self.name}: unit draw out of range: {u}")
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return int(round(v)) if self.integer else float(v)


@dataclass(frozen=True)
class SearchSpace:
    name: str
    parameters: tuple[Parameter, ...]

    def __len__(self) -> int:
        return len(self.parameters)

    def from_unit(self, u) -> dict:
        if len(u) != len(self.parameters):
            raise ValueError(f"expected {len(self.parameters)} unit draws, got {len(u)}")
        return {p.name: p.from_unit(float(x)) for p, x in zip(self.parameters, u)}

    def as_dict(self) -> dict:
        """For the run record — a result is not reproducible without its search space."""
        return {
            "name": self.name,
            "parameters": [
                {"name": p.name, "low": p.low, "high": p.high, "log": p.log,
                 "integer": p.integer, "note": p.note}
                for p in self.parameters
            ],
        }


# --- stage 2: the plan's primary Phase-5 target ------------------------------
STAGE2 = SearchSpace(
    name="stage2_margin",
    parameters=(
        Parameter(
            "model.margin", 0.5, 10.0, log=True,
            note="upstream 3.0. Log-scaled because it is compared against a "
                 "reconstruction error whose own scale is set by the representation, so "
                 "the meaningful moves are multiplicative.",
        ),
        Parameter(
            "model.margin_weight", 0.1, 20.0, log=True,
            note="upstream 2.0. Searched jointly with the margin and never alone: the "
                 "two trade off directly, which is exactly why hpo_guard refuses a 1-D "
                 "grid over either.",
        ),
        Parameter(
            "training.lr", 1e-5, 3e-3, log=True,
            note="upstream stage-1 used 1e-3; the fine-tune LR is not documented. "
                 "Centred below the pretrain value since this is a fine-tune.",
        ),
        Parameter(
            "training.iterations", 5_000, 60_000, log=True, integer=True,
            note="upstream ran 10 epochs and was best at epoch 9, so longer training "
                 "was never tested. Length is a target, not a constant.",
        ),
    ),
)

# --- stage 1: secondary, and only meaningful once the objective is masked ----
STAGE1 = SearchSpace(
    name="stage1_masked",
    parameters=(
        Parameter("training.lr", 1e-5, 3e-3, log=True, note="upstream 1e-3."),
        Parameter("training.weight_decay", 1e-7, 1e-3, log=True, note="upstream 1e-5."),
        Parameter(
            "model.mask_ratio", 0.15, 0.75,
            note="Phase 4.1. Linear, not log: this is a fraction, and the interesting "
                 "range is a factor of five wide, not five decades.",
        ),
        Parameter("model.dropout", 0.0, 0.4, note="upstream 0.20."),
    ),
)

SEARCH_SPACES = {"stage2": STAGE2, "stage1": STAGE1}
