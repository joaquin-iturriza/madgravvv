"""Measured parameter counts and the C2 budget check.

Constraint C2: any replacement component must have approximately the parameter count
of the component it replaces. Improving a result by growing the model is trivial and
uninteresting, and the upstream pipeline's compactness is a stated design goal (it
runs on an Arduino UNO Q at ~3 W, ~740 ms per tile).

The plan is explicit that the ~1e5-per-component figures quoted for the baseline are
*inferred from the described topology, not measured*. So the reference numbers this
module compares against must come from `count_checkpoint` over the vendored `.pt`
files, recorded once into `config/param_budget.yaml` — never from an estimate typed
into a docstring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class BudgetVerdict:
    """The C2 line of a per-experiment record (item 3)."""

    component: str
    baseline: int
    candidate: int
    tolerance: float
    passed: bool

    @property
    def ratio(self) -> float:
        return self.candidate / self.baseline if self.baseline else float("inf")

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"C2 {mark}: {self.component} {self.baseline:,} -> {self.candidate:,} "
            f"({self.ratio:+.1%} of baseline, tolerance +/-{self.tolerance:.0%})"
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "ratio": self.ratio}


def count_parameters(module, trainable_only: bool = False) -> int:
    """Parameter count of a live `torch.nn.Module`.

    Counts buffers nowhere: BatchNorm running statistics are state, not capacity, and
    counting them would make a BN-heavy baseline look larger than it is.
    """
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_checkpoint(path: str | Path, prefix: str | None = None) -> int:
    """Parameter count of a vendored `.pt` checkpoint, without instantiating the model.

    This is how the baseline numbers get measured. `prefix` restricts the count to one
    component of a combined checkpoint (e.g. `"encoder."`).

    Buffers are excluded by name where they are recognisable (BatchNorm's
    `running_mean` / `running_var` / `num_batches_tracked`), for the same reason
    `count_parameters` excludes them.
    """
    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    state = obj
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, Mapping) and key in state:
            state = state[key]
    if not isinstance(state, Mapping):
        raise TypeError(f"{path}: cannot find a state dict in a {type(obj).__name__}")

    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    total = 0
    for name, tensor in state.items():
        if not hasattr(tensor, "numel"):
            continue
        if name.endswith(buffer_suffixes):
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        total += int(tensor.numel())
    return total


def per_component_counts(model_dict: Mapping[str, object]) -> dict[str, int]:
    """`{name: count}` over a dict of live modules — the table a run record carries."""
    return {name: count_parameters(m) for name, m in model_dict.items()}


def check_budget(
    component: str,
    candidate: int,
    baseline: int,
    tolerance: float = 0.10,
    strict: bool = True,
) -> BudgetVerdict:
    """Compare a candidate's measured count against the measured baseline.

    `tolerance` is the fractional band that still counts as "approximately the same
    parameter count". 10% is the default: wide enough that a differently-factorised
    block is not rejected over rounding, narrow enough that no real capacity gain
    hides inside it.

    With `strict` (the default) a failure raises, so a run that violates C2 cannot
    quietly produce a headline number.
    """
    if baseline <= 0:
        raise ValueError(f"{component}: baseline count must be positive, got {baseline}")
    passed = abs(candidate - baseline) <= tolerance * baseline
    verdict = BudgetVerdict(component, baseline, candidate, tolerance, passed)
    if strict and not passed:
        raise ValueError(
            f"{verdict}\nC2 forbids buying an improvement with parameters. Either "
            f"shrink the candidate back into the band, or state explicitly in the run "
            f"record that this is an out-of-budget control run and not a proposal."
        )
    return verdict


def load_reference(path: str | Path) -> dict[str, int]:
    """Read the measured baseline counts recorded from the vendored weights."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    counts = data.get("components", data)
    return {k: int(v) for k, v in counts.items()}
