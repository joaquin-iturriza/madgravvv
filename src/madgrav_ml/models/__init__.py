"""Model components: the stage-1/stage-2 CAE, the glitch arm, the HM/LM specialists.

Every component here is subject to constraint C2 — a replacement must have
approximately the parameter count of the thing it replaces. `param_budget` is where
that is measured and enforced; nothing in this package is exempt from it. The
reference counts live in `config/param_budget.yaml` and were measured from the
vendored checkpoints, not inferred from the topology.

`param_budget` imports eagerly and is torch-free on purpose: the C2 numbers, and the
whole `eval/` harness, must be usable on the local machine, which has no CUDA stack.
The model classes are resolved lazily so that importing this package does not drag in
torch.
"""

from .param_budget import (
    BudgetVerdict,
    check_budget,
    count_checkpoint,
    count_parameters,
    load_reference,
    per_component_counts,
)

_LAZY = {
    "BaselineCAE": ".cae",
    "margin_loss": ".cae",
    "stage2_loss": ".cae",
    "GlitchArm": ".arms",
    "SpecialistCNN": ".arms",
    "SeedEnsemble": ".arms",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "BaselineCAE",
    "BudgetVerdict",
    "GlitchArm",
    "SeedEnsemble",
    "SpecialistCNN",
    "check_budget",
    "count_checkpoint",
    "count_parameters",
    "load_reference",
    "margin_loss",
    "per_component_counts",
    "stage2_loss",
]
