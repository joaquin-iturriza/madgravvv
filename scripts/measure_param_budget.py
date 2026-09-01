#!/usr/bin/env python3
"""Measure the vendored MADGRAV checkpoints and regenerate `config/param_budget.yaml`.

Constraint C2 is enforced against measured numbers. The plan is explicit that the
~1e5-per-component figures are inferred from the described topology, not measured, so
this script is the first thing to run in a fresh checkout — and re-run if the
reference repo is ever updated.

Runs without torch: a `.pt` is a zip of a pickle, and the tensor *shapes* are
recoverable from the pickle's rebuild calls without instantiating anything. That
matters because the counts are needed on a local machine that has no CUDA stack.
"""

from __future__ import annotations

import io
import math
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

REFERENCE = Path(".reference/MADGRAV")
CHECKPOINTS = {
    "cae": REFERENCE / "assets/models/baseline_cae_weaksup_best.pt",
    "glitch_arm": REFERENCE / "lr_cascade/p1v42/arm_deploy_seed0.pt",
    "specialist_hm": REFERENCE / "search_mode/hm_native_seed0.pt",
    "specialist_lm": REFERENCE / "search_mode/lm_native_seed0.pt",
}


class _Stub:
    def __init__(self, *a, **k):
        pass


class _Tensor:
    """Placeholder capturing only the shape from `torch._utils._rebuild_tensor_v2`."""

    def __init__(self, storage, offset, size, stride, *a, **k):
        self.size = tuple(size)

    def numel(self) -> int:
        return math.prod(self.size) if self.size else 1


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _Tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, hooks: data
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module.startswith(("torch", "numpy")):
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub

    def persistent_load(self, pid):
        return None


def shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        obj = _Unpickler(io.BytesIO(z.read(name))).load()
    out: dict[str, tuple[int, ...]] = {}

    def walk(node, prefix=""):
        if isinstance(node, _Tensor):
            out[prefix.rstrip(".")] = node.size
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")

    walk(obj)
    return out


def count(path: Path, prefix: str | None = None) -> int:
    return sum(
        math.prod(s) if s else 1
        for k, s in shapes(path).items()
        if not k.endswith(BUFFER_SUFFIXES) and (prefix is None or k.startswith(prefix))
    )


def main() -> int:
    missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
    if missing:
        print(
            "missing reference checkpoints:\n  " + "\n  ".join(missing) +
            "\n\nVendor the upstream repo first:  bash scripts/vendor_reference.sh",
            file=sys.stderr,
        )
        return 1

    counts = {name: count(p) for name, p in CHECKPOINTS.items()}
    counts["cae_classifier"] = count(CHECKPOINTS["cae"], prefix="classifier")
    counts["cae_conv"] = counts["cae"] - counts["cae_classifier"]

    for name in sorted(counts):
        print(f"{name:>18s}  {counts[name]:>10,d}")

    print("\nUpdate config/param_budget.yaml if these differ from what it holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
