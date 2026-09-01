#!/usr/bin/env bash
# One-time environment build on a CC-IN2P3 LOGIN NODE (compute nodes have internet too,
# but building once on a login node keeps runs to a pure read).
#
#   ssh ccin2p3
#   cd /sps/lpnhe/jiturrizaramirez01/madgrav && bash scripts/setup_env.sh
#
# THE ONE THING THAT MATTERS HERE: the torch wheel must ship sm_70 (Volta), or nothing
# runs on the gpu_v100 partition. The current default PyPI wheel is CUDA 13, which
# DROPPED Volta; cu124 (torch 2.6) ships sm_70 (V100) and sm_90 (H100).
#
# Installing torch from the cu124 index first is NOT enough on its own. `mup` depends on
# torchvision, and resolving torchvision from the default index drags in a newer torch
# and silently uninstalls the cu124 one — observed here: torch 2.6.0+cu124 replaced by
# torch 2.13.0, i.e. a venv that imports fine, passes tests on the login node, and then
# fails on every V100 job. So:
#   1. install torch AND torchvision together from the cu124 index;
#   2. pin both with PIP_CONSTRAINT for the project install, so nothing can move them;
#   3. verify sm_70 is in the compiled arch list, and fail loudly if it is not.
#
# The venv is built from the anaconda module's python, which has a proper rpath, so
# calling .venv/bin/python later needs no `module load`.
set -euo pipefail
PROJ="${PROJ:-/sps/lpnhe/jiturrizaramirez01/madgrav}"
cd "$PROJ"

# `module` is a shell function, and a non-interactive `ssh host 'bash script.sh'` does
# not source the profile that defines it. Source it explicitly so this works both when
# run by hand on a login node and when driven through scripts/remote.sh.
if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
module load Programming_Languages/anaconda/3.12
python -m venv .venv

PY=.venv/bin/python
$PY -m pip install --quiet --upgrade pip

# 1. the CUDA-12.4 pair, together
$PY -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# 2. freeze them so the project install cannot resolve around them
CONSTRAINTS=$(mktemp)
trap 'rm -f "$CONSTRAINTS"' EXIT
$PY - <<'PYEOF' > "$CONSTRAINTS"
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PYEOF
echo "--- pinning ---"; cat "$CONSTRAINTS"
PIP_CONSTRAINT="$CONSTRAINTS" $PY -m pip install -e ".[dev,deep,gw]"

# 3. verify, and refuse to hand back a venv that cannot run on the default partition
echo
echo "=== check ==="
$PY - <<'PYEOF'
import sys
import torch

print("torch", torch.__version__, "| cuda", torch.version.cuda)

# `get_arch_list()` returns [] on a login node (no CUDA driver), so it cannot be the
# primary test — this script is meant to be run exactly there. The CUDA major version
# is the reliable signal: 12.x ships sm_70, 13.x dropped it. The arch list is still
# checked when it is populated, which is the case in a GPU job.
cuda = torch.version.cuda or ""
major = int(cuda.split(".")[0]) if cuda.split(".")[0].isdigit() else 0
if major >= 13:
    sys.exit(
        f"FATAL: torch was built against CUDA {cuda}, which dropped Volta (sm_70), so "
        f"it cannot run on the gpu_v100 partition. Something re-resolved torch off the "
        f"cu124 index. Delete .venv and re-run; if it recurs, find which dependency "
        f"pulled the newer torch (`pip install -e . --dry-run` shows the resolution)."
    )
archs = torch.cuda.get_arch_list()
if archs:
    print("arch list:", archs)
    if not any(a.startswith("sm_70") for a in archs):
        sys.exit(f"FATAL: no sm_70 in the compiled arch list {archs}.")
else:
    print("arch list: empty (no CUDA driver here — expected on a login node)")

import gwpy, hydra  # noqa: F401
print("gwpy", gwpy.__version__, "| hydra ok")
PYEOF
$PY -m pytest tests/ -q
