#!/usr/bin/env bash
# One-time environment build on a CC-IN2P3 LOGIN NODE (compute nodes have internet too,
# but building once on a login node keeps runs to a pure read).
#
#   ssh ccin2p3
#   cd /sps/lpnhe/jiturrizaramirez01/madgrav && bash scripts/setup_env.sh
#
# Order matters. Install torch FIRST from the CUDA-12.4 index: the default wheel is now
# CUDA 13, which DROPPED Volta (sm_70) and so will NOT run on the gpu_v100 partition.
# cu124 (torch 2.6) ships sm_70 (V100) and sm_90 (H100). Installing it first satisfies
# torch>=2.2 so the editable install below leaves it alone.
#
# The venv is built from the anaconda module's python, which has a proper rpath, so
# calling .venv/bin/python later needs no `module load`.
set -euo pipefail
PROJ="${PROJ:-/sps/lpnhe/jiturrizaramirez01/madgrav}"
cd "$PROJ"

module load Programming_Languages/anaconda/3.12
python -m venv .venv

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
.venv/bin/python -m pip install -e ".[dev,deep,gw]"

echo
echo "=== check ==="
.venv/bin/python -c "import torch, gwpy, hydra; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
.venv/bin/python -m pytest tests/ -q
