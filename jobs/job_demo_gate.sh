#!/bin/bash
#SBATCH --job-name=madgrav_demo_gate
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# THE REPRODUCTION GATE (improvement plan section 3.4). Nothing else starts until this
# passes. It recovers GW190521 from the ~256 s segment bundled with the upstream repo,
# using the vendored weights, and must produce:
#
#     net sigma ~ 7.7     HM ~ 0.99     LM ~ 0.95     verdict RECOVERED
#
# WHY IT IS NOT A FORMALITY HERE. Upstream's environment.yml pins torch 1.12.1 /
# CUDA 11.2 and states that the frozen weights are calibration-locked to that build:
# "a different torch/CUDA build can shift the GPU forward pass and break the frozen FAR
# calibration". Our venv is torch 2.6.0+cu124, because CUDA 11.2 has no wheel that runs
# on this cluster's GPUs. Upstream's own instruction for that case is to "match it as
# closely as your GPUs allow and re-validate the demo before trusting FARs" -- so this
# job IS that re-validation, and its numbers decide whether our build can be used to
# quote anything at all.
#
# GPU, not CPU. The upstream README is explicit that CPU forward is not byte-identical
# to the calibrated GPU path, so a CPU number cannot validate anything. SM_ALLOW_CPU is
# left unset here on purpose: if the GPU is missing, this job should fail loudly rather
# than silently produce an uncalibrated number that looks like a pass.
#
# Usage: sbatch jobs/job_demo_gate.sh
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs runs/demo_gate

echo "=== madgrav reproduction gate on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| arch', torch.cuda.get_arch_list())"
echo "--- upstream pinned: torch 1.12.1 / CUDA 11.2 (see .reference/MADGRAV/environment.yml) ---"
echo

cd "$PROJ/.reference/MADGRAV"
MADGRAV_ROOT="$PROJ/.reference/MADGRAV" DEV=cuda:0 PYTHON="$PY" \
  bash demo/run_demo.sh 2>&1 | tee "$PROJ/runs/demo_gate/demo_$(date +%Y%m%d_%H%M%S).log"

echo
echo "=== expected: net sigma ~7.7, HM ~0.99, LM ~0.95, verdict RECOVERED ==="
echo "=== if the numbers differ materially, STOP: the plan makes every downstream    ==="
echo "=== comparison meaningless until the discrepancy is understood (section 3.4).  ==="
