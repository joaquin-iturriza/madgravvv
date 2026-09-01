#!/bin/bash
#SBATCH --job-name=madgrav_bg
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Time-slide background for one configuration.
#
# READ THIS BEFORE EDITING: the background must be scored by the SAME selection the
# foreground goes through. If the gate changed, the slides must be re-run — a FAR
# measured under a different selection than the foreground is meaningless, and nothing
# downstream can detect it. Point this job at the same config the foreground used.
#
#   sbatch jobs/job_background.sh selection=runs/madgrav/<run>/config.yaml
#
# GPU, not CPU: the upstream README is explicit that the GPU forward pass is the
# calibrated path and CPU forward is not byte-identical. FAR runs are GPU-only.
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav background on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
$PY -u -m madgrav_ml.experiments.matched_far "$@"
echo "=== done ==="
