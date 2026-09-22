#!/bin/bash
#SBATCH --job-name=madgrav_bg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
#SBATCH --gres=gpu:1
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
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
PY=python                       # env activated by sites/activate.sh
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav background on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
$PY -u -m madgrav_ml.experiments.matched_far "$@"
echo "=== done ==="
