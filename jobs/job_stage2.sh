#!/bin/bash
#SBATCH --job-name=madgrav_stage2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
#SBATCH --gres=gpu:1
# Stage-2 margin fine-tune. Needs a stage-1 checkpoint:
#
#   sbatch jobs/job_stage2.sh model.init_from=runs/madgrav/<stage1_run>/models/model_best.pt
#
# m and lambda are the HPO targets (Phase 5); pass them as overrides:
#   sbatch jobs/job_stage2.sh model.margin=2.5 model.margin_weight=1.5 seed=1
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
PY=python                       # env activated by sites/activate.sh
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav stage-2 on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
$PY -u run.py --config-name=stage2 "$@"
echo "=== done ==="
