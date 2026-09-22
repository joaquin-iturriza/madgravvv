#!/bin/bash
#SBATCH --job-name=madgrav_seeds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=runs/_logs/%x_%A_%a.out
#SBATCH --error=runs/_logs/%x_%A_%a.out
#SBATCH --gres=gpu:1
# Seed array. No claimed improvement is reportable from a single seed — the record
# refuses a 'keep' verdict below three — so this is the normal way to run anything
# that will be quoted, not an extra step.
#
#   sbatch jobs/job_seeds.sh exp_type=stage1 model.objective=masked
#   sbatch --array=0-4 jobs/job_seeds.sh          # 5 seeds, matching the glitch arm
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
PY=python                       # env activated by sites/activate.sh
cd "$PROJ"
mkdir -p runs/_logs

SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "=== madgrav seed $SEED on $(hostname) | args: $* ==="
$PY -u run.py seed=$SEED tag=seed$SEED "$@"
echo "=== done ==="
