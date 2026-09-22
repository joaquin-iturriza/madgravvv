#!/bin/bash
#SBATCH --job-name=madgrav_stage1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
#SBATCH --gres=gpu:1
# Stage-1 self-supervised CAE on the precomputed tile bank.
#
# This is the second half of the section 3.4 reproduction gate: a reimplemented stage-1
# run whose score distribution can be compared against the vendored weights on the same
# held-out tiles. It is not a proposal -- nothing here is meant to beat anything. It is
# the baseline every later comparison is made against, so an unexplained gap here
# poisons everything downstream.
#
# Usage:
#   site submit <site> madgrav jobs/job_stage1_bank.sh
#   site submit <site> madgrav jobs/job_stage1_bank.sh model.objective=masked
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
PY=python                       # env activated by sites/activate.sh
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav stage-1 on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u run.py exp_type=stage1 data=tiles "$@"
echo "=== done ==="
