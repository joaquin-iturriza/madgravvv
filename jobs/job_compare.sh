#!/bin/bash
#SBATCH --job-name=madgrav_compare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
#SBATCH --gres=gpu:1
# Score two front ends over the tile banks. GPU because the README is explicit that the
# GPU forward pass is the calibrated path for the frozen weights and CPU is not
# byte-identical -- a comparison against those weights must run where they are calibrated.
#
#   scripts/remote.sh sbatch jobs/job_compare.sh --checkpoint <run>/models/model_best.pt
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
cd "$PROJ"
mkdir -p runs/_logs
echo "=== madgrav compare on $(hostname) | args: $* ==="
python -u scripts/compare_front_ends.py "$@"
echo "=== done ==="
