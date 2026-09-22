#!/bin/bash
#SBATCH --job-name=madgrav_gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
#SBATCH --gres=gpu:1
# The foreground half of an efficiency-at-fixed-FAR measurement: coincident injections
# through the same tiling and the same model as the background scan. Same reasoning for
# the partition as jobs/job_scan_background.sh -- CPU pool for the Q-transform, GPU for
# the calibrated forward pass.
#
#   site submit <site> madgrav jobs/job_scan_injections.sh \
#       --checkpoint runs/madgrav/<run>/models/model_best.pt --n-injections 4000
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
cd "$PROJ"
mkdir -p runs/_logs data_cache/injections
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=== madgrav glitch gate on $(hostname) | args: $* ==="
python -u scripts/apply_glitch_gate.py "$@"
echo "=== done ==="
