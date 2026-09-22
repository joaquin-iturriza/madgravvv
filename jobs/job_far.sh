#!/bin/bash
#SBATCH --job-name=madgrav_far
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Time slides + FAR curve. Pure numpy over the cached score series, so CPU only -- the
# expensive part (the Q-transform) was paid by the background scan.
#
#   scripts/remote.sh sbatch jobs/job_far.sh scripts/far_curve.py --n-lags 7000
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
cd "$PROJ"
mkdir -p runs/_logs
SCRIPT="$1"; shift
python -u "$SCRIPT" "$@"
echo "=== done ==="
