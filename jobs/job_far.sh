#!/bin/bash
#SBATCH --job-name=madgrav_far
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Time slides + FAR curve. Pure numpy over the cached score series, so CPU only -- the
# expensive part (the Q-transform) was paid by the background scan.
#
#   scripts/remote.sh sbatch jobs/job_far.sh scripts/far_curve.py --n-lags 7000
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
cd "$PROJ"
mkdir -p runs/_logs
SCRIPT="$1"; shift
$PROJ/.venv/bin/python -u "$SCRIPT" "$@"
echo "=== done ==="
