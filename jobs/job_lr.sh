#!/bin/bash
#SBATCH --job-name=madgrav_lr
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Likelihood-ratio ranking over the slides. CPU only, and deliberately WITHOUT the
# single-thread BLAS pinning the other jobs use: the inner loop is one big matrix
# product per lag (the restricted-lag coherence), which is exactly what threaded BLAS is
# for. Pinning it to one thread here would cost an order of magnitude.
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
cd "$PROJ"
mkdir -p runs/_logs
# First argument is the script, as in job_far.sh, so this one job covers both the fit
# and the measurement.
SCRIPT="${1:-scripts/far_lr.py}"; [ -f "$SCRIPT" ] && shift || SCRIPT=scripts/far_lr.py
echo "=== madgrav LR on $(hostname) | ${SLURM_CPUS_PER_TASK:-?} cores | $SCRIPT | args: $* ==="
$PROJ/.venv/bin/python -u "$SCRIPT" "$@"
echo "=== done ==="
