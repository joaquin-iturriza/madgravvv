#!/bin/bash
#SBATCH --job-name=madgrav_gate
#SBATCH --partition=gpu_v100
#SBATCH --account=lpnhe
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# The foreground half of an efficiency-at-fixed-FAR measurement: coincident injections
# through the same tiling and the same model as the background scan. Same reasoning for
# the partition as jobs/job_scan_background.sh -- CPU pool for the Q-transform, GPU for
# the calibrated forward pass.
#
#   scripts/remote.sh sbatch jobs/job_scan_injections.sh \
#       --checkpoint runs/madgrav/<run>/models/model_best.pt --n-injections 4000
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
cd "$PROJ"
mkdir -p runs/_logs data_cache/injections
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=== madgrav glitch gate on $(hostname) | args: $* ==="
$PROJ/.venv/bin/python -u scripts/apply_glitch_gate.py "$@"
echo "=== done ==="
