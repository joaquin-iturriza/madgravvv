#!/bin/bash
#SBATCH --job-name=madgrav_compare
#SBATCH --partition=gpu_v100
#SBATCH --account=lpnhe
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Score two front ends over the tile banks. GPU because the README is explicit that the
# GPU forward pass is the calibrated path for the frozen weights and CPU is not
# byte-identical -- a comparison against those weights must run where they are calibrated.
#
#   scripts/remote.sh sbatch jobs/job_compare.sh --checkpoint <run>/models/model_best.pt
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
cd "$PROJ"
mkdir -p runs/_logs
echo "=== madgrav compare on $(hostname) | args: $* ==="
$PROJ/.venv/bin/python -u scripts/compare_front_ends.py "$@"
echo "=== done ==="
