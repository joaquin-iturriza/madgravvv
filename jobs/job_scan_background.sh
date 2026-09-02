#!/bin/bash
#SBATCH --job-name=madgrav_bgscan
#SBATCH --partition=gpu_v100
#SBATCH --account=lpnhe
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/_logs/%x_%A_%a.out
#SBATCH --error=runs/_logs/%x_%A_%a.out
# Background scan: CPU pool builds tiles, GPU scores them.
#
# Both resources are needed at once and neither partition alone is right. The
# Q-transform is ~250 ms of pure CPU per tile and dominates; the CAE forward is
# milliseconds on a GPU and ~60 ms on a CPU, which would triple the wall clock. And the
# README is explicit that the GPU forward is the calibrated path for the frozen weights,
# so any number meant to be comparable with them has to be produced on one.
#
#   scripts/remote.sh sbatch --array=0-3 jobs/job_scan_background.sh \
#       --checkpoint runs/madgrav/<run>/models/model_best.pt
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
cd "$PROJ"
mkdir -p runs/_logs data_cache/background
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=== madgrav background scan on $(hostname) | shard ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} | args: $* ==="
$PROJ/.venv/bin/python -u scripts/scan_background.py \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
echo "=== done ==="
