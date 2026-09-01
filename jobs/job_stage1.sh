#!/bin/bash
#SBATCH --job-name=madgrav_stage1
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Stage-1 self-supervised CAE: the reimplementation of the front end the upstream
# release does not ship. Reproducing it is the gate in front of every experiment.
#
# Usage: sbatch [overrides] jobs/job_stage1.sh [hydra overrides...]
#   sbatch jobs/job_stage1.sh seed=42
#   sbatch jobs/job_stage1.sh model.objective=masked model.mask_patch=[256,8]
#   sbatch --partition=gpu_h100 --gres=gpu:h100:1 jobs/job_stage1.sh
#
# CC-IN2P3 (Lyon) specifics:
#   * SLURM; GPU partitions gpu_v100 (32 GB, 16 nodes, default) and gpu_h100 (80 GB,
#     3 nodes, scarce). QOS gpu, account lpnhe.
#   * Compute nodes have internet and mount /sps, so GWOSC fetches work from a job —
#     but pre-warm data_cache/strain on a login node so runs only read it. Refetching
#     the same strain from 30 parallel jobs is antisocial and slow.
#   * The venv is self-contained (scripts/setup_env.sh); no `module load` needed here.
#   * Keep caches OFF /pbs/home (tiny). Everything lives under /sps.
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav stage-1 on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
$PY -u run.py exp_type=stage1 "$@"
echo "=== done ==="
