#!/bin/bash
#SBATCH --job-name=madgrav_stage1
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Stage-1 self-supervised CAE on the precomputed tile bank.
#
# This is the second half of the section 3.4 reproduction gate: a reimplemented stage-1
# run whose score distribution can be compared against the vendored weights on the same
# held-out tiles. It is not a proposal -- nothing here is meant to beat anything. It is
# the baseline every later comparison is made against, so an unexplained gap here
# poisons everything downstream.
#
# Usage:
#   scripts/remote.sh sbatch jobs/job_stage1_bank.sh
#   scripts/remote.sh sbatch jobs/job_stage1_bank.sh model.objective=masked
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav stage-1 on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u run.py exp_type=stage1 data=tiles "$@"
echo "=== done ==="
