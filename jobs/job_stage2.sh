#!/bin/bash
#SBATCH --job-name=madgrav_stage2
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
# Stage-2 margin fine-tune. Needs a stage-1 checkpoint:
#
#   sbatch jobs/job_stage2.sh model.init_from=runs/madgrav/<stage1_run>/models/model_best.pt
#
# m and lambda are the HPO targets (Phase 5); pass them as overrides:
#   sbatch jobs/job_stage2.sh model.margin=2.5 model.margin_weight=1.5 seed=1
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav stage-2 on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
$PY -u run.py --config-name=stage2 "$@"
echo "=== done ==="
