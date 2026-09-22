#!/bin/bash
#SBATCH --job-name=madgrav_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
#SBATCH --gres=gpu:1
# Training-throughput measurement: batch size, mixed precision, memory format. Decides
# the operating point that the searched hyperparameters are then tuned around; it is not
# itself a hyperparameter sweep.
#
#   scripts/remote.sh sbatch jobs/job_bench.sh
#   scripts/remote.sh sbatch jobs/job_bench.sh --channels 2 --size 512 256
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
PROJ="$PROJECT_DIR"
PY=python                       # env activated by sites/activate.sh
cd "$PROJ"
mkdir -p runs/_logs

echo "=== madgrav throughput bench on $(hostname) | args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
$PY -u scripts/bench_throughput.py "$@"
echo "=== done ==="
