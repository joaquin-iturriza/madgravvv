#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Ten-second infrastructure check: env activates, GPU visible. Not a training run.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
echo "site=$CCORCH_SITE host=$(hostname) project=$PROJECT_DIR data=$DATA_DIR"
python -c "import sys,torch;print(f\"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
