#!/bin/bash
#SBATCH --job-name=madgrav_tiles
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Precompute the noise-tile bank. CPU partition: the cost is the Q-transform (252 ms per
# tile, measured), which is pure CPU and embarrassingly parallel. A GPU would sit idle.
#
# 16 cores, not 32. The bottleneck is not arithmetic — it is the shared /sps filesystem
# during worker start-up, and BLAS oversubscription once running. The thread limits below
# matter: without them, 16 workers each spawning 16 BLAS threads is 256 threads fighting
# over 16 cores.
#
# Usage:
#   scripts/remote.sh sbatch jobs/job_build_tiles.sh
#   scripts/remote.sh sbatch jobs/job_build_tiles.sh --split hpo_val --n-tiles 2000 \
#       --out data_cache/tiles/val
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs

# One BLAS thread per worker: the parallelism is the process pool, not the linear algebra.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== madgrav tile build on $(hostname) | cores=${SLURM_CPUS_PER_TASK} | args: $* ==="
df -h /sps/lpnhe | tail -1
$PY -u scripts/build_tile_cache.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
echo "=== banks ==="
du -sh data_cache/tiles/* 2>/dev/null
