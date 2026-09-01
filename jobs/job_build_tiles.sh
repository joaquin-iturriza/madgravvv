#!/bin/bash
#SBATCH --job-name=madgrav_tiles
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Precompute the noise-tile bank. CPU partition and many cores: the cost is the
# Q-transform (~4.6 s per tile, measured), which is pure CPU and embarrassingly parallel.
# A GPU would sit idle.
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

echo "=== madgrav tile build on $(hostname) | cores=${SLURM_CPUS_PER_TASK} | args: $* ==="
df -h /sps/lpnhe | tail -1
$PY -u scripts/build_tile_cache.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
echo "=== banks ==="
du -sh data_cache/tiles/* 2>/dev/null
