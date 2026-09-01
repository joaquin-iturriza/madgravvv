#!/bin/bash
#SBATCH --job-name=madgrav_fetch
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Warm the strain cache from GWOSC. CPU partition -- this is a download, and holding a
# V100 idle on the network for an hour would be antisocial in a different way.
#
# WHY A JOB AND NOT A LOGIN NODE. The first attempt ran on a login node and was killed
# partway through with no message: the log simply stops after segment 30 of 46. Login
# nodes are for `--dry-run` and single-segment checks; an hour-long process belongs in
# the scheduler, where it also survives an ssh drop by construction.
#
# WHY --jobs 2 AND NOT 4. At four concurrent workers GWOSC rate-limited us -- 58
# "Too much trials for .../api/v2/event-versions" and 29 of 30 segments failing. gwpy
# issues an API query per fetch_open_data call, so concurrency and chunking multiply the
# request rate together. Two workers, one request per segment.
#
# Resumable: cached segments are skipped, so re-running after any failure is the
# recovery procedure. Training fold only by default -- see scripts/fetch_strain.py for
# why not downloading the evaluation fold is the cheapest enforcement of C4.
#
# Usage: scripts/remote.sh sbatch jobs/job_fetch_strain.sh [extra args]
set -e
PROJ=/sps/lpnhe/jiturrizaramirez01/madgrav
PY=$PROJ/.venv/bin/python
cd "$PROJ"
mkdir -p runs/_logs data_cache/strain

echo "=== madgrav strain fetch on $(hostname) | args: $* ==="
df -h /sps/lpnhe | tail -1
$PY -u scripts/fetch_strain.py --jobs 2 "$@"
echo "=== cache now ==="
ls data_cache/strain/*.npz 2>/dev/null | wc -l
du -sh data_cache/strain
