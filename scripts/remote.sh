#!/usr/bin/env bash
# remote.sh — run a job-control command in the madgrav project dir on CC-IN2P3.
#
# Why this exists: the CC-IN2P3 admins do not allow AI/external-assistant sessions to
# run *on* the cluster (https://doc.cc.in2p3.fr/en/Daily-usage/users.html#ai-and-external-services-at-cnrs).
# So the assistant runs LOCALLY against an sshfs mount of the project, and only the few
# commands that genuinely need the scheduler (sbatch / squeue / sacct) are sent over
# ssh. All file editing, log reading and tailing happen on the local mount with no ssh
# at all. Keep what goes over the wire minimal and scheduler-only.
#
# Usage:
#   scripts/remote.sh sbatch --parsable jobs/job_stage1.sh
#   scripts/remote.sh squeue --me
#   scripts/remote.sh sacct -j <id> --format=JobID,State,ExitCode,Elapsed
#   scripts/remote.sh 'POLL=30 scripts/wait_for_slurm.sh <jobid>'   # background wait
#
# The command runs after `cd <project>` on the login node, so relative paths
# (jobs/..., runs/...) resolve exactly as a manual login-node submit would.
#
# Env overrides: CCIN2P3_HOST (ssh alias, default ccin2p3),
#                CCIN2P3_PROJ (remote project dir).
set -euo pipefail
HOST="${CCIN2P3_HOST:-ccin2p3}"
PROJ="${CCIN2P3_PROJ:-/sps/lpnhe/jiturrizaramirez01/madgrav}"

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/remote.sh <command to run in project dir on the cluster>" >&2
  exit 2
fi

# A single string arg passes through verbatim (so the remote shell sees env-prefixes
# like `POLL=30 ...`); several args are re-quoted individually.
if [ "$#" -eq 1 ]; then
  remote_cmd="$1"
else
  remote_cmd="$(printf '%q ' "$@")"
fi

exec ssh "$HOST" "cd '$PROJ' && ${remote_cmd}"
