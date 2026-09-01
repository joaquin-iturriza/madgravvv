#!/usr/bin/env bash
# wait_for_slurm.sh — block cheaply until the given SLURM job(s) leave the queue, then
# print their final state and a tail of each log. Run this ON the cluster, via
# scripts/remote.sh, in the background:
#
#   jid=$(scripts/remote.sh sbatch --parsable jobs/job_stage1.sh)
#   scripts/remote.sh "POLL=30 scripts/wait_for_slurm.sh $jid"    # run_in_background
#
# With no job id it waits on all of the caller's jobs. Never poll this from the
# assistant side in a loop of tool calls — that is what this script exists to avoid.
set -uo pipefail
POLL="${POLL:-30}"
TAIL="${TAIL:-40}"
JOBS="$*"

while :; do
  if [ -n "$JOBS" ]; then
    n=$(squeue --noheader --job "${JOBS// /,}" 2>/dev/null | wc -l)
  else
    n=$(squeue --me --noheader 2>/dev/null | wc -l)
  fi
  [ "${n:-0}" -eq 0 ] && break
  sleep "$POLL"
done

echo "=== final state ==="
if [ -n "$JOBS" ]; then
  sacct -j "${JOBS// /,}" --format=JobID%20,JobName%24,State,ExitCode,Elapsed,MaxRSS
else
  sacct --starttime now-1days --format=JobID%20,JobName%24,State,ExitCode,Elapsed
fi

echo "=== log tails ==="
for j in $JOBS; do
  for f in runs/_logs/*_"${j}".out; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    tail -n "$TAIL" "$f"
  done
done
