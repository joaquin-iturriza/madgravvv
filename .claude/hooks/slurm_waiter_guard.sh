#!/usr/bin/env bash
# Stop hook — refuse to end a turn with SLURM jobs in flight and no background waiter.
#
# Why: CLAUDE.md ("Waiting on jobs — always background, never hand-poll") says: submit, then
# launch `scripts/wait_for_slurm.sh <jids>` with run_in_background so the harness re-invokes
# me exactly once when the jobs finish. It explicitly forbids promising "I'll report when they
# land" WITHOUT a mechanism. The model (me) submitted two 16-trial sweeps and then did exactly
# that — no waiter, just a promise. That silently drops the result on the floor: the turn ends,
# nothing re-invokes me, and the user has to notice and prod.
#
# Rule enforced: if the site registry shows any of my runs queued/running and no
# wait_for_slurm.sh process is alive, block the Stop and make me launch the waiter.
#
# Escape hatch: `touch .claude/.no_waiter_needed` to allow one Stop with jobs in flight
# (deliberate fire-and-forget, e.g. the user said they'll check themselves). Auto-cleared.
set -uo pipefail
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BYPASS="$REPO/.claude/.no_waiter_needed"

if [ -f "$BYPASS" ]; then
  rm -f "$BYPASS"
  exit 0
fi

# Jobs live on three sites now; the `site` registry is the one place that knows
# them all. Refresh it (bounded), then list this project's non-terminal runs as
# "<run-id> <job> <STATE>". If the tool is missing or the sites are unreachable,
# fail open rather than hang the Stop.
command -v site >/dev/null 2>&1 || exit 0
timeout 90 site poll >/dev/null 2>&1 || true
jobs=$(timeout 30 site runs --project madgrav --limit 100 2>/dev/null \
       | awk '$4 ~ /^(SUBMITTED|PENDING|RUNNING|IDLE|CONFIGURING|UNKNOWN)$/ {print $1, $6, $4}' \
       | grep -viE 'prebuild' || true)
[ -z "$jobs" ] && exit 0

# A waiter alive FOR THESE RUNS? Only a `site poll` whose command line names the
# run id counts (another session's or project's poll proves nothing), or a run
# listed in .claude/.slurm_monitor_jobs by a Monitor, or the on-site waiter.
MON="$REPO/.claude/.slurm_monitor_jobs"
unwatched=""
for rid in $(awk '{print $1}' <<<"$jobs"); do
  if pgrep -f "site poll.*$rid" >/dev/null 2>&1 \
     || pgrep -f "wait_for_slurm.sh.*$(awk -v r="$rid" '$1==r {print $2}' <<<"$jobs")" >/dev/null 2>&1 \
     || { [ -f "$MON" ] && grep -qx "$rid" "$MON"; }; then
    continue
  fi
  unwatched="$unwatched $rid"
done
[ -z "$unwatched" ] && exit 0
jobs=$(awk -v u=" $unwatched " 'index(u, " "$1" ")' <<<"$jobs")
# Or a Monitor-tool watcher: on this WSL2 laptop Claude Code stops background Bash tasks
# within minutes ("low memory" with 6 GB free), so multi-hour jobs are watched with a
# persistent Monitor instead. It has no process name to pgrep, so arming one records the
# watched job ids in .claude/.slurm_monitor_jobs (one per line); every queued job must be
# listed there. Remove or rewrite the file when the watch ends.

n=$(printf '%s\n' "$jobs" | wc -l | tr -d ' ')
{
  echo "BLOCKED by slurm_waiter_guard: $n of your SLURM job(s) are still in the queue and NO"
  echo "background waiter is running. Ending the turn now means nothing will re-invoke you when"
  echo "they finish — the result gets dropped and the user has to chase it."
  echo ""
  printf '%s\n' "$jobs" | head -8 | sed 's/^/    /'
  [ "$n" -gt 8 ] && echo "    ... ($n total)"
  echo ""
  echo "CLAUDE.md (Waiting on jobs): submit, then launch the waiter IN THE BACKGROUND —"
  echo "    a background shell (run_in_background) that repeats \`site poll <run>\` every"
  echo "    30-60 s until the state is terminal, then reads \`site logs <run>\`;"
  echo "    or a persistent Monitor listing the run id(s) in .claude/.slurm_monitor_jobs."
  echo "Do NOT promise 'I'll report when they land' without that mechanism, and do NOT hand-poll."
  echo ""
  echo "If the jobs are genuinely fire-and-forget: touch .claude/.no_waiter_needed and stop again."
} >&2
exit 2
