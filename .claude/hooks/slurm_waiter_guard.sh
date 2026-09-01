#!/usr/bin/env bash
# Stop hook — refuse to end a turn with SLURM jobs in flight and no background waiter.
#
# Ported from Foundational_Amplitudes. CLAUDE.md ("Waiting on jobs") says: submit, then
# launch the waiter with run_in_background so the harness re-invokes you exactly once
# when the jobs finish. It explicitly forbids promising "I'll report when they land"
# without a mechanism. The model there submitted two sweeps and did exactly that — no
# waiter, just a promise. That silently drops the result: the turn ends, nothing
# re-invokes anyone, and the user has to notice and prod.
#
# ADAPTED FOR THIS PROJECT'S EXECUTION MODEL. FA runs on the cluster, so it can call
# squeue directly. Here the assistant runs locally and drives SLURM over ssh, so the
# check has to go through scripts/remote.sh — one multiplexed round-trip per Stop.
# It FAILS OPEN on any ssh problem: a hook that blocks every turn because the mount or
# the network is down is worse than the miss it prevents.
#
# The waiter itself is a local `ssh ... wait_for_slurm.sh` process, so pgrep on the
# local side is the right place to look for it.
#
# Escape hatch: `touch .claude/.no_waiter_needed` to allow one Stop with jobs in flight
# (deliberate fire-and-forget, e.g. the user said they will check themselves).
# Auto-cleared on use.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
BYPASS="$REPO/.claude/.no_waiter_needed"
REMOTE="$REPO/scripts/remote.sh"

# Already blocked once in this stop sequence -> let it through (a Stop hook cannot block
# twice, and this is also the deliberate-pause escape).
input="$(cat 2>/dev/null || true)"
case "$input" in *'"stop_hook_active"'*true*) exit 0 ;; esac

if [ -f "$BYPASS" ]; then
  rm -f "$BYPASS"
  exit 0
fi

# A waiter alive locally? (the backgrounded ssh holding wait_for_slurm.sh)
pgrep -f "wait_for_slurm.sh" >/dev/null 2>&1 && exit 0

[ -x "$REMOTE" ] || exit 0

# Fail open: a timeout, a dead mount or a dropped key must not block the turn.
jobs=$(timeout 25 "$REMOTE" 'squeue --me -h -o "%i %j %T"' 2>/dev/null) || exit 0
jobs=$(printf '%s\n' "$jobs" | sed '/^[[:space:]]*$/d')
[ -z "$jobs" ] && exit 0

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
  echo "    scripts/remote.sh \"POLL=30 scripts/wait_for_slurm.sh <jid> [<jid> ...]\"   # run_in_background: true"
  echo "Do NOT promise 'I'll report when they land' without that mechanism, and do NOT hand-poll squeue."
  echo "Reading runs/_logs/*.out while a job runs is a local file op and is always fine."
  echo ""
  echo "If the jobs are genuinely fire-and-forget: touch .claude/.no_waiter_needed and stop again."
} >&2
exit 2
