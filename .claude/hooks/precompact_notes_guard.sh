#!/usr/bin/env bash
# PreCompact hook — refuse to compact while the working-state notes are stale.
#
# WHY THIS EXISTS. Compaction discards everything not written down. Over a long
# autonomous run this session will compact many times, and each time the things that
# actually prevent repeated mistakes are the ones most easily lost: that the shipped LR
# coefficients leak on our background, that HPO_VAL is not clean background, that the
# lag ladder wraps at n/s-1, that a guard was written and never verified to fire. Those
# are not in the code and not in results.tex; they live in `.claude/working_state.md`
# and nowhere else.
#
# A hook cannot write them — it does not know what is in the model's head. What it CAN
# do is refuse to proceed while they are stale, which is what turns "I should keep notes"
# into "I cannot lose state without noticing". Same shape as slurm_waiter_guard: the
# guard does not do the work, it makes skipping the work impossible to do quietly.
#
# Staleness is measured against the last commit rather than wall-clock: a session that
# has committed since it last wrote notes has, by definition, done something worth
# recording.
#
# Escape hatch: `.claude/.no_notes_needed` — for a compaction that genuinely follows no
# new work. Adding it is a record that the user approved this.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

NOTES=".claude/working_state.md"
[ -f .claude/.no_notes_needed ] && exit 0

input=$(cat)
trigger=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("trigger",""))
except Exception: print("")' 2>/dev/null)

if [ ! -f "$NOTES" ]; then
  {
    echo "BLOCKED by precompact_notes_guard: $NOTES does not exist."
    echo
    echo "Compaction is about to discard the working context. Write the notes first:"
    echo "  - what is running right now (job ids, what they will produce)"
    echo "  - what was just decided and why, especially anything that overturned an"
    echo "    earlier conclusion"
    echo "  - what failed and the cause, so it is not retried"
    echo "  - the immediate next action"
    echo
    echo "Then compact again. (.claude/.no_notes_needed waives this.)"
  } >&2
  exit 2
fi

# Newer than the last commit? Then it reflects the work since that commit.
last_commit_epoch=$(git log -1 --format=%ct 2>/dev/null || echo 0)
notes_epoch=$(stat -c %Y "$NOTES" 2>/dev/null || echo 0)

if [ "$notes_epoch" -lt "$last_commit_epoch" ]; then
  age_min=$(( (last_commit_epoch - notes_epoch) / 60 ))
  {
    echo "BLOCKED by precompact_notes_guard: $NOTES is older than the last commit"
    echo "by ${age_min} min, so work has happened that it does not describe."
    echo
    echo "Update it before compacting — the summariser keeps the shape of the"
    echo "conversation, not the reasons behind the decisions in it. Record:"
    echo "  - jobs in flight and what they answer"
    echo "  - conclusions that CHANGED, and what changed them"
    echo "  - failures and their causes"
    echo "  - the next concrete action"
    echo
    echo "trigger=${trigger:-unknown}"
  } >&2
  exit 2
fi
exit 0
