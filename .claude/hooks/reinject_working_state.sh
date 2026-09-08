#!/usr/bin/env bash
# SessionStart(matcher=compact) — put the working state back into context.
#
# This hook's stdout is the ONE mechanism whose output survives a compaction, so it is
# the whole reason .claude/working_state.md is worth maintaining. It therefore must not
# fail quietly: a `cat ... 2>/dev/null` with a relative path produces nothing whenever
# cwd is not the project root, and the failure is invisible by construction -- context
# simply comes back thinner with no indication why.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || { echo "[working-state] cannot cd to project dir"; exit 0; }
NOTES=".claude/working_state.md"
if [ ! -f "$NOTES" ]; then
  echo "[working-state] $NOTES is missing; nothing was carried across this compaction."
  exit 0
fi
cat "$NOTES"
