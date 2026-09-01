#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit) hook — block writes to the persistent-memory dir.
#
# Why: this project keeps ALL guidance in CLAUDE.md (ground rule #4). Auto-memory is
# disabled via settings (autoMemoryEnabled:false); this hook is the hard backstop so
# nothing — model or harness — silently re-scatters notes into the memory tree.
# Blocks the edit and tells the model to put it in CLAUDE.md instead.
set -uo pipefail

# Extract tool_input.file_path with sed only -- NO python/jq dependency (neither is guaranteed on
# the PATH the harness runs hooks with; a guard that needs an absent binary silently no-ops).
input=$(cat)
fp=$(printf '%s' "$input" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
[ -z "$fp" ] && exit 0

case "$fp" in
  */.claude/projects/*/memory/*|*/memory/MEMORY.md|*/memory/*.md)
    echo "Blocked: persistent memory is disabled for madgrav (all guidance lives in CLAUDE.md). Put this in CLAUDE.md (or docs/results.tex) instead." >&2
    exit 2
    ;;
esac

# Also block NEW top-level notes files (rule #4: one centralized CLAUDE.md, no scattered
# notes). A stray HANDOFF.md slipped through this hook once — the memory-dir pattern alone
# was too narrow. Allowed at the repo root: CLAUDE.md and README.md; everything else *.md
# belongs in docs/ (results.tex) or CLAUDE.md.
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
case "$fp" in
  "$root"/*.md)
    base=${fp#"$root"/}
    case "$base" in
      */*|CLAUDE.md|README.md) ;;   # subdir files and the allowed roots pass
      *)
        echo "Blocked: no new top-level .md files (CLAUDE.md rule #4 — one centralized location). Fold this into CLAUDE.md or docs/results.tex instead." >&2
        exit 2
        ;;
    esac
    ;;
esac
exit 0
