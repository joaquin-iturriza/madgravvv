#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit) hook — worktree reminder (ADVISORY, non-blocking).
#
# Why: "open a worktree for new feature work" is a proactive step with no natural
# trigger, so it gets forgotten. This supplies the trigger at the moment it matters:
# the first edit of trunk code on `ccin2p3`. It injects a reminder and lets the edit
# proceed, so a false positive on a quick edit costs one line, not a hard stop.
set -uo pipefail
REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

# Exempt meta / lightweight files that do not warrant a feature worktree.
case "$fp" in
  */.claude/*|*/CLAUDE.md|*.md|*/scratchpad/*|*/configs/*) exit 0 ;;
esac

# Only nudge when actually sitting on the trunk branch.
br=$(git -C "$REPO" symbolic-ref --quiet --short HEAD 2>/dev/null)
[ "$br" = "ccin2p3" ] || exit 0

msg="Worktree reminder: editing trunk file directly on ccin2p3. Per the git workflow, new experiment/feature work should go in a worktree (git worktree add worktrees/wt-<feat> -b <feat> ccin2p3). If this is a quick standalone edit, proceed; otherwise create a worktree first."
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":%s}}\n' \
  "$(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')"
exit 0
