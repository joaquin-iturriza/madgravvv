#!/usr/bin/env bash
# Stop hook — nudge (once) when a turn ends with uncommitted changes (modified tracked
# files or new-but-not-ignored files; anything git status --porcelain reports).
#
# Why: CLAUDE.md says commit finished work as it lands, without asking. Relying on
# the model to remember is unreliable (it has silently ended turns with edits sitting
# uncommitted). This makes the CHECKPOINT deterministic without pretending a hook can
# author a commit: it never commits or stages anything — it only blocks the stop once
# and hands the reason back to the model, which then writes a proper, scoped commit
# (or, if this is a deliberate mid-task pause, ends the turn again to pass through).
#
# Escape from an infinite loop / mid-task pause: when a Stop hook blocks, the next stop
# arrives with stop_hook_active=true. We allow that one through, so the nudge fires at
# most once per stop sequence. `main` (generated artifact) and detached HEAD are skipped.
set -uo pipefail

# Already nudged this stop sequence -> let it through (prevents an infinite loop and is
# the deliberate-pause escape: end the turn again and it passes).
input="$(cat 2>/dev/null || true)"
case "$input" in
  *'"stop_hook_active"'*true*) exit 0 ;;
esac

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$REPO" 2>/dev/null || exit 0

br="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" || exit 0   # detached -> skip
[ -z "$br" ] && exit 0
[ "$br" = "main" ] && exit 0                                          # generated artifact

# Nothing uncommitted (respects .gitignore, incl. modified + new-but-not-ignored) -> ok.
[ -z "$(git status --porcelain 2>/dev/null)" ] && exit 0

printf '%s' '{"decision":"block","reason":"Turn ending with uncommitted changes on branch '"$br"'. Per CLAUDE.md, commit the finished work now (small scope, clear message, authored as the user, no AI attribution); run git status to see what changed and git add the intended paths. If this is a deliberate mid-task pause, end the turn again and this checkpoint will let it through."}'
exit 0
