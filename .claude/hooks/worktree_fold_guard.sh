#!/usr/bin/env bash
# PreToolUse(Bash) hook — refuse to delete a worktree whose RESULTS have not been folded
# back into the trunk.
#
# Ported from Foundational_Amplitudes, where it exists because the loss already happened.
# `git merge` brings back a worktree's CODE and nothing else. Run directories,
# predictions, checkpoints and figures are gitignored, so they exist only inside the
# worktree and are destroyed by `git worktree remove` / `rm -rf`. A 12-trial sweep behind
# a published figure was lost that way; the figure's panel now renders empty and cannot
# be rebuilt without re-running it on GPU.
#
# It matters more here. `runs/<run>/fold_audit.jsonl` is the evidence that the evaluation
# fold was never used for tuning, and `summary.json` is the only record of what a run
# trained on. Losing them does not merely cost a re-run — it makes the numbers
# unciteable, because nothing remains to show the fold discipline held.
#
# The hook blocks the deletion and points at scripts/fold_worktree.sh. It then re-checks
# the FILESYSTEM rather than trusting a memo: a name-keyed "already folded" record goes
# stale as soon as a worktree of the same name is recreated, and it cannot notice an
# INCOMPLETE fold, which is how a missing extension nearly destroyed 327 config files
# while the guard reported everything fine.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
FOLDER="$REPO/scripts/fold_worktree.sh"

# Single source of truth for WHICH files count as results: read RESULT_EXTS out of the
# fold script rather than keeping a second list here. Two hand-maintained lists drifting
# apart is precisely what caused the original data loss.
EXTS=$(sed -n 's/^RESULT_EXTS="\(.*\)"$/\1/p' "$FOLDER" 2>/dev/null)
[ -z "$EXTS" ] && EXTS="json jsonl yaml yml npz npy png pdf csv pkl pt txt log out sh"
DIRS=$(sed -n 's/^RESULT_DIRS="\(.*\)"$/\1/p' "$FOLDER" 2>/dev/null)
[ -z "$DIRS" ] && DIRS="runs figures"

input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0

# Only look at commands that destroy a worktree.
case "$cmd" in
  *"worktree remove"*) ;;
  *rm\ -rf\ *wt-*|*rm\ -fr\ *wt-*) ;;
  *) exit 0 ;;
esac

# Which worktree(s)? Ask git for the real list and match its paths against the command,
# rather than guessing at a naming convention.
paths=$(git -C "$REPO" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}')
[ -z "$paths" ] && exit 0

for wt in $paths; do
  [ "$wt" = "$REPO" ] && continue
  base=$(basename "$wt")
  case "$cmd" in *"$wt"*|*"$base"*) ;; *) continue ;; esac
  [ -d "$wt" ] || continue

  pending=""
  targets=""
  for d in $DIRS; do [ -d "$wt/$d" ] && targets="$targets $wt/$d"; done
  [ -z "$targets" ] && continue

  while IFS= read -r src; do
    ext="${src##*.}"
    case " $EXTS " in *" $ext "*) ;; *) continue ;; esac
    rel="${src#"$wt"/}"
    # tracked files come back through the merge; only untracked results are at risk
    if git -C "$wt" ls-files --error-unmatch "$rel" >/dev/null 2>&1; then continue; fi
    if [ ! -e "$REPO/$rel" ]; then pending="$rel"; break; fi
  done < <(find $targets -type f 2>/dev/null)
  [ -z "$pending" ] && continue

  {
    echo "BLOCKED by worktree_fold_guard: worktree '$base' still holds results that exist"
    echo "NOWHERE ELSE. Merging the branch does not move them — they are gitignored, so"
    echo "removing the worktree destroys them permanently."
    echo
    echo "That includes fold_audit.jsonl and summary.json, which are the evidence that the"
    echo "evaluation fold was never used for tuning. Losing those does not just cost a re-run;"
    echo "it makes the run's numbers unciteable."
    echo
    echo "Fold them into the trunk first:"
    echo "  bash scripts/fold_worktree.sh $wt              # see what would be copied"
    echo "  bash scripts/fold_worktree.sh $wt --apply      # copy them"
    echo
    echo "First still-missing file: $pending"
    echo "Then retry the removal."
  } >&2
  exit 2
done
exit 0
