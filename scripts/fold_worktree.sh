#!/usr/bin/env bash
# Fold a worktree's RESULTS back into the trunk before the worktree is deleted.
#
# WHY THIS EXISTS. Code in a worktree comes back through `git merge`. Everything else
# does not: run directories, predictions, checkpoints, figures and the fold audit trail
# are all gitignored, so they live only inside the worktree and die with
# `git worktree remove`. In Foundational_Amplitudes this already destroyed a 12-trial
# sweep, leaving a published figure that renders empty and cannot be rebuilt without
# re-running it on GPU.
#
# It is worse here than there. `runs/<run>/fold_audit.jsonl` is the evidence that the
# evaluation fold was never used for tuning — the thing offered to an external reviewer
# — and `summary.json` is the only record of what a run trained on. Losing those does
# not just cost a re-run: it makes the numbers unciteable, because nothing can prove the
# fold discipline held.
#
# Merging the branch is NOT enough. Run this before removing any worktree.
#
# Usage:
#   scripts/fold_worktree.sh ../wt-foo            # dry run: list what would be copied
#   scripts/fold_worktree.sh ../wt-foo --apply    # copy it into the trunk
#
# It copies only files MISSING from the trunk, or whose worktree mtime is newer than the
# trunk copy's. It never deletes and never overwrites a trunk file with an older or
# identical one, so running it twice is safe.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"

WT="${1:-}"
APPLY=0
[ "${2:-}" = "--apply" ] && APPLY=1
if [ -z "$WT" ]; then
  echo "usage: $0 <worktree-path> [--apply]" >&2
  exit 2
fi
[ -d "$WT" ] || { echo "no such worktree: $WT" >&2; exit 2; }
WT=$(cd "$WT" && pwd)
[ "$WT" = "$REPO" ] && { echo "refusing to fold the trunk into itself" >&2; exit 2; }

# Directories that carry results rather than source: gitignored (so a merge does not move
# them) and expensive or impossible to regenerate.
#
# data_cache/ is deliberately ABSENT. It is hundreds of GB of strain refetchable from
# GWOSC, so folding it would copy the one thing that is both enormous and genuinely
# regenerable. Warm the trunk's cache instead.
RESULT_DIRS="runs figures"
# Extensions worth folding when they turn up inside those dirs.
# yaml is not optional: config.yaml is the only record of what a run trained on. jsonl
# covers fold_audit.jsonl, which is the fold-discipline evidence. pt covers checkpoints,
# which are large but cost GPU-hours to reproduce.
RESULT_EXTS="json jsonl yaml yml npz npy png pdf csv pkl pt txt log out sh"

echo "folding results:  $WT"
echo "            into: $REPO"
[ "$APPLY" = 0 ] && echo "(DRY RUN -- pass --apply to copy)"
echo

# Tracked files come back through `git merge`; they are NOT at risk and must not be
# copied (their worktree mtime is just the checkout time, which would look "newer" and
# could clobber a trunk file with staler content). Only untracked/ignored files are lost.
tracked=$(mktemp)
git -C "$WT" ls-files > "$tracked" 2>/dev/null

n_new=0; n_newer=0; n_same=0; n_tracked=0; bytes=0
tmp=$(mktemp)
for d in $RESULT_DIRS; do
  [ -d "$WT/$d" ] || continue
  find "$WT/$d" -type f 2>/dev/null >> "$tmp"
done

while IFS= read -r src; do
  [ -f "$src" ] || continue
  ext="${src##*.}"
  case " $RESULT_EXTS " in *" $ext "*) ;; *) continue ;; esac
  rel="${src#"$WT"/}"
  if grep -qxF "$rel" "$tracked" 2>/dev/null; then
    n_tracked=$((n_tracked+1)); continue
  fi
  dst="$REPO/$rel"
  if [ ! -e "$dst" ]; then
    n_new=$((n_new+1))
    sz=$(stat -c%s "$src" 2>/dev/null || echo 0); bytes=$((bytes+sz))
    [ "$n_new" -le 20 ] && echo "  NEW    $rel"
    if [ "$APPLY" = 1 ]; then mkdir -p "$(dirname "$dst")" && cp -p "$src" "$dst"; fi
  elif [ "$src" -nt "$dst" ]; then
    n_newer=$((n_newer+1))
    [ "$n_newer" -le 10 ] && echo "  NEWER  $rel"
    if [ "$APPLY" = 1 ]; then cp -p "$src" "$dst"; fi
  else
    n_same=$((n_same+1))
  fi
done < "$tmp"
rm -f "$tmp" "$tracked"

echo
echo "  new in worktree      : $n_new  (~$((bytes/1024)) KiB)"
echo "  newer than trunk     : $n_newer"
echo "  already in trunk     : $n_same"
echo "  tracked (via merge)  : $n_tracked"
if [ "$APPLY" = 1 ]; then
  echo
  echo "COPIED. Commit anything that belongs in git, then the worktree is safe to remove:"
  echo "  git worktree remove $WT"
  # No "folded" record is written on purpose: worktree_fold_guard.sh re-checks the
  # filesystem instead, so an incomplete fold cannot be masked by a memo saying it is done.
elif [ $((n_new + n_newer)) -gt 0 ]; then
  echo
  echo "Nothing copied yet. Re-run with --apply before removing this worktree."
fi
