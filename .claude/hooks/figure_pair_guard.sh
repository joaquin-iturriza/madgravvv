#!/usr/bin/env bash
# Stop hook — every figure must exist as BOTH .png and .pdf (matching basename).
#
# Ported from Foundational_Amplitudes: the user always wants each figure in both
# formats, and relying on the model to remember is unreliable (it shipped png-only).
# This checks at end of turn and blocks the stop if a figure touched THIS SESSION is
# missing its counterpart, so the missing format gets produced before finishing.
#
# CLAUDE.md states the rule under Experiment standards ("every figure as PDF + PNG").
# It is not cosmetic: the PNG is what gets read back in a session, and the PDF is what
# goes into results.tex and eventually to the upstream author. Producing one and not
# the other means the figure is either unreadable here or unusable there.
#
# Figures are gitignored (*.png, and *.pdf under docs/), so git cannot see them — this
# works off the filesystem plus an mtime marker, so ONLY figures created or modified
# since the last check are inspected. Pre-existing single-format figures are never
# retroactively flagged.
#
# Heavy trees are pruned: runs/, data_cache/, .reference/, .venv/, .git/.
#
# Escape hatch: list basenames or repo-relative paths to skip, one per line, in
# .claude/figure_pair_ignore.txt.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
MARKER="$REPO/.claude/.figpair_last"
IGNORE="$REPO/.claude/figure_pair_ignore.txt"
cd "$REPO" 2>/dev/null || exit 0

input="$(cat 2>/dev/null || true)"
case "$input" in *'"stop_hook_active"'*true*) exit 0 ;; esac

# First run: establish a baseline, do not retroactively flag existing figures.
if [ ! -e "$MARKER" ]; then
  : > "$MARKER"; exit 0
fi

# `command find`, not `find`: on this machine the user's profile defines `find` as a
# shell function wrapping bfs, and an exported function propagates into this hook. bfs
# honours a different prune dialect, so the guard walked into .venv/ and .reference/ and
# spat thousands of stat errors over sshfs. A hook must not depend on which find it gets.
#
# Directories are matched by NAME at any depth rather than by ./path, so a worktree's
# copy of runs/ or .venv/ is pruned too. Pruning matters here for the reason CLAUDE.md
# gives under Conventions: this is an sshfs mount, and .venv/ alone is tens of thousands
# of files, each stat costing a network round trip.
changed=$(command find . \
  \( -type d \( -name .git -o -name runs -o -name data_cache -o -name .reference \
       -o -name .venv -o -name worktrees -o -name __pycache__ \) \) -prune -o \
  -type f \( -name '*.png' -o -name '*.pdf' \) -newer "$MARKER" -print 2>/dev/null)
[ -z "$changed" ] && { : > "$MARKER"; exit 0; }

missing=""
for f in $changed; do
  base="${f%.*}"
  name=$(basename "$base")
  if [ -f "$IGNORE" ] && grep -qxF -e "$name" -e "${f#./}" -e "${base#./}" "$IGNORE" 2>/dev/null; then
    continue
  fi
  [ -f "$base.png" ] && [ -f "$base.pdf" ] && continue
  want="pdf"; [ -f "$base.pdf" ] && want="png"
  case "$missing" in *"$base.$want"*) continue ;; esac
  missing="$missing
    $base.$want   (missing; $base.$([ "$want" = pdf ] && echo png || echo pdf) exists)"
done

[ -z "$missing" ] && { : > "$MARKER"; exit 0; }

{
  echo "BLOCKED by figure_pair_guard: a figure written this session exists in only one format."
  printf '%s\n' "$missing"
  echo ""
  echo "CLAUDE.md, Experiment standards: every figure ships as BOTH .png and .pdf, same basename,"
  echo "same directory. The PNG is what gets read back in a session; the PDF is what goes into"
  echo "docs/results.tex and eventually to the upstream author. Save both in the same call:"
  echo "    fig.savefig(base + '.png', dpi=150); fig.savefig(base + '.pdf')"
  echo "and make the plotting helper emit both by default rather than fixing it per figure."
  echo ""
  echo "Genuine single-format exceptions go in .claude/figure_pair_ignore.txt (basename or path)."
} >&2
exit 2
