#!/usr/bin/env bash
#
# publish_main.sh — regenerate the public `main` branch as a curated subset of the
# `ccin2p3` development trunk, then push it. Ported from Foundational_Amplitudes.
#
# `main` is a BUILD ARTIFACT of `ccin2p3`: never edit it by hand and never merge
# `ccin2p3 -> main`. To change what is public, edit the PUBLIC_PATHS allowlist below and
# re-run. Anything not in the allowlist is removed from `main`; everything in it is
# synced from the trunk. History on `main` is preserved (each publish is a new commit).
#
# WHAT `main` IS FOR HERE. This project is a contribution back to MADGRAV, so `main` is
# what gets pointed at in that conversation: the retrainable front end, the evaluation
# harness, and the configs — usable in his structure, without our cluster paths, our
# SLURM scripts, our reviewer config or our lab notebook.
#
# VISIBILITY CAVEAT (inherited from FA, and it bites here). A single GitHub repo shares
# visibility across ALL branches. Stripping `main`'s tree does NOT hide anything: the
# `ccin2p3` trunk is equally public, so `jobs/`, `.claude/` and `docs/results.tex` are
# readable by anyone regardless of this allowlist. The allowlist controls what a reader
# lands on, not what they can reach. Anything that must actually stay private has to be
# gitignored (as `docs/improvement-plan.md` is) or live in a separate repo.
#
# Usage:  scripts/publish_main.sh [--no-push]
#
set -euo pipefail

SRC="ccin2p3"          # development trunk (source of truth)
DST="main"             # public branch (generated)
PUSH=1
[[ "${1:-}" == "--no-push" ]] && PUSH=0

# --- the public core: only these top-level paths are published ---------------
# Excluded on purpose: jobs/ and scripts/ (CC-IN2P3-specific), .claude/ (our reviewer and
# guard config), docs/ (the lab notebook and the gitignored plan), CLAUDE.md (this
# cluster's operating manual), figures/.
PUBLIC_PATHS=(
  .gitignore
  .gitattributes
  pyproject.toml
  README.md
  # --- core run path ---
  run.py
  src
  config
  # --- the tests are part of the contribution: they are what shows the fold
  #     discipline and the C2 budget are enforced rather than asserted ---
  tests
)

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# `main` must exist locally; seed it from origin/main (or from SRC) the first time.
if ! git show-ref --verify --quiet "refs/heads/${DST}"; then
  if git show-ref --verify --quiet "refs/remotes/origin/${DST}"; then
    git branch "${DST}" "origin/${DST}"
  else
    git branch "${DST}" "${SRC}"
  fi
fi

wt="$(mktemp -d)"
cleanup() { git worktree remove --force "$wt" >/dev/null 2>&1 || true; }
trap cleanup EXIT

git worktree add --force "$wt" "${DST}" >/dev/null

(
  cd "$wt"

  # 1. remove any tracked top-level entry that is not in the allowlist
  while IFS= read -r entry; do
    keep=0
    for p in "${PUBLIC_PATHS[@]}"; do [[ "$entry" == "$p" ]] && keep=1 && break; done
    [[ $keep -eq 0 ]] && git rm -rq "$entry"
  done < <(git ls-tree --name-only HEAD)

  # 2. sync every allowlisted path from the trunk (picks up new files + updates)
  for p in "${PUBLIC_PATHS[@]}"; do
    git checkout "${SRC}" -- "$p" 2>/dev/null || echo "warn: '$p' not found on ${SRC}, skipping"
  done

  git add -A
  if git diff --cached --quiet; then
    echo "main is already in sync with the public subset of ${SRC} — nothing to publish."
    exit 0
  fi

  git commit -q -m "Publish: sync public core from ${SRC}@$(git rev-parse --short "${SRC}")"
  echo "Published $(git rev-parse --short HEAD) on ${DST}."
  if [[ $PUSH -eq 1 ]]; then
    git push origin "${DST}"
  else
    echo "(--no-push: review, then 'git push origin ${DST}')"
  fi
)
