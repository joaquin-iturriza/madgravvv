#!/usr/bin/env bash
# Vendor the upstream MADGRAV repo into .reference/ (gitignored, read-only context).
#
# We do not fork it and we do not commit it: this project is a contribution back to it,
# so the upstream tree stays a pristine checkout that can be diffed against. The
# vendored weights are what the C2 parameter budget and the section-3.4 reproduction
# gate are measured against.
#
# Foundational_Amplitudes is vendored at its **jeanzay** branch, not `main`. `main` there
# is a generated, stripped publication artifact: it carries no CLAUDE.md and no .claude/,
# so a default clone silently gives you a repo that looks like it has no project rules.
# The dev trunk is `jeanzay`, and that is where the guards this project's .claude/ is
# ported from actually live. (Getting this wrong once is how the ported hooks were nearly
# missed altogether.)
set -euo pipefail

vendor() {  # $1=dest  $2=url  $3=branch (optional)
  local dest="$1" url="$2" branch="${3:-}"
  mkdir -p "$(dirname "$dest")"
  if [ -d "$dest/.git" ]; then
    echo "updating $dest${branch:+ ($branch)}"
    if [ -n "$branch" ]; then
      git -C "$dest" fetch -q --depth 1 origin "$branch"
      git -C "$dest" checkout -q -B "$branch" FETCH_HEAD
    else
      git -C "$dest" pull --ff-only
    fi
  else
    echo "cloning $url${branch:+ ($branch)} -> $dest"
    if [ -n "$branch" ]; then
      git clone -q --filter=blob:none --depth 1 --branch "$branch" "$url" "$dest"
    else
      git clone -q --filter=blob:none --depth 1 "$url" "$dest"
    fi
  fi
}

vendor "${1:-.reference/MADGRAV}" "${MADGRAV_URL:-https://github.com/ginguglia/MADGRAV.git}"
vendor ".reference/Foundational_Amplitudes" \
       "${FA_URL:-https://github.com/joaquin-iturriza/Foundational_Amplitudes.git}" \
       "${FA_BRANCH:-jeanzay}"

echo
echo "vendored. Next:  python scripts/measure_param_budget.py"
