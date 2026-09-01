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

# --- provision: replace the upstream tree's dangling scratch symlinks -------------
#
# UPSTREAM PORTABILITY BUG, worth reporting back. MADGRAV commits nine symlinks under
# search_mode/ that point at the author's own scratch storage:
#
#     search_mode/streams_o4a -> /scratch/fs201312/gi54209/streams_o4a_evt56
#     search_mode/strain      -> /scratch/fs201312/gi54209/strain_o3a_56       (+ 7 more)
#
# On any machine that is not his they are dangling. That breaks `demo/run_demo.sh` on a
# fresh clone, because driver_streams.py does `os.makedirs(OUT, exist_ok=True)` at import
# and `exist_ok=True` does NOT swallow FileExistsError when the path exists as a dangling
# symlink rather than a directory. So the documented two-minute demo cannot run anywhere
# except the author's machine, and the failure looks like a bug in your environment.
#
# We replace them with real directories. They are run-output and scratch locations, not
# data we need: the demo reads its strain from demo/strain via SM_STRAIN.
#
# NOTE for a full search run (not the demo): an empty directory here means "not
# provisioned", not "no data found". If a search comes back with nothing, check this
# first rather than concluding the search found nothing.
provision_dangling_symlinks() {
  local root="$1" n=0
  [ -d "$root" ] || return 0
  while IFS= read -r link; do
    [ -e "$link" ] && continue          # resolves fine, leave it alone
    printf '  provisioning %s (was -> %s)\n' "${link#"$root"/}" "$(readlink "$link")"
    rm -f "$link" && mkdir -p "$link" && n=$((n+1))
  done < <(command find "$root" -type l 2>/dev/null)
  [ "$n" -gt 0 ] && echo "  replaced $n dangling symlink(s) with real directories"
  return 0
}

echo
echo "provisioning ${1:-.reference/MADGRAV}:"
provision_dangling_symlinks "${1:-.reference/MADGRAV}"

echo
echo "vendored. Next:  python scripts/measure_param_budget.py"
