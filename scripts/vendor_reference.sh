#!/usr/bin/env bash
# Vendor the upstream MADGRAV repo into .reference/ (gitignored, read-only context).
#
# We do not fork it and we do not commit it: this project is a contribution back to it,
# so the upstream tree stays a pristine checkout that can be diffed against. The
# vendored weights are what the C2 parameter budget and the section-3.4 reproduction
# gate are measured against.
set -euo pipefail
DEST="${1:-.reference/MADGRAV}"
URL="${MADGRAV_URL:-https://github.com/ginguglia/MADGRAV.git}"

mkdir -p "$(dirname "$DEST")"
if [ -d "$DEST/.git" ]; then
  echo "updating $DEST"
  git -C "$DEST" pull --ff-only
else
  echo "cloning $URL -> $DEST"
  git clone --filter=blob:none --depth 1 "$URL" "$DEST"
fi

echo
echo "vendored. Next:  python scripts/measure_param_budget.py"
