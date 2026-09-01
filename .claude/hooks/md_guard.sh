#!/usr/bin/env bash
# PreToolUse(Write) hook — block creation of NEW scattered notes/report docs.
#
# Ported from Foundational_Amplitudes, where the lesson was learned the hard way: the
# model violated "one centralized document" by rationalizing "it's a report, not
# guidance", and then abused a blanket notes/ exemption to drop a new .md there. So the
# rule is enforced at the moment of creation instead of relying on the model not to
# invent exceptions.
#
# Here it matters for a second reason. This project's durable record is `docs/results.tex`
# plus each run's `summary.json`, and `summary.json` is written by ExperimentRecord,
# which REFUSES an unsupportable claim. A findings-2026-09.md written next to some plots
# is a way to record a result while bypassing that validation entirely — no fold
# assignment, no FAR, no seed count, no measured parameter counts. That is the failure
# this blocks.
#
# Scope: Write (creation) of a NEW .md/.markdown/.tex/.rst that does not already exist.
# Editing or overwriting an existing file is always fine — that is not scattering.
# Exemptions: CLAUDE.md, README*, the harness plan dir, and .claude/{agents,commands,
# skills}/*.md, which are harness config in Claude Code's required format, not prose.
#
# Escape hatch (deliberate and auditable): to create an approved new doc, add its
# repo-relative or absolute path to .claude/md_allowlist.txt, one per line. That makes
# "the user approved this doc" an explicit recorded act rather than an in-the-moment
# rationalization.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
ALLOWLIST="$REPO/.claude/md_allowlist.txt"

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

case "$fp" in
  *.md|*.markdown|*.tex|*.rst) ;;
  *) exit 0 ;;
esac

case "$fp" in
  */CLAUDE.md|*/README.md|*/README*.md|*/.claude/plans/*) exit 0 ;;
  */.claude/agents/*.md|*/.claude/commands/*.md|*/.claude/skills/*.md) exit 0 ;;
esac

# The vendored upstream tree is not ours to police.
case "$fp" in */.reference/*) exit 0 ;; esac

# Only NEW files are "scattering".
[ -e "$fp" ] && exit 0

rel=${fp#"$REPO"/}
if [ -f "$ALLOWLIST" ]; then
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$line" ] && continue
    if [ "$line" = "$fp" ] || [ "$line" = "$rel" ]; then exit 0; fi
  done < "$ALLOWLIST"
fi

{
  echo "BLOCKED by md_guard: refusing to create the new doc file '$rel'."
  echo "CLAUDE.md ground rule #4: guidance lives in CLAUDE.md and findings live in docs/results.tex — never a fresh .md/.tex next to some code or plots. 'It's a report, not guidance' is NOT an exception."
  echo "For a RESULT there is a second reason: the record of a run is runs/<run>/summary.json, written by ExperimentRecord, which refuses to serialise a claim with no fold assignment, no measured parameter counts, no primary metrics, or a 'keep' verdict under three seeds. A hand-written notes file bypasses all of that."
  echo "Write a section into docs/results.tex instead. Only if the user has EXPLICITLY approved a brand-new file: record its path in .claude/md_allowlist.txt, then retry. Otherwise ask first."
} >&2
exit 2
