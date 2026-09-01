#!/usr/bin/env bash
# PreToolUse(Bash|Write) hook — refuse to configure a RUN with plotting disabled.
#
# Ported from Foundational_Amplitudes, where the cost was measured. The model shipped a
# plotting-disable flag across seven scripts and both sweep configs, copy-pasted from an
# old hand-off script, never once checking what it gated. It gated the one curve showing
# whether the quantity under investigation was descending or plateaued — during a live
# investigation into exactly that. Roughly twenty GPU-hours and six wrong hypotheses
# went into reconstructing by hand a plot that had been switched off.
#
# The failure mode is not "chose to disable plots". It is "carried a flag forward
# without ever asking what it does", which by construction the model is not thinking
# about. Only a hook catches that.
#
# It bites harder here, because the plots are the diagnostics for things no scalar
# shows: whether the stage-2 margin term is actually separating the two score
# distributions or just inflating both, and where on the efficiency-vs-total-mass curve
# a change helped. A run that trained fine with plotting off has to be repeated.
#
# Fires when a real RUN is configured with plotting off: Bash invoking run.py with such
# an override, Bash sbatch of a script carrying one, or Write of a .sh/.yaml whose
# content carries one. Deliberately does NOT fire on grep/sed/rg/cat/git, so searching
# for or REMOVING the flag stays possible, and never polices .claude/hooks/ (its own
# source) or the vendored .reference/ tree.
#
# Escape hatch: .claude/plot_disable_allowlist.txt (one substring or path per line).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
ALLOWLIST="$REPO/.claude/plot_disable_allowlist.txt"

# Pattern for "plotting turned off", assembled so this file never contains a literal match.
K1='plot'
OFF_RE="(^|[[:space:]])${K1}=(false|False)|(^|[[:space:]])${K1}:[[:space:]]*'?(false|False)|${K1}ting\.[a-zA-Z_]+=(false|False)|${K1}ting\.[a-zA-Z_]+:[[:space:]]*'?(false|False)"

input=$(cat)

# One extraction, three fields, each on its own line and the body last — `read` splits on
# whitespace, so packing a multi-word command into one `read` variable silently truncates
# it to its first token (which is how the first version of this hook never fired).
parsed=$(printf '%s' "$input" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get("tool_input", {}) or {}
    cmd = ti.get("command", "") or ""
    fp = ti.get("file_path", "") or ""
    content = ti.get("content", "") or ""
    if cmd:
        print("bash"); print("-"); print(cmd)
    elif fp:
        print("write"); print(fp); print(content)
    else:
        print("none"); print("-"); print("")
except Exception:
    print("none"); print("-"); print("")' 2>/dev/null)

tool=$(printf '%s' "$parsed" | sed -n '1p')
path=$(printf '%s' "$parsed" | sed -n '2p')
text=$(printf '%s' "$parsed" | tail -n +3)
[ "${tool:-none}" = "none" ] && exit 0
[ -z "$text" ] && exit 0

# Never police this hook's own source, or the vendored upstream tree.
case "${path:-}" in */.claude/hooks/*|*/.reference/*) exit 0 ;; esac

if [ "$tool" = "bash" ]; then
  # Searching for, or cleaning up, the flag must stay possible.
  printf '%s' "$text" | grep -qE '(^|[[:space:]|;])(grep|rg|sed|awk|cat|less|git|find)([[:space:]]|$)' && exit 0
  # Only a real run: run.py directly, or an sbatch of a script that carries the flag.
  if printf '%s' "$text" | grep -qE 'run\.py'; then
    :
  elif printf '%s' "$text" | grep -qE '(^|[^[:alnum:]_])sbatch([^[:alnum:]_]|$)'; then
    for s in $(printf '%s' "$text" | tr ' \t\n' '\n\n\n' | grep -E '\.sh$' || true); do
      p="$s"; [ -f "$p" ] || p="$REPO/$s"; [ -f "$p" ] || continue
      text="$text
$(cat "$p")"
    done
  else
    exit 0
  fi
else
  case "${path:-}" in *.sh|*.yaml|*.yml) ;; *) exit 0 ;; esac
fi

printf '%s' "$text" | grep -qE "$OFF_RE" || exit 0

if [ -f "$ALLOWLIST" ]; then
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$line" ] && continue
    case "${path:-}$text" in *"$line"*) exit 0 ;; esac
  done < "$ALLOWLIST"
fi

{
  echo "BLOCKED by plot_guard: this configures a run with plotting disabled."
  echo "CLAUDE.md, Conventions: keep plotting ON. The plots are the diagnostics no scalar shows —"
  echo "whether the stage-2 margin is separating the two score distributions or just inflating both,"
  echo "and where on the efficiency-vs-total-mass curve a change actually helped. A run that trained"
  echo "fine with plotting off has to be repeated to answer the question you asked it."
  echo "If the user explicitly wants plots off for this one thing, add a matching substring or path to"
  echo ".claude/plot_disable_allowlist.txt, then retry."
} >&2
exit 2
