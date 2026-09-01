#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit) hook — the C1-C5 hard constraints, mechanically.
#
# The improvement plan lists five constraints that are the upstream author's design
# decisions. A proposal that breaks one is not usable by him however good the metric,
# and the failure is silent: a run that violates C5 trains fine and produces plausible
# numbers that are simply not comparable to his. So the two constraints that CAN be
# checked mechanically are checked here, at the moment the code is written.
#
# What this hook does NOT do: judge C1 (no multi-detector information upstream of the
# per-detector score), C2 (parameter budget) or C3 (stage 1 stays unsupervised). Those
# are semantic and belong to the repo-reviewer and to the runtime check in
# models/param_budget.py. A grep that tried would fire on every mention of the words.
#
# DENY on:
#   C5  — an `ml4gw` import or dependency. The upstream README says so explicitly: its
#         whitening changes the coherence statistic and therefore the results.
#
# WARN (additionalContext, non-blocking) on:
#   headline AUC/ROC in an experiment or a report path. AUC is a legitimate development
#   diagnostic and an illegitimate headline; the hook cannot tell which one is being
#   written, so it reminds rather than blocks.
#
# FAIL OPEN. settings.json invokes this as `... || true`, on purpose: a deny is stdout
# JSON with exit 0, so the suffix cannot swallow one, and a syntax error here must not
# make every Edit fail — including the edit that would fix this file.
set -uo pipefail

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

# The vendored upstream tree is not ours to police.
case "$fp" in */.reference/*) exit 0 ;; esac

payload=$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    ti = json.load(sys.stdin).get("tool_input",{})
    parts = [ti.get("content",""), ti.get("new_string","")]
    parts += [e.get("new_string","") for e in ti.get("edits",[]) or []]
    print("\n".join(p for p in parts if isinstance(p,str)))
except Exception: print("")' 2>/dev/null)
[ -z "$payload" ] && exit 0

emit() {  # $1 = deny|context, $2 = message
  esc=$(printf '%s' "$2" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
  if [ "$1" = deny ]; then
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}\n' "$esc"
  else
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":%s}}\n' "$esc"
  fi
}

# --- C5: no ml4gw -----------------------------------------------------------
if printf '%s' "$payload" | grep -Eq '(^|[^A-Za-z0-9_])(import[[:space:]]+ml4gw|from[[:space:]]+ml4gw|["'"'"']ml4gw)'; then
  emit deny "BLOCKED by constraint C5: this adds an ml4gw import or dependency. The upstream MADGRAV README says explicitly not to add ml4gw — its whitening differs, which changes the coherence statistic and therefore the results, so anything trained or measured through it is not comparable to the author's numbers. Whitening lives in src/madgrav_ml/data/representation.py and is deliberately numpy/scipy/gwpy only. If you need something ml4gw provides, implement it there or use gwpy. (ml4gw and Aframe remain fine as EXTERNAL BENCHMARKS to compare against in docs/results.tex — just not as a dependency of this pipeline.)"
  exit 0
fi

# --- AUC as a headline ------------------------------------------------------
case "$fp" in
  */experiments/*|*/report/*|*/docs/*)
    if printf '%s' "$payload" | grep -Eq 'roc_auc_score|roc_curve|average_precision_score|\bAUC\b'; then
      emit context "Reminder (plan section 3.1): AUC/ROC is a development diagnostic here, never a headline. The accepted currency is detection efficiency at fixed FAR and sensitive volume VT, with a single-detector variant of each because constraint C1 makes per-detector sensitivity the primary target. If this AUC is going into secondary_metrics for development, carry on; if it is about to be quoted as a result, it should not be."
      exit 0
    fi
    ;;
esac

exit 0
