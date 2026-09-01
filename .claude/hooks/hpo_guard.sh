#!/usr/bin/env bash
# PreToolUse(Bash) hook — block hand-rolled HYPERPARAMETER grids submitted via sbatch.
#
# Ported from Foundational_Amplitudes, where the model violated this repeatedly — even
# after an explicit instruction — by rationalizing "this isn't an HPO, it's a diagnostic
# to test a hypothesis" and then reaching for an sbatch --array over lr / lambda.
# Framing an HP question as a mechanism question does not stop it being an HP search.
#
# It corrupts conclusions. A 1-D grid over `margin` at fixed `margin_weight` and lr
# answers "best margin GIVEN those fixed values", not "does this work at its own HPs".
# The stage-2 HPs interact by construction — m and lambda trade off directly against
# each other — so a 1-D scan comes back flat and manufactures a false "the margin
# doesn't matter" verdict.
#
# HERE IT IS ALSO A C4 RISK, which is worse. An sbatch --array over an HP has no fold
# record: nothing opens FoldGuard.hpo(), nothing lands in fold_audit.jsonl, and nothing
# stops a trial being scored on the evaluation fold. The plan is explicit that
# hyperparameter optimization scored on the evaluation fold does not merely weaken a
# result, it invalidates the quoted FAR. The audit trail is the evidence shown to a
# reviewer, and a grid submitted this way leaves none.
#
# Scope: Bash commands that `sbatch` a script which BOTH (a) is a job array
# (#SBATCH --array) and (b) feeds a known HP a per-task shell variable. Arrays over
# NON-HP axes stay allowed — seeds, objective, representation, data tag, warm-start
# checkpoint, run name, ablation flags — because those are controlled ablations, not
# HP searches. jobs/job_seeds.sh is exactly such an array and must not trip this.
#
# Escape hatch (deliberate and auditable): if the user explicitly approves a one-off
# grid, add the script's path to .claude/hpo_grid_allowlist.txt.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
ALLOWLIST="$REPO/.claude/hpo_grid_allowlist.txt"

input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0

printf '%s' "$cmd" | grep -qE '(^|[^[:alnum:]_])sbatch([^[:alnum:]_]|$)' || exit 0

# Hyperparameters. Deliberately EXCLUDES run-design and ablation axes (seed, iterations,
# model.objective, representation, data.source) — arrays over those are legitimate.
HP_RE='(model\.(margin|margin_weight|dropout|mask_ratio)|training\.(lr|weight_decay|batchsize|clip_grad_norm|reduceplateau_factor|reduceplateau_patience|cosanneal_warmup_frac|cosanneal_eta_min))'

scripts=$(printf '%s' "$cmd" | tr ' \t\n' '\n\n\n' | grep -E '\.sh$' || true)

for s in $scripts; do
  path="$s"
  [ -f "$path" ] || path="$REPO/$s"
  [ -f "$path" ] || continue

  rel=${path#"$REPO"/}
  if [ -f "$ALLOWLIST" ]; then
    skip=0
    while IFS= read -r line; do
      line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
      [ -z "$line" ] && continue
      if [ "$line" = "$path" ] || [ "$line" = "$rel" ]; then skip=1; break; fi
    done < "$ALLOWLIST"
    [ "$skip" -eq 1 ] && continue
  fi

  grep -qE '^#SBATCH[[:space:]]+--array' "$path" || continue
  hit=$(grep -nE "${HP_RE}=[\"']?\\\$" "$path" | head -3 || true)
  [ -z "$hit" ] && continue

  {
    echo "BLOCKED by hpo_guard: '$rel' is a hand-rolled HYPERPARAMETER GRID (sbatch --array varying an HP)."
    echo "Offending line(s):"
    printf '%s\n' "$hit" | sed 's/^/    /'
    echo ""
    echo "Two separate problems:"
    echo "  1. C4. An array like this has no fold record — nothing opens FoldGuard.hpo(), nothing"
    echo "     lands in fold_audit.jsonl, and nothing prevents a trial being scored on the"
    echo "     evaluation fold. HPO scored on the evaluation fold does not weaken the result, it"
    echo "     invalidates the quoted FAR. Run the search inside 'with guard.hpo(label, trial=i)',"
    echo "     reading HPO_TRAIN / HPO_VAL only, so every trial is logged with its fold."
    echo "  2. A 1-D grid at fixed other-HPs cannot find a joint optimum. m and lambda trade off"
    echo "     directly against each other; scanning one at a fixed value of the other will come"
    echo "     back flat and manufacture a false 'this HP does not matter' conclusion."
    echo ""
    echo "Reframing an HP question as 'just a diagnostic / just testing a mechanism' does not make"
    echo "it a controlled ablation. Arrays over seeds, objective, representation, data source or"
    echo "ablation flags are still fine — jobs/job_seeds.sh is one and does not trip this."
    echo "If the user EXPLICITLY approved this one-off grid: add '$rel' to .claude/hpo_grid_allowlist.txt, then retry."
  } >&2
  exit 2
done

exit 0
