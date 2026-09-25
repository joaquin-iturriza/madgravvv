#!/usr/bin/env bash
# PreToolUse(Bash) hook — the agent never hand-picks a cluster, and never piles a batch on one.
#
# Why: ~/work/CLAUDE.md says work is placed by `site pick <project> [--n K]` (free GPUs,
# queue ahead, my own queued jobs, deployed + verified setup; Jean Zay only when asked).
# The model still typed `site submit ccin2p3 ...` from habit, or picked once and sent 60
# sweeps to that one site until its queue was full. CLAUDE.md alone did not stop either.
#
# Rule: `site submit auto ...` is always fine (it places each job live). A named
# submission uses one unit of the quota the last `site pick <project>` left in
# ~/.local/share/ccorch/placement/<project>.json (`--n K` plans K units across sites;
# default 1), valid for 60 minutes. Counted as submissions: `site submit <site> <project>`
# and `site run <site> <project> -- ... sweep_manager.py submit|sbatch|condor_submit...`.
# Raw sbatch / condor_submit from a local session are refused outright.
set -uo pipefail
PLACEMENT_DIR="$HOME/.local/share/ccorch/placement"
MAX_AGE=3600

input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0

if printf '%s' "$cmd" | grep -qE '(^|[;&|(]|\$\()[[:space:]]*(sbatch|condor_submit|condor_submit_dag)([^[:alnum:]_]|$)'; then
  {
    echo "BLOCKED by placement_guard: raw sbatch/condor_submit. Submit through the site tool so the"
    echo "run is registered and placed:   site pick <project> [--n K]   then   site submit auto <project> <script> [-- args]"
  } >&2
  exit 2
fi

subs=$( { printf '%s' "$cmd" | grep -oE '(^|[;&|(]|\$\()[[:space:]]*site[[:space:]]+submit[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+' \
           | sed -E 's/.*site[[:space:]]+submit[[:space:]]+//';
         if printf '%s' "$cmd" | grep -qE 'sweep_manager\.py[[:space:]]+submit|generate_sweep\.py[^;|&]*--submit|condor_submit|sbatch'; then
           printf '%s' "$cmd" | grep -oE '(^|[;&|(]|\$\()[[:space:]]*site[[:space:]]+run[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+' \
             | sed -E 's/.*site[[:space:]]+run[[:space:]]+//';
         fi; } | sort -u )
[ -z "$subs" ] && exit 0

while read -r site project; do
  [ -z "$site" ] && continue
  case "$site" in auto|-*) continue ;; esac
  f="$PLACEMENT_DIR/$project.json"
  verdict=$(python3 - "$f" "$site" "$MAX_AGE" <<'PY'
import json, sys, time, os
f, site, max_age = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not os.path.exists(f):
    print("no placement recorded for this project"); sys.exit(0)
try:
    d = json.load(open(f))
except Exception as e:
    print("unreadable placement file: %s" % e); sys.exit(0)
age = int(time.time()) - int(d.get("ts", 0))
quota = d.get("quota") or ({d["site"]: 1} if d.get("site") else {})
if age > max_age:
    print("last `site pick` is %d min old; pick again" % (age // 60)); sys.exit(0)
left = int(quota.get(site, 0))
if left <= 0:
    have = ", ".join("%s %d" % kv for kv in quota.items() if kv[1] > 0) or "nothing"
    print("no quota left for %r in the last plan (left: %s)" % (site, have)); sys.exit(0)
quota[site] = left - 1
d["quota"] = quota
json.dump(d, open(f, "w"))
print("ok")
PY
)
  [ "$verdict" = "ok" ] && continue
  {
    echo "BLOCKED by placement_guard: submission to '$site' for '$project' — $verdict."
    echo "Placement is not the agent's call, and a batch is spread over sites by headroom. Run:"
    echo "    site pick $project --n <number of jobs/trials>     # or: site submit auto $project <script> [-- args]"
    echo "and submit only what the plan allots to each site (--allow-jeanzay ONLY if the user asked for Jean Zay)."
  } >&2
  exit 2
done <<< "$subs"

exit 0
