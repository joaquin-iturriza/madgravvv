#!/usr/bin/env bash
# PreToolUse(Bash) hook — the agent never hand-picks a cluster.
#
# Why: ~/work/CLAUDE.md says work is placed by `site pick <project>` (availability, queue,
# deployed + verified setup, Jean Zay only when asked). The model still typed
# `site submit ccin2p3 ...` from habit, and CLAUDE.md alone did not stop it.
#
# Rule: `site submit auto ...` is always fine. `site submit <named site> <project> ...` is
# allowed only when the last `site pick <project>` (or `site submit auto`) chose THAT site
# within the last 60 minutes; `site` records that in ~/.local/share/ccorch/placement/<project>.json.
# Raw scheduler submissions (sbatch / condor_submit) from a local session are refused outright:
# they bypass the run registry as well as placement.
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

# raw scheduler calls in command position
if printf '%s' "$cmd" | grep -qE '(^|[;&|(]|\$\()[[:space:]]*(sbatch|condor_submit)([^[:alnum:]_]|$)'; then
  {
    echo "BLOCKED by placement_guard: raw sbatch/condor_submit. Submit through the site tool so the"
    echo "run is registered and placed:   site pick <project>   then   site submit auto <project> <script> [-- args]"
  } >&2
  exit 2
fi

# every `site submit <site> <project>` occurrence in the command
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
if d.get("site") != site:
    print("last `site pick` chose %r (%d min ago), not %r" % (d.get("site"), age // 60, site)); sys.exit(0)
if age > max_age:
    print("last `site pick` chose %r but %d min ago; pick again" % (site, age // 60)); sys.exit(0)
print("ok")
PY
)
  [ "$verdict" = "ok" ] && continue
  {
    echo "BLOCKED by placement_guard: 'site submit $site $project' — $verdict."
    echo "Placement is not the agent's call. Run:"
    echo "    site pick $project            # or: site submit auto $project <script> [-- args]"
    echo "and submit to the site it picks (add --allow-jeanzay ONLY if the user asked for Jean Zay)."
  } >&2
  exit 2
done < <(printf '%s' "$cmd" | grep -oE '(^|[;&|(]|\$\()[[:space:]]*site[[:space:]]+submit[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+' \
         | sed -E 's/.*site[[:space:]]+submit[[:space:]]+//')

exit 0
