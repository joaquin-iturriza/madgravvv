#!/bin/bash
# madgrav: the GWOSC strain cache under $DATA_DIR/strain (~262 GB for O3a). Lives on CC-IN2P3 by design; fetched by a job.
# Contract (run by `site env`, checked by `site pick`): with sites/activate.sh sourced,
#   bash sites/setup.sh            prepare this site for the project (idempotent)
#   bash sites/setup.sh --verify   fast, read-only: exit 0 iff runnable, one status line
set -u

verify() {
  local n; n=$(find "$DATA_DIR/strain" -maxdepth 2 -type f 2>/dev/null | head -5 | wc -l)
  if [ "$n" -eq 0 ]; then echo "missing: strain cache at $DATA_DIR/strain (site submit <site> madgrav jobs/job_fetch_strain.sh; CC-IN2P3 holds it)"; return 1; fi
  echo "ok: strain cache at $DATA_DIR/strain"
}
[ "${1:-}" = "--verify" ] && { verify; exit $?; }
mkdir -p "$DATA_DIR" runs/_logs
verify || true      # 262 GB is a decision, not a setup step
exit 0
