# Resolve the project root and this site's python env at RUNTIME.
# Source from any job script:
#     _CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
#     source "$_CCORCH_ROOT/sites/activate.sh"
# Only the env activation differs per cluster; this is the one place it is named.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PROJECT_DIR
if [ -z "${CCORCH_SITE:-}" ]; then
  case "$PROJECT_DIR" in
    /sps/*)        CCORCH_SITE=ccin2p3 ;;
    /lustre/*)     CCORCH_SITE=jeanzay ;;
    /eos/*|/afs/*) CCORCH_SITE=lxplus ;;
    *)             CCORCH_SITE=local ;;
  esac
fi
export CCORCH_SITE
case "$CCORCH_SITE" in
  ccin2p3)
    source "$PROJECT_DIR/.venv/bin/activate"
    export WORK="${WORK:-/sps/lpnhe/jiturrizaramirez01}"
    export SCRATCH="${SCRATCH:-/sps/lpnhe/jiturrizaramirez01/tmp}"
    export DATA_DIR="${DATA_DIR:-/sps/lpnhe/jiturrizaramirez01/madgrav/data_cache}"
    export SUBMIT_DIR="${SUBMIT_DIR:-/sps/lpnhe/jiturrizaramirez01/madgrav}"
    ;;
  jeanzay)
    module load pytorch-gpu/py3/2.6.0 2>/dev/null || true
    source "$PROJECT_DIR/.venv/bin/activate"
    export WORK="${WORK:-/lustre/fswork/projects/rech/itg/ulm49ia}"
    export SCRATCH="${SCRATCH:-/lustre/fsn1/projects/rech/itg/ulm49ia}"
    export DATA_DIR="${DATA_DIR:-/lustre/fswork/projects/rech/itg/ulm49ia/madgrav/data_cache}"
    export SUBMIT_DIR="${SUBMIT_DIR:-/lustre/fswork/projects/rech/itg/ulm49ia/madgrav}"
    ;;
  lxplus)
    source "$PROJECT_DIR/.venv/bin/activate"
    export WORK="${WORK:-/eos/user/j/joiturri}"
    export SCRATCH="${SCRATCH:-/eos/user/j/joiturri/tmp}"
    export DATA_DIR="${DATA_DIR:-/eos/user/j/joiturri/madgrav/data_cache}"
    export SUBMIT_DIR="${SUBMIT_DIR:-/afs/cern.ch/user/j/joiturri/madgrav}"
    ;;
  *)
    [ -d "$PROJECT_DIR/.venv" ] && source "$PROJECT_DIR/.venv/bin/activate"
    export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
    export SUBMIT_DIR="${SUBMIT_DIR:-$PROJECT_DIR}"
    ;;
esac
