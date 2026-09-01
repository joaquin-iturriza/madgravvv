#!/usr/bin/env bash
# review_backlog.sh — BATCHED pillar review. Spaced out, not per-change.
#
# Three pillars of this repo each have a reviewer subagent:
#   CLAUDE.md          -> claudemd-keeper   (keep it an operating manual, not a notebook)
#   docs/*.tex         -> notes-editor      (author voice, figures, claims)
#   code/config/jobs   -> repo-reviewer     (correctness, fold leakage, C1-C5, hygiene)
#
# Why batched: reviewing every commit-sized diff burns a subagent per edit and reviews
# nothing in context. Instead each pillar carries a WATERMARK — the commit at which it was
# last reviewed, plus the size of the change reviewed then. The Stop hook accumulates
# everything since that watermark and only asks for a reviewer once the backlog is worth a
# pass. The reviewer then sees the WHOLE backlog at once, which is also a better review: it
# can judge a section, not a hunk — a config default that changed in one commit while a
# script still assumes the old value is invisible to a per-hunk reviewer.
#
# WHERE THE TEETH ARE. A Stop hook can only ever NUDGE: the harness sets stop_hook_active on
# the next stop so a hook cannot block a turn twice, otherwise it would loop forever. So
# `check` is a reminder that can be walked past by just ending the turn again. The actual
# gate is `gate`, a PreToolUse(Edit|Write) hook: while a pillar is over threshold, EDITS TO
# THAT PILLAR'S FILES ARE DENIED. Passing the turn buys nothing, because the next edit to the
# overdue file is refused until its reviewer has run. Commits, and work on every other
# pillar, stay unblocked. (This replaces the old per-commit review_gate.sh.)
#
# Modes:
#   check                 (Stop hook, default) nudge once if any pillar is over threshold
#   gate                  (PreToolUse(Edit|Write)) DENY edits to an over-threshold pillar
#   begin <reviewer>      reviewer takes the lock: lifts `gate` so it can apply its own fixes
#   advance <reviewer>    reviewer PASSED; resets its watermark and drops the lock
#   status                human-readable backlog table (also: what /review-now would run)
#   init                  set every watermark to "everything so far is reviewed"
#
# Escape hatch, deliberately explicit: `begin <reviewer>` lifts the gate for that pillar until
# the next `advance`. It is the reviewer's normal first step, and it is also how a human says
# "I am editing this myself, stand down" — an auditable act, not a silent bypass.
#
# PERF — this repo lives on an sshfs mount of CC-IN2P3, where git calls that walk the tree or
# history pay network latency. Measured here: `git rev-list --count <sha>..HEAD -- <paths>`
# ~8 s, `git diff --numstat` ~2 s, `ls-files --others` ~1 s. Since `gate` runs on EVERY edit,
# a naive port would add ~10 s to each one. So:
#   (1) results are CACHED per pillar — the commit count keyed on HEAD (it cannot change
#       without a commit), the line count on a short TTL;
#   (2) the commit count short-circuits on a cheap pathspec-free upper bound: if ALL commits
#       since the watermark are under the threshold, the pillar's own count must be too;
#   (3) counting uses --numstat, not a full diff blob.
# `advance`/`init` re-seed the cache as they move the watermark, so the change takes effect
# immediately; `begin` leaves it alone, since taking the lock does not move anything.
#
# Shell only, no jq (not guaranteed on the hook PATH); python3 is used ONLY to parse the hook
# JSON on stdin, where a regex would be fooled by a Write payload containing "file_path".
#
# FAIL OPEN. settings.json invokes both hook modes as `... || true`, on purpose. A deny is
# delivered as JSON on stdout with exit 0, so the suffix cannot swallow one — all it does is
# neutralize a nonzero exit from the script itself. Without it, a syntax error in THIS FILE
# makes every Edit fail with a hook error, including the edit that would fix the file. That
# is not hypothetical: it happened while writing it. A review reminder must never be able to
# lock the repo. (Same reason `.claude/*` is excluded from the code pillar below.)
# The cost: exit-code signalling is inert here. If you ever rewrite a decision to use the
# exit-2 PreToolUse convention instead of stdout JSON, it will silently no-op — drop the
# `|| true` in the same edit.
set -uo pipefail
# Pathspecs below are stored space-separated and expanded UNQUOTED for word splitting. Without
# this, bash would also glob them and `*.py` would collapse to the repo root's .py files,
# silently narrowing the code pillar to nothing.
set -f

STATE_REL=".claude/.review_state"
# Seconds a cached line count stays valid. Generous on purpose: recomputing costs several
# seconds over sshfs, and drifting a few minutes past a threshold before the gate notices is
# harmless — the Stop hook re-checks at the end of every turn anyway.
TTL="${REVIEW_TTL:-300}"

# --- pillar table -----------------------------------------------------------
# name | line threshold | commit threshold | git pathspecs
#
# The code pathspec is SOURCE only. Globs match at any depth in git, so `*.py` covers every
# module without listing directories — which is also why notebooks/ and figures/ must be
# excluded explicitly: they are ungated in `pillar_for_path`, and counting their .py into the
# backlog would let exploratory churn deny edits to src/ and hand the reviewer files its own
# routing says are not its business. `.claude/*` is excluded for the same reason, and because
# gating the hooks behind the hook they configure is a way to lock yourself out of fixing it.
# Run artifacts need no exclusion (outputs/, data_cache/, *.pkl, *.pt are gitignored, so a
# diff never sees them). configs/ IS reviewed: JSON configs
# are a declared part of the pipeline contract, not generated data.
PILLARS='claudemd-keeper|80|8|CLAUDE.md
notes-editor|80|8|docs/*.tex
repo-reviewer|200|12|src config scripts jobs tests *.py :(exclude)notebooks/* :(exclude)figures/* :(exclude).claude/* :(exclude).reference/*'

pillar_field() { printf '%s\n' "$PILLARS" | awk -F'|' -v n="$1" -v f="$2" '$1==n{print $f}'; }
pillar_names() { printf '%s\n' "$PILLARS" | awk -F'|' '{print $1}'; }

# Repo root from this script's OWN location (../.. from .claude/hooks/), which costs nothing
# and carries no hardcoded per-machine path; `git rev-parse` only as a fallback for an odd
# invocation. Every git call here is ~0.3-1.3 s over sshfs, so the cheap answer comes first —
# `gate` must be able to wave through an ungated file having spent no git at all.
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -e "${REPO:-/nonexistent}/.git" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && exit 0
cd "$REPO" 2>/dev/null || exit 0

# HEAD sha, or empty in a repo with no commits yet (every mode then no-ops).
head_sha() { git rev-parse HEAD 2>/dev/null; }

# --- hook payload parsing ---------------------------------------------------
# python3, not a glob: a Write payload's `content` can itself contain the literal text
# "file_path" or "stop_hook_active", and a pattern match would happily pick the wrong one.
# Both helpers fail toward "do nothing" if the payload is unparseable.

hook_file_path() {
  python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null
}

# "1" when the harness says it already blocked once in this stop sequence. Unparseable input
# answers "1" on purpose: skipping one nudge just defers it a turn, whereas blocking a stop
# the harness has already blocked is how a Stop hook loops forever.
hook_stop_active() {
  python3 -c 'import sys,json
try: print("0" if json.load(sys.stdin).get("stop_hook_active") is not True else "1")
except Exception: print("1")' 2>/dev/null
}

# `status` sets this to bypass every shortcut: exact commit counts (no upper-bound trick) and
# a fresh line count (no TTL cache). It is a rare, human-facing command, so it pays full price
# rather than reporting a number that is minutes out of date.
EXACT=0

# --- raw measurement --------------------------------------------------------

# Changed lines in the pillar's paths since $1, counting untracked-but-not-ignored files at
# their full length (a brand-new module is a change, even uncommitted). --numstat prints "-"
# for binary files; treat that as zero rather than letting it poison the sum.
raw_lines() {
  local since="$1" paths="$2" n u f w
  n=$(git diff --numstat "$since" -- $paths 2>/dev/null | awk \
      '{a=$1; d=$2; if(a=="-")a=0; if(d=="-")d=0; s+=a+d} END{print s+0}')
  u=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    [ -f "$f" ] || continue
    w=$(wc -l < "$f" 2>/dev/null); u=$(( u + ${w:-0} ))
  done < <(git ls-files --others --exclude-standard -- $paths 2>/dev/null)
  echo $(( ${n:-0} + u ))
}

# Commits since $1 touching the pillar's paths. $3 = the pillar's commit threshold: if the
# pathspec-free count (cheap) is already below it, the pathspec-scoped count cannot reach it
# either, so skip the expensive history walk and return the upper bound — the DUE verdict is
# identical, only the displayed number is loose, which `status` corrects via EXACT.
raw_commits() {
  local since="$1" paths="$2" ct="$3" tot
  tot=$(git rev-list --count "$since"..HEAD 2>/dev/null) || tot=0
  if [ "$EXACT" != 1 ] && [ "${tot:-0}" -lt "$ct" ]; then echo "${tot:-0}"; return; fi
  git rev-list --count "$since"..HEAD -- $paths 2>/dev/null || echo 0
}

# --- state ------------------------------------------------------------------

# watermark: "<sha> <baseline_lines> <baseline_commits>"
state_file() { echo "$REPO/$STATE_REL/$1"; }
cache_file() { echo "$REPO/$STATE_REL/$1.cache"; }
lock_file()  { echo "$REPO/$STATE_REL/$1.lock"; }

read_state() { local f; f=$(state_file "$1"); [ -f "$f" ] && cat "$f" || echo ""; }

write_state() {
  mkdir -p "$REPO/$STATE_REL"
  printf '%s %s %s\n' "$2" "$3" "$4" > "$(state_file "$1")"
}

# Seconds after which a held lock is treated as abandoned and surfaced by `check`. A live
# review, or a human who said "I am editing this myself", is left in peace; a reviewer that
# blocked, crashed or was interrupted would otherwise leave its pillar disarmed indefinitely —
# neither gated nor nudged — with the evidence in a gitignored dir nobody looks at.
LOCK_STALE="${REVIEW_LOCK_STALE:-3600}"

# Age of a pillar's lock in seconds; empty when not held or unstattable.
lock_age() {
  local f t; f=$(lock_file "$1"); [ -f "$f" ] || return 0
  t=$(stat -c %Y "$f" 2>/dev/null) || return 0
  [ -z "$t" ] && return 0
  echo $(( $(date +%s) - t ))
}

write_cache() {   # $1=who $2=head $3=lines_epoch $4=lines $5=commits
  mkdir -p "$REPO/$STATE_REL"
  printf '%s %s %s %s\n' "$2" "$3" "$4" "$5" > "$(cache_file "$1")"
}

# Set a pillar's watermark to "as of right now, nothing is owed". $2 = HEAD, if already known.
reset_pillar() {
  local who="$1" head="${2:-}" paths lines
  paths=$(pillar_field "$who" 4)
  [ -z "$head" ] && head=$(head_sha)
  [ -z "$head" ] && return
  lines=$(raw_lines "$head" "$paths")       # the uncommitted remainder just reviewed
  write_state "$who" "$head" "$lines" 0
  # watermark == HEAD, so zero commits are owed by definition and the line count is fresh:
  # seed the cache rather than making the next caller pay for it again.
  write_cache "$who" "$head" "$(date +%s)" "$lines" 0
}

# Backlog for a pillar -> echoes "lines commits". Empty state self-initializes to zero owed.
# cache line: "<head_sha> <lines_epoch> <lines> <commits>"
# $2 = HEAD, if the caller already resolved it (saves a git call per pillar).
backlog_for() {
  local who="$1" head="${2:-}" paths st sha base_l tl tc c_head c_ts c_l c_c now ct fresh
  paths=$(pillar_field "$who" 4)
  ct=$(pillar_field "$who" 3)
  [ -z "$head" ] && head=$(head_sha)
  [ -z "$head" ] && { echo "0 0"; return; }
  st=$(read_state "$who")
  if [ -z "$st" ]; then reset_pillar "$who" "$head"; echo "0 0"; return; fi
  sha=$(echo "$st" | awk '{print $1}')
  base_l=$(echo "$st" | awk '{print $2+0}')

  now=$(date +%s)
  c_head=""; c_ts=0; c_l=""; c_c=""
  if [ -f "$(cache_file "$who")" ]; then
    read -r c_head c_ts c_l c_c < "$(cache_file "$who")" 2>/dev/null
  fi
  fresh=0
  [ "$c_head" = "$head" ] && [ -n "${c_l:-}" ] && [ -n "${c_c:-}" ] && fresh=1
  # A corrupt cache must fall back to recomputing, not sail through as a non-numeric value:
  # `$(( ))` would silently yield 0 and `[ x -ge N ]` would error to false, i.e. allow-all.
  case "${c_l:-}${c_c:-}" in *[!0-9]*) fresh=0 ;; esac

  # The watermark commit can only vanish (rebase/gc) via an operation that also moves HEAD,
  # which invalidates the cache — so this check belongs on the recompute path only.
  if [ "$fresh" != 1 ] || [ "$EXACT" = 1 ]; then
    git cat-file -e "$sha^{commit}" 2>/dev/null \
      || { reset_pillar "$who" "$head"; echo "0 0"; return; }
  fi

  if [ "$fresh" = 1 ] && [ "$EXACT" != 1 ]; then
    tc="$c_c"                                   # commits change only when HEAD does
  else
    tc=$(raw_commits "$sha" "$paths" "$ct")
  fi
  if [ "$fresh" = 1 ] && [ "$EXACT" != 1 ] && [ $(( now - ${c_ts:-0} )) -lt "$TTL" ]; then
    tl="$c_l"
  else
    tl=$(raw_lines "$sha" "$paths")
    c_ts="$now"
  fi

  write_cache "$who" "$head" "${c_ts:-$now}" "$tl" "$tc"

  # subtract what was already reviewed at the watermark; clamp (a revert can go negative)
  echo "$(( tl > base_l ? tl - base_l : 0 )) $tc"
}

is_due() {                       # $1=lines $2=commits $3=who  -> 0 when due
  local l="$1" c="$2" who="$3" lt ct
  lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
  [ "$l" -ge "$lt" ] || [ "$c" -ge "$ct" ]
}

# Which reviewer owns a repo-relative path (empty = ungated).
pillar_for_path() {
  case "$1" in
    CLAUDE.md)                                      echo claudemd-keeper ;;
    docs/*.tex)                                     echo notes-editor ;;
    .claude/*|.reference/*)                         echo "" ;;   # hook config / vendored upstream
    runs/*|data_cache/*|figures/*|notebooks/*)      echo "" ;;   # generated / exploratory
    src/*|config/*|scripts/*|jobs/*|tests/*|*.py)   echo repo-reviewer ;;
    *)                                              echo "" ;;
  esac
}

# Model suggestion for a DUE pillar, sized on its own backlog. Floor sonnet (a backlog is by
# construction already at threshold, so there is no haiku-sized case); opus for a large span
# or when core numerics are in it, where a miss is costly. Only ever called on the due path,
# so the extra --name-only call is off the hot path.
suggest_model() {
  local who="$1" lines="$2" sha="$3" paths
  paths=$(pillar_field "$who" 4)
  case "$who" in
    repo-reviewer)
      # untracked files too: a backlog that is entirely NEW modules under models/ or
      # features/ is exactly the case that wants opus, and --name-only never shows them.
      { git diff --name-only "$sha" -- $paths 2>/dev/null
        git ls-files --others --exclude-standard -- $paths 2>/dev/null; } \
        | grep -Eq 'src/madgrav_ml/(models|data|eval|experiments|report)/' \
        && { echo opus; return; }
      [ "$lines" -gt 400 ] && echo opus || echo sonnet ;;
    claudemd-keeper) [ "$lines" -gt 200 ] && echo opus || echo sonnet ;;
    notes-editor)    [ "$lines" -gt 300 ] && echo opus || echo sonnet ;;
    *) echo sonnet ;;
  esac
}

pillar_what() {
  case "$1" in
    claudemd-keeper) echo "CLAUDE.md stays an operating manual — no results, no session log, no bloat" ;;
    notes-editor)    echo "docs/results.tex — author voice, figures earning their place, claims sound" ;;
    repo-reviewer)   echo "code correctness, fold leakage, the C1-C5 constraints, committed artifacts" ;;
    *)               echo "pillar review" ;;
  esac
}

mode="${1:-check}"
case "$mode" in

  begin)
    who="${2:-}"
    [ -z "$who" ] && { echo "usage: review_backlog.sh begin <reviewer>" >&2; exit 2; }
    pillar_field "$who" 1 | grep -q . || { echo "unknown reviewer '$who'" >&2; exit 2; }
    mkdir -p "$REPO/$STATE_REL"; : > "$(lock_file "$who")"
    echo "[review-backlog] $who holds the lock — edits to its pillar are allowed until 'advance'"
    exit 0
    ;;

  advance)
    who="${2:-}"
    [ -z "$who" ] && { echo "usage: review_backlog.sh advance <reviewer>" >&2; exit 2; }
    pillar_field "$who" 1 | grep -q . || { echo "unknown reviewer '$who'" >&2; exit 2; }
    reset_pillar "$who"
    rm -f "$(lock_file "$who")"
    echo "[review-backlog] $who watermark advanced to $(git rev-parse --short HEAD) — backlog cleared"
    exit 0
    ;;

  gate)
    fp=$(hook_file_path)
    [ -z "$fp" ] && exit 0
    rel=${fp#"$REPO"/}
    case "$rel" in /*) exit 0 ;; esac          # outside the repo (scratchpad etc.)
    who=$(pillar_for_path "$rel")
    [ -z "$who" ] && exit 0                    # ungated file: answered with zero git calls
    # A lock means a review cycle is open, so stand down — but only while it is plausibly
    # live. Past LOCK_STALE the lock is treated as abandoned and the gate re-arms, otherwise
    # one crashed reviewer would permanently downgrade its pillar to a nudge (which this
    # file's own header notes can be walked past). Re-running `begin` refreshes the mtime.
    age=$(lock_age "$who")
    { [ -n "$age" ] && [ "$age" -lt "$LOCK_STALE" ]; } && exit 0
    head=$(head_sha); [ -z "$head" ] && exit 0

    set -- $(backlog_for "$who" "$head")
    l="${1:-0}"; c="${2:-0}"
    is_due "$l" "$c" "$who" || exit 0

    # The verdict may rest on a TTL-cached line count and an upper-bound commit count. Before
    # denying an edit and quoting numbers, re-measure exactly and re-test — a backlog that
    # shrank inside the TTL must not block on a number that contradicts the reason given.
    EXACT=1; set -- $(backlog_for "$who" "$head"); l="${1:-0}"; c="${2:-0}"
    is_due "$l" "$c" "$who" || exit 0
    st=$(read_state "$who"); sha=$(echo "$st" | awk '{print $1}')
    short=$(git rev-parse --short "$sha" 2>/dev/null || echo "$sha")
    lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
    m=$(suggest_model "$who" "$l" "$sha")
    msg="BLOCKED by review-backlog: '$rel' belongs to the $who pillar, which has ~$l unreviewed"
    msg="$msg changed lines over $c commit(s) since $short — past its $lt-line / $ct-commit threshold."
    msg="$msg Editing it further would pile more onto a backlog nobody has read."
    msg="$msg  Run the $who subagent (suggest: $m) on the WHOLE backlog now:"
    msg="$msg \`git diff $short -- $(pillar_field "$who" 4)\`."
    msg="$msg It takes the lock with 'review_backlog.sh begin $who', applies its fixes, and on a pass"
    msg="$msg clears the backlog with 'review_backlog.sh advance $who' — after which this edit succeeds."
    msg="$msg Do not run 'advance' yourself on a reviewer's behalf."

    esc=$(printf '%s' "$msg" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$esc"
    exit 0
    ;;

  init)
    for who in $(pillar_names); do reset_pillar "$who"; rm -f "$(lock_file "$who")"; done
    echo "[review-backlog] all watermarks set to $(git rev-parse --short HEAD), locks dropped — clean slate"
    exit 0
    ;;

  status)
    EXACT=1
    head=$(head_sha)
    printf '%-18s %10s %8s   %s\n' REVIEWER LINES COMMITS 'STATE'
    for who in $(pillar_names); do
      lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
      set -- $(backlog_for "$who" "$head")
      l="${1:-0}"; c="${2:-0}"
      if is_due "$l" "$c" "$who"; then s="DUE (>= $lt lines or $ct commits)"; else s="ok"; fi
      age=$(lock_age "$who")
      if [ -n "$age" ] && [ "$age" -lt "$LOCK_STALE" ]; then
        s="LOCKED ${age}s — review cycle open, gate stood down ($s)"
      elif [ -n "$age" ]; then
        s="LOCKED ${age}s — STALE, gate re-armed, review still owed ($s)"
      fi
      printf '%-18s %10s %8s   %s\n' "$who" "$l/$lt" "$c/$ct" "$s"
    done
    exit 0
    ;;

  check)
    # Already nudged in this stop sequence -> let it through (no infinite loop; also the
    # deliberate-pause escape). The watermark is untouched, so it fires again next turn.
    # `!= 0`, not `= 1`: an empty answer means the helper itself failed (no python3 on the
    # hook PATH), and that must defer like every other failure. Only a present, explicit
    # false lets the nudge through — the case that loops is the one to fail safe on.
    [ "$(hook_stop_active)" != 0 ] && exit 0

    br="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" || exit 0
    [ -z "$br" ] && exit 0
    [ "$br" = "main" ] && exit 0        # generated artifact, never authored here

    head=$(head_sha); [ -z "$head" ] && exit 0
    due=""; stale=""
    for who in $(pillar_names); do
      # A locked pillar is mid-review, so it is not nudged. That is correct while the cycle is
      # live; once the lock goes stale the review is still owed, and `gate` has already
      # re-armed on its own — so surface it rather than skipping it forever.
      if [ -f "$(lock_file "$who")" ]; then
        age=$(lock_age "$who")
        [ -n "$age" ] && [ "$age" -ge "$LOCK_STALE" ] && stale="$stale|$who:$(( age / 60 ))"
        continue
      fi
      set -- $(backlog_for "$who" "$head")
      l="${1:-0}"; c="${2:-0}"
      is_due "$l" "$c" "$who" || continue
      # That verdict may rest on a TTL-cached line count and an upper-bound commit count.
      # Before demanding a subagent — or quoting a number — re-measure exactly and re-test:
      # a backlog that shrank inside the TTL must not produce a review request for nothing,
      # nor a self-contradicting "~5 unreviewed lines, past its 200-line threshold".
      EXACT=1; set -- $(backlog_for "$who" "$head"); EXACT=0
      l="${1:-0}"; c="${2:-0}"
      is_due "$l" "$c" "$who" && due="$due|$who:$l:$c"
    done

    if [ -z "$due" ] && [ -n "$stale" ]; then
      msg="ABANDONED REVIEW CYCLE. A reviewer took its pillar lock and never ran \`advance\`, and"
      msg="$msg the lock has since gone stale, so the edit gate has re-armed by itself — the review is"
      msg="$msg still owed and the watermark has not moved:"
      IFS='|' read -ra items <<< "${stale#|}"
      for it in "${items[@]}"; do
        [ -z "$it" ] && continue
        msg="$msg  * ${it%%:*} — lock held ${it##*:} min."
      done
      msg="$msg  Re-run that subagent to finish the cycle (it calls \`advance\` on a pass), or if the"
      msg="$msg findings are already applied, run it again on the current backlog. Only drop the lock"
      msg="$msg by hand (\`rm .claude/.review_state/<name>.lock\`) if the user is editing that pillar"
      msg="$msg themselves and asked the reviewers to stand down."
      esc=$(printf '%s' "$msg" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
      printf '{"decision":"block","reason":"%s"}\n' "$esc"
      exit 0
    fi
    [ -z "$due" ] && exit 0

    msg="BATCHED REVIEW DUE. Accumulated unreviewed changes have crossed the threshold for the"
    msg="$msg pillar(s) below. Run each subagent now; each reviews its ENTIRE backlog at once (not"
    msg="$msg just the last edit), and on a pass calls \`review_backlog.sh advance <name>\` itself to"
    msg="$msg clear it. Backlog, scope and suggested model per reviewer:"
    IFS='|' read -ra items <<< "${due#|}"
    for it in "${items[@]}"; do
      [ -z "$it" ] && continue
      who="${it%%:*}"; rest="${it#*:}"; l="${rest%%:*}"; c="${rest##*:}"   # already exact
      st=$(read_state "$who"); sha=$(echo "$st" | awk '{print $1}')
      short=$(git rev-parse --short "$sha" 2>/dev/null || echo "$sha")
      m=$(suggest_model "$who" "$l" "$sha")
      msg="$msg  * $who (suggest: $m) — ~$l changed lines over $c commit(s) since $short;"
      msg="$msg $(pillar_what "$who"). Its backlog: \`git diff $short -- $(pillar_field "$who" 4)\`."
    done
    msg="$msg  The model suggestion is a size+area prior (sonnet floor, opus ceiling); the dispatcher"
    msg="$msg makes the final call. RUN THEM NOW, in this turn, without asking the user first — a hook"
    msg="$msg cannot spawn a subagent, so this message IS the automation and you are the part that"
    msg="$msg executes it. Ending the turn instead is not a pause: the PreToolUse gate will refuse your"
    msg="$msg next edit to these files until the reviewer has run. \`review_backlog.sh status\` lists all."
    if [ -n "$stale" ]; then
      msg="$msg  ALSO: a lock has been held without an \`advance\` on"
      IFS='|' read -ra items <<< "${stale#|}"
      for it in "${items[@]}"; do
        [ -z "$it" ] && continue
        msg="$msg ${it%%:*} (${it##*:} min)"
      done
      msg="$msg — that lock is stale, so its gate has re-armed while the review is still owed."
    fi

    esc=$(printf '%s' "$msg" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"decision":"block","reason":"%s"}\n' "$esc"
    exit 0
    ;;

  *) echo "usage: review_backlog.sh {check|gate|begin <r>|advance <r>|status|init}" >&2; exit 2 ;;
esac
