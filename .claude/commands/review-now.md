---
description: Run the pillar reviewers over everything unreviewed since their last watermark
allowed-tools: Bash, Task
---

Show the current backlogs:

!`bash .claude/hooks/review_backlog.sh status`

Now run the reviewer subagents for the pillars listed above, **in parallel** (one message,
several Task calls). Include a pillar if `$ARGUMENTS` names it; if `$ARGUMENTS` is empty,
include every pillar whose backlog is non-zero — not only those marked `DUE`, since this is a
deliberate manual review.

- `claudemd-keeper` — backlog: `git diff <watermark> -- CLAUDE.md`
- `notes-editor` — backlog: `git diff <watermark> -- 'docs/*.tex'`
- `repo-reviewer` — backlog: `git diff <watermark> -- <the code pathspec in review_backlog.sh>`

Take each watermark SHA from the state files in `.claude/.review_state/` (first field). Tell
each subagent it is reviewing an accumulated backlog, not a single change, that it takes the
lock with `review_backlog.sh begin <name>` first, and that on a pass it clears its own
watermark via `review_backlog.sh advance <name>`.

Then report back: one line per reviewer with its verdict, followed by only the items that
need my decision.
