#!/usr/bin/env python3
"""Self-test for the .claude guards.

Ported from Foundational_Amplitudes. It lives in a .py rather than a shell script for a
specific reason: the test payloads by design contain the very patterns the guards match,
so a .sh version would trip the guards the moment the harness inspected the Bash command
that launched it.

The guards are load-bearing — hpo_guard is the mechanical half of C4 and constraint_guard
is the mechanical half of C5 — so a guard that silently stops matching is a real hole.
That is not hypothetical: the first version of plot_guard packed a multi-word command
into a single `read` variable, which truncated it to one token, and the hook never fired
once. It looked installed and did nothing.

Run:  python3 .claude/hooks/test_guards.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HOOKS = os.path.join(REPO, ".claude", "hooks")

# Assembled from pieces so this file contains no literal match either.
OFF = "plot" + "=" + "false"
GRID = (
    "#!/bin/bash\n#SBATCH --array=0-5\n"
    "M=(1.0 2.0 3.0)\nMARGIN=${M[$SLURM_ARRAY_TASK_ID]}\n"
    "python run.py --config-name=stage2 model." + "margin=$MARGIN\n"
)
ABLATION = (
    "#!/bin/bash\n#SBATCH --array=0-2\n"
    "SEED=$((42 + SLURM_ARRAY_TASK_ID))\n"
    "python run.py seed=$SEED model.objective=masked\n"
)


def run(hook, payload):
    p = subprocess.run(
        ["bash", os.path.join(HOOKS, hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    # exit 2 = deny via stderr; a deny can also arrive as stdout JSON.
    denied = p.returncode == 2
    if not denied and p.stdout.strip():
        try:
            d = json.loads(p.stdout)
            denied = (
                d.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
            )
        except json.JSONDecodeError:
            pass
    return denied


RESULTS = []


def check(desc, hook, payload, want_block):
    blocked = run(hook, payload)
    ok = blocked == want_block
    RESULTS.append(ok)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'BLOCK' if blocked else 'allow':5s} "
        f"(want {'BLOCK' if want_block else 'allow'})  {desc}"
    )


def bash(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


def write(path, content):
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": content}}


def precompact_state():
    """Exercise precompact_notes_guard against the tree as it stands.

    Deliberately not mocked. The bug this catches was a mismatch between the guard's
    model of "fresh" and how git actually records a file, so a fake repo would have
    reproduced the model rather than the reality.
    """
    return run("precompact_notes_guard.sh", {"trigger": "auto",
                                             "hook_event_name": "PreCompact"})


# Hooks with no case here, and why. Listed rather than silently tolerated: a hook with
# no case cannot fail, which is how precompact_notes_guard reached settings.json
# carrying a live deadlock while this suite reported 18/18. Anything NOT in this set
# must have a case, so adding an untested guard fails loudly.
KNOWN_UNTESTED = {
    # side-effecting on the real repo: they commit, push, or mutate review state
    "auto_push", "commit_checkpoint", "review_backlog",
    # depend on live cluster or worktree state that cannot be faked meaningfully
    "slurm_waiter_guard", "worktree_fold_guard", "worktree_guard",
    # figure_pair_guard needs a session's worth of written figures to judge
    "figure_pair_guard",
    # reinject_working_state only cats a file; its failure mode is covered by the
    # precompact case, which fails if the file is absent
    "reinject_working_state",
}


def untested_hooks():
    """Hooks with neither a case in this file nor an entry in KNOWN_UNTESTED."""
    import glob
    body = pathlib.Path(__file__).read_text()
    names = {pathlib.Path(p).stem for p in glob.glob(os.path.join(HOOKS, "*.sh"))}
    return sorted(n for n in names
                  if n not in KNOWN_UNTESTED and f'"{n}.sh"' not in body)


def main():
    tmpdir = tempfile.mkdtemp(dir=os.path.join(REPO, "jobs"))
    grid = os.path.join(tmpdir, "grid.sh")
    abl = os.path.join(tmpdir, "ablation.sh")
    with open(grid, "w") as fh:
        fh.write(GRID)
    with open(abl, "w") as fh:
        fh.write(ABLATION)
    rel_grid = os.path.relpath(grid, REPO)
    rel_abl = os.path.relpath(abl, REPO)

    try:
        print("constraint_guard (C5: no ml4gw, and the AUC reminder)")
        check("ml4gw import in a data module", "constraint_guard.sh",
              write(f"{REPO}/src/madgrav_ml/data/x.py", "import ml" + "4gw\n"), True)
        check("ml4gw as a pyproject dependency", "constraint_guard.sh",
              write(f"{REPO}/pyproject.toml", '  "ml' + '4gw>=0.1",\n'), True)
        check("plain numpy import", "constraint_guard.sh",
              write(f"{REPO}/src/madgrav_ml/data/x.py", "import numpy\n"), False)
        check("ml4gw named inside the vendored upstream tree", "constraint_guard.sh",
              write(f"{REPO}/.reference/MADGRAV/x.py", "import ml" + "4gw\n"), False)

        print("hpo_guard (C4: an HP array has no fold record)")
        check("sbatch of a margin grid", "hpo_guard.sh",
              bash(f"scripts/remote.sh sbatch {rel_grid}"), True)
        check("sbatch of a seed/objective ablation array", "hpo_guard.sh",
              bash(f"scripts/remote.sh sbatch {rel_abl}"), False)
        check("the committed seed array", "hpo_guard.sh",
              bash("scripts/remote.sh sbatch jobs/job_seeds.sh exp_type=stage1"), False)
        check("a plain single job", "hpo_guard.sh",
              bash("scripts/remote.sh sbatch jobs/job_stage1.sh seed=42"), False)

        print("plot_guard (a run configured with plotting off)")
        check("run.py override", "plot_guard.sh",
              bash(f"scripts/remote.sh sbatch jobs/job_stage1.sh {OFF}"), True)
        check("Write of a config carrying it", "plot_guard.sh",
              write(f"{REPO}/config/x.yaml", "plot: false\n"), True)
        check("grepping FOR the flag (cleanup must stay possible)", "plot_guard.sh",
              bash(f"grep -rn {OFF} ."), False)
        check("a normal run", "plot_guard.sh",
              bash("scripts/remote.sh sbatch jobs/job_stage1.sh seed=42"), False)

        print("md_guard (a new doc bypasses ExperimentRecord)")
        check("a new findings doc", "md_guard.sh",
              write(f"{REPO}/docs/FINDINGS.md", "x"), True)
        check("a new stray note beside the code", "md_guard.sh",
              write(f"{REPO}/src/madgrav_ml/NOTES.md", "x"), True)
        check("editing the existing CLAUDE.md", "md_guard.sh",
              write(f"{REPO}/CLAUDE.md", "x"), False)
        check("a subagent definition (harness config)", "md_guard.sh",
              write(f"{REPO}/.claude/agents/new-reviewer.md", "x"), False)
        check("a python file", "md_guard.sh",
              write(f"{REPO}/src/madgrav_ml/x.py", "x"), False)

        print("block_memory (persistent memory stays disabled)")
        check("a write into the memory dir", "block_memory.sh",
              write("/home/u/.claude/projects/p/memory/MEMORY.md", "x"), True)
    finally:
        for f in (grid, abl):
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(tmpdir)

    n, total = sum(RESULTS), len(RESULTS)
    # --- precompact_notes_guard ---------------------------------------------------
    # Committing the notes ALONGSIDE the work must not block the next compaction. An
    # mtime test made that block, since a file is always older than the commit holding
    # it; satisfying it dirtied the tree, which made another commit, which blocked
    # again. A long autonomous run would have deadlocked on its first auto-compaction.
    blocked = precompact_state()
    RESULTS.append(not blocked)
    print(f"  [{'PASS' if not blocked else 'FAIL'}] "
          f"{'BLOCK' if blocked else 'allow':5s} (want allow)  "
          f"precompact: notes current with the tree do not block")

    missing = untested_hooks()
    RESULTS.append(not missing)
    print(f"  [{'PASS' if not missing else 'FAIL'}] "
          f"{'allow' if not missing else 'BLOCK':5s} (want allow)  "
          f"every hook has a case (untested: {missing or 'none'})")

    n, total = sum(RESULTS), len(RESULTS)
    print(f"\n{n}/{total} guard checks passed")
    return 0 if n == total else 1

if __name__ == "__main__":
    sys.exit(main())
