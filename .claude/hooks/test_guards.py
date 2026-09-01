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
    print(f"\n{n}/{total} guard checks passed")
    return 0 if n == total else 1


if __name__ == "__main__":
    sys.exit(main())
