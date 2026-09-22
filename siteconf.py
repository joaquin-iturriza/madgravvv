"""siteconf -- where am I, and what does this cluster want.

    import siteconf
    siteconf.SITE            # "ccin2p3" | "jeanzay" | "lxplus" | "local"
    siteconf.PROJECT_DIR     # this checkout's root, wherever it lives
    siteconf.SWEEP_DIR, .SUBMIT_DIR, .SCRATCH, .DATA_DIR, .SETUP_COMMANDS, .CLUSTER

    siteconf.resolve(cfg)    # fill a sweep config's paths/cluster from the site
    siteconf.slurm_header(cluster, job_name, out, err)   # one #SBATCH block, right for the site

Facts come from sites/sites.yaml, the only file (with sites/activate.sh) that
names a cluster. The site is CCORCH_SITE when `site submit` exported it, else
sniffed from this file's own location. Nothing here hardcodes a path.
"""
import os

import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_SNIFF = (("/sps/", "ccin2p3"), ("/lustre/", "jeanzay"), ("/eos/", "lxplus"), ("/afs/", "lxplus"))


def _sniff(path):
    for prefix, name in _SNIFF:
        if path.startswith(prefix):
            return name
    return "local"


SITE = os.environ.get("CCORCH_SITE") or _sniff(PROJECT_DIR)

with open(os.path.join(PROJECT_DIR, "sites", "sites.yaml")) as _f:
    _ALL = yaml.safe_load(_f) or {}
_S = dict(_ALL.get(SITE) or {})

# A "local" checkout has no absolute paths in the table: derive them.
_S.setdefault("project_dir", PROJECT_DIR)
_S.setdefault("sweep_dir", os.path.join(_S["project_dir"], "sweeps"))
_S.setdefault("submit_dir", os.path.dirname(_S["sweep_dir"]))
_S.setdefault("scratch", os.path.join(_S["project_dir"], "tmp"))
_S.setdefault("data_dir", os.path.join(_S["project_dir"], "data"))
_S.setdefault("setup_commands", ["source %s/sites/activate.sh" % _S["project_dir"]])
_S.setdefault("cluster", {"scheduler": "none"})

SWEEP_DIR = os.environ.get("SWEEP_DIR", _S["sweep_dir"])
SUBMIT_DIR = _S["submit_dir"]           # where a scheduler must be driven from (AFS at CERN)
SCRATCH = os.environ.get("SCRATCH", _S["scratch"])
DATA_DIR = os.environ.get("DATA_DIR", _S["data_dir"])
SETUP_COMMANDS = list(_S["setup_commands"])
CLUSTER = dict(_S["cluster"])
CPU = dict(_S.get("cpu") or {})
RESULTS_DIR = _S.get("results_dir", SWEEP_DIR)

# job-owned cluster keys: these may legitimately appear in a sweep config
JOB_KEYS = ("time", "cpus_per_task", "request_gpus", "auto_submit", "mem", "scheduler")
# site-owned: never in a sweep config; always from here
SITE_KEYS = ("partition", "account", "qos", "gres", "gpu_flag", "constraint",
             "max_cpus_per_gpu", "python_bin")

_VARS = {"PROJECT_DIR": lambda: _S["project_dir"], "SWEEP_DIR": lambda: SWEEP_DIR,
         "DATA_DIR": lambda: DATA_DIR, "SCRATCH": lambda: SCRATCH,
         "SUBMIT_DIR": lambda: SUBMIT_DIR}
_OTHER_ROOTS = sorted({(o or {}).get("project_dir") for o in _ALL.values()
                       if (o or {}).get("project_dir")}, key=len, reverse=True)


def _expand_str(v):
    if "${" in v:
        for name, fn in _VARS.items():
            v = v.replace("${%s}" % name, fn())
    # a config generated on another cluster still carries that cluster's tree
    for root in _OTHER_ROOTS:
        if root != _S["project_dir"] and v.startswith(root + "/"):
            return _S["project_dir"] + v[len(root):]
    return v


def _expand(node):
    """Recursively expand ${VAR} and re-root foreign absolute paths, in place."""
    if isinstance(node, dict):
        for k, v in node.items():
            node[k] = _expand_str(v) if isinstance(v, str) else (_expand(v) or v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            node[i] = _expand_str(v) if isinstance(v, str) else (_expand(v) or v)
    return node


def resolve(cfg):
    """Fill the site-owned half of a sweep config in place and return it.

    A config written on one cluster and read on another keeps its job settings
    (time, cpus, gpu count, mem override) and gets this site's paths and
    scheduler directives. Job-owned keys already present are NOT overridden.
    """
    paths = cfg.get("paths") or {}
    cfg["paths"] = paths
    paths.setdefault("project_dir", _S["project_dir"])
    for k in ("afs_sweep_dir", "eos_sweep_dir", "python_env", "python_bin"):
        paths.pop(k, None)                      # site-owned; re-derived below
    sd = paths.get("sweep_dir")
    if sd:
        for other in _ALL.values():
            op = (other or {}).get("sweep_dir")
            if op and sd.startswith(op) and op != SWEEP_DIR:
                paths["sweep_dir"] = SWEEP_DIR + sd[len(op):]
                break
    if CLUSTER.get("scheduler") == "htcondor":
        # CERN: submission files + locks on AFS, bulk results on EOS
        paths["afs_sweep_dir"] = paths.pop("sweep_dir", None) or SWEEP_DIR
        paths["eos_sweep_dir"] = RESULTS_DIR
    elif not sd:
        paths["sweep_dir"] = SWEEP_DIR
    if not paths.get("setup_commands"):
        paths["setup_commands"] = list(SETUP_COMMANDS)
    paths["python_env"] = "%s/sites/activate.sh" % _S["project_dir"]   # `source`-able
    paths["python_bin"] = "python"               # env is activated by setup_commands
    cl = cfg.get("cluster") or {}
    cfg["cluster"] = cl
    for k in SITE_KEYS:
        cl.pop(k, None)                     # a stale value from another cluster
    for k, v in CLUSTER.items():
        cl.setdefault(k, v)
    cl.setdefault("request_gpus", 1)
    _expand(cfg)
    return cfg


def gpu_directive(n=1):
    """The site's way of asking for n GPUs (CC: --gpus, Jean Zay: --gres)."""
    flag = CLUSTER.get("gpu_flag") or "--gres=gpu:{n}"
    return "#SBATCH " + flag.format(n=int(n))


def clamp_cpus(cpus, gpus=1):
    cap = CLUSTER.get("max_cpus_per_gpu")
    if cap and int(cpus) > int(cap) * int(gpus):
        return int(cap) * int(gpus)
    return int(cpus)


def slurm_header(cluster, job_name, out, err, extra=()):
    """A complete #SBATCH block for THIS site from a (job-owned) cluster dict.

    Emits qos/mem only where the site has them, --gpus vs --gres per site,
    -C where the site's entitlement needs it, clamps cpus to the site's per-GPU
    ceiling, and never names a partition or account that isn't this cluster's.
    """
    cl = {**CLUSTER, **{k: v for k, v in (cluster or {}).items() if k in JOB_KEYS}}
    gpus = int(cl.get("request_gpus", 1) or 0)
    cpus = clamp_cpus(cl.get("cpus_per_task", 8), max(gpus, 1))
    lines = ["#!/bin/bash", "#SBATCH --job-name=%s" % job_name]
    if cl.get("partition"):
        lines.append("#SBATCH --partition=%s" % cl["partition"])
    if cl.get("account"):
        lines.append("#SBATCH --account=%s" % cl["account"])
    if cl.get("constraint"):
        lines.append("#SBATCH -C %s" % cl["constraint"])
    if cl.get("qos"):
        lines.append("#SBATCH --qos=%s" % cl["qos"])
    lines += ["#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1"]
    if gpus:
        lines.append(gpu_directive(gpus))
    lines.append("#SBATCH --cpus-per-task=%d" % cpus)
    lines.append("#SBATCH --time=%s" % cl.get("time", "20:00:00"))
    if CLUSTER.get("mem") is not None:          # site accepts/needs --mem
        lines.append("#SBATCH --mem=%s" % (cl.get("mem") or CLUSTER["mem"]))
    lines += ["#SBATCH --output=%s" % out, "#SBATCH --error=%s" % err]
    lines += list(extra)
    return "\n".join(lines) + "\n"


def cpu_header(job_name, out, err, cpus=32, time="04:00:00", extra=()):
    """#SBATCH block for CPU-only work on the site's free/cheap CPU partition."""
    lines = ["#!/bin/bash", "#SBATCH --job-name=%s" % job_name]
    part = CPU.get("partition", CLUSTER.get("partition"))
    if part:
        lines.append("#SBATCH --partition=%s" % part)
    acct = CPU.get("account", CLUSTER.get("account"))
    if acct:
        lines.append("#SBATCH --account=%s" % acct)
    lines += ["#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=%d" % int(cpus)]
    if CPU.get("mem_per_cpu"):
        lines.append("#SBATCH --mem-per-cpu=%s" % CPU["mem_per_cpu"])
    lines += ["#SBATCH --time=%s" % time, "#SBATCH --hint=nomultithread",
              "#SBATCH --output=%s" % out, "#SBATCH --error=%s" % err]
    lines += list(extra)
    return "\n".join(lines) + "\n"


def setup_lines(cfg=None):
    cmds = (cfg or {}).get("paths", {}).get("setup_commands") or SETUP_COMMANDS
    return "\n".join(cmds)
