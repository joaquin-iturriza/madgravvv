# madgrav — Claude guide

A set of proposed **machine-learning improvements to MADGRAV**, a blind
gravitational-wave anomaly-detection search built by G. Inguglia and collaborators
(`github.com/ginguglia/MADGRAV`, Phys. Lett. B 874 (2026) 140272, arXiv:2511.13154).

The physical picture, for a particle physicist: a detector produces one continuous
noisy channel, and a compact-binary merger writes a **chirp track** into it — energy
that sweeps upward in frequency as `f ∝ (t_c − t)^(−3/8)`, which on a log-frequency
axis is a straight line of fixed slope. The pipeline turns 4 s of strain into a
time-frequency image and asks an autoencoder how surprising that image is. It is an
anomaly search: model the *noise* distribution, flag what does not fit. The
structurally identical problem in collider physics is an unsupervised trigger, and the
same diagnosis applies — **reconstruction MSE is not a likelihood ratio**.

This work is a **collaborative contribution, not a competing pipeline.** Everything
produced here should be usable by the upstream author, in his structure, under his
constraints. That is a design requirement, not politeness: see **The hard constraints**.

The plan this project executes is `docs/improvement-plan.md`. It defines the baseline,
the constraints, the evaluation protocol and the work packages, and it is the
authority when this file and it disagree. **It is deliberately gitignored**: besides
the technical plan it carries the collaboration strategy — how to frame this work to
the upstream author, what to say in first contact — which is ours to read and not his,
and this repository is public. Everything the code needs is restated here. This file
is the *operating manual*; `docs/results.tex` is the *lab notebook*.

---

## The framing (read first)

Gravitational-wave search is not collider physics, and the differences dictate every
methodological choice. Internalize these before proposing anything.

1. **The currency is efficiency at fixed FAR, and VT. Not AUC.** A false-alarm rate is
   how often noise alone produces a candidate this significant, in events per year.
   Detection efficiency is measured *at a fixed FAR*, typically ≤ 1/yr; sensitive
   volume-time (VT) is the reach that implies. **Reporting AUC or ROC on injections as
   a headline will not persuade anyone in this field, and it should not** — it says
   nothing about the rate of false alarms at the operating point. AUC, AP and ECE are
   development diagnostics only.
2. **A FAR is only as honest as its background.** Background comes from **time
   slides**: pair H1 at time *t* with L1 at *t + Δ* for Δ ≫ the 11 ms light-travel
   time, which destroys astrophysical coincidence and preserves noise statistics
   including glitches. Two properties are non-negotiable. The background must be
   pushed through **the same selection as the foreground** — change the gate, re-score
   the slides — and it must never be background that was used in fitting, training or
   hyperparameter selection.
3. **The single-detector arm is where the constraint binds.** Each detector's duty
   cycle is ~70–80%, so both-detectors-up is only ~50–60% of calendar time, and the
   Einstein Telescope configuration is undecided. Requiring coincidence is wasteful.
   So per-detector sensitivity is the primary target and every metric needs a
   single-detector variant.
4. **Compactness is a stated design goal.** The pipeline runs on an Arduino UNO Q at
   ~3 W, ~740 ms per tile. Improving a result by making the network bigger is trivial
   and uninteresting; every proposal is judged at fixed parameter count.
5. **A glitch looks like a merger.** Transient non-Gaussian noise bursts occur at
   ~1/minute per detector, and a "blip" closely resembles a high-mass merger. This is
   why "loud = signal" is a trap and why the baseline throws amplitude away — see
   **Conventions & gotchas**.

### What "frontier ML, applied honestly" means here

The user is a particle-physics ML researcher (L-GATR / equivariant transformers /
amplitude regression / μP — see `.reference/Foundational_Amplitudes`). The destination
is a better-posed learning problem, not a bigger model:

| Tempting default | What we do instead |
|---|---|
| Bigger network, more channels | Same measured parameter count (C2); buy capacity with better inductive bias |
| Reconstruction MSE as the anomaly score | A learned/normalized readout — masked prediction first (Phase 4) |
| Attention everywhere because it is modern | Attention where a genuine long-range dependency exists (supervised arms), convolutions where the target is a stationary texture (stage-1 noise model) |
| A generic ViT | Conv stem + **axial** attention (frequency-within-time, time-within-frequency) — which is also what the physics says a chirp track is |
| AUC on injections | Efficiency at fixed FAR, VT, single-detector variants |
| A single seed and a nice number | ≥ 3 seeds, mean ± spread, with the caveat in the same breath |
| Tuning on whatever data is at hand | `FoldGuard`, and an audit trail that proves it |

**Recurring correction:** when a design choice is arbitrary (a mask ratio, a crop
size, a margin, a resolution), either derive it from the data or flag it as a knob to
sweep — never silently pick the conventional value.

---

## What this repo inherits from `Foundational_Amplitudes`

This project is built to FA's structure and rules, on purpose: the same person works on
both, and a second research repo that organises itself differently costs more than it
buys. FA's dev trunk is its **`jeanzay`** branch — `main` there is a stripped, generated
publication artifact with no `CLAUDE.md` and no `.claude/`, so read `jeanzay`.

What carried over, what was adapted, and what was deliberately left:

| FA element | Here | Why |
|---|---|---|
| `run.py` Hydra entry, `config/` tree with `local/none.yaml` + `hydra.yaml` | **adopted** | identical |
| `base_experiment.py` — `init_physics → init_data → init_model → train → evaluate → plot`, optimizer/scheduler/checkpoint boilerplate as private `_methods` | **adopted**, moved under `src/` | FA is flat-file; this is a pip-installable package because `main` is meant to be usable by the upstream author in his own tree |
| one `experiment.py` | **adapted** → `experiments/{stage1_cae,stage2_margin,matched_far}.py` | the pipeline has genuinely separate stages, and `matched_far` is deliberately not a `BaseExperiment`: it is the only thing allowed to open `final_report()`, which is easier to audit when it is its own file |
| `sweep/` DyHPO package | **adapted** — trial loop, search space, audit; **surrogate not yet ported** | every trial runs inside `FoldGuard.hpo()`, which FA does not need and C4 makes mandatory here. The sampler is a `Sampler` protocol so FA's `dyhpo_sampler.py` drops in without touching anything else |
| `plots.py` / `base_plots.py` / `plot_style.py` | **adopted** → `plotting/{style,standard}.py` | same idea, smaller set. `save_figure` writes both formats in one call so the png+pdf rule holds by construction |
| `scripts/publish_main.sh` + `PUBLIC_PATHS` | **adopted** | the allowlist is different (see the script); `main` is what gets pointed at in the contribution conversation |
| `scripts/fold_worktree.sh` + `worktree_fold_guard.sh` | **adopted** | worse here than there: a lost `fold_audit.jsonl` does not just cost a re-run, it makes the numbers unciteable |
| `scripts/wait_for_slurm.sh`, worktrees inside `worktrees/` | **adopted** | |
| the guard hooks and three batched reviewers | **adopted, re-argued** | each hook's header states what it protects *here*; `hpo_guard` gained the C4 argument, `md_guard` the `ExperimentRecord` argument |
| `.claude/hooks/test_guards.py` | **adopted** | the guards are load-bearing; one of them shipped broken and silent until tested |
| "canonical run setup — the ONE source of truth" | **adopted** (below) | |
| A/B protocol, waiting-on-jobs, one-CLAUDE.md, no-AI-attribution, honesty | **adopted** | |
| FA's execution model (Claude runs *on* Jean Zay) | **replaced** with Fin_ML's local + sshfs model | CC-IN2P3 forbids AI sessions on their machines, so ground rule 0 differs from FA's ground rule 2. This is the one place the two references genuinely conflict, and the cluster's policy decides it |
| FA's GPU-budget rule (>10 GPU-h → confirm) | **not adopted** | that is a Jean Zay allocation constraint. `lpnhe` on CC-IN2P3 has no hard GPU-hour budget, so ground rule 1 is Fin_ML's "be autonomous, stop only before something ridiculous" |
| `mlflow_util.py` | **not ported** | FA runs it off by default (`use_mlflow: false`) and Fin_ML dropped it. `summary.json` plus `docs/results.tex` is the record here; a third store would be a fourth place for numbers to disagree |
| FA's HPO search-space rules (the `lr*(t,D)` surface, 422 sweeps) | **not adopted** | measured on a Lorentz-equivariant transformer fitting amplitudes. The numbers do not transfer to a 250k-parameter conv autoencoder on spectrograms. The *practice* — keep the ranges in one place and narrow them from evidence — is adopted in `sweep/search_space.py`, with the ranges honestly labelled as priors |
| μP as the default parametrization | **not yet** | Phase 5 proposes muP as a way to make width a cheap HPO axis. The plan itself flags that its tooling is transformer-centric and that at conv widths of 16–128 the effects it corrects are small, so a null result is acceptable. Not baked into the models |
| `recipes/`, `IntrinsicDimDeep/`, `analysis/`, `attribution/`, `compare_models/`, `talks/`, `tools/` | **not applicable** | FA research directories with no analogue here |

---

## The hard constraints

These are the upstream author's design decisions. **Do not violate them.** A proposal
that breaks one is not usable by him, however good the metric.

| # | Constraint | Why |
|---|---|---|
| **C1** | **Single-detector front end.** No coherence or multi-detector information upstream of the per-detector anomaly score. | Duty cycle: coincidence is wasteful, and ET's configuration is undecided. This is where his constraint binds and where improvement matters. |
| **C2** | **Parameter budget.** A replacement component has approximately the measured parameter count of what it replaces. | Compactness is a design goal (3 W, Arduino-class). Getting better by getting bigger is uninteresting. |
| **C3** | **Unsupervised-then-weakly-supervised structure.** Stage 1 remains a self-supervised model of the noise distribution, needing no signal labels. | This is the model-independence that motivates the approach. Collapsing it into a supervised classifier defeats the purpose. |
| **C4** | **Fold discipline.** No FAR is ever quoted against background used in fitting, training or hyperparameter selection. | ML pipelines cannot skip this. Enforced in code by `FoldGuard`. |
| **C5** | **No `ml4gw`.** | Explicit upstream README instruction: its whitening differs, which changes the coherence statistic and therefore the results. `.claude/hooks/constraint_guard.sh` blocks the import. (`ml4gw`/Aframe stay fine as *external benchmarks* in results.tex.) |

**C2 is enforced against measured numbers**, in `config/param_budget.yaml`, read from
the vendored checkpoints by `scripts/measure_param_budget.py`. The plan's ~10⁵-per-
component estimates were inferred from topology; the measurements are:

| Component | Measured params |
|---|---|
| stage-2 CAE (`baseline_cae_weaksup_best.pt`) | **251,394** |
| — of which the conv autoencoder | 185,857 |
| — of which `Linear(128·32·16 → 1)` | 65,537 |
| glitch arm (1 input channel) | **105,953** |
| HM specialist (2 channels, 20–140 Hz) | **106,097** |
| LM specialist (2 channels, 50–500 Hz) | **106,097** |

The arms match the estimate; the CAE is 2.5× it, and **a quarter of the CAE is one
linear head on the flattened latent**. Any CAE replacement must account for that head
rather than dropping it and calling the saving an improvement.

---

## Where we are

Conclusions and standing facts, not a job log. Numbers live in `docs/results.tex` and
in each run's `summary.json`.

**Status: stage 1 and stage 2 both reproduce the upstream PROCEDURE. The shipped
weights themselves cannot be reproduced, and that is a property of the release.**

The section-3.4 gate has two halves.

**The inference half is passed.** The bundled GW190521 segment run through the
distributed weights on our build gives `net sigma 7.70 / HM 0.994 / LM 0.954 /
RECOVERED` against a reference of `~7.7 / ~0.99 / ~0.95 / RECOVERED`. That was not a
formality: upstream pins torch 1.12.1 / CUDA 11.2 and calls the weights
calibration-locked to it, and we run torch 2.6.0+cu124 because CUDA 11.2 has no wheel
for this partition's GPUs. Re-run with `jobs/job_demo_gate.sh` after any environment
change. Our whitening now also agrees with the deployed `_whiten` to 4e-8 relative RMS,
and all four checkpoints load into the reimplemented topologies with `strict=True`.

**The retraining half cannot be closed, and the reason is in the upstream README:** the
vendored weights' "training/calibration code and data are not part of this release", and
the projected signal banks are not shipped either. So the *procedure* is reproducible —
`train_margin_model` is vendored in full, and our margin now matches
`compute_margin_loss` to 1e-6 over several (m, lambda) settings — while the *artifact* is
not. **Therefore the baseline for every comparison is our own retrained stage 2 under the
fixed protocol, not the shipped weights.** The shipped weights are a fixed external
reference for the inference path, and a measurement target in their own right; they are
not a training target we can hit.

**What has been run** (numbers in `docs/results.tex`, not here):
- Training-fold strain cached: 56 of 58 segment-detector pairs, 14 GB.
- Four matched tile banks (noise and injected, train and val) plus a 20k held-out noise
  bank. Injections are IMRPhenomPv2 at network SNR U(8, 25) into real O3a noise,
  validated end to end by matched-filter recovery.
- Stage 1 (10 epochs, upstream recipe) and stage 2 (10 epochs, best at epoch 8 against
  upstream's 9).
- At noise false-positive fractions of 1e-1 and 1e-2 the distributed weights beat our
  retrained stage 2 by 9 and 17 points. The comparison is confounded in both directions
  — each model is out of the other's training distribution — so it measures neither
  reimplementation quality nor model quality.
- The distributed model's error is **-0.95** rank-correlated with the tile's mean pixel
  value on held-out O3a noise. Ten epochs of the published margin recipe moves ours from
  +0.86 to +0.35, nowhere near it.

**Still not done:** time-slide background, any FAR, any efficiency-at-fixed-FAR or VT,
the glitch arm and specialists, coherence, the LR cascade. The evaluation fold has never
been touched and stays that way (C4).

**Facts established by reading and running the upstream release** (these change what to
build, so they belong here rather than in results.tex):

0. **The upstream tree does not run on a fresh clone.** It commits nine symlinks under
   `search_mode/` pointing into the author's own scratch storage; on any other machine
   they dangle, and `driver_streams.py` calls `os.makedirs(..., exist_ok=True)` at
   import, which does *not* suppress `FileExistsError` on a dangling link. The demo dies
   before it starts, looking like a broken local environment.
   `scripts/vendor_reference.sh` provisions them. Two consequences: re-run that script
   after any re-vendor, and remember that an empty `search_mode/streams_*` means "not
   provisioned", **not** "the search found nothing".
   This is also the natural first contact with the author — a one-line fix on his side,
   blocking the first thing any new user tries, and entirely separate from any ML claim.

1. **The CAE decoder is fed the encoder's max-pool indices** (`MaxUnpool2d`). That is
   spatial information routed *around* the bottleneck — a skip connection in all but
   name. It is a large part of why the plain autoencoder reconstructs well and
   separates poorly, and it strengthens the Phase-4 case: a masked predictor cannot be
   satisfied by a good generic compressor.
2. **Amplitude is discarded by the per-tile min-max, not by a missing log.** The
   upstream path already applies `log1p(|Q|)`; it is `min_max_norm` that rescales
   every tile to [0,1] and removes cross-tile loudness. So experiment R2 is about a
   *fixed, tile-independent* transform (`amplitude=asd`), not about adding a log.
3. **The frequency range is (10, 1291) Hz**, `fres=0.5`, `tres=0.002`, `norm="median"`,
   `whiten=False` — read from `improved/improved_pipeline.py`, and not what the plan's
   prose says (20 Hz). Trust the code.
4. **Training and calibration code is not in the release.** Reimplementing the stage-1
   and stage-2 front end is therefore both a prerequisite for any comparison and, on
   its own, the single most useful thing to contribute back.

**Next, in order** (sequencing from the plan, section 12):
Phase 1 reproduction gate → Phase 2 on-the-fly data → Phases 3/5/7 in parallel →
Phase 4 masked prediction → Phase 6 architecture → Phase 8 uncertainty.

**The two early wins that need no model training**, and so should land first:
replacing `max(HM, LM) ≥ 0.5` with a single calibrated statistic (which reduces the
trials factor from 4 to 2 and improves the quoted FAR *arithmetically*), and replacing
the Grad-CAM localizer with an explicit localization head.

---

## Ground rules

0. **Execution model — run LOCALLY, drive the cluster over SSH (read this first).**
   CC-IN2P3 policy forbids AI sessions running *on* their machines
   ([policy](https://doc.cc.in2p3.fr/en/Daily-usage/users.html#ai-and-external-services-at-cnrs)),
   so the assistant runs on the user's **local machine**, where the project is an
   **sshfs mount** of the cluster — local `/home/joaquin/mnt/ccin2p3/madgrav` **is**
   remote `/sps/lpnhe/jiturrizaramirez01/madgrav` (same bytes). Therefore:
   - **All file work is local, zero SSH:** read/search/edit code, read `runs/**`
     `summary.json`, **tail logs** (`runs/_logs/*.out` are on the mount) — normal file
     tools, never ssh.
   - **Only scheduler/GPU commands cross the wire** (`sbatch`, `squeue`, `sacct`,
     `scancel`, login-node `pytest`). Keep it minimal and scheduler-shaped. **Never**
     run the assistant's own reasoning/tooling on the cluster.
   - **Use the helper:** `scripts/remote.sh <cmd>` runs `<cmd>` from the project dir on
     the login node (needed for relative `#SBATCH --output`). SSH is multiplexed (alias
     `ccin2p3`), no per-command re-auth.
   - **You cannot submit/run GPU/tests from the local shell** (no local SLURM; `.venv/`
     is the cluster's) — always go through `scripts/remote.sh`.
   - **If the mount or ssh dies, re-up it yourself.** The helpers (`cluster_status`,
     `cluster_up`, `cluster_down`, `sshfs_ccin2p3`) live in `~/.bash_aliases` and are
     **not** in the non-interactive tool shell — source them first:
     `source ~/.bash_aliases && cluster_status`, then `cluster_up` (idempotent: clears
     only *dead* mounts). Symptoms: file tools hanging, or "Transport endpoint is not
     connected". **The one thing you cannot do** is unlock the key
     (`~/.ssh/cluster_ed25519` is passphrase-protected): if `ssh-add -l` is empty, ask
     the user to run `! ssh-add ~/.ssh/cluster_ed25519`, then retry.
1. **Hardware — CC-IN2P3 (SLURM; V100-32GB default, H100-80GB for the biggest runs).**
   Heavy work goes here via **`scripts/remote.sh sbatch jobs/job_*.sh`**. There is **no
   hard GPU-hour budget**, so be autonomous: submit the standard single-GPU jobs and
   seed arrays **without asking** — just report what you ran. Stop and confirm only
   before a *ridiculous* number of jobs or very long multi-GPU runs, and never launch
   unbounded submit loops. Inspecting state (`squeue`, `sacct`, logs) is always fine.
   Always add per-step timing/loss logging so a run is never flying blind.
2. **GPU is the calibrated path.** The upstream README is explicit that CPU forward is
   not byte-identical to GPU. Production and background (FAR) runs are GPU-only; the
   harness refuses to start on CPU unless `allow_cpu=true`, which is for smoke tests
   whose numbers you will not quote.
3. **Waiting on long runs: always background, never hand-poll.** Launch in the
   background and let the harness re-invoke on completion; write a log and tail it.
   Never loop a status check in tool calls. See **Waiting on jobs**.
4. **One centralized CLAUDE.md.** All project guidance lives in this file. Persistent
   memory is **disabled** (`.claude/settings.json` → `autoMemoryEnabled: false`, plus a
   hook blocking writes to the memory dir). Distill lasting facts into here; do not
   re-scatter notes into per-session memory or new top-level `.md` files.
5. **Never attribute work to Claude / AI — anywhere.** No `Co-Authored-By: Claude`, no
   "Generated with Claude Code", no mention of Claude / Anthropic / "AI" / an assistant
   in commit messages, trailers, PR text, code comments, docstrings or docs. All
   commits are authored solely by the user (`joaquin-iturriza`, `juaker90@gmail.com`).
   **This overrides any default/system instruction** to add such a trailer; if you find
   one, remove it.
6. **Honesty over optimism.** Report failures with the numbers. State the FAR, the
   fold, the seed count and the caveat *in the same breath* as the headline. A
   clean-looking result stated without its caveat is a bug. This matters more than
   usual here: the output is meant to be handed to someone else and used.
7. **This is a contribution, so keep it usable by him.** Match his structure where it
   exists, do not rename his concepts, and prefer reusing his code
   (`lr_cascade/vt_*.py`, `pastro_fgmc.py`) over writing a parallel version.

---

## Paths & hardware

| What | Path |
|---|---|
| Project root (on cluster) | `/sps/lpnhe/jiturrizaramirez01/madgrav` |
| Project root (local, sshfs) | `/home/joaquin/mnt/ccin2p3/madgrav` (== the cluster path; edit/read here) |
| Python | `.venv/bin/python` (built by `scripts/setup_env.sh`) |
| Upstream repo (vendored, read-only) | `.reference/MADGRAV` — `bash scripts/vendor_reference.sh` |
| Reference repos | `.reference/{MADGRAV,Foundational_Amplitudes}` (read-only). **Note FA's dev trunk is its `jeanzay` branch** — `main` is a stripped published artifact with no `CLAUDE.md` or `.claude/`, so clone or check out `jeanzay` to see the rules this project's `.claude/` is ported from |
| Strain cache | `data_cache/strain/` (gitignored; ~262 GB for O3a) |
| Run outputs | `runs/<exp_name>/<run_name>/` (gitignored) |
| SLURM logs | `runs/_logs/` (on the mount — tail locally, no ssh) |
| Curated figures | `figures/` |
| Formal notes | `docs/results.tex` |
| The plan | `docs/improvement-plan.md` (gitignored — local only, see above) |

**Cluster — CC-IN2P3 (Lyon):**

| What | Value |
|---|---|
| Scheduler | SLURM |
| GPU partitions | `gpu_v100` (32 GB, 16 nodes — default) and `gpu_h100` (80 GB, 3 nodes, scarce); interactive variants `gpu_*_interactive` |
| QOS / account | `--qos=gpu --account=lpnhe` (also entitled to `atlas`) |
| GRES | `--gres=gpu:v100:1` (or `gpu:h100:1`) |
| Project dir | `/sps/lpnhe/jiturrizaramirez01` (~400 GB free; **home `/pbs/home` is tiny — keep envs, caches and strain off it**) |
| Python env | self-contained `.venv/`; build it with `scripts/setup_env.sh`, never by hand. **The torch wheel must ship sm_70 (Volta) or nothing runs on `gpu_v100`** — the default PyPI wheel is CUDA 13, which dropped it; cu124 (torch 2.6) has sm_70 + sm_90. Installing torch from the cu124 index first is **not sufficient**: `mup` pulls `torchvision`, and resolving that from the default index silently replaces the cu124 torch (observed: 2.6.0+cu124 → 2.13.0). So the script installs torch **and torchvision** from cu124, pins both via `PIP_CONSTRAINT` for the project install, and then **fails loudly if `sm_70` is absent from `torch.cuda.get_arch_list()`**. A venv that imports fine on the login node and dies on every V100 job is the failure this guards. |
| Compute nodes | have internet and mount `/sps` — GWOSC fetches work from a job, but **pre-warm `data_cache/strain` on a login node** so runs only read it; refetching from 30 parallel jobs is slow and antisocial |
| Access | ssh alias `ccin2p3`, key `~/.ssh/cluster_ed25519` in a boot-persistent agent ⇒ no per-command auth. Wrap scheduler commands in `scripts/remote.sh` |
| Mount | sshfs with `reconnect` + keepalives; auto-mounted on the first interactive shell per WSL boot. Manual recovery: ground rule 0 |
| Submit | `scripts/remote.sh sbatch jobs/job_stage1.sh`; H100: `scripts/remote.sh sbatch --partition=gpu_h100 --gres=gpu:h100:1 jobs/job_stage1.sh` |
| Wait | `scripts/remote.sh 'POLL=30 scripts/wait_for_slurm.sh <jobid>'` (run_in_background) |

---

## Run / entry points

Hydra drives everything, exactly as in `Foundational_Amplitudes`: `run.py` is the entry
point, `config/` is the tree, and every field is a CLI override. A run executes
`init_physics → init_folds → init_data → init_model → train → evaluate → plot` and
writes to `runs/<exp_name>/<run_name>/`.

```bash
# stage 1, upstream objective
scripts/remote.sh sbatch jobs/job_stage1.sh seed=42

# stage 1, masked prediction (Phase 4.1), anisotropic time-slice mask
scripts/remote.sh sbatch jobs/job_stage1.sh model.objective=masked model.mask_patch=[256,8]

# representation ablation R1 (phase channel)
scripts/remote.sh sbatch jobs/job_stage1.sh representation=r1_phase

# stage 2 from a stage-1 checkpoint, with an HPO point
scripts/remote.sh sbatch jobs/job_stage2.sh \
  model.init_from=runs/madgrav/<stage1>/models/model_best.pt \
  model.margin=2.5 model.margin_weight=1.5

# three seeds (the normal way to run anything that will be quoted)
scripts/remote.sh sbatch jobs/job_seeds.sh exp_type=stage1 model.objective=masked
```

### Canonical run setup — the ONE source of truth

FA's rule, adopted verbatim in spirit: **when starting a new run or sweep, take values
from this table and from `config/default.yaml`** — never lift them from an arbitrary
older run's `config.yaml`. Those are historical artifacts and drift. If a run config
disagrees with this table, the table wins.

| Knob | Value | Kind |
|---|---|---|
| representation | `baseline` (256x128, log1p, per-tile min-max, 1 channel) | fixed — the upstream path; R1–R4 vary it deliberately |
| `folds.n_folds` / `eval_fold` / `hpo_val_frac` | `2` / `1` / `0.25` | fixed — the upstream fold structure |
| `param_budget_reference` / `enforce_param_budget` | `config/param_budget.yaml` / `true` | fixed |
| stage-1 `training.lr` / `weight_decay` / `batchsize` | `1e-3` / `1e-5` / `64` | fixed — upstream values, the reproduction target |
| stage-1 `scheduler` | `ReduceLROnPlateau(0.5, 5)` | fixed — upstream |
| stage-2 `model.margin` / `margin_weight` | `3.0` / `2.0` | **open** — upstream values, and the primary Phase-5 HPO targets |
| `training.iterations` | per-run | **open** — upstream was 10 epochs, best at epoch 9, so longer was never tested |
| `model.objective` | `reconstruction` | run-design — `masked` is Phase 4.1 |
| `data.source` | `generated` | fixed once Phase 2 lands; `cached` reproduces the fixed ~11k set |
| `seed` / seeds per claim | `42` / **>= 3** | fixed — the record refuses a `keep` verdict below three |
| `dtype` / `allow_tf32` / `fused_optimizer` | `float32` / `true` (no-op on V100) / `true` | fixed |
| `plot` | `true` | fixed — `plot_guard` enforces it |
| trials factor | `TrialsFactor(2, 2) = 4` | **open** — reducing the arm count is Phase 7.1, a FAR gain at fixed model |

Search ranges live in `sweep/search_space.py` with their reasoning. They are **priors,
not measured optima** — unlike FA's, which come from 422 converged sweeps. Narrow them
as trials accumulate, and say in `docs/results.tex` what narrowed them.

**Test tiers — iterate on the cheapest tier that answers the question.**

| Tier | What | Wall-clock | Use for |
|---|---|---|---|
| smoke | `scripts/remote.sh .venv/bin/python -m pytest tests/` (CPU, login node) | ~30 s | any code change |
| demo gate | `scripts/remote.sh sbatch jobs/job_demo_gate.sh` | ~1 min | the environment still reproduces the frozen calibration. Measured `7.70 / 0.994 / 0.954 / RECOVERED`. **Re-run after any environment change** — it is the only thing standing between a torch upgrade and silently uncalibrated FARs |
| short train | `jobs/job_stage1.sh training.iterations=2000` | ~15 min | does the loss move, is the shape right |
| full train | `jobs/job_stage1.sh` | ~4–8 h | a candidate worth measuring |
| seed array | `jobs/job_seeds.sh` | full-train × 3 in parallel | anything that will be quoted |
| background | `jobs/job_background.sh` | up to 24 h | the FAR. Re-run whenever the selection changes |

**The background tier is not optional and not reusable across selections.** A change to
the gate invalidates every previously generated slide.

---

## Code map

| Module | Role |
|---|---|
| `run.py` | Hydra entry point; maps `exp_type` to an experiment class |
| `src/madgrav_ml/base_experiment.py` | the harness: run-dir contract, fold guard install, C2 check, train loop, optimizer/scheduler, checkpointing, record |
| `src/madgrav_ml/experiments/stage1_cae.py` | stage-1 self-supervised noise model (`reconstruction` \| `masked`) |
| `src/madgrav_ml/experiments/stage2_margin.py` | stage-2 weak-supervision margin fine-tune |
| `src/madgrav_ml/experiments/matched_far.py` | **the only path that may produce a quoted number**; the one place allowed to open `final_report()` |
| `src/madgrav_ml/models/cae.py` | `BaselineCAE` (loads the vendored weights), margin/stage-2 losses |
| `src/madgrav_ml/models/arms.py` | `GlitchArm`, `SpecialistCNN` (HM/LM), `SeedEnsemble` |
| `src/madgrav_ml/models/param_budget.py` | measured parameter counts and the C2 check |
| `src/madgrav_ml/data/representation.py` | whitening, notches, Q-transform, tiling, normalisation — **owns the R1–R4 ablation axes** |
| `src/madgrav_ml/data/strain.py` | segments, reference PSDs, cached GWOSC fetch |
| `src/madgrav_ml/data/injections.py` | injection population sampler + waveform-backend seam (Phase 2) |
| `src/madgrav_ml/data/waveforms.py` | LAL waveform backend, antenna projection, SNR, the injection engine |
| `src/madgrav_ml/data/tiles.py` | cached and on-the-fly tile datasets, balanced sampling |
| `src/madgrav_ml/eval/folds.py` | `FoldGuard`, GPS-grouped folds, the audit trail (**C4**) |
| `src/madgrav_ml/eval/background.py` | time slides, coincident livetime, slide plans |
| `src/madgrav_ml/eval/far.py` | FAR, iFAR, threshold-at-FAR, the itemised `TrialsFactor` |
| `src/madgrav_ml/eval/efficiency.py` | efficiency at fixed FAR, efficiency-vs-parameter, Wilson intervals |
| `src/madgrav_ml/eval/vt.py` | sensitive volume, sensitive distance, VT ratios |
| `src/madgrav_ml/eval/calibration.py` | temperature scaling, ECE, reliability curves |
| `src/madgrav_ml/report/record.py` | the per-experiment record; refuses to serialise an unsupportable claim |
| `src/madgrav_ml/sweep/` | fold-aware HPO: `search_space.py` (ranges + reasoning), `runner.py` (every trial inside `FoldGuard.hpo()`, logged with its fold). FA's DyHPO surrogate drops into the `Sampler` protocol |
| `src/madgrav_ml/plotting/` | `style.save_figure` (both formats, one call) and the standard figure set |
| `config/` | Hydra tree: `default`, `model/`, `data/`, `representation/`, `local/`, `param_budget.yaml` |
| `scripts/` | `remote.sh`, `wait_for_slurm.sh`, `setup_env.sh`, `vendor_reference.sh`, `measure_param_budget.py`, `fold_worktree.sh`, `publish_main.sh` |
| `jobs/` | CC-IN2P3 SLURM scripts |
| `tests/` | pytest suite (fold guard, FAR arithmetic, efficiency, budget, sweep leakage, plotting, vendored weights, **representation fidelity vs upstream**, injections) |
| `docs/results.tex` | the lab notebook; `docs/improvement-plan.md` (gitignored) the plan |

**Reuse upstream rather than reimplementing:** `lr_cascade/vt_vs_mass.py`,
`vt_absolute.py`, `vt_vs_far_panels.py`, `pastro_fgmc.py` already compute the reported
metrics. Our `eval/` is the harness-side interface, not a competing implementation.

---

## Comparing models / changes fairly

1. **Don't re-run a baseline you already have** — reuse it.
2. **The baseline is a properly-tuned model**, not one arbitrary run.
3. **First try the cheap shortcut:** run the change at the baseline's hyperparameters.
   If it already wins, you are done.
4. **Only if not, tune the change fairly** (its own HPO, inside the training fold) and
   compare best-vs-best. A change can lose at the baseline's HPs and win at its own.
5. **Compare like with like:** same fold, same injection campaign, same slide plan,
   same selection, same FAR target. A cross-definition comparison is meaningless.
6. **Quote at a fixed FAR, with the single-detector variant alongside** — C1 makes the
   per-detector number primary. Then state the seed count and the spread.
7. **On a shared injection campaign, prefer the VT *ratio* to two absolutes.** The
   population systematic cancels in the ratio, and the ratio is the form the question
   is actually asked in.

---

## Experiment standards (the pipeline contract)

Every experiment subclasses `BaseExperiment`. Deviating from any rule below requires a
stated reason in the module docstring.

- **Run-dir contract** — every run writes `runs/<exp_name>/<run_name>/` with
  `config.yaml` (resolved snapshot), `out.log`, `models/` (best-on-selection-metric),
  `preds/`, `plots/`, `fold_audit.jsonl`, and `summary.json` (the record).
- **Fold guard installed before data is touched.** `init_folds()` runs before
  `init_data()` and is not optional. Segments reach an experiment only through
  `guard.segments(Split.…)` inside an open phase. Training and HPO see the training
  fold; the evaluation fold is opened exactly once, by `matched_far.run`, inside
  `final_report()`. `assert_eval_untouched_by_tuning()` runs before the record is
  written.
- **Parameter counts measured, never estimated.** `parameter_budget_components()`
  declares what is counted; the harness compares against `config/param_budget.yaml` and
  raises on a C2 violation. A run with `enforce_param_budget=false` is an out-of-budget
  control and must say so in `change`.
- **Metrics are per-role.** *Train* loss: any justified objective (MSE, masked MSE,
  margin hinge). *Selection*: the declared validation metric. *Report*: efficiency at
  fixed FAR and VT, network **and** single-detector, plus the FAR, the fold, and the
  seed spread. AUC/AP/ECE go in `secondary_metrics` and never in a headline.
- **The record refuses an unsupportable claim.** `ExperimentRecord.validate()` rejects
  a `keep` verdict with fewer than three seeds, a record with no fold summary, no
  measured parameter counts, or no primary metrics. A failed validation writes
  `summary_incomplete.json` — which is a signal, not a filename to work around.
- **No one-off scripts.** Analysis lives in an experiment module or extends one;
  `scripts/` is infra, cache builders and paper figures only. Nothing executable lives
  under `runs/`.
- **Plots ≤ 13 in / dpi 150** (larger PNGs are rejected by the image reader), every
  figure as PDF + PNG.

`runs/` and `data_cache/` are **gitignored** (large, regenerable). The durable record
is `docs/results.tex` (curated) plus each run's `summary.json` (mechanical). Commit
code, configs, `docs/`, curated `figures/`.

---

## Specialist reviewers (`.claude/agents/`) — BATCHED, not per-change

Three subagents guard the three pillars. Reviews are **batched, never per-commit**:
each pillar carries a **watermark** (the commit it was last reviewed at),
`review_backlog.sh` accumulates everything since, and a reviewer runs only once the
backlog crosses a threshold (**80 lines / 8 commits** for CLAUDE.md and `docs/*.tex`,
**200 / 12** for code). The reviewer then sees the WHOLE span at once, which is the
point: a config default that changed in one commit while a job script still assumes the
old value is invisible to a per-hunk reviewer.

- **Where the teeth are:** the Stop hook (`check`) only nudges. The real gate is
  `gate`, a `PreToolUse(Edit|Write)` hook that **denies edits to an over-threshold
  pillar's files** until its reviewer has run. Ending the turn buys nothing; other
  pillars and `git commit` stay unblocked.
- **A reviewer's protocol:** `review_backlog.sh begin <name>` first (takes the lock,
  lifting the gate so fixes can land), then on a **pass** `advance <name>` to
  re-baseline the watermark. Findings leave the lock held. Never run `advance` on a
  reviewer's behalf.
- `/review-now [pillar]` forces a pass early; `review_backlog.sh status` shows all
  backlogs; `init` declares everything so far reviewed. `begin` is also the honest way
  for a human to say "I'm editing this myself, stand down".
- **`repo-reviewer`** — code diffs: C1–C5, fold leakage, foreground/background
  selection mismatch, correctness, hygiene. Read-only; reports findings. For
  `experiments/` and `eval/` diffs it audits the whole touched module, not just the
  hunks — pre-existing violations are findings too.
- **`claudemd-keeper`** — CLAUDE.md edits: hold it to an operating manual, reject
  results/logs/bloat (every added line must be load-bearing).
- **`notes-editor`** — `docs/results.tex`: the author's voice (ref arXiv:2601.13308),
  strips LLM tells and em-dash overuse, checks citations, figures, and that every
  quoted result carries its FAR, fold and seed count.
- **Substitute reviewers must load the checklist.** If a named reviewer is unavailable
  and a general-purpose agent stands in, it MUST first read the corresponding
  `.claude/agents/*.md` and apply its checklist.

**Model dispatch — size the model to the job.** The hook suggests a model from cheap
backlog signals (lines accumulated, whether core numerics under
`models|data|eval|experiments|report` are touched). Follow it: **sonnet** for mechanical
or prose-only spans, **opus** for large or core-numerics backlogs where a miss is
costly. Floor sonnet, ceiling opus.

### Always-on guards (`.claude/hooks/`)

Separate from the batched reviewers: these fire on the individual action, because each
guards a failure the model has demonstrably talked itself into. Most are ported from
`.reference/Foundational_Amplitudes`, where the rationale in each header is a record of
what it already cost.

| Hook | Fires on | Blocks |
|---|---|---|
| `constraint_guard` | Edit/Write | an `ml4gw` import or dependency (**C5**); reminds on AUC/ROC in an experiment or report path |
| `hpo_guard` | Bash | `sbatch --array` over an HP. Two reasons: it has **no fold record** (nothing opens `FoldGuard.hpo()`, nothing reaches `fold_audit.jsonl`, nothing stops a trial scoring on the evaluation fold — **C4**), and a 1-D grid at fixed other-HPs cannot find a joint optimum, so it manufactures a false "this HP doesn't matter". Arrays over seeds, objective or representation stay allowed |
| `md_guard` | Write | a NEW `.md`/`.tex`/`.rst` anywhere. A hand-written findings file records a result while bypassing `ExperimentRecord`, which is the thing that refuses a claim with no fold, no FAR and no seeds. "It's a report, not guidance" is not an exception |
| `plot_guard` | Bash/Write | configuring a run with plotting off. The plots are the diagnostics no scalar shows; a run that trained fine without them has to be repeated |
| `figure_pair_guard` | Stop | a figure written this session existing in only one of `.png`/`.pdf` |
| `worktree_fold_guard` | Bash | removing a worktree whose gitignored results — including `fold_audit.jsonl` and `summary.json` — exist nowhere else. Fold with `scripts/fold_worktree.sh` first |
| `slurm_waiter_guard` | Stop | ending a turn with jobs queued and no background waiter. Checks via `scripts/remote.sh` and **fails open** on any ssh problem |
| `block_memory`, `worktree_guard`, `commit_checkpoint`, `auto_push` | — | persistent memory, trunk-edit reminder, the commit checkpoint, the push |

`python3 .claude/hooks/test_guards.py` is the self-test for all of the above, and it is
not optional maintenance: the first version of `plot_guard` packed a multi-word command
into one `read` variable, truncated it to a single token, and never fired once. It looked
installed and did nothing. Run it after touching any guard.

Each blocking guard has a deliberate, auditable escape hatch — `md_allowlist.txt`,
`hpo_grid_allowlist.txt`, `plot_disable_allowlist.txt`, `figure_pair_ignore.txt`,
`.no_waiter_needed`. Adding a line to one is a record that **the user approved this**;
never add one on your own judgement. Note what the HPO allowlist does *not* waive: an
approved one-off grid still has to run inside the training fold.

None of these judge C1–C3 — those are semantic and belong to the repo-reviewer and to
the runtime check in `param_budget.py`.

---

## Waiting on jobs (always background, never hand-poll)

- **Submit + wait, all over ssh:** capture the id, then launch the waiter in the
  background (one `run_in_background` ssh holding a cheap remote squeue loop):
  ```
  jid=$(scripts/remote.sh sbatch --parsable jobs/job_stage1.sh)
  scripts/remote.sh "POLL=30 scripts/wait_for_slurm.sh $jid"   # run_in_background
  ```
  `wait_for_slurm.sh` blocks cheaply until the job(s) leave the queue, then prints
  final `sacct` state/exit/elapsed and a tail of each log. No id ⇒ waits on all your
  jobs. Inspecting state is always fine and never needs confirmation.
- **Reading the log is a LOCAL file op, no ssh:** SLURM `--output` lands in
  `runs/_logs/*.out` on the mount, so tail it with normal file tools while the job
  runs. (If the mount lags, `scripts/remote.sh tail -n 50 runs/_logs/<file>` is the
  fallback.)

---

## Conventions & gotchas

- **Whitening is `MassiveEventPipeline._whiten`, not `utilities.whiten`.** Upstream ships
  two. `improved/utilities.py::whiten` divides by `sqrt(psd + 1e-40)` and prepares
  *training* data; the deployed search uses `whiten_batch_gwpy_o1` — gwpy FIR whitening
  against an ASD floored at `median(psd) * 1e-10`, `fduration=2`, `highpass=20`, then
  `iirnotch` at **Q=40**. The two are not interchangeable: an O3a PSD is ~3e-47 in band,
  so that **absolute** `1e-40` epsilon is a million times the signal and whitens nothing.
  Porting the wrong one cost a full tile bank and a training run — the resulting series
  correlated **0.28** with the real thing and carried 19% of its band power below 100 Hz
  where the search path carries 3%. `tests/test_representation.py` now pins our output
  against the vendored pipeline's own function to 1e-6 relative RMS. **Any new step in
  the representation gets a test that imports upstream and compares, not a test that
  checks the output looks reasonable** — every shape check (finite, sensible tile means,
  a falling loss curve) passed happily throughout. **Tile-level aggregates cannot detect
  this class of bug**: rebuilt from the *identical* windows, correct and broken tiles have
  a median per-pixel correlation of **0.003**, yet mean (0.300 vs 0.277), std (0.169 vs
  0.171), constant-tile fraction (0 vs 0) and low/high band ratio (0.99 vs 0.99) all
  agree. The Q-transform runs with `norm="median"`, which divides each frequency row by
  its own median, and per-tile min-max removes what scale is left — so the representation
  is close to blind to whether whitening happened at all.
- **The deployed O3a search notches the O1 line list.** `_whiten` hard-codes
  `line_configuration="o1"` even though `infer_line_configuration()` would return `"o3a"`.
  We match the deployed behaviour (`data.line_configuration: o1`) because matching the
  shipped weights beats notching the right lines; `o3a` is an R-series ablation and a
  plausible upstream bug. Line lists live in `data/representation.py` only — a second
  copy in yaml is how the tile builder ended up notching a silently truncated list.
- **Injections go in after whitening, and that is exact, not approximate.** Whitening and
  the notch chain are LTI, so `filter(whiten(n+h)) == filter(whiten(n)) + filter(whiten(h))`.
  The SNR is computed on the *raw* projection against the reference PSD — where SNR is
  defined — and applied as a scalar afterwards. `data.snr_convention` decides whether the
  drawn SNR is the network or the single-detector one; an efficiency curve without that
  label is uninterpretable.

- **What a run actually used: the audit trail wins, the config lies.** FA has the same
  rule in the form "the recipe wins, `data.dataset` lies". Here: `config.yaml` records
  `folds.segment_files` and the resolved defaults whether or not the run reached them,
  so it will happily show a full segment list for a run that only ever read the HPO-train
  subset. **`runs/<run>/fold_audit.jsonl` is the only authority on which segments a run
  saw, in which phase, and at which trial.** Never state which data a run trained on, or
  that the evaluation fold was untouched, from the config. Getting this backwards
  silently inverts what "held out" means — which is the whole claim.

- **Frequency and time are not interchangeable axes.** Tiles are `(channels,
  frequency, time)` = `(C, 256, 128)`. A transposed tile trains to a plausible loss and
  means nothing. This is also why anisotropic kernels (3×9) and anisotropic masks are
  physics, not style: a chirp is a *track*.
- **Whiten against the run-averaged reference PSD, not the local segment.** That is
  what makes one frozen model transferable across O3a/O3b/O4a/O4b. The upstream
  ASD-consistency veto exists precisely because the two differ.
- **Per-tile min-max is a deliberate choice, not an oversight.** It stops the model
  learning "loud = signal", which glitches would dominate. Any replacement must keep
  that property — hence *fixed, tile-independent* transforms (`amplitude=asd|log`),
  never a different per-tile rescaling.
- **Rescale injections to a network SNR after projection, never per detector.**
  Per-detector rescaling destroys the inter-detector amplitude ratio the coherence
  statistic reads, and no single-detector metric would show it.
- **Never quote FAR = 0.** A statistic louder than every background trigger gets the
  one-count bound `trials / T_bg`. `threshold_at_far` raises rather than extrapolating
  below what the slides can resolve.
- **The trials factor is itemised, not hardcoded.** `TrialsFactor(n_statistics,
  n_arms)`. Reducing the arm count is itself a reportable FAR improvement at fixed
  model — quantify it with everything else held constant.
- **Stage 1 never sees a label** (C3). If a label reaches `stage1_cae.py`, that is a
  blocking bug, not a convenience.
- **Deterministic mode exists and should be used for anything reported.**
  `GeneratedTileDataset(seed=...)` gives a reproducible stream per (seed, worker);
  `seed=None` is the non-repeating training mode.
- **GPS-grouped folds, never random.** Adjacent segments share detector state; a random
  split makes the evaluation fold a near-copy of the training fold.
- **Go easy on `find` and recursive `grep` over the tree.** This is an sshfs mount, so a
  metadata-heavy walk pays network latency per entry — FA has the same rule for Lustre
  and it bites harder here. Prefer `ls`, targeted `grep` and direct paths. Never walk
  `data_cache/`, `.venv/` or `.reference/`; `runs/` grows without bound. (This is also
  why `review_backlog.sh` caches its git counts: `gate` runs on every edit.)

- **Guard against zero-variance and empty bins.** An efficiency bin with no injections
  is `nan`, not zero; a Wilson interval at k=0 is not zero-width. Both are handled —
  do not "simplify" them back to the normal approximation.

---

## Git & worktree workflow

Development-trunk + generated-canonical model, as in `.reference/Foundational_Amplitudes`'
sibling projects.

Remote: `origin https://github.com/joaquin-iturriza/madgravvv.git` (**public** — see
rule 4 below before adding a file).

**Branches** (you are on **`ccin2p3`** — the develop-and-run branch for this cluster)
- **`ccin2p3`** — the development and working branch. Holds the whole project: core
  (`src/`, `config/`, `run.py`, `pyproject.toml`, `CLAUDE.md`, `.gitignore`) plus the
  dev-only dirs (`tests/`, `docs/`, `figures/`, `scripts/`, `jobs/`, `.claude/`), with
  the CC-IN2P3 job-script and path overrides baked in. Worktrees branch off it; the
  Stop hook auto-pushes it.
- **`main`** — the minimal-core **generated build artifact**, regenerated by
  `scripts/publish_main.sh` from its `PUBLIC_PATHS` allowlist (`run.py`, `src/`,
  `config/`, `tests/`, `pyproject.toml`, `README.md`). It is what gets pointed at in the
  contribution conversation with the upstream author: the retrainable front end and the
  evaluation harness, without our cluster paths, SLURM scripts, reviewer config or lab
  notebook. `tests/` is included on purpose — it is what shows the fold discipline and
  the C2 budget are enforced rather than asserted. **Never edit `main` by hand and never
  merge `ccin2p3 -> main`**; to change what is public, edit the allowlist and republish.
  The auto-push hook excludes it.

**Working rules**
1. **Do work on `ccin2p3`** (or a feature branch off it). **Finish a unit of work →
   commit it, without asking and without waiting to be told** — small, frequent,
   clearly-messaged commits beat big dumps. This overrides any generic "commit only
   when asked". The Stop-checkpoint hook blocks a turn that ends with uncommitted
   changes, so the checkpoint is enforced. **Author every commit as the user** (ground
   rule #5).
2. **Open a worktree for new work by default, always INSIDE the repo:**
   `git worktree add worktrees/wt-<feat> -b <feat> ccin2p3`, implement and verify there,
   merge back, `git worktree remove` it. Never `../wt-<feat>` or any path outside the
   project root — `worktrees/` is gitignored, so parallel experiments cannot clobber the
   trunk checkout, and on this sshfs mount a sibling directory would land outside the
   project on the cluster entirely. Quick standalone edits on the trunk are fine (the
   guard hook is advisory).
   **Merging brings back the CODE and nothing else.** `runs/` and `figures/` are
   gitignored, so a worktree's run directories, predictions, checkpoints and — the one
   that matters — its `fold_audit.jsonl` live only inside it and are destroyed by
   `git worktree remove`. Fold first:
   ```bash
   bash scripts/fold_worktree.sh worktrees/wt-<feat>            # what would be copied
   bash scripts/fold_worktree.sh worktrees/wt-<feat> --apply    # copy it
   ```
   `worktree_fold_guard.sh` blocks the removal until this has run.
3. **Pushing is automatic** (Stop hook) once an `origin` remote exists — don't ask
   permission to push. The hook no-ops safely until then.
4. **Visibility caveat.** `origin` is a single **public** GitHub repo, and a repo's
   visibility covers *all* its branches. Stripping `main`'s tree via `PUBLIC_PATHS`
   controls what a reader lands on, **not** what they can reach: `ccin2p3` is equally
   public, so `jobs/`, `.claude/` and `docs/results.tex` are readable regardless. Anything
   that must actually stay private has to be **gitignored** — as `docs/improvement-plan.md`
   is, because it carries the collaboration strategy about the upstream author — or live
   in a separate repo. Check this before adding a file, not after pushing it.
5. **Never commit run artifacts, strain, checkpoints, or `.reference/`.** `.gitignore`
   covers them. Commit code, configs, `docs/`, curated `figures/`.

(Ground rule #5 always holds: never attribute commits/PRs to Claude/AI.)

---

## Glossary

| Term | Meaning |
|---|---|
| ASD / PSD | Amplitude / power spectral density. The noise spectrum; effectively the noise covariance, diagonal in Fourier space for stationary noise. |
| Strain h(t) | Fractional arm-length change ΔL/L read out by the interferometer, ~10⁻²¹. |
| Whitening | Dividing the Fourier transform by the ASD so noise is unit-variance and flat. |
| Q-transform | Constant-Q time-frequency transform; Q = centre frequency / bandwidth. |
| Chirp | Upward-sweeping frequency track from an inspiralling binary; `f ∝ (t_c − t)^(−3/8)`. |
| CBC | Compact binary coalescence — merging black holes or neutron stars. |
| Glitch | Transient non-Gaussian noise burst, ~1/minute per detector. A "blip" closely resembles a high-mass merger. |
| Injection | Simulated waveform added to real noise. The MC-signal-overlaid-on-data equivalent. |
| Approximant | Semi-analytic waveform model, e.g. IMRPhenomPv2, IMRPhenomXPHM. |
| SNR | Matched-filter signal-to-noise ratio; network SNR is the quadrature sum. ~8 is the conventional detection threshold. |
| Time slide | Pairing H1 at *t* with L1 at *t + Δ*, Δ ≫ 11 ms. The background-estimation method. |
| FAR | False alarm rate, in events/year. |
| VT | Sensitive volume × time. The reach metric. |
| p_astro | Probability a candidate is astrophysical rather than noise; computed via FGMC. |
| H1 / L1 | LIGO Hanford / Livingston, ~3000 km apart, ~10 ms light travel time. |
| O3a … O4b | LIGO observing runs 3 and 4, each split in two. |
| GWOSC / GWTC | Open science centre (public strain) / the official transient catalog. |
| ET | Einstein Telescope, the planned third-generation European detector. |
