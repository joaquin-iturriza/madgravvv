# madgrav

Machine-learning improvements to **MADGRAV**, a blind gravitational-wave anomaly
search ([ginguglia/MADGRAV](https://github.com/ginguglia/MADGRAV), Phys. Lett. B 874
(2026) 140272, [arXiv:2511.13154](https://arxiv.org/abs/2511.13154)).

Two things this provides that the upstream release does not:

1. **A retrainable stage-1 / stage-2 front end.** The upstream package ships frozen
   weights but not the training or calibration code, so its convolutional autoencoder
   cannot be retrained. The model definitions here load the vendored checkpoints
   exactly, which makes a reimplementation validatable before anything is changed.
2. **A matched-FAR evaluation harness.** Detection efficiency and sensitive volume at
   a fixed false-alarm rate, measured against time-slide background pushed through the
   *same* selection as the foreground, with GPS-time-grouped folds and an audit trail
   proving the evaluation fold was never used for tuning.

Everything is built to the upstream author's constraints, so that results are usable
by him rather than merely favourable: a single-detector front end, a fixed parameter
budget, a self-supervised stage 1, strict fold discipline, and no `ml4gw`. See
`CLAUDE.md` for the operating manual and `docs/improvement-plan.md` for the plan.

## Install

```bash
bash scripts/vendor_reference.sh        # clone upstream into .reference/ (read-only)
python scripts/measure_param_budget.py  # measure the vendored weights -> C2 reference

# on a CC-IN2P3 login node
bash scripts/setup_env.sh               # venv + cu124 torch + editable install
```

`scripts/setup_env.sh` installs torch from the CUDA-12.4 index **first**, on purpose:
the current default wheel is CUDA 13, which dropped Volta (sm_70) and so will not run
on the V100 partition.

## Quickstart

Training is driven by [Hydra](https://hydra.cc). The entry point is `run.py`; the
config tree lives under `config/`.

```bash
# stage 1, upstream objective
python run.py exp_type=stage1 seed=42

# stage 1 with masked patch prediction instead of autoencoding
python run.py exp_type=stage1 model.objective=masked model.mask_patch=[256,8]

# representation ablation R1: keep the Q-transform phase as a second channel
python run.py exp_type=stage1 representation=r1_phase

# stage 2 margin fine-tune
python run.py --config-name=stage2 model.init_from=runs/madgrav/<run>/models/model_best.pt
```

On the cluster, submit rather than running directly:

```bash
scripts/remote.sh sbatch jobs/job_stage1.sh model.objective=masked
scripts/remote.sh sbatch jobs/job_seeds.sh exp_type=stage1   # 3 seeds
```

A run executes `init_physics → init_folds → init_data → init_model → train → evaluate
→ plot` and writes `config.yaml`, `out.log`, `models/`, `preds/`, `plots/`,
`fold_audit.jsonl` and `summary.json` under `runs/<exp_name>/<run_name>/`.

## Layout

| Path | What |
|------|------|
| `run.py` | Hydra entry point |
| `src/madgrav_ml/base_experiment.py` | generic harness: train loop, run-dir contract, fold guard, C2 check |
| `src/madgrav_ml/experiments/` | stage 1, stage 2, and the matched-FAR evaluation |
| `src/madgrav_ml/models/` | CAE, glitch arm, HM/LM specialists, parameter budget |
| `src/madgrav_ml/data/` | whitening, Q-transform representation, strain access, injections |
| `src/madgrav_ml/eval/` | fold guard, time slides, FAR, efficiency, VT, calibration |
| `src/madgrav_ml/report/` | the per-experiment record |
| `config/` | Hydra configs, and the measured parameter budget |
| `scripts/`, `jobs/` | cluster plumbing and SLURM submission |
| `docs/` | the improvement plan and the running notes |

## License

The upstream MADGRAV project is MIT-licensed. This repository is an independent
contribution to it and is not a fork; the upstream tree is vendored read-only under
`.reference/` and is not redistributed here.
