"""The shared experiment harness.

Structure and method names follow `Foundational_Amplitudes`' `base_experiment.py`, so
the two projects read the same way: a Hydra config drives

    init_physics -> init_data -> init_model -> train -> evaluate -> plot

with every piece of ML boilerplate as a private `_method`. What is different here is
what the harness *enforces*, because this project quotes a false-alarm rate and the
upstream one does not:

* a `FoldGuard` must be installed before data is touched (constraint C4);
* measured parameter counts are recorded and checked against the C2 budget;
* the run directory contract is fixed, and the run ends by writing an
  `ExperimentRecord` that refuses to serialise if a claim would be unsupportable.

Subclasses implement `init_physics`, `init_data`, `init_model`, `_init_loss`,
`_batch_loss`, `evaluate` and `plot`.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import madgrav_ml.logger as logger_module
from madgrav_ml.eval.folds import FoldGuard
from madgrav_ml.logger import FORMATTER, LOGGER, MEMORY_HANDLER
from madgrav_ml.misc import NaNError, cosine_warmup_scheduler, get_device, set_seed
from madgrav_ml.models.param_budget import check_budget, count_parameters, load_reference
from madgrav_ml.report.record import ExperimentRecord

torch.autograd.set_detect_anomaly(False)


class BaseExperiment:
    """One run. `__call__` executes it and routes every exception to the logger."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.guard: FoldGuard | None = None
        self.param_counts: dict[str, int] = {}
        self.budget_verdicts: list = []

    # ---- lifecycle ---------------------------------------------------------

    def __call__(self):
        try:
            self._init()
            self.full_run()
        except Exception:
            LOGGER.exception("Exiting with error")
            raise
        finally:
            if not logger_module.LOGGING_INITIALIZED:
                # nothing was ever flushed; dump the buffered records so a failure
                # during init is not silent
                stream = logging.StreamHandler()
                stream.setLevel(logging.DEBUG)
                MEMORY_HANDLER.setTarget(stream)
                MEMORY_HANDLER.close()

    def full_run(self):
        t0 = time.time()
        LOGGER.debug(OmegaConf.to_yaml(self.cfg))
        self._save_config("config.yaml")

        self.init_physics()
        self.init_folds()
        self.init_data()
        self._init_dataloader()
        self.init_model()
        self._check_param_budget()
        self._init_loss()

        if self.device.type == "cuda":
            free_mem, total_mem = torch.cuda.mem_get_info()
            LOGGER.info(f"VRAM: {free_mem / 1024**2:.0f} MB free of {total_mem / 1024**2:.0f} MB")
            torch.cuda.reset_peak_memory_stats()

        if self.cfg.train:
            self._init_optimizer()
            self._init_scheduler()
            self.train()
            self._save_model()

        if self.cfg.evaluate:
            self.evaluate()
        if self.cfg.plot and self.cfg.save:
            self.plot()

        if self.device.type == "cuda":
            LOGGER.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

        self.gpu_hours = (time.time() - t0) / 3600.0 if self.device.type == "cuda" else 0.0
        self._write_record()

        dt = time.time() - t0
        LOGGER.info(
            f"Finished {self.cfg.exp_name}/{self.cfg.run_name} "
            f"after {dt / 60:.1f} min ({dt / 3600:.2f} h)"
        )

    # ---- init --------------------------------------------------------------

    def _init(self):
        self._init_experiment()
        self._init_directory()
        self._init_logger()
        self._init_backend()

    def _init_experiment(self):
        if self.cfg.run_name is None:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = self.cfg.get("tag", self.cfg.exp_type)
            with open_dict(self.cfg):
                self.cfg.run_name = f"{now}_{tag}_{np.random.randint(0, 10000):04d}"
        if self.cfg.get("run_dir", None) is None:
            with open_dict(self.cfg):
                self.cfg.run_dir = os.path.join(
                    self.cfg.base_dir, "runs", self.cfg.exp_name, self.cfg.run_name
                )
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)
            LOGGER.info(f"Using seed {self.cfg.seed}")

    def _init_directory(self):
        if not self.cfg.save:
            LOGGER.info("save=False — no outputs will be written")
            return
        run_dir = Path(self.cfg.run_dir).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(
                f"run directory {run_dir} already exists and is not empty. Runs are "
                f"immutable records; pick a new run_name rather than overwriting one."
            )
        for sub in ("", "models", "preds", "plots"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

    def _init_logger(self):
        if logger_module.LOGGING_INITIALIZED:
            return
        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        if self.cfg.save:
            fh = logging.FileHandler(Path(self.cfg.run_dir) / "out.log")
            fh.setFormatter(FORMATTER)
            fh.setLevel(logging.DEBUG)
            LOGGER.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setLevel(LOGGER.level)
        sh.setFormatter(FORMATTER)
        LOGGER.addHandler(sh)
        MEMORY_HANDLER.setTarget(sh)
        MEMORY_HANDLER.close()
        LOGGER.removeHandler(MEMORY_HANDLER)
        LOGGER.propagate = False
        logger_module.LOGGING_INITIALIZED = True

    def _init_backend(self):
        self.device = get_device()
        LOGGER.info(f"Using device {self.device}")
        if self.device.type != "cuda" and not self.cfg.get("allow_cpu", False):
            raise RuntimeError(
                "No GPU. The upstream README is explicit that the GPU forward pass is "
                "the calibrated path and CPU forward is not byte-identical, so "
                "production and background (FAR) runs must be on GPU. Set "
                "allow_cpu=true only for a smoke test whose numbers you will not quote."
            )
        if self.device.type == "cuda" and self.cfg.training.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def init_folds(self):
        """Install the FoldGuard. Runs before `init_data`, and is not optional.

        A subclass may override to build folds differently, but must set `self.guard`.
        Nothing downstream reads segments except through it.
        """
        from madgrav_ml.data.strain import load_segments

        fcfg = self.cfg.folds
        segs = []
        for ifo, path in fcfg.segment_files.items():
            segs.extend(load_segments(path, ifo=ifo))
        self.guard = FoldGuard.from_segments(
            segs,
            eval_fold=fcfg.eval_fold,
            n_folds=fcfg.n_folds,
            hpo_val_frac=fcfg.hpo_val_frac,
            hpo_bg_frac=fcfg.get("hpo_bg_frac", 0.30),
            audit_path=os.path.join(self.cfg.run_dir, "fold_audit.jsonl")
            if self.cfg.save
            else None,
        )
        s = self.guard.summary()
        LOGGER.info(
            f"Folds: {s['n_folds']} GPS-grouped, eval={s['eval_fold']}, "
            f"train livetime {s['train_livetime_s'] / 86400:.1f} d, "
            f"eval livetime {s['eval_livetime_s'] / 86400:.1f} d"
        )

    def _check_param_budget(self):
        """Measure this run's components and hold them to C2."""
        modules = self.parameter_budget_components()
        self.param_counts = {name: count_parameters(m) for name, m in modules.items()}
        for name, n in self.param_counts.items():
            LOGGER.info(f"Parameters — {name}: {n:,}")
        ref_path = self.cfg.get("param_budget_reference", None)
        if ref_path is None:
            LOGGER.warning(
                "no param_budget_reference set — C2 is unenforced for this run. That "
                "is fine for a smoke test and not fine for anything reported."
            )
            return
        reference = load_reference(ref_path)
        strict = bool(self.cfg.get("enforce_param_budget", True))
        tol = float(self.cfg.get("param_budget_tolerance", 0.10))
        for name, n in self.param_counts.items():
            if name not in reference:
                LOGGER.warning(f"no C2 reference for component {name!r}; skipping")
                continue
            v = check_budget(name, n, reference[name], tolerance=tol, strict=strict)
            LOGGER.info(str(v))
            self.budget_verdicts.append(v)

    def parameter_budget_components(self) -> dict:
        """`{name: module}` for the C2 check. Override when a run has several models."""
        return {"model": self.model} if getattr(self, "model", None) is not None else {}

    # ---- optimisation ------------------------------------------------------

    def _init_optimizer(self):
        t = self.cfg.training
        groups = [{"params": self.model.parameters(), "lr": t.lr}]
        extra = {}
        if t.get("fused_optimizer", True) and torch.cuda.is_available() and t.optimizer in ("Adam", "AdamW"):
            extra["fused"] = True
        common = dict(betas=tuple(t.betas), eps=t.eps, weight_decay=t.weight_decay, **extra)
        if t.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(groups, **common)
        elif t.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(groups, **common)
        elif t.optimizer == "RAdam":
            common.pop("fused", None)
            self.optimizer = torch.optim.RAdam(groups, **common)
        else:
            raise ValueError(f"optimizer {t.optimizer} not implemented")
        LOGGER.debug(f"Optimizer {t.optimizer}, lr={t.lr}")

    def _init_scheduler(self):
        t = self.cfg.training
        if t.scheduler is None:
            self.scheduler = None
        elif t.scheduler == "CosineAnnealingLR":
            warmup = int(t.get("cosanneal_warmup_frac", 0.0) * t.iterations) or t.get(
                "cosanneal_warmup_steps", 0
            )
            self.scheduler = cosine_warmup_scheduler(
                self.optimizer, warmup, T_max=t.iterations, eta_min=t.cosanneal_eta_min
            )
        elif t.scheduler == "ReduceLROnPlateau":
            # the upstream stage-1/stage-2 setting: factor 0.5, patience 5
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=t.reduceplateau_factor,
                patience=t.reduceplateau_patience,
                # Upstream passes threshold=1e-3 in both train_unsup_model and
                # train_margin_model. Torch's default is 1e-4, i.e. ten times more
                # sensitive, which would drop the LR on improvements upstream ignores.
                threshold=t.get("reduceplateau_threshold", 1e-3),
            )
        elif t.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=t.lr * t.onecycle_max_lr,
                pct_start=t.onecycle_pct_start,
                total_steps=int(t.iterations),
            )
        else:
            raise ValueError(f"scheduler {t.scheduler} not implemented")
        LOGGER.debug(f"Scheduler {t.scheduler}")

    # ---- training ----------------------------------------------------------

    def train(self):
        t = self.cfg.training
        self.train_loss, self.val_loss, self.train_lr = [], [], []
        best, best_step = float("inf"), -1
        self.model.train()
        it = self._cycle(self.train_loader)

        for step in range(t.iterations):
            data = next(it)
            loss, parts = self._step(data, step)
            self.train_loss.append(loss)
            self.train_lr.append(self.optimizer.param_groups[0]["lr"])

            if step % t.log_every_n_steps == 0:
                extra = " ".join(f"{k}={v:.4g}" for k, v in parts.items())
                LOGGER.info(f"step {step:>7d}  loss={loss:.5g}  {extra}")

            if t.validate_every_n_steps and step % t.validate_every_n_steps == 0:
                vl = self._validate(step)
                self.val_loss.append((step, vl))
                if vl < best:
                    best, best_step = vl, step
                    self._save_model(step="best")
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # A subclass whose model-selection criterion is not its loss sets
                    # `_scheduler_metric` inside `_validate`. Stage 2 does: upstream
                    # selects on a detection criterion but steps the LR on the
                    # validation loss, and driving ReduceLROnPlateau with the selection
                    # criterion instead would silently rewrite the LR schedule.
                    self.scheduler.step(getattr(self, "_scheduler_metric", None) or vl)

            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()

        LOGGER.info(f"Best validation loss {best:.5g} at step {best_step}")
        self.best_val_loss, self.best_step = best, best_step
        if t.es_load_best_model and best_step >= 0 and self.cfg.save:
            self._load_model("best")

    def _step(self, data, step):
        self.optimizer.zero_grad(set_to_none=True)
        loss, parts = self._batch_loss(data)
        if not torch.isfinite(loss):
            raise NaNError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        if self.cfg.training.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.clip_grad_norm
            )
        self.optimizer.step()
        return float(loss.detach()), parts

    @torch.no_grad()
    def _validate(self, step):
        self.model.eval()
        losses = []
        for data in self.val_loader:
            loss, _ = self._batch_loss(data)
            losses.append(float(loss))
        self.model.train()
        vl = float(np.mean(losses)) if losses else float("nan")
        LOGGER.info(f"step {step:>7d}  val={vl:.5g}")
        return vl

    @staticmethod
    def _cycle(iterable):
        while True:
            for item in iterable:
                yield item

    # ---- persistence -------------------------------------------------------

    def _save_config(self, filename):
        if not self.cfg.save:
            return
        path = Path(self.cfg.run_dir) / filename
        path.write_text(OmegaConf.to_yaml(self.cfg))

    def _save_model(self, step="end", filename=None):
        if not self.cfg.save:
            return
        filename = filename or f"model_{step}.pt"
        path = Path(self.cfg.run_dir) / "models" / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": getattr(self, "optimizer", None).state_dict()
                if getattr(self, "optimizer", None) is not None
                else None,
                "step": step,
                "param_counts": self.param_counts,
            },
            path,
        )
        LOGGER.debug(f"Saved {path}")

    def _load_model(self, step="best"):
        path = Path(self.cfg.run_dir) / "models" / f"model_{step}.pt"
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        LOGGER.info(f"Loaded {path}")

    def _write_record(self):
        """Write `summary.json`. Refuses on an unsupportable claim — see ExperimentRecord."""
        if not self.cfg.save:
            return
        rec = ExperimentRecord(
            name=self.cfg.run_name,
            hypothesis=self.cfg.get("hypothesis", ""),
            change=self.cfg.get("change", ""),
            parameters={
                "counts": self.param_counts,
                "c2": [v.as_dict() for v in self.budget_verdicts],
            },
            folds=self.guard.summary() if self.guard is not None else {},
            primary=getattr(self, "primary_metrics", {}),
            secondary=getattr(self, "secondary_metrics", {}),
            seeds=list(self.cfg.get("seeds", [])) or ([self.cfg.seed] if self.cfg.seed is not None else []),
            compute_gpu_hours=getattr(self, "gpu_hours", None),
            verdict=self.cfg.get("verdict", "needs-more-work"),
            reasoning=self.cfg.get("reasoning", "not yet assessed"),
            config=json.loads(json.dumps(OmegaConf.to_container(self.cfg, resolve=True), default=str)),
        )
        if self.guard is not None:
            self.guard.assert_eval_untouched_by_tuning()
        try:
            path = rec.save(self.cfg.run_dir)
            LOGGER.info(f"Wrote {path}")
        except ValueError as exc:
            # Do not fail the run over an incomplete record — but do not pretend it is
            # complete either. Write it under a name that cannot be mistaken for one.
            LOGGER.warning(f"Record incomplete, writing summary_incomplete.json:\n{exc}")
            (Path(self.cfg.run_dir) / "summary_incomplete.json").write_text(
                json.dumps(rec.as_dict(), indent=2, sort_keys=True, default=str)
            )

    # ---- subclass hooks ----------------------------------------------------

    def init_physics(self):
        """Resolve detector/run constants and the reference PSDs."""

    def init_data(self):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        """Emit the standard figure set. Override to add more, never to replace it."""
        from madgrav_ml.plotting import plot_standard

        written = plot_standard(
            self.cfg.run_dir,
            train_loss=getattr(self, "train_loss", None),
            val_loss=getattr(self, "val_loss", None),
            lr=getattr(self, "train_lr", None),
            noise_scores=getattr(self, "noise_scores", None),
            signal_scores=getattr(self, "signal_scores", None),
        )
        for png, _pdf in written:
            LOGGER.info(f"Wrote {png} (+ .pdf)")

    def _init_dataloader(self):
        raise NotImplementedError

    def _init_loss(self):
        raise NotImplementedError

    def _batch_loss(self, data) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError
