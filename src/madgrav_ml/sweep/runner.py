"""The trial loop, with the fold discipline enforced rather than trusted.

Every trial runs inside `FoldGuard.hpo(label, trial=i)`. In that phase the guard permits
`HPO_TRAIN` and `HPO_VAL` and refuses `TRAIN` and `EVAL`, so a trial cannot reach the
evaluation fold even by mistake, and each access is appended to `fold_audit.jsonl` tagged
with its trial number. The trial log this writes plus that audit file are the two things
an external reviewer needs to check the claim that tuning never saw the evaluation
background — which the plan calls the strongest available signal that a contributor
understands what kind of project this is.

`Sampler` is a protocol with one method, so porting FA's DyHPO surrogate later replaces
that object alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ..eval.folds import FoldGuard
from ..logger import LOGGER
from .search_space import SearchSpace


@dataclass
class Trial:
    """One evaluated point. Serialised in full — a trial with no fold record is noise."""

    index: int
    params: dict
    objective: float | None = None
    state: str = "pending"          # pending | done | failed
    seconds: float | None = None
    fold: dict = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Sampler(Protocol):
    """Proposes the next point. The seam where FA's DyHPO surrogate drops in."""

    def suggest(self, space: SearchSpace, history: list[Trial]) -> dict:
        ...

    def observe(self, trial: Trial) -> None:
        ...


class RandomSampler:
    """Scrambled Sobol, falling back to plain uniform when scipy is unavailable.

    Quasi-random rather than uniform because at the trial counts this project can afford
    — tens, not thousands — a uniform draw leaves visible holes and clusters in a 4-D
    space, and a low-discrepancy sequence covers it far more evenly for free. It is a
    baseline, not the destination: it does not learn from `observe`.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._engine = None

    def _unit(self, dim: int, n: int) -> np.ndarray:
        if self._engine is None:
            try:
                from scipy.stats import qmc

                self._engine = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
            except Exception:  # pragma: no cover - only on a scipy-less environment
                LOGGER.warning("scipy.stats.qmc unavailable; falling back to uniform draws")
                self._engine = np.random.default_rng(self.seed)
        if hasattr(self._engine, "random"):
            out = self._engine.random(n)
            return np.asarray(out).reshape(n, dim)
        return self._engine.uniform(size=(n, dim))  # pragma: no cover

    def suggest(self, space: SearchSpace, history: list[Trial]) -> dict:
        return space.from_unit(self._unit(len(space), 1)[0])

    def observe(self, trial: Trial) -> None:
        """No-op: this sampler does not learn. A surrogate would fit here."""


class SweepRunner:
    """Run `n_trials` of `objective` inside the training fold, logging every one.

    `objective(params, guard) -> float` is minimised. It receives the open guard so it
    can read `Split.HPO_TRAIN` and `Split.HPO_VAL`; reading anything else raises, which
    is the point.

    A failed trial is recorded as failed and the sweep continues. Silently dropping it
    would leave a trial log that disagrees with the number of jobs run, and the log is
    supposed to be auditable.
    """

    def __init__(
        self,
        space: SearchSpace,
        guard: FoldGuard,
        out_dir: str | Path,
        sampler: Sampler | None = None,
        label: str = "",
    ):
        self.space = space
        self.guard = guard
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sampler = sampler or RandomSampler()
        self.label = label or space.name
        self.trials: list[Trial] = []
        self.log_path = self.out_dir / "trials.jsonl"

    def run(self, objective: Callable[[dict, FoldGuard], float], n_trials: int) -> Trial:
        for i in range(n_trials):
            params = self.sampler.suggest(self.space, self.trials)
            trial = Trial(index=i, params=params)
            t0 = time.time()
            try:
                # The whole trial lives inside the HPO phase. Nothing here can reach the
                # evaluation fold, and every read is logged against this trial number.
                with self.guard.hpo(f"{self.label}", trial=i):
                    trial.objective = float(objective(params, self.guard))
                trial.state = "done"
            except Exception as exc:
                trial.state = "failed"
                trial.error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(f"trial {i} failed: {trial.error}")
            trial.seconds = time.time() - t0
            trial.fold = {
                "eval_fold": self.guard.eval_fold,
                "hpo_val_frac": self.guard.hpo_val_frac,
                "eval_fold_reads": self.guard._final_reports,
            }
            self.trials.append(trial)
            self.sampler.observe(trial)
            self._append(trial)
            if trial.state == "done":
                LOGGER.info(
                    f"trial {i:>3d}  obj={trial.objective:.6g}  "
                    + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in params.items())
                )

        # The evaluation fold must be exactly as untouched as when we started.
        self.guard.assert_eval_untouched_by_tuning()
        return self.best()

    def _append(self, trial: Trial) -> None:
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(trial.as_dict(), sort_keys=True, default=str) + "\n")

    def best(self) -> Trial:
        done = [t for t in self.trials if t.state == "done" and t.objective is not None]
        if not done:
            raise RuntimeError(
                f"no trial completed in sweep {self.label!r} — "
                f"{len(self.trials)} attempted, all failed"
            )
        return min(done, key=lambda t: t.objective)

    def summary(self) -> dict:
        """For the run record. Carries the search space: a best-HP set without the space
        it was found in cannot be reproduced or judged."""
        done = [t for t in self.trials if t.state == "done"]
        return {
            "label": self.label,
            "search_space": self.space.as_dict(),
            "n_trials": len(self.trials),
            "n_done": len(done),
            "n_failed": len(self.trials) - len(done),
            "best": self.best().as_dict() if done else None,
            "trial_log": str(self.log_path),
            "fold": self.guard.summary(),
        }
