"""Hydra entry point. Mirrors `Foundational_Amplitudes/run.py`.

    python run.py exp_type=stage1 model.objective=masked training.lr=3e-4
    python run.py --config-name=stage2 model.margin=2.5 model.margin_weight=1.5

Every field in `config/` is overridable from the CLI, and the resolved config is
snapshotted into the run directory, so a run is reproducible from its own outputs.
"""

import warnings

warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")

import hydra
import torch

EXPERIMENTS = {
    "stage1": "madgrav_ml.experiments.stage1_cae:Stage1CAEExperiment",
    "stage2": "madgrav_ml.experiments.stage2_margin:Stage2MarginExperiment",
}


def _resolve(exp_type: str):
    if exp_type not in EXPERIMENTS:
        raise ValueError(
            f"exp_type {exp_type!r} not implemented; known: {sorted(EXPERIMENTS)}"
        )
    module, cls = EXPERIMENTS[exp_type].split(":")
    return getattr(__import__(module, fromlist=[cls]), cls)


@hydra.main(config_path="config", config_name="madgrav", version_base=None)
def main(cfg):
    match cfg.training.dtype:
        case "float64":
            torch.set_default_dtype(torch.float64)
        case "float32":
            torch.set_default_dtype(torch.float32)
        case _:
            raise ValueError(f"dtype {cfg.training.dtype} not implemented")

    _resolve(cfg.exp_type)(cfg)()


if __name__ == "__main__":
    main()
