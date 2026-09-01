"""Small shared utilities (ported from Foundational_Amplitudes)."""

import math
from collections.abc import Mapping

import torch
from torch.optim.lr_scheduler import LambdaLR


class NaNError(BaseException):
    """Raised when training encounters a NaN in the loss or in the model weights."""


def get_device() -> torch.device:
    """CUDA if available, CPU otherwise.

    Note the upstream README's warning: the GPU forward pass is the *calibrated* path,
    and CPU forward is not byte-identical. Production and background (FAR) runs must
    run on GPU; CPU is for smoke tests only.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dict with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def frequency_check(step, every_n_steps, skip_initial=False):
    """Whether an action due `every_n_steps` should fire at (one-indexed) `step`."""
    if every_n_steps is None or every_n_steps == 0:
        return False
    if skip_initial and step == 0:
        return False
    return step % every_n_steps == 0


def cosine_warmup_scheduler(optimizer, warmup_steps, T_max, eta_min=0):
    """Cosine annealing with linear warmup, as a LambdaLR on the optimizer's base lr."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, T_max - warmup_steps))
        base = optimizer.defaults["lr"]
        floor = eta_min / base if base else 0.0
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def set_seed(seed):
    """Seed python/numpy/torch. Every claimed improvement is reported over >= 3 seeds."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
