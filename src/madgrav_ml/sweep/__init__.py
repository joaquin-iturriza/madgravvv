"""Hyperparameter search, with the fold discipline built into the loop.

FA keeps a `sweep/` package around a DyHPO multi-fidelity surrogate. The structure is
adopted here, and one thing is added that FA does not need: **every trial runs inside an
open `FoldGuard.hpo()` phase**, so a trial physically cannot read the evaluation fold and
every trial is written to an audit trail with its fold assignment.

That is not decoration. `hpo_guard.sh` blocks a hand-rolled `sbatch --array` over an HP
partly because such an array has no fold record at all; this package is the thing it
points at instead.

What is ported and what is not: the trial loop, the search space, the audit trail and the
fold enforcement are here. The DyHPO **surrogate** is not — `SweepRunner` currently draws
from a Sobol-scrambled quasi-random sampler behind the `Sampler` protocol. Porting
`sweep/dyhpo_sampler.py` from `.reference/Foundational_Amplitudes` is a drop-in
replacement for that one object and changes nothing else here.
"""

from .search_space import SEARCH_SPACES, Parameter, SearchSpace
from .runner import RandomSampler, Sampler, SweepRunner, Trial

__all__ = [
    "SEARCH_SPACES",
    "Parameter",
    "RandomSampler",
    "Sampler",
    "SearchSpace",
    "SweepRunner",
    "Trial",
]
