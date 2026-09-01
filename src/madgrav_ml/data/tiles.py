"""Tile datasets: the fixed cached kind and the on-the-fly generated kind.

Both expose the same interface so an experiment can swap them from config, which is
what makes the Phase-2 comparison ("does unlimited data beat the 11k fixed set?") a
one-line change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # torch is an optional extra; the eval harness must import without it
    import torch
    from torch.utils.data import Dataset, IterableDataset
except ImportError:  # pragma: no cover - exercised only in torch-free environments
    torch = None
    Dataset = object
    IterableDataset = object


@dataclass
class TileBatch:
    """A batch of tiles with their labels and provenance.

    `source` is carried because the fold audit needs to know which segment a tile came
    from; a tile with no provenance cannot be assigned to a fold and so cannot appear
    in anything that quotes a FAR.
    """

    x: "np.ndarray | torch.Tensor"
    y: "np.ndarray | torch.Tensor | None"
    source: list[str]


class CachedTileDataset(Dataset):
    """Tiles precomputed to disk — the baseline's fixed ~11k set.

    Expects an `.npz` per shard with arrays `x` (N, C, F, T), `y` (N,) and `source`
    (N,). Kept deliberately dumb: caching is a performance decision, and anything
    clever here would make the on-the-fly comparison unfair.
    """

    def __init__(self, paths: list[str | Path]):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("CachedTileDataset needs at least one shard")
        self._index: list[tuple[int, int]] = []
        self._shards: dict[int, dict] = {}
        for si, p in enumerate(self.paths):
            with np.load(p) as z:
                n = z["x"].shape[0]
            self._index.extend((si, i) for i in range(n))

    def __len__(self) -> int:
        return len(self._index)

    def _shard(self, si: int) -> dict:
        if si not in self._shards:
            with np.load(self.paths[si]) as z:
                self._shards[si] = {k: z[k] for k in ("x", "y", "source") if k in z}
        return self._shards[si]

    def __getitem__(self, i: int):
        si, j = self._index[i]
        s = self._shard(si)
        x = torch.from_numpy(np.asarray(s["x"][j], dtype=np.float32))
        y = torch.tensor(float(s["y"][j])) if "y" in s else torch.tensor(float("nan"))
        return x, y


class GeneratedTileDataset(IterableDataset):
    """Effectively infinite non-repeating tiles: noise from the allowed fold, plus
    freshly drawn injections and glitches.

    `noise_provider(rng) -> whitened strain` must draw only from segments the current
    `FoldGuard` phase permits. That is not enforced here on purpose — the guard owns
    that decision, and duplicating the check in two places is how the two copies drift
    apart. Wire it as `noise_provider=lambda rng: sampler(guard.segments(Split.TRAIN), rng)`.

    With `seed=None` every worker gets an independent non-repeating stream. With an
    integer seed the stream is deterministic per (seed, worker), which is what a single
    reported run needs.
    """

    def __init__(
        self,
        noise_provider,
        tile_fn,
        signal_fraction: float = 0.5,
        injector=None,
        seed: int | None = None,
        length: int | None = None,
    ):
        if not 0.0 <= signal_fraction <= 1.0:
            raise ValueError(f"signal_fraction must be in [0,1], got {signal_fraction}")
        if signal_fraction > 0 and injector is None:
            raise ValueError("signal_fraction > 0 needs an injector")
        self.noise_provider = noise_provider
        self.tile_fn = tile_fn
        self.signal_fraction = signal_fraction
        self.injector = injector
        self.seed = seed
        self.length = length

    def _rng(self) -> np.random.Generator:
        worker = 0
        if torch is not None:
            info = torch.utils.data.get_worker_info()
            worker = info.id if info is not None else 0
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, worker])

    def __iter__(self):
        rng = self._rng()
        n = 0
        while self.length is None or n < self.length:
            strain = self.noise_provider(rng)
            label = 0.0
            if self.signal_fraction and rng.random() < self.signal_fraction:
                strain = self.injector(strain, rng)
                label = 1.0
            x = self.tile_fn(strain)
            if torch is not None:
                yield torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(label)
            else:
                yield np.asarray(x, dtype=np.float32), label
            n += 1

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError(
                "GeneratedTileDataset is unbounded by design; pass `length` if a "
                "sized epoch is needed (e.g. to make a step count comparable to the "
                "cached-dataset baseline)"
            )
        return self.length


def balanced_sampler_weights(labels: np.ndarray) -> np.ndarray:
    """Per-sample weights that equalise the classes.

    The baseline's 10:1 glitch:signal imbalance is one of the two things Phase 2 is
    meant to fix (the other being the sample size). Focal loss is the alternative;
    ablate them against each other rather than stacking both without measuring.
    """
    y = np.asarray(labels).ravel()
    classes, counts = np.unique(y, return_counts=True)
    weight = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight[v] for v in y], dtype=np.float64)
