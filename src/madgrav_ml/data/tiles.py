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
    """Tiles precomputed to disk — a bank built by `scripts/build_tile_cache.py`.

    Loaded eagerly into one array. That is a deliberate choice over lazy per-shard
    loading: a 20k-tile bank at 1x256x128 float32 is 2.6 GB, which fits comfortably, and
    the lazy version has a trap. With `num_workers > 0` each DataLoader worker gets its
    own copy of the dataset object and lazily loads whatever shards it touches — and
    under shuffling every worker touches every shard, so the memory is (bank size) x
    (workers). Eight workers over this bank would be 21 GB to read data that needs no
    per-item work at all.

    So: load once, and use `num_workers=0`. Reading from a bank is an array slice; there
    is nothing to parallelise.
    """

    def __init__(self, paths, max_gb: float = 8.0):
        """`paths` is a list of shard files, or a glob like
        `data_cache/tiles/train/*.npz`. A glob because a bank is written as however many
        shards it needs, and a config enumerating them goes stale the moment the bank is
        rebuilt at a different size."""
        import glob as _glob

        if isinstance(paths, (str, Path)):
            paths = sorted(_glob.glob(str(paths)))
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError(
                "CachedTileDataset got no shards. Build the bank first: "
                "scripts/remote.sh sbatch jobs/job_build_tiles.sh"
            )

        xs, ys = [], []
        for p in self.paths:
            with np.load(p) as z:
                xs.append(np.asarray(z["x"], dtype=np.float32))
                ys.append(np.asarray(z["y"], dtype=np.float32) if "y" in z
                          else np.full(len(xs[-1]), np.nan, dtype=np.float32))
        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        gb = self.x.nbytes / 1e9
        if gb > max_gb:
            raise MemoryError(
                f"bank is {gb:.1f} GB, over the {max_gb} GB guard. Either raise max_gb "
                f"deliberately or build a smaller bank — silently swapping is worse than "
                f"failing here."
            )

    @property
    def size_gb(self) -> float:
        return self.x.nbytes / 1e9

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.x[i]), torch.tensor(float(self.y[i]))


class GeneratedTileDataset(IterableDataset):
    """Effectively infinite non-repeating tiles: noise from the allowed fold, plus
    freshly drawn injections and glitches.

    `noise_provider(rng)` must draw only from segments the current `FoldGuard` phase
    permits. That is not enforced here on purpose — the guard owns that decision, and
    duplicating the check in two places is how the two copies drift apart. Wire it as
    `noise_provider=lambda rng: sampler(guard.segments(Split.TRAIN), rng)`.

    It returns either a whitened strain array, or `(strain, context)` where the context
    describes the window it came from. The context exists for the injector: the antenna
    pattern and the geocentre time delay are functions of the detector and the GPS time,
    so an injection into a window that cannot say which detector it is or when it
    happened is not a projected signal, it is a waveform pasted onto noise. Stage 1
    ignores it (`signal_fraction=0`), which is why the second element stays optional
    rather than forcing a change on a path that already works.

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
            drawn = self.noise_provider(rng)
            strain, context = drawn if isinstance(drawn, tuple) else (drawn, None)
            label = 0.0
            if self.signal_fraction and rng.random() < self.signal_fraction:
                strain = self.injector(strain, rng, context)
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
