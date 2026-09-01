"""Stage 1: unsupervised reconstruction model of the detector-noise distribution.

The upstream stage 1 is a convolutional autoencoder trained on noise-only tiles with
an MSE reconstruction objective — 10 epochs, Adam lr 1e-3, weight decay 1e-5, batch
64, ReduceLROnPlateau(factor 0.5, patience 5). None of that training code is in the
release, so this module is the reimplementation, and reproducing it is the gate that
section 3.4 of the plan puts in front of every experiment.

Constraint C3: stage 1 stays self-supervised. It sees noise only and never a signal
label. `objective: masked` swaps the autoencoding objective for masked patch
prediction (Phase 4.1) without touching that property — a masked predictor cannot see
the patch it must predict, so it cannot satisfy the objective by learning a generic
compressor, which is the failure mode the plain autoencoder has.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from madgrav_ml.base_experiment import BaseExperiment
from madgrav_ml.eval.folds import Split
from madgrav_ml.logger import LOGGER
from madgrav_ml.models.cae import BaselineCAE


def random_mask(shape, ratio: float, patch: tuple[int, int], generator=None) -> torch.Tensor:
    """Boolean mask over a (B, 1, F, T) tile, True where the patch is hidden.

    `patch` is (frequency, time). Anisotropy is the point: masking a frequency band
    across a time span and masking a time slice across all frequencies probe different
    structures, and the plan asks for them ablated separately. A chirp is a track, so a
    mask that removes a *time* slice is the one it cannot inpaint from surrounding noise.
    """
    b, _, f, t = shape
    pf, pt = patch
    gf, gt = max(1, f // pf), max(1, t // pt)
    grid = torch.rand(b, 1, gf, gt, generator=generator) < ratio
    return F.interpolate(grid.float(), size=(f, t), mode="nearest").bool()


class Stage1CAEExperiment(BaseExperiment):
    """Self-supervised noise model. Objective is `reconstruction` or `masked`."""

    def init_physics(self):
        from madgrav_ml.data.representation import TileSpec

        self.spec = TileSpec(**self.cfg.representation)
        LOGGER.info(
            f"Representation: {self.spec.n_channels}ch, {self.spec.size[0]}x{self.spec.size[1]}, "
            f"amplitude={self.spec.amplitude}, phase_channel={self.spec.phase_channel}"
        )

    def init_data(self):
        """Noise-only tiles from the training fold. The eval fold is not opened here.

        Which HPO subset is used depends on the phase the caller opened: a plain
        training run reads TRAIN, an HPO trial reads HPO_TRAIN / HPO_VAL. That is the
        guard's decision, not this method's.
        """
        from madgrav_ml.data.tiles import CachedTileDataset, GeneratedTileDataset

        with self.guard.training(f"stage1:{self.cfg.run_name}"):
            train_segments = self.guard.segments(Split.HPO_TRAIN)
            val_segments = self.guard.segments(Split.HPO_VAL)
        LOGGER.info(
            f"Noise pool: {len(train_segments)} train / {len(val_segments)} val segments"
        )

        if self.cfg.data.source == "cached":
            self.train_set = CachedTileDataset(self.cfg.data.train_shards)
            self.val_set = CachedTileDataset(self.cfg.data.val_shards)
        elif self.cfg.data.source == "generated":
            # Phase 2: effectively infinite non-repeating noise. signal_fraction=0
            # because C3 forbids signal labels at stage 1.
            from madgrav_ml.data.representation import make_tile

            self.train_set = GeneratedTileDataset(
                noise_provider=self._noise_provider(train_segments),
                tile_fn=lambda w: make_tile(w, self.spec),
                signal_fraction=0.0,
                seed=self.cfg.seed,
            )
            self.val_set = GeneratedTileDataset(
                noise_provider=self._noise_provider(val_segments),
                tile_fn=lambda w: make_tile(w, self.spec),
                signal_fraction=0.0,
                seed=(self.cfg.seed or 0) + 1,
                length=self.cfg.data.val_tiles,
            )
        else:
            raise ValueError(f"unknown data.source {self.cfg.data.source!r}")

    def _noise_provider(self, segments):
        """Draw a whitened tile-length stretch of real noise from `segments`.

        Reads from the local cache via `SegmentReader` — never from GWOSC. A random
        four-second window is a cache miss under the `(ifo, start, end)` key, so calling
        `fetch_strain` here would issue one archive request per training tile.
        """
        from madgrav_ml.data.representation import notch_and_highpass, whiten
        from madgrav_ml.data.strain import SegmentReader, available_segments, load_reference_psd

        fs = self.spec.sample_rate
        cache = self.cfg.data.strain_cache
        window = float(self.cfg.data.window_seconds)

        have = available_segments(segments, cache)
        missing = len(segments) - len(have)
        if not have:
            raise FileNotFoundError(
                f"none of the {len(segments)} segments are cached under {cache}. "
                f"Warm it first: scripts/remote.sh sbatch jobs/job_fetch_strain.sh"
            )
        if missing:
            # Loud, not silent: training on a fraction of the fold you think you have is
            # the kind of thing that turns into an unexplained result three weeks later.
            LOGGER.warning(
                f"{missing} of {len(segments)} segments are not cached and will not be "
                f"sampled ({sum(s.duration for s in have) / 86400:.2f} d available)"
            )

        reader = SegmentReader(cache, capacity=int(self.cfg.data.get("reader_capacity", 2)))
        psds = {ifo: load_reference_psd(p) for ifo, p in self.cfg.data.reference_psd.items()}
        lines = {ifo: tuple(v) for ifo, v in self.cfg.data.notch_lines.items()}

        def provider(rng):
            seg, raw = reader.random_window(have, rng, window, fs)
            w = whiten(raw, fs, reference_psd=psds[seg.ifo])
            return notch_and_highpass(w, fs, lines=lines[seg.ifo])

        return provider

    def _init_dataloader(self):
        from torch.utils.data import DataLoader

        t = self.cfg.training
        kw = dict(batch_size=t.batchsize, num_workers=t.num_workers, pin_memory=True)
        self.train_loader = DataLoader(self.train_set, **kw)
        self.val_loader = DataLoader(self.val_set, **kw)

    def init_model(self):
        self.model = BaselineCAE(
            in_channels=self.spec.n_channels, dropout=self.cfg.model.dropout
        ).to(self.device)
        if self.cfg.model.get("init_from", None):
            state = torch.load(self.cfg.model.init_from, map_location=self.device,
                               weights_only=False)
            state = state.get("model", state)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            LOGGER.info(
                f"Loaded {self.cfg.model.init_from} "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
            )

    def _init_loss(self):
        self.objective = self.cfg.model.objective
        if self.objective not in ("reconstruction", "masked"):
            raise ValueError(f"unknown stage-1 objective {self.objective!r}")
        self.mask_ratio = self.cfg.model.get("mask_ratio", 0.5)
        self.mask_patch = tuple(self.cfg.model.get("mask_patch", (16, 8)))

    def _batch_loss(self, data):
        x = data[0].to(self.device, non_blocking=True)
        if self.objective == "reconstruction":
            recon, _ = self.model(x)
            loss = F.mse_loss(recon, x)
            return loss, {"mse": float(loss.detach())}

        mask = random_mask(x.shape, self.mask_ratio, self.mask_patch).to(x.device)
        recon, _ = self.model(x.masked_fill(mask, 0.0))
        # scored on the hidden patches only: predicting the visible part is free and
        # would dilute the signal the objective is supposed to carry
        loss = F.mse_loss(recon[mask], x[mask])
        return loss, {"masked_mse": float(loss.detach())}

    @torch.no_grad()
    def evaluate(self):
        """Per-tile score distribution on held-out noise — the input to the sigma calibration.

        Nothing here touches the evaluation fold: stage-1 evaluation is a diagnostic on
        the training fold's validation subset. The FAR-carrying numbers come later,
        from `experiments.matched_far`.
        """
        self.model.eval()
        errs = []
        for data in self.val_loader:
            x = data[0].to(self.device)
            errs.append(self.model.reconstruction_error(x, reduction="none").cpu().numpy())
        errs = np.concatenate(errs) if errs else np.array([])
        self.noise_scores = errs   # consumed by plot(): the score-separation figure
        self.secondary_metrics = {
            "noise_score_mean": float(errs.mean()) if errs.size else float("nan"),
            "noise_score_std": float(errs.std()) if errs.size else float("nan"),
            "noise_score_p99": float(np.percentile(errs, 99)) if errs.size else float("nan"),
            "n_val_tiles": int(errs.size),
        }
        LOGGER.info(f"Noise score distribution: {self.secondary_metrics}")
        if self.cfg.save:
            np.save(f"{self.cfg.run_dir}/preds/noise_scores.npy", errs)
