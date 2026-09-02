"""Stage 2: the weak-supervision margin fine-tune.

A reproduction, so every number here is read off `improved_pipeline.py` rather than the
paper. What that changes, relative to a reasonable reading of the description:

  * the margin is MULTIPLICATIVE and the noise term inside the hinge is DETACHED --
    `relu(m * mean(err_noise).detach() - mean(err_signal))`, see `models/cae.py`;
  * training pairs a noise loader with a signal loader and takes one batch of each per
    step (`zip(noise_train_loader, sig_train_loader)`), so both classes get the full
    batch size rather than half of a mixed batch;
  * the learning rate is `cfg.lr` = 1e-3, the SAME as stage 1, not a reduced fine-tuning
    rate;
  * the checkpoint is chosen by a DETECTION criterion, not by the loss --
    `val_inj_n_above_3sigma`, the number of validation injections scoring above
    `mean + 3 std` of the validation noise, tie-broken by the mean score separation.
    The LR scheduler still steps on the validation loss.

Upstream ran 10 epochs with `weaksup_es_patience = 10`, so early stopping could never
fire; "best at epoch 9 of 10" means longer training was never tested, which makes
training length a first-class HPO target alongside m and lambda (Phase 5).

This is the only place signal labels enter the front end. Stage 1 stays label-free (C3).
"""

from __future__ import annotations

import numpy as np
import torch

from madgrav_ml.experiments.stage1_cae import Stage1CAEExperiment
from madgrav_ml.logger import LOGGER
from madgrav_ml.models.cae import stage2_loss


class _Paired:
    """Zip two loaders into (noise_batch, signal_batch), re-iterable.

    `zip` stops at the shorter loader, which is upstream's behaviour and is why the two
    banks are built to the same size.
    """

    def __init__(self, noise, signal):
        self.noise, self.signal = noise, signal

    def __iter__(self):
        return zip(iter(self.noise), iter(self.signal))

    def __len__(self):
        return min(len(self.noise), len(self.signal))


class Stage2MarginExperiment(Stage1CAEExperiment):
    """Fine-tune a stage-1 CAE against noise/signal pairs with a hinge margin.

    Inherits the stage-1 data plumbing and overrides only the label-bearing parts, so
    the two stages cannot drift apart in their representation or fold handling.
    """

    def init_data(self):
        """Two banks: noise tiles and injected tiles from the *same* fold.

        `data.source == "cached"` reads a precomputed pair built by
        `scripts/build_tile_cache.py --inject`, whose windows are drawn from the same
        RNG stream as the noise bank's, so tile *i* of one is tile *i* of the other with
        a signal added. Otherwise both are generated on the fly (Phase 2).
        """
        from madgrav_ml.data.tiles import CachedTileDataset, GeneratedTileDataset
        from madgrav_ml.eval.folds import Split

        with self.guard.training(f"stage2:{self.cfg.run_name}"):
            train_segments = self.guard.segments(Split.HPO_TRAIN)
            val_segments = self.guard.segments(Split.HPO_VAL)

        if self.cfg.data.source == "cached":
            self.train_set = CachedTileDataset(self.cfg.data.train_shards)
            self.val_set = CachedTileDataset(self.cfg.data.val_shards)
            self.signal_train_set = CachedTileDataset(self.cfg.data.signal_train_shards)
            self.signal_val_set = CachedTileDataset(self.cfg.data.signal_val_shards)
            for name, noise, signal in (
                ("train", self.train_set, self.signal_train_set),
                ("val", self.val_set, self.signal_val_set),
            ):
                LOGGER.info(f"{name}: {len(noise)} noise tiles, {len(signal)} signal tiles")
                if len(noise) != len(signal):
                    # Not fatal — zip truncates — but it means the two banks were not
                    # built as a matched pair, and the margin would then be separating
                    # two different draws of the fold as much as noise from signal.
                    LOGGER.warning(
                        f"{name} banks are not the same size ({len(noise)} vs "
                        f"{len(signal)}); they are not a matched pair"
                    )
            return

        from madgrav_ml.data.representation import make_tile

        common = dict(
            tile_fn=lambda w: make_tile(w, self.spec),
            injector=self._injector(),
        )
        self.train_set = GeneratedTileDataset(
            noise_provider=self._noise_provider(train_segments), signal_fraction=0.0,
            seed=self.cfg.seed, **common)
        self.signal_train_set = GeneratedTileDataset(
            noise_provider=self._noise_provider(train_segments), signal_fraction=1.0,
            seed=self.cfg.seed, **common)
        self.val_set = GeneratedTileDataset(
            noise_provider=self._noise_provider(val_segments), signal_fraction=0.0,
            seed=(self.cfg.seed or 0) + 1, length=self.cfg.data.val_tiles, **common)
        self.signal_val_set = GeneratedTileDataset(
            noise_provider=self._noise_provider(val_segments), signal_fraction=1.0,
            seed=(self.cfg.seed or 0) + 1, length=self.cfg.data.val_tiles, **common)

    def _init_dataloader(self):
        """Four loaders, zipped into two. `self.val_loader` stays the paired one so the
        base class's validation call site is unchanged; the unpaired noise and signal
        val loaders are kept because the selection criterion needs them separately."""
        super()._init_dataloader()
        from torch.utils.data import DataLoader, IterableDataset

        t = self.cfg.training

        def make(ds, shuffle, drop_last=True):
            # drop_last on TRAIN only. On validation it would silently discard the tail
            # of the bank, which then no longer lines up with the per-injection metadata
            # the efficiency-versus-SNR cut needs (1984 scores against 2002 rows).
            iterable = isinstance(ds, IterableDataset)
            return DataLoader(ds, batch_size=t.batchsize, num_workers=0,
                              shuffle=(shuffle and not iterable), drop_last=drop_last)

        self.noise_train_loader = self.train_loader
        self.noise_val_loader = make(self.val_set, False, drop_last=False)
        self.signal_train_loader = make(self.signal_train_set, True)
        self.signal_val_loader = make(self.signal_val_set, False, drop_last=False)
        self.train_loader = _Paired(self.noise_train_loader, self.signal_train_loader)
        self.val_loader = _Paired(self.noise_val_loader, self.signal_val_loader)

    def _injector(self):
        """Add a freshly drawn, projected, SNR-scaled waveform to a noise window."""
        from madgrav_ml.data.injections import ParameterSampler

        backend = self.cfg.data.get("waveform_backend", None)
        if backend is None:
            raise NotImplementedError(
                "no waveform backend configured (data.waveform_backend). Set it to "
                "'lal' to use data/waveforms.py::LALWaveformBackend. Running stage 2 "
                "without one would train on noise labelled as signal."
            )
        if backend != "lal":
            raise ValueError(
                f"unknown data.waveform_backend {backend!r}; only 'lal' is implemented"
            )

        from madgrav_ml.data.waveforms import build_engine

        sampler = ParameterSampler(seed=self.cfg.seed,
                                   snr_range=tuple(self.cfg.data.snr_range))
        engine = build_engine(self.cfg.data, self.spec.sample_rate)
        LOGGER.info(
            f"Injections: {engine.backend.approximant_name} from "
            f"{engine.backend.f_lower} Hz, SNR {tuple(self.cfg.data.snr_range)} on the "
            f"'{engine.snr_convention}' convention over {sorted(engine.psds)}"
        )
        default_ifo = sorted(engine.psds)[0]

        def inject(strain, rng, context=None):
            ifo = context.ifo if context is not None else default_ifo
            gps = None if context is None else 0.5 * (context.start + context.end)
            # Redraw on an antenna-pattern null: measure-zero, but hit a few times in a
            # million-tile campaign, and crashing three hours in over a source that
            # happened to sit in a blind spot is a waste. A silent skip returning pure
            # noise with a signal label would be far worse, so the retry is bounded.
            for _ in range(8):
                params = sampler.draw(rng)
                try:
                    return engine.inject(strain, params, ifo, gps)
                except ValueError as exc:
                    if "zero SNR" not in str(exc):
                        raise
            raise RuntimeError(
                "eight consecutive draws had zero SNR in every detector; the SNR band "
                "or the reference PSDs are almost certainly wrong"
            )

        return inject

    def _init_loss(self):
        self.margin = float(self.cfg.model.margin)
        self.lam = float(self.cfg.model.margin_weight)
        LOGGER.info(f"Stage-2 margin m={self.margin} (multiplicative), lambda={self.lam}")

    # ---- loss and selection -------------------------------------------------

    def _errors(self, batch):
        x = batch[0].to(self.device, non_blocking=True)
        recon, _ = self.model(x)
        return ((recon - x) ** 2).flatten(1).mean(1)

    def _batch_loss(self, data):
        noise_batch, signal_batch = data
        return stage2_loss(self._errors(noise_batch), self._errors(signal_batch),
                           self.margin, self.lam)

    @torch.no_grad()
    def _scores(self, loader) -> np.ndarray:
        return np.concatenate([self._errors(b).cpu().numpy() for b in loader])

    @torch.no_grad()
    def _validate(self, step):
        """Upstream's selection criterion, returned as something to MINIMISE.

        `val_inj_n_above_3sigma` counts validation injections scoring above
        `mean + 3 std` of the validation noise; ties break on the mean separation. The
        base loop keeps the checkpoint with the smallest returned value, so this returns
        the negated criterion. `n` is a non-negative integer and the tie-break is
        squashed into (0, 1) by a strictly monotone map, which makes `-(n + frac)` an
        exact lexicographic encoding rather than a weighted blend of the two.

        `_scheduler_metric` carries the validation LOSS out separately: upstream steps
        ReduceLROnPlateau on the loss while selecting on the criterion, and conflating
        them would rewrite the LR schedule.
        """
        self.model.eval()
        losses = [float(self._batch_loss(d)[0]) for d in self.val_loader]
        self._scheduler_metric = float(np.mean(losses)) if losses else float("nan")

        noise = self._scores(self.noise_val_loader)
        signal = self._scores(self.signal_val_loader)
        self.model.train()

        threshold = noise.mean() + 3.0 * noise.std()
        n_above = int((signal > threshold).sum())
        separation = float(signal.mean() - noise.mean())
        frac = 0.5 * (1.0 + np.tanh(separation / (noise.std() + 1e-12)))

        LOGGER.info(
            f"step {step:>7d}  val={self._scheduler_metric:.5g}  "
            f"above3sigma={n_above}/{len(signal)}  sep={separation:.5g}  "
            f"noise_mu={noise.mean():.5g}"
        )
        self.val_detection = (step, n_above, len(signal), separation)
        return -(n_above + frac)

    # ---- evaluation ---------------------------------------------------------

    def evaluate(self):
        """Noise and injection score distributions, and the detection efficiency.

        These are DIAGNOSTICS on the training fold's validation subset, not results.
        The efficiency quoted here is against a threshold set by the validation noise
        itself, which is not a false-alarm rate: a FAR needs the time-slide background
        and the trials factor, and it comes from `experiments.matched_far`. The record
        will refuse to serialise this run as a result, and that is correct.
        """
        self.model.eval()
        noise = self._scores(self.noise_val_loader)
        signal = self._scores(self.signal_val_loader)
        self.model.train()
        self.noise_scores, self.signal_scores = noise, signal

        metrics = {
            "noise_score_mean": float(noise.mean()),
            "noise_score_std": float(noise.std()),
            "signal_score_mean": float(signal.mean()),
            "n_val_noise": int(noise.size),
            "n_val_signal": int(signal.size),
        }
        # Efficiency at a threshold set by the noise quantile, not by 3 sigma alone:
        # a Gaussian sigma on a distribution with excess kurtosis 25 is not a 0.13%
        # tail, so quote the quantile the threshold actually sits at as well.
        for label, thresh in (
            ("3sigma", noise.mean() + 3.0 * noise.std()),
            ("noise_p99", float(np.percentile(noise, 99.0))),
            ("noise_p999", float(np.percentile(noise, 99.9))),
        ):
            metrics[f"efficiency_at_{label}"] = float((signal > thresh).mean())
            metrics[f"threshold_{label}"] = float(thresh)
        metrics["noise_quantile_of_3sigma"] = float(
            (noise <= noise.mean() + 3.0 * noise.std()).mean()
        )

        snr = self._signal_metadata("network_snr")
        if snr is not None and len(snr) == len(signal):
            thresh = metrics["threshold_noise_p999"]
            edges = [8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
            curve = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (snr >= lo) & (snr < hi)
                curve.append((lo, hi, int(m.sum()),
                              float((signal[m] > thresh).mean()) if m.any() else float("nan")))
            metrics["efficiency_vs_snr_at_noise_p999"] = [
                {"snr_low": lo, "snr_high": hi, "n": n, "efficiency": e}
                for lo, hi, n, e in curve
            ]
            LOGGER.info("Efficiency vs network SNR at the 99.9th noise percentile:")
            for lo, hi, n, e in curve:
                LOGGER.info(f"  SNR {lo:4.1f}-{hi:4.1f}  n={n:5d}  eff={e:.3f}")

        self.secondary_metrics = metrics
        LOGGER.info(
            f"noise mu={metrics['noise_score_mean']:.5g} "
            f"sigma={metrics['noise_score_std']:.5g}; "
            f"efficiency at 3 sigma = {metrics['efficiency_at_3sigma']:.3f}, "
            f"at the 99.9th noise percentile = "
            f"{metrics['efficiency_at_noise_p999']:.3f}"
        )
        if self.cfg.save:
            np.save(f"{self.cfg.run_dir}/preds/noise_scores.npy", noise)
            np.save(f"{self.cfg.run_dir}/preds/signal_scores.npy", signal)

    def _signal_metadata(self, key: str):
        """Pull a per-injection column out of the signal shards.

        The bank stores the drawn parameters alongside the tiles precisely so an
        efficiency curve can be cut on them afterwards. Order is the shard order, which
        is also the dataset order, because the builder uses `imap` rather than
        `imap_unordered`.
        """
        import glob

        pattern = self.cfg.data.get("signal_val_shards", None)
        if not pattern:
            return None
        cols = []
        for path in sorted(glob.glob(str(pattern))):
            with np.load(path) as z:
                if key not in z.files:
                    return None
                cols.append(z[key])
        return np.concatenate(cols) if cols else None
