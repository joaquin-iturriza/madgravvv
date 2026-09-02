"""Stage 2: the weak-supervision margin fine-tune.

Upstream: hinge margin m = 3.0, weight lambda = 2.0, pushing signal-tile reconstruction
error above noise-tile error; best at epoch 9 of 10, seed 42. That "best at 9 of 10"
is worth reading twice — it means longer training was never tested, which makes
training length a first-class HPO target alongside m and lambda (Phase 5).

This is the *only* place signal labels enter the front end. Stage 1 stays label-free
(C3), and the margin term is what turns a reconstruction model into a detector without
collapsing it into a supervised classifier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from madgrav_ml.experiments.stage1_cae import Stage1CAEExperiment
from madgrav_ml.logger import LOGGER
from madgrav_ml.models.cae import stage2_loss


class Stage2MarginExperiment(Stage1CAEExperiment):
    """Fine-tune a stage-1 CAE against noise/signal pairs with a hinge margin.

    Inherits the stage-1 data plumbing and overrides only the label-bearing parts, so
    the two stages cannot drift apart in their representation or fold handling.
    """

    def init_data(self):
        """Paired batches: noise tiles and injected tiles from the *same* fold.

        The injections are drawn on the fly (Phase 2) unless `data.source == "cached"`,
        in which case the fixed upstream bank is used and the comparison between the
        two is the experiment.
        """
        from madgrav_ml.data.tiles import CachedTileDataset, GeneratedTileDataset
        from madgrav_ml.eval.folds import Split

        with self.guard.training(f"stage2:{self.cfg.run_name}"):
            train_segments = self.guard.segments(Split.HPO_TRAIN)
            val_segments = self.guard.segments(Split.HPO_VAL)

        if self.cfg.data.source == "cached":
            self.train_set = CachedTileDataset(self.cfg.data.train_shards)
            self.val_set = CachedTileDataset(self.cfg.data.val_shards)
            return

        from madgrav_ml.data.representation import make_tile

        self.train_set = GeneratedTileDataset(
            noise_provider=self._noise_provider(train_segments),
            tile_fn=lambda w: make_tile(w, self.spec),
            signal_fraction=self.cfg.data.signal_fraction,
            injector=self._injector(),
            seed=self.cfg.seed,
        )
        self.val_set = GeneratedTileDataset(
            noise_provider=self._noise_provider(val_segments),
            tile_fn=lambda w: make_tile(w, self.spec),
            signal_fraction=self.cfg.data.signal_fraction,
            injector=self._injector(),
            seed=(self.cfg.seed or 0) + 1,
            length=self.cfg.data.val_tiles,
        )

    def _injector(self):
        """Add a freshly drawn, projected, SNR-scaled waveform to a noise window.

        The waveform backend lives behind `data.waveform_backend` so IMRPhenomPv2
        (baseline) and IMRPhenomXPHM (the upstream injection banks) are a config switch.
        If no backend is configured this raises rather than silently injecting nothing —
        a stage-2 run whose "signal" tiles are pure noise would train to a plausible
        loss curve and be worthless.
        """
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

        sampler = ParameterSampler(
            seed=self.cfg.seed,
            snr_range=tuple(self.cfg.data.snr_range),
        )
        engine = build_engine(self.cfg.data, self.spec.sample_rate)
        LOGGER.info(
            f"Injections: {engine.backend.approximant_name} from "
            f"{engine.backend.f_lower} Hz, SNR {tuple(self.cfg.data.snr_range)} on the "
            f"'{engine.snr_convention}' convention over {sorted(engine.psds)}"
        )
        default_ifo = sorted(engine.psds)[0]
        # Redraw on an antenna-pattern null. It is a measure-zero event that is
        # nonetheless hit a few times in a million-tile campaign, and a crash three
        # hours into a run over a source that simply happened to sit in a blind spot is
        # a waste; a silent skip that returned pure noise with a signal label would be
        # far worse, so the retry is bounded and then raises.
        def inject(strain, rng, context=None):
            ifo = context.ifo if context is not None else default_ifo
            # The antenna pattern is evaluated at the coalescence, which sits at the
            # centre of the window, not at its first sample.
            gps = None if context is None else 0.5 * (context.start + context.end)
            for _ in range(8):
                # The dataset's rng, not the sampler's own: it is seeded per worker, so
                # this is what keeps eight DataLoader workers from drawing eight copies
                # of the same injection sequence.
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
        LOGGER.info(f"Stage-2 margin m={self.margin}, lambda={self.lam}")

    def _batch_loss(self, data):
        x, y = data[0].to(self.device, non_blocking=True), data[1].to(self.device)
        is_signal = y > 0.5
        if is_signal.all() or (~is_signal).all():
            # A batch with only one class makes the hinge meaningless (the noise mean
            # it compares against would come from nothing). Fall back to plain MSE on
            # whatever is there rather than producing a silently wrong gradient.
            recon, _ = self.model(x)
            loss = F.mse_loss(recon, x)
            return loss, {"mse": float(loss.detach()), "margin": 0.0}

        x_noise, x_sig = x[~is_signal], x[is_signal]
        recon_noise, _ = self.model(x_noise)
        recon_sig, _ = self.model(x_sig)
        err_sig = F.mse_loss(recon_sig, x_sig, reduction="none").flatten(1).mean(1)
        loss, parts = stage2_loss(recon_noise, x_noise, err_sig, self.margin, self.lam)
        return loss, parts
