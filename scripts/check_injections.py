#!/usr/bin/env python
"""End-to-end check of the injection path on real cached strain.

Builds signal tiles at a ladder of SNRs from real O3a noise and reports, for each:
the recovered matched-filter SNR in the whitened window, the fraction of rho^2 that
survives the centre crop the network actually sees, and how far the tile's statistics
move from a pure-noise tile. Writes a figure.

This exists because "the loss went down" is not evidence that an injection campaign is
correct. A wrong LAL epoch, a dropped antenna factor or a mis-scaled amplitude all
produce a stage-2 run that trains to a plausible curve on tiles containing nothing.

  scripts/remote.sh .venv/bin/python scripts/check_injections.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import scipy.ndimage  # noqa: F401
import scipy.signal  # noqa: F401
from gwpy.frequencyseries import FrequencySeries  # noqa: F401
from gwpy.timeseries import TimeSeries  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.data.injections import ParameterSampler  # noqa: E402
from madgrav_ml.data.representation import (  # noqa: E402
    TileSpec, make_tile, notch, notch_lines_for, whiten,
)
from madgrav_ml.data.strain import (  # noqa: E402
    SegmentReader, available_segments, load_reference_psd, load_segments,
)
from madgrav_ml.data.waveforms import (  # noqa: E402
    InjectionEngine, LALWaveformBackend, optimal_snr, snr2_fraction_in_crop,
)
from madgrav_ml.eval.folds import FoldGuard, Split  # noqa: E402
from madgrav_ml.plotting.style import save_figure  # noqa: E402

SEGMENTS = REPO / ".reference/MADGRAV/search_mode/o3a_bg_segments_56.json"
PSD_DIR = REPO / ".reference/MADGRAV/data/o3a_search_prep"
SNR_LADDER = (8.0, 12.0, 20.0, 40.0)
WINDOW = 4.0


def matched_filter_snr(data, template, sigma, fs, keep=2.0):
    n = len(data)
    k = int(keep * fs)
    lo = (n - k) // 2
    d, t = data[lo:lo + k], template[lo:lo + k]
    corr = np.fft.irfft(np.fft.rfft(d) * np.conj(np.fft.rfft(t)), n=k)
    return float(np.max(np.abs(corr)) / (np.sqrt(np.sum(t ** 2)) * sigma))


def main() -> int:
    spec = TileSpec()
    fs = spec.sample_rate
    psds = {i: load_reference_psd(PSD_DIR / f"reference_psd_{i}.npz") for i in ("H1", "L1")}
    engine = InjectionEngine(
        backend=LALWaveformBackend(),
        psds=psds,
        notch_lines={i: notch_lines_for(i, "o1") for i in psds},
        sample_rate=fs,
        window_seconds=WINDOW,
        snr_convention="network",
    )

    segs = load_segments(SEGMENTS, ifo="H1") + load_segments(SEGMENTS, ifo="L1")
    guard = FoldGuard.from_segments(segs, eval_fold=1, n_folds=2)
    with guard.training("check-injections"):
        wanted = guard.segments(Split.HPO_TRAIN)
    have = available_segments(wanted, REPO / "data_cache/strain")
    reader = SegmentReader(REPO / "data_cache/strain", capacity=1)

    rng = np.random.default_rng(7)
    seg, raw = reader.random_window(have, rng, WINDOW, fs)
    lines = notch_lines_for(seg.ifo, "o1")
    noise = notch(whiten(raw, fs, reference_psd=psds[seg.ifo]), fs, lines=lines)
    centre = noise[len(noise) // 4:3 * len(noise) // 4]
    sigma = float(np.std(centre))
    gps = 0.5 * (seg.start + seg.end)
    print(f"window {seg.ifo} GPS {seg.start:.0f}  whitened noise sigma = {sigma:.3f} "
          f"(1.0 means the reference PSD describes this stretch)")

    sampler = ParameterSampler(seed=11)
    base = sampler.draw(np.random.default_rng(11))
    noise_tile = make_tile(noise, spec)[0]

    rows, tiles = [], [("noise only", noise_tile)]
    for target in SNR_LADDER:
        p = type(base)(**{**base.as_dict(), "network_snr": target})
        h = engine.whitened_signal(p, seg.ifo, gps)
        data = noise + h
        rec = matched_filter_snr(data, h, sigma, fs)
        wave = engine.backend.generate_window(p, fs, WINDOW)
        proj = engine.scale_factor(wave, seg.ifo, p, gps) * engine.project(wave, seg.ifo, p, gps)
        det_snr = optimal_snr(proj, fs, psds[seg.ifo], engine.f_low)
        in_ctx = snr2_fraction_in_crop(h, fs, spec.context_seconds)
        in_crop = snr2_fraction_in_crop(h, fs, spec.crop_seconds)
        tile = make_tile(data, spec)[0]
        rows.append((target, det_snr, rec, in_ctx, in_crop,
                     float(tile.mean() - noise_tile.mean()),
                     float(np.abs(tile - noise_tile).max())))
        tiles.append((f"network SNR {target:g}", tile))

    print(f"\nsource: m1={base.mass1:.1f} m2={base.mass2:.1f} "
          f"chi1z={base.spin1z:+.2f} chi2z={base.spin2z:+.2f} "
          f"iota={base.inclination:.2f} dt={base.time_shift:+.3f}s in {seg.ifo}\n")
    print(f"{'net SNR':>8} {'this det':>9} {'recovered':>10} {'rho2 in 2s':>11} "
          f"{'rho2 in 1s':>11} {'d(tile mean)':>13} {'max |dpix|':>11}")
    for t, d, r, ctx, c, dm, mx in rows:
        print(f"{t:8.1f} {d:9.2f} {r:10.2f} {ctx:11.3f} {c:11.3f} {dm:+13.4f} {mx:11.4f}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tiles), figsize=(2.4 * len(tiles), 2.9), sharey=True)
    for ax, (title, tile) in zip(np.atleast_1d(axes), tiles):
        ax.imshow(tile, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time bin")
    np.atleast_1d(axes)[0].set_ylabel("frequency bin")
    fig.suptitle(f"IMRPhenomPv2 injections into real {seg.ifo} O3a noise "
                 f"(m1={base.mass1:.0f}, m2={base.mass2:.0f})", fontsize=10)
    out = REPO / "runs/_checks/injection_ladder"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out)
    print(f"\nwrote {out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
