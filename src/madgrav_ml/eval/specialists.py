"""The CNN glitch gate: HM/LM specialists placed by the glitch arm's Grad-CAM.

The last veto in the deployed chain, and on our measurements the one aimed squarely at
what limits us. Section~8 of `docs/results.tex` found the background reaching net sigma
32 while no injection passes 20; this is the stage whose entire job is to decide whether
a loud thing is a glitch.

  `is_glitch = max(HM, LM) < 0.5`

Ported from `search_mode/driver_blindscan.py::cnn_hm_lm`. The construction is more
involved than the other vetoes and every step of it matters:

1. Both detectors' whitened 4 s windows go through the ordinary tile path, and the
   GLITCH ARM's Grad-CAM attention peak on the H1 tile gives a time column `t0`. That is
   the "learned signal locator" — upstream's comment is explicit that an energy argmax
   was tried and collapses onto noise for marginal signals.
2. The Q-transform magnitude is recomputed at NATIVE resolution (not the 256x128 tile):
   log1p|Q| over the central 1 s, ~2563 frequency rows at 0.5 Hz and ~500 time columns.
3. Both detectors are cropped to a band and to 113 columns starting at `a`, the native
   column corresponding to `t0`. BOTH legs use the H1 t0 — the specialists are shown the
   same instant in both detectors, which is what makes the two channels comparable.
4. Each crop is min-max normalised on its own, the two are stacked as channels, and the
   pair is fed to a specialist: HM over 20-140 Hz, LM over 50-500 Hz.

Note what the gate throws away: `max(HM, LM) >= 0.5` reduces two calibrated
probabilities to one bit, and the two arms are why the false-alarm rate carries a trials
factor of 2. Replacing the maximum with a single calibrated statistic is plan
section 9.1 — a FAR improvement at fixed model, costing no parameters.
"""

from __future__ import annotations

import numpy as np

GLITCH_THRESH = 0.5
HM_BAND_HZ = (20.0, 140.0)
LM_BAND_HZ = (50.0, 500.0)
CROP_COLUMNS = 113          # WT
CAM_CLAMP = (14, 113)       # the interior columns the attention peak is clamped to
CAM_SMOOTH_SIGMA = 2.0


def cam_t0(arm, tiles: np.ndarray, device, chunk: int = 256) -> np.ndarray:
    """Grad-CAM attention peak time column per tile. `tiles` is (n, 256, 128).

    Hooks the fourth (last) convolutional block, weights its feature map by the mean
    gradient of the summed logit, rectifies, upsamples to tile shape, and sums over
    frequency to get attention per time column. The peak is smoothed and clamped to the
    interior so the crop that follows always fits inside the native magnitude.

    This runs a BACKWARD pass, so it cannot sit inside `torch.no_grad()`.
    """
    import torch
    import torch.nn.functional as F

    tiles = np.asarray(tiles, dtype=np.float32)
    if tiles.ndim == 2:
        tiles = tiles[None]
    from scipy.ndimage import gaussian_filter1d

    out = np.empty(len(tiles), dtype=int)
    for c0 in range(0, len(tiles), chunk):
        sub = tiles[c0:c0 + chunk]
        x = torch.from_numpy(sub[:, None]).float().to(device).requires_grad_(True)
        feat: dict = {}

        def hook(_m, _i, o):
            feat["a"] = o
            o.retain_grad()

        handle = arm.blocks[3].register_forward_hook(hook)
        arm.zero_grad()
        logit = arm(x)
        logit.sum().backward()
        handle.remove()

        a = feat["a"]
        weights = a.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * a).sum(1, keepdim=True).clamp(min=0)
        cam = F.interpolate(cam, size=tiles.shape[1:], mode="bilinear",
                            align_corners=False)[:, 0]
        attention = cam.sum(1).detach().cpu().numpy()
        for i in range(len(sub)):
            peak = np.argmax(gaussian_filter1d(attention[i], CAM_SMOOTH_SIGMA))
            out[c0 + i] = int(np.clip(peak, *CAM_CLAMP))
    return out


def native_magnitude(whitened: np.ndarray, spec) -> np.ndarray:
    """log1p|Q| at native resolution over the central crop — no resize.

    The specialists read the Q-transform at full frequency resolution, not the 256x128
    tile the autoencoder sees. Feeding them the tile would change what 113 columns and a
    20-140 Hz band mean.
    """
    from madgrav_ml.data.representation import _crop_bounds, centre_crop, qtransform

    qi = centre_crop(whitened, spec.sample_rate, spec.context_seconds)
    total_seconds = len(qi) / float(spec.sample_rate)
    mag = np.log1p(np.abs(qtransform(qi, spec))).astype(np.float32)
    start, stop = _crop_bounds(mag.shape[1], total_seconds, spec.crop_seconds)
    return mag[:, start:stop]


def column_for(t0: int, n_columns: int) -> int:
    """Tile time column -> native column. `(t0 - 14) / 128 * T`, upstream's mapping."""
    return int(round((t0 - CAM_CLAMP[0]) / 128.0 * n_columns))


def band_crop(mag: np.ndarray, column: int, band: tuple[float, float],
              f_low: float = 10.0, f_res: float = 0.5,
              width: int = CROP_COLUMNS) -> np.ndarray:
    """Select `band` in frequency and `width` columns from `column`, then min-max.

    The frequency axis is reconstructed as `f_low + f_res * k` rather than read off the
    spectrogram, which is what upstream does and is correct for a gwpy q_transform given
    an explicit `fres`. Zero-padding past the end of the magnitude, rather than clipping
    the window, keeps every crop the same width whatever the attention peak was.
    """
    n_cols = mag.shape[1]
    faxis = f_low + f_res * np.arange(mag.shape[0])
    rows = mag[(faxis >= band[0]) & (faxis <= band[1])]
    out = np.zeros((rows.shape[0], width), dtype=np.float32)
    lo, hi = max(0, column), min(n_cols, column + width)
    if hi > lo:
        out[:, lo - column:hi - column] = rows[:, lo:hi]
    return ((out - out.min()) / (out.max() - out.min() + 1e-9)).astype(np.float32)


def specialist_inputs(mag_h1: np.ndarray, mag_l1: np.ndarray, t0: int) -> dict:
    """The two stacked (2, f, 113) tensors the HM and LM specialists take.

    Both detectors are cropped at the SAME column, derived from H1's attention peak.
    Localising each detector independently would let the two channels show different
    instants, and the specialist would be asked to compare things that never coincided.
    """
    column = column_for(t0, mag_h1.shape[1])
    return {
        "hm": np.stack([band_crop(mag_h1, column, HM_BAND_HZ),
                        band_crop(mag_l1, column, HM_BAND_HZ)]),
        "lm": np.stack([band_crop(mag_h1, column, LM_BAND_HZ),
                        band_crop(mag_l1, column, LM_BAND_HZ)]),
    }


def specialist_scores(hm_net, lm_net, inputs: dict, device) -> tuple[float, float]:
    """P(signal) from each specialist. Sigmoid of the logit, as upstream applies it."""
    import torch

    with torch.no_grad():
        out = []
        for net, key in ((hm_net, "hm"), (lm_net, "lm")):
            x = torch.from_numpy(inputs[key][None]).float().to(device)
            out.append(float(torch.sigmoid(net(x)).item()))
    return out[0], out[1]


def is_glitch(hm, lm, thresh: float = GLITCH_THRESH) -> np.ndarray:
    """`max(HM, LM) < 0.5`. One bit out of two calibrated probabilities — see the
    module docstring; recovering the discarded information is plan section 9.1."""
    return np.maximum(np.asarray(hm), np.asarray(lm)) < thresh
