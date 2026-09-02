"""The CNN glitch gate, checked against upstream's own Grad-CAM and crop.

This is the veto aimed at what actually limits the search — the loud-noise tail — and
its input construction has more places to go wrong than everything else combined: an
attention peak in tile columns mapped into native columns, a band selected off a
reconstructed frequency axis, a crop that both detectors must share, and a per-crop
min-max. A quiet error in any of them produces plausible probabilities that mean nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from madgrav_ml.data.representation import TileSpec  # noqa: E402
from madgrav_ml.eval import specialists as S  # noqa: E402
from madgrav_ml.models.arms import GlitchArm, SpecialistCNN  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / ".reference/MADGRAV"

pytestmark = pytest.mark.skipif(not REFERENCE.exists(),
                                reason="run scripts/vendor_reference.sh")


def morph_roi():
    path = str(REFERENCE / "search_mode")
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("MADGRAV_ROOT", str(REFERENCE))
    return pytest.importorskip("morph_roi")


def arm():
    m = GlitchArm()
    m.load_state_dict(torch.load(REFERENCE / "lr_cascade/p1v42/arm_deploy_seed0.pt",
                                 map_location="cpu"), strict=True)
    return m.eval()


def specialist(name):
    m = SpecialistCNN()
    m.load_state_dict(torch.load(REFERENCE / f"search_mode/{name}_native_seed0.pt",
                                 map_location="cpu"), strict=True)
    return m.eval()


def tiles(n=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 256, 128)).astype(np.float32)
    # a bright track, so the attention peak has something to find and is not pure noise
    for i in range(n):
        col = 30 + 20 * i
        x[i, 40:120, col:col + 6] = 1.0
    return x


# --- the localiser ------------------------------------------------------------


def test_cam_t0_matches_upstream():
    """Bit-level against `morph_roi.cam_t0_batch` with the distributed arm weights.

    Both the arm and the Grad-CAM have to be right for this to agree: the hook is on the
    fourth block, the weights are the mean gradient of the summed logit, and the peak is
    smoothed before the argmax.
    """
    mr = morph_roi()
    a, x = arm(), tiles()
    mine = S.cam_t0(a, x, torch.device("cpu"))
    theirs = mr.cam_t0_batch(a, x, torch.device("cpu"))
    assert np.array_equal(mine, theirs), (mine, theirs)


def test_cam_t0_is_clamped_to_the_interior():
    """The clamp is what guarantees the 113-column crop fits inside the magnitude."""
    a = arm()
    x = np.zeros((3, 256, 128), np.float32)
    x[0, :, 0] = 1.0      # attention pulled hard to the left edge
    x[1, :, 127] = 1.0    # and to the right
    t0 = S.cam_t0(a, x, torch.device("cpu"))
    assert t0.min() >= S.CAM_CLAMP[0] and t0.max() <= S.CAM_CLAMP[1]


def test_cam_t0_follows_the_bright_track():
    """A localiser that ignored its input would make every crop identical."""
    a = arm()
    t0 = S.cam_t0(a, tiles(n=4), torch.device("cpu"))
    assert len(set(t0.tolist())) > 1, t0


# --- the crop -----------------------------------------------------------------


def upstream_crop(mag, a, T, flo, fhi, WT=113):
    """Transcription of `driver_blindscan._crop`, for a line-by-line comparison."""
    fax = 10.0 + 0.5 * np.arange(mag.shape[0])
    m = mag[(fax >= flo) & (fax <= fhi)]
    out = np.zeros((m.shape[0], WT), dtype=np.float32)
    sa = max(0, a)
    sb = min(T, a + WT)
    if sb > sa:
        out[:, sa - a:sb - a] = m[:, sa:sb]
    return ((out - out.min()) / (out.max() - out.min() + 1e-9)).astype(np.float32)


@pytest.mark.parametrize("column", [0, 37, 200, 380])
@pytest.mark.parametrize("band", [S.HM_BAND_HZ, S.LM_BAND_HZ])
def test_band_crop_matches_upstream(column, band):
    rng = np.random.default_rng(1)
    mag = rng.random((2563, 500)).astype(np.float32)
    assert np.array_equal(S.band_crop(mag, column, band),
                          upstream_crop(mag, column, mag.shape[1], *band))


def test_band_crop_shapes_are_the_bands_they_claim():
    mag = np.random.default_rng(2).random((2563, 500)).astype(np.float32)
    # 20-140 Hz at 0.5 Hz from 10 Hz -> rows 20..260 inclusive
    assert S.band_crop(mag, 100, S.HM_BAND_HZ).shape == (241, 113)
    assert S.band_crop(mag, 100, S.LM_BAND_HZ).shape == (901, 113)


def test_column_mapping_matches_upstream():
    for t0 in (14, 50, 113):
        assert S.column_for(t0, 500) == int(round((t0 - 14) / 128 * 500))


def test_both_detectors_are_cropped_at_the_same_column():
    """The specialists compare two detectors at one instant. Localising each leg
    separately would show them different moments and the comparison would be empty."""
    rng = np.random.default_rng(3)
    h1 = rng.random((2563, 500)).astype(np.float32)
    l1 = rng.random((2563, 500)).astype(np.float32)
    got = S.specialist_inputs(h1, l1, t0=60)
    col = S.column_for(60, 500)
    assert np.array_equal(got["hm"][0], S.band_crop(h1, col, S.HM_BAND_HZ))
    assert np.array_equal(got["hm"][1], S.band_crop(l1, col, S.HM_BAND_HZ))
    assert got["hm"].shape[0] == 2 and got["lm"].shape[0] == 2


# --- the gate -----------------------------------------------------------------


def test_specialists_accept_their_native_band_shapes():
    """AdaptiveAvgPool means the nets take any height, but the two bands really are
    different heights, so a swap would run silently rather than raise."""
    rng = np.random.default_rng(4)
    inputs = S.specialist_inputs(rng.random((2563, 500)).astype(np.float32),
                                 rng.random((2563, 500)).astype(np.float32), t0=64)
    hm, lm = S.specialist_scores(specialist("hm"), specialist("lm"), inputs,
                                 torch.device("cpu"))
    assert 0.0 <= hm <= 1.0 and 0.0 <= lm <= 1.0


def test_gate_is_the_max_of_two_probabilities():
    assert bool(S.is_glitch(0.4, 0.4))
    assert not bool(S.is_glitch(0.4, 0.6))
    assert not bool(S.is_glitch(0.6, 0.4))
    assert np.array_equal(S.is_glitch(np.array([0.1, 0.9]), np.array([0.2, 0.2])),
                          np.array([True, False]))


def test_native_magnitude_is_not_the_tile():
    """The specialists read full frequency resolution; feeding them the 256x128 tile
    would silently change what '20-140 Hz' and '113 columns' mean."""
    spec = TileSpec()
    rng = np.random.default_rng(5)
    mag = S.native_magnitude(rng.normal(0, 1, 4 * spec.sample_rate), spec)
    assert mag.shape[0] > 2000, mag.shape
    assert 400 < mag.shape[1] < 600, mag.shape
