"""The plotting layer: both formats, one call, and the size cap enforced.

These are cheap tests for an expensive mistake. `figure_pair_guard` blocks a turn when a
figure exists in only one format, and `save_figure` is what makes that impossible by
construction — so if `save_figure` ever regresses to writing one format, the guard turns
from a safety net into a permanent blocker.
"""

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from madgrav_ml.plotting import MAX_INCHES, plot_loss, plot_score_separation, save_figure


def test_save_figure_writes_both_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png, pdf = save_figure(fig, tmp_path / "probe")
    assert png.exists() and pdf.exists()
    assert png.stem == pdf.stem


def test_save_figure_rejects_an_extension(tmp_path):
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="without an extension"):
        save_figure(fig, tmp_path / "probe.png")


def test_save_figure_rejects_an_unreadable_size(tmp_path):
    fig, _ = plt.subplots(figsize=(MAX_INCHES + 2, 4))
    with pytest.raises(ValueError, match="cap is"):
        save_figure(fig, tmp_path / "huge")


def test_standard_plots_produce_pairs(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    png, pdf = plot_loss([1.0, 0.5, 0.3], [(0, 1.1), (2, 0.4)], [1e-3, 5e-4, 1e-4],
                         out_dir=tmp_path)
    assert png.exists() and pdf.exists()
    png, pdf = plot_score_separation(rng.normal(size=500), rng.normal(loc=3, size=200),
                                     out_dir=tmp_path)
    assert png.exists() and pdf.exists()
