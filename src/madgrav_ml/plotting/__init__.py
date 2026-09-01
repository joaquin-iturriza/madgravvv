"""Standard figures, with fixed filenames and both output formats.

FA keeps `plots.py` / `base_plots.py` / `plot_style.py`; this is the same split,
smaller. The reason it exists as a shared layer rather than per-experiment matplotlib is
the same one that made FA's experiments incomparable before its harness: figures drawn
ad hoc per experiment cannot be put side by side, and the axis you forgot to fix is
always the one that mattered.
"""

from .standard import (
    plot_efficiency_vs_far,
    plot_efficiency_vs_mass,
    plot_loss,
    plot_reliability,
    plot_score_separation,
    plot_standard,
)
from .style import FIGSIZE, MAX_INCHES, save_figure, use_style

__all__ = [
    "FIGSIZE",
    "MAX_INCHES",
    "plot_efficiency_vs_far",
    "plot_efficiency_vs_mass",
    "plot_loss",
    "plot_reliability",
    "plot_score_separation",
    "plot_standard",
    "save_figure",
    "use_style",
]
