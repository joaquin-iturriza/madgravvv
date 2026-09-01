"""Figure style and the one function that writes a figure.

`save_figure` is the only sanctioned way to write a plot, and it emits **both** `.png`
and `.pdf` in a single call. That is deliberate: CLAUDE.md requires both formats and
`.claude/hooks/figure_pair_guard.sh` blocks the end of a turn if one is missing, so
routing every figure through one function makes the rule satisfied by construction
rather than remembered. Calling `fig.savefig` directly is what the guard is there to
catch.

The PNG is what gets read back inside a session; the PDF is what goes into
`docs/results.tex` and eventually to the upstream author. Neither is optional.
"""

from __future__ import annotations

import os
from pathlib import Path

# Plots larger than this are rejected by the image reader, so a figure that exceeds it
# is unreadable in-session — which is most of what a development figure is for.
MAX_INCHES = 13.0
DPI = 150
FIGSIZE = (7.0, 4.5)

# Colour-blind-safe, and deliberately small: a categorical palette longer than this is a
# sign the figure is carrying more series than a reader can follow.
COLORS = ("#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#949494")


def use_style() -> None:
    """Apply the project's matplotlib defaults. Idempotent."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "axes.prop_cycle": mpl.cycler(color=list(COLORS)),
        }
    )


def save_figure(fig, base: str | os.PathLike, close: bool = True) -> tuple[Path, Path]:
    """Write `<base>.png` and `<base>.pdf`. Returns both paths.

    `base` carries no extension. Passing one is a mistake worth catching loudly, since
    it would otherwise produce `foo.png.png` and a guard failure nobody understands.

    The figure is checked against `MAX_INCHES` before writing rather than after, so an
    oversized figure fails where the size was chosen, not three functions later.
    """
    base = Path(base)
    if base.suffix in (".png", ".pdf"):
        raise ValueError(
            f"save_figure takes a base path without an extension, got {base}. It writes "
            f"both formats; naming one of them defeats the point."
        )
    w, h = fig.get_size_inches()
    if max(w, h) > MAX_INCHES:
        raise ValueError(
            f"figure is {w:.1f}x{h:.1f} in; the cap is {MAX_INCHES} in because larger "
            f"images are rejected by the reader, making the figure unreadable in-session."
        )
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return png, pdf
