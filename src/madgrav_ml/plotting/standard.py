"""The standard figure set, fixed filenames, emitted by every run.

FA's rule is that experiments emit the same plots under the same names so runs can be
compared without re-deriving the axes each time. The set here is chosen for what this
project actually has to answer:

    loss                the training curve, linear and log, with the LR overlaid
    score_separation    the noise and signal score distributions. THE stage-2 diagnostic:
                        the margin term can raise the signal score without separating
                        anything, and no scalar in the log distinguishes those two cases
    efficiency_vs_far   efficiency against the FAR axis, which is the axis the field
                        reads. A change that helps at FAR 10/yr and not at 1/yr is a
                        change that does not help
    efficiency_vs_mass  efficiency against total mass — where a change bought its gain
    reliability         the reliability diagram, before and after calibration

Ad-hoc extra plots are fine in addition, never instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import FIGSIZE, save_figure, use_style


def plot_loss(train_loss, val_loss=None, lr=None, out_dir="plots", name="loss"):
    """Training curve, linear and log, with the learning rate on a twin axis.

    Log scale is not decoration: a reconstruction MSE that has stopped improving in the
    third decimal place looks flat on a linear axis and is still descending on a log one,
    and the upstream stage-1 schedule (ReduceLROnPlateau) makes that distinction the
    difference between "converged" and "the LR collapsed early".
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 1.6, FIGSIZE[1]))
    steps = np.arange(len(train_loss))
    for ax, logy in zip(axes, (False, True)):
        ax.plot(steps, train_loss, label="train")
        if val_loss is not None and len(val_loss):
            vs, vl = zip(*val_loss)
            ax.plot(vs, vl, marker="o", ms=3, label="val")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right")
    if lr is not None and len(lr):
        tw = axes[0].twinx()
        tw.plot(steps, lr, color="0.6", lw=1.0, ls="--")
        tw.set_ylabel("learning rate", color="0.4")
        tw.grid(False)
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / name)


def plot_score_separation(noise_scores, signal_scores=None, out_dir="plots",
                          name="score_separation", threshold=None):
    """Per-tile score distributions for noise and (if present) signal.

    Read it for separation, not for location. The stage-2 margin objective is a hinge on
    the *difference* of the two means, so it can satisfy itself by inflating both
    distributions together — which improves the training loss and buys no detection
    efficiency whatsoever. That failure is invisible in the loss curve and obvious here.
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    noise = np.asarray(noise_scores, dtype=float)
    lo, hi = np.percentile(noise, [0.1, 99.9])
    if signal_scores is not None and len(signal_scores):
        sig = np.asarray(signal_scores, dtype=float)
        lo = min(lo, float(np.percentile(sig, 0.1)))
        hi = max(hi, float(np.percentile(sig, 99.9)))
    bins = np.linspace(lo, hi, 80)

    ax.hist(noise, bins=bins, histtype="step", density=True, label=f"noise (n={noise.size})")
    if signal_scores is not None and len(signal_scores):
        sig = np.asarray(signal_scores, dtype=float)
        ax.hist(sig, bins=bins, histtype="step", density=True, label=f"signal (n={sig.size})")
        # Separation in units of the noise width — the quantity the margin is supposed to
        # move, quoted so a run's log and its figure cannot disagree.
        d = (sig.mean() - noise.mean()) / (noise.std() + 1e-12)
        ax.set_title(f"separation = {d:.2f} sigma_noise")
    if threshold is not None:
        ax.axvline(threshold, color="0.3", ls="--", lw=1.0, label="threshold")
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / name)


def plot_efficiency_vs_far(curves, out_dir="plots", name="efficiency_vs_far"):
    """Efficiency against FAR, log-x, descending — the axis the field reads.

    `curves` is `{label: [Efficiency, ...]}`. The FAR axis is where a change has to hold
    up: helping at 10/yr and not at 1/yr is not helping, and only this view shows it.
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, effs in curves.items():
        far = np.array([e.far_target for e in effs], dtype=float)
        eff = np.array([e.efficiency for e in effs], dtype=float)
        lo = np.array([e.lo for e in effs], dtype=float)
        hi = np.array([e.hi for e in effs], dtype=float)
        order = np.argsort(far)
        ax.plot(far[order], eff[order], marker="o", ms=3, label=label)
        ax.fill_between(far[order], lo[order], hi[order], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("FAR [/yr]")
    ax.set_ylabel("detection efficiency")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / name)


def plot_efficiency_vs_mass(curves, out_dir="plots", name="efficiency_vs_mass"):
    """Efficiency in bins of total mass. `curves` is `{label: [Efficiency, ...]}`.

    Bins with no injections come through as NaN and are simply not drawn — an empty bin
    is an unmeasured point, not a zero, and plotting it as zero has ended arguments in
    the wrong direction before.
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, effs in curves.items():
        x = np.arange(len(effs), dtype=float)
        eff = np.array([e.efficiency for e in effs], dtype=float)
        err = np.array([[e.efficiency - e.lo for e in effs],
                        [e.hi - e.efficiency for e in effs]], dtype=float)
        good = np.isfinite(eff)
        ax.errorbar(x[good], eff[good], yerr=err[:, good], marker="o", ms=3,
                    capsize=2, label=label)
    if curves:
        first = next(iter(curves.values()))
        ax.set_xticks(np.arange(len(first)))
        ax.set_xticklabels([e.label.split("[")[-1].rstrip(")") for e in first],
                           rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(r"total mass $M_\mathrm{tot}$ [$M_\odot$]")
    ax.set_ylabel("detection efficiency")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / name)


def plot_reliability(probabilities, labels, n_bins=15, out_dir="plots",
                     name="reliability", extra=None):
    """Reliability diagram with the ECE quoted. `extra` adds a post-calibration curve."""
    import matplotlib.pyplot as plt

    from ..eval.calibration import expected_calibration_error, reliability_curve

    use_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1.0, label="perfect")
    for lbl, p in [("raw", probabilities)] + list((extra or {}).items()):
        pred, obs, cnt = reliability_curve(p, labels, n_bins=n_bins)
        ece = expected_calibration_error(p, labels, n_bins=n_bins)
        ax.plot(pred, obs, marker="o", ms=3, label=f"{lbl} (ECE {ece:.3f})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, Path(out_dir) / name)


def plot_standard(run_dir, train_loss=None, val_loss=None, lr=None,
                  noise_scores=None, signal_scores=None):
    """Emit whatever of the standard set this run has the inputs for.

    Called from `BaseExperiment.plot`. Missing inputs are skipped rather than faked: a
    stage-1 run has no signal scores, and drawing an empty panel for one would suggest a
    measurement that was never made.
    """
    out = Path(run_dir) / "plots"
    written = []
    if train_loss is not None and len(train_loss):
        written.append(plot_loss(train_loss, val_loss, lr, out_dir=out))
    if noise_scores is not None and len(noise_scores):
        written.append(plot_score_separation(noise_scores, signal_scores, out_dir=out))
    return written
