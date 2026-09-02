#!/usr/bin/env python
"""Compare a retrained front end against the distributed weights on held-out noise.

Both models score the same validation tile bank. What is compared is the *shape* of the
noise-only score distribution (the tail is what sets the false-alarm rate), the rank
agreement between the two, and what each model's error tracks in the tile.

Read the caveat before quoting anything from this: a stage-1 checkpoint and the
distributed stage-2 checkpoint are not the same object, and a sign reversal between
them is expected, not anomalous. This becomes the Section 3.4 comparison only once the
margin fine-tune has been reproduced.

  scripts/remote.sh .venv/bin/python scripts/compare_front_ends.py \
      --checkpoint runs/madgrav/<run>/models/model_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from madgrav_ml.data.tiles import CachedTileDataset  # noqa: E402
from madgrav_ml.models.cae import BaselineCAE  # noqa: E402
from madgrav_ml.plotting.style import save_figure  # noqa: E402

VENDORED = REPO / ".reference/MADGRAV/assets/models/baseline_cae_weaksup_best.pt"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy import; ties broken by order, which is fine
    here because the scores are continuous and ties are measure-zero."""
    ra = np.empty(len(a)); ra[np.argsort(a)] = np.arange(len(a))
    rb = np.empty(len(b)); rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def score(model, tiles, device, batch=128) -> np.ndarray:
    out = []
    model.eval().to(device)
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            x = torch.from_numpy(tiles[i:i + batch]).to(device)
            out.append(model.reconstruction_error(x, reduction="none").cpu().numpy())
    return np.concatenate(out)


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--val-shards", default="data_cache/tiles/val/*.npz")
    ap.add_argument("--signal-shards", default="data_cache/tiles/val_signal/*.npz",
                    help="matched injected bank; pass '' to compare on noise only")
    ap.add_argument("--out", type=Path, default=REPO / "runs/_checks/front_end_comparison")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = CachedTileDataset(args.val_shards)
    tiles = np.stack([val[i][0].numpy() for i in range(len(val))])
    print(f"{len(tiles)} validation tiles, shape {tiles.shape[1:]}, on {device}")

    def load(path):
        """Our checkpoints wrap the weights with the optimiser state and step; the
        distributed one is a bare state dict. `strict=True` either way — a silently
        partial load is how a comparison ends up being against a half-initialised net."""
        blob = torch.load(path, map_location="cpu")
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model = BaselineCAE()
        model.load_state_dict(state, strict=True)
        return model

    ours, theirs = load(args.checkpoint), load(VENDORED)

    s_ours, s_theirs = score(ours, tiles, device), score(theirs, tiles, device)
    z_ours, z_theirs = standardise(s_ours), standardise(s_theirs)

    def stats(z):
        from scipy.stats import kurtosis, skew
        return (np.median(z), np.percentile(z, 99), np.percentile(z, 99.9),
                z.max(), skew(z), kurtosis(z))

    print(f"\n{'standardised score':<22}{'retrained':>12}{'distributed':>14}")
    names = ("median", "99th pct", "99.9th pct", "maximum", "skewness", "excess kurtosis")
    for name, a, b in zip(names, stats(z_ours), stats(z_theirs)):
        print(f"{name:<22}{a:>+12.2f}{b:>+14.2f}")
    print(f"\nSpearman rank correlation between the two: {spearman(s_ours, s_theirs):+.3f}")

    flat = tiles[:, 0]
    props = {
        "mean pixel value": flat.mean(axis=(1, 2)),
        "pixel standard deviation": flat.std(axis=(1, 2)),
        "fraction of pixels above 0.5": (flat > 0.5).mean(axis=(1, 2)),
    }
    print(f"\n{'tile property (Spearman rho with error)':<40}{'retrained':>12}{'distributed':>14}")
    for name, v in props.items():
        print(f"{name:<40}{spearman(s_ours, v):>+12.2f}{spearman(s_theirs, v):>+14.2f}")

    # --- detection, at MATCHED noise false-positive fractions --------------------
    #
    # The only fair way to compare two models whose scores live on different scales.
    # Each model's threshold is set by its OWN noise distribution, so both are quoted
    # at the same fraction of noise tiles surviving. This is not yet a false-alarm
    # RATE -- that needs the time-slide background and the trials factor -- but it is
    # the like-for-like comparison the reproduction requirement asks for.
    curves = {}
    if args.signal_shards:
        import glob

        sig = CachedTileDataset(args.signal_shards)
        sig_tiles = np.stack([sig[i][0].numpy() for i in range(len(sig))])
        snr = np.concatenate([np.load(f)["network_snr"]
                              for f in sorted(glob.glob(args.signal_shards))])
        print(f"\n{len(sig_tiles)} injections, network SNR "
              f"{snr.min():.1f}-{snr.max():.1f}")
        g_ours, g_theirs = score(ours, sig_tiles, device), score(theirs, sig_tiles, device)

        print(f"\n{'noise false-positive fraction':<32}{'retrained':>12}{'distributed':>14}")
        for fp in (0.1, 0.01, 0.001):
            row = []
            for s_noise, s_sig in ((s_ours, g_ours), (s_theirs, g_theirs)):
                t = np.quantile(s_noise, 1.0 - fp)
                row.append(float((s_sig > t).mean()))
            curves[fp] = row
            print(f"{'efficiency at ' + format(fp, '.3f'):<32}{row[0]:>12.3f}{row[1]:>14.3f}")

        edges = [8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
        print(f"\nefficiency vs network SNR at a 0.1% noise false-positive fraction")
        print(f"{'SNR bin':<14}{'n':>7}{'retrained':>12}{'distributed':>14}")
        t_ours = np.quantile(s_ours, 0.999)
        t_theirs = np.quantile(s_theirs, 0.999)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (snr >= lo) & (snr < hi)
            if not m.any():
                continue
            print(f"{f'{lo:.0f}-{hi:.0f}':<14}{int(m.sum()):>7}"
                  f"{float((g_ours[m] > t_ours).mean()):>12.3f}"
                  f"{float((g_theirs[m] > t_theirs).mean()):>14.3f}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    bins = np.linspace(-3, max(z_ours.max(), z_theirs.max()), 80)
    axes[0].hist(z_ours, bins=bins, histtype="step", label="retrained (stage 1)")
    axes[0].hist(z_theirs, bins=bins, histtype="step", label="distributed (stage 2)")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("standardised reconstruction error"); axes[0].set_ylabel("tiles")
    axes[1].scatter(z_ours, z_theirs, s=3, alpha=0.3)
    axes[1].set_xlabel("retrained (standardised)"); axes[1].set_ylabel("distributed (standardised)")
    axes[1].set_title(f"Spearman {spearman(s_ours, s_theirs):+.3f}", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out)
    print(f"\nwrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
