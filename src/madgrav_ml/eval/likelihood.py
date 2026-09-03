"""The likelihood-ratio cascade: the last upstream component, and the one that stops
throwing information away.

Every stage before this reduced a continuous quantity to a bit. Coherence became
`>= tcoh`, the centroids became `< f_cut`, the two specialists became
`max(HM, LM) >= 0.5`. Each cut is worth a large factor in background
(Section~9 of `docs/results.tex`), and each also discards the difference between a
trigger that barely passed and one that passed by a mile.

The cascade instead ranks on

    loglr = beta_0 + ((F - mu) / sd) . beta_{1:}

a logistic model over seven standardised features:

    sigma_H1, sigma_L1, coherence, centroid_H1, centroid_L1,
    gate(g_H1, sigma_H1), gate(g_L1, sigma_L1)

where `g` is the 5-seed glitch-arm ensemble logit on the detector's own tile, and
`gate(g, s) = clip(g, -6, 6) * clip(s/3, 0, 1)` suppresses the arm's opinion on triggers
that are not loud enough for it to have one.

The coefficients are FROZEN and distributed with the package, in
`data/o3a_frozen_lr_off200.npz`, as two folds. Upstream scores a trigger from fold g with
the model fitted on fold 1-g, so nothing is ever ranked by a model that saw it. We keep
that discipline: the fold of a background span decides which of the two models scores it.

Fitting our own would be a different experiment and a much easier one to get wrong --
the coefficient on coherence is constrained non-negative in the upstream fit, which is a
prior about physics, not something a fit discovers on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GCLIP = 6.0
FEATURE_NAMES = ("sigma_H1", "sigma_L1", "coherence", "centroid_H1", "centroid_L1",
                 "gated_arm_H1", "gated_arm_L1")


def arm_gate(g: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """`clip(g, -6, 6) * clip(sigma/3, 0, 1)` — the arm's opinion, weighted by loudness.

    The ramp matters: below sigma = 0 the arm contributes nothing at all, and above
    sigma = 3 it contributes fully. A glitch classifier asked about a trigger that is not
    loud in the first place is being asked a question its training never posed, and this
    is how upstream declines to use the answer.
    """
    return np.clip(np.asarray(g, dtype=float), -GCLIP, GCLIP) * np.clip(
        np.asarray(sigma, dtype=float) / 3.0, 0.0, 1.0)


def features(sigma_h1, sigma_l1, coherence, centroid_h1, centroid_l1,
             arm_h1, arm_l1) -> np.ndarray:
    """The seven-column feature matrix, in upstream's order."""
    return np.column_stack([
        np.atleast_1d(sigma_h1), np.atleast_1d(sigma_l1), np.atleast_1d(coherence),
        np.atleast_1d(centroid_h1), np.atleast_1d(centroid_l1),
        arm_gate(arm_h1, sigma_h1), arm_gate(arm_l1, sigma_l1),
    ]).astype(float)


def log_likelihood_ratio(f: np.ndarray, mu, sd, beta) -> np.ndarray:
    """`beta[0] + ((F - mu)/sd) . beta[1:]`."""
    f = np.atleast_2d(np.asarray(f, dtype=float))
    return beta[0] + ((f - np.asarray(mu)) / np.asarray(sd)) @ np.asarray(beta)[1:]


def load_frozen(path: str | Path) -> dict:
    """The distributed two-fold model. Returns `{0: (mu, sd, beta), 1: ...}`."""
    with np.load(path) as z:
        out = {g: (z[f"mu{g}"], z[f"sd{g}"], z[f"be{g}"]) for g in (0, 1)}
        floor = float(z["floor"]) if "floor" in z.files else None
    for g, (mu, sd, beta) in out.items():
        if len(mu) != len(FEATURE_NAMES) or len(beta) != len(FEATURE_NAMES) + 1:
            raise ValueError(
                f"fold {g}: expected {len(FEATURE_NAMES)} features and "
                f"{len(FEATURE_NAMES)+1} coefficients, got {len(mu)} and {len(beta)}"
            )
    out["floor"] = floor
    return out


def score_held_out(f: np.ndarray, fold: np.ndarray, frozen: dict) -> np.ndarray:
    """Score each row with the model fitted on the OTHER fold.

    Not a nicety. Both folds' models are shipped, and using the one that saw a trigger's
    own fold would rank noise against a model tuned partly on that noise — which is the
    fold-discipline failure this project's `FoldGuard` exists to prevent one level up.
    """
    f = np.atleast_2d(np.asarray(f, dtype=float))
    fold = np.asarray(fold)
    out = np.empty(len(f), dtype=float)
    for g in (0, 1):
        m = fold == g
        if m.any():
            out[m] = log_likelihood_ratio(f[m], *frozen[1 - g])
    return out
