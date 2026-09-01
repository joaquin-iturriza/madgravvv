"""Post-hoc probability calibration and the diagnostics that show it worked.

Why this is not cosmetic here: the logistic likelihood ratio consumes P(signal) from
the supervised arms as a *feature*. A classifier trained on ~11k examples is badly
overconfident, and a miscalibrated feature degrades the LR fit in a way AUC never
reveals. Fixing it is free and is one of the plan's early structural wins.

Fit on a held-out slice of the training fold, never on the evaluation fold.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE with equal-width bins: mean |confidence - accuracy|, weighted by bin count."""
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probabilities {p.shape} and labels {y.shape} disagree")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        ece += (n / p.size) * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean predicted, observed frequency, count) per bin — the reliability diagram."""
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pred, obs, cnt = [], [], []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        pred.append(p[sel].mean() if n else np.nan)
        obs.append(y[sel].mean() if n else np.nan)
        cnt.append(n)
    return np.array(pred), np.array(obs), np.array(cnt)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, max_iter: int = 200, lr: float = 0.05
) -> float:
    """Temperature scaling: one parameter T minimising NLL of sigmoid(logit / T).

    A single parameter cannot change the ranking, so AUC is invariant and only the
    calibration moves — which is the point. Fit on held-out data from the *training*
    fold.
    """
    z = np.asarray(logits, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    log_t = 0.0
    for _ in range(max_iter):
        t = np.exp(log_t)
        s = 1.0 / (1.0 + np.exp(-z / t))
        # d(NLL)/d(log T) = sum (s - y) * (-z / T)
        grad = float(np.mean((s - y) * (-z / t)))
        log_t -= lr * grad
        if abs(grad) < 1e-9:
            break
    return float(np.exp(log_t))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-z / temperature))
