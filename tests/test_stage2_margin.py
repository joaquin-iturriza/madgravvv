"""The stage-2 margin, checked against upstream's own `compute_margin_loss`.

The margin is the whole of stage 2, and three of its properties are invisible in the
paper's description and change training completely if read wrong:

  * `margin * noise` is MULTIPLICATIVE. Noise MSE here is ~2e-3, so an additive
    `relu(3 + noise - signal)` is saturated for every batch that will ever exist: the
    relu never switches off and the gradient degenerates to "raise signal error without
    bound", a term with no fixed point rather than a margin.
  * the noise term inside the hinge is DETACHED, so the margin can raise signal error
    but cannot satisfy itself by lowering noise error.
  * it is `relu` of the batch MEANS, not the mean of per-tile relus.

We got all three wrong on the first pass. These tests compare against the vendored
implementation rather than against a restatement of it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from madgrav_ml.models.cae import margin_loss, stage2_loss  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / ".reference/MADGRAV"

pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(), reason="run scripts/vendor_reference.sh"
)


def upstream():
    path = str(REFERENCE / "improved")
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("MADGRAV_ROOT", str(REFERENCE))
    return pytest.importorskip("improved_pipeline")


class Tiny(nn.Module):
    """A trainable stand-in whose forward returns a reconstruction only.

    Upstream's `compute_reconstruction_loss` does `recon = model(qt)`; `BaselineCAE`
    here returns `(recon, logit)`, so the comparison needs a model with upstream's
    calling convention. The margin arithmetic is what is under test, not the encoder.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def batches(seed=0, n=8, size=16):
    g = torch.Generator().manual_seed(seed)
    x_n = torch.rand(n, 1, size, size, generator=g)
    x_s = torch.rand(n, 1, size, size, generator=g)
    return x_n, x_s


def errors(model, x):
    return ((model(x) - x) ** 2).flatten(1).mean(1)


@pytest.mark.parametrize("margin,lam", [(3.0, 2.0), (1.5, 0.5), (5.0, 1.0)])
def test_matches_upstream_compute_margin_loss(margin, lam):
    ip = upstream()
    torch.manual_seed(0)
    model = Tiny()
    x_n, x_s = batches()

    reference, ref_noise, ref_signal, ref_margin, _, _ = ip.compute_margin_loss(
        model, (x_n,), (x_s,), torch.device("cpu"), margin, lam
    )
    mine, parts = stage2_loss(errors(model, x_n), errors(model, x_s), margin, lam)

    assert torch.allclose(mine, reference, rtol=1e-6, atol=1e-9)
    assert parts["noise"] == pytest.approx(ref_noise, rel=1e-6)
    assert parts["signal"] == pytest.approx(ref_signal, rel=1e-6)
    assert parts["margin"] == pytest.approx(ref_margin, rel=1e-6, abs=1e-12)


def test_margin_is_multiplicative_not_additive():
    """At the scale the reconstruction error actually lives at, the two differ hugely."""
    noise = torch.full((8,), 2.0e-3)
    signal = torch.full((8,), 5.0e-3)
    # multiplicative: relu(3 * 2e-3 - 5e-3) = 1e-3, an unsatisfied but finite margin
    assert float(margin_loss(noise, signal, 3.0)) == pytest.approx(1.0e-3, rel=1e-6)
    # an additive reading would give relu(3 + 2e-3 - 5e-3) ~ 3.0, i.e. permanently
    # saturated for every batch that will ever be seen
    assert float(margin_loss(noise, signal, 3.0)) < 0.01


def test_margin_switches_off_when_satisfied():
    """A signal error already three times the noise contributes exactly zero gradient."""
    noise = torch.full((8,), 2.0e-3, requires_grad=True)
    signal = torch.full((8,), 9.0e-3, requires_grad=True)
    hinge = margin_loss(noise, signal, 3.0)
    assert float(hinge) == 0.0
    hinge.backward()
    assert torch.count_nonzero(signal.grad) == 0


def test_noise_term_inside_the_hinge_is_detached():
    """The margin may raise signal error; it may not lower noise error to satisfy itself.

    Only the plain MSE term trains the noise branch, so the gradient of the hinge alone
    with respect to the noise errors must be exactly zero.
    """
    noise = torch.full((8,), 2.0e-3, requires_grad=True)
    signal = torch.full((8,), 1.0e-3, requires_grad=True)
    margin_loss(noise, signal, 3.0).backward()
    assert noise.grad is None or torch.count_nonzero(noise.grad) == 0
    assert torch.count_nonzero(signal.grad) > 0


def test_hinge_is_on_the_batch_mean_not_per_tile():
    """One hinge per batch: a batch satisfying the margin on average contributes nothing,
    even when individual tiles fall short. The mean-of-relus reading would not."""
    noise = torch.full((4,), 1.0e-3)
    # mean(signal) = 5e-3 > 3 * 1e-3, but two of the four tiles are below the margin
    signal = torch.tensor([0.5e-3, 0.5e-3, 9.5e-3, 9.5e-3])
    assert float(margin_loss(noise, signal, 3.0)) == 0.0
    per_tile = torch.relu(3.0 * noise.mean() - signal).mean()
    assert float(per_tile) > 0.0


def test_total_is_noise_plus_lambda_hinge():
    noise = torch.full((8,), 2.0e-3)
    signal = torch.full((8,), 1.0e-3)
    total, parts = stage2_loss(noise, signal, 3.0, 2.0)
    assert float(total) == pytest.approx(2.0e-3 + 2.0 * (3 * 2.0e-3 - 1.0e-3), rel=1e-6)
    assert parts["margin"] == pytest.approx(5.0e-3, rel=1e-6)
