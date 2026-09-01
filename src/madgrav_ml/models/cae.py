"""The stage-1/stage-2 convolutional autoencoder, reimplemented so it can be trained.

The upstream release ships `assets/models/baseline_cae_weaksup_best.pt` but not the
code that produced it, so this is a from-the-weights reimplementation: the module
names and tensor shapes match the checkpoint exactly, which means the vendored
weights load into it and the reimplementation can be validated against them before
anything is changed (plan section 3.4).

Measured from the checkpoint (not estimated):

    conv encoder/decoder   185,857 params
    classifier head         65,537 params   (Linear(128*32*16 -> 1))
    ------------------------------------
    total                  251,394 params

Two things worth flagging, because they bear directly on Phase 4:

* The decoder is `MaxUnpool2d` fed with the encoder's pooling **indices**. Those
  indices are spatial information routed around the bottleneck — a skip connection in
  all but name. It is a large part of why the plain autoencoder reconstructs well and
  separates poorly, which is exactly the failure mode the masked-prediction objective
  is meant to remove.
* A quarter of the parameter budget is a linear head on the flattened latent. Any
  replacement proposal has to account for it: dropping the head "for free" is a 26%
  parameter saving that would not be a fair like-for-like comparison under C2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCAE(nn.Module):
    """Three conv/pool blocks down, three unpool/deconv blocks up, plus a linear head.

    Input is a single-channel 256 x 128 (frequency x time) Q-transform tile; the
    latent is 128 x 32 x 16. Shapes and submodule names follow the vendored
    checkpoint so `load_state_dict(..., strict=True)` succeeds against it.
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.20):
        super().__init__()
        d = dropout
        self.in_channels = in_channels
        self.latent_dim = 128 * 32 * 16

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(d)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(d)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(d)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (128, 32, 16))
        self.classifier = nn.Linear(self.latent_dim, 1)

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout2d(d)

        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.drop5 = nn.Dropout2d(d)

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, in_channels, 3, padding=1)

    def encode(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        s1 = x.size()
        x, i1 = self.pool1(x)

        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        s2 = x.size()
        x, i2 = self.pool2(x)

        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        s3 = x.size()
        x, i3 = self.pool3(x)

        return self.flatten(x), (i1, i2, i3), (s1, s2, s3)

    def decode(self, z, indices, sizes):
        i1, i2, i3 = indices
        s1, s2, s3 = sizes
        x = self.unflatten(z)
        x = self.unpool1(x, i3, output_size=s3)
        x = self.drop4(F.relu(self.bn4(self.deconv1(x))))
        x = self.unpool2(x, i2, output_size=s2)
        x = self.drop5(F.relu(self.bn5(self.deconv2(x))))
        x = self.unpool3(x, i1, output_size=s1)
        return self.deconv3(x)

    def forward(self, x):
        """Returns (reconstruction, logit). The logit is the stage-2 weak-supervision head."""
        z, idx, sizes = self.encode(x)
        recon = self.decode(z, idx, sizes)
        logit = self.classifier(z).squeeze(-1)
        return recon, logit

    @torch.no_grad()
    def reconstruction_error(self, x, reduction: str = "mean") -> torch.Tensor:
        """Per-tile MSE — the quantity the per-detector significance is built from.

        Kept as an explicit method because the whole of Phase 4 is about replacing
        *this readout*, not the encoder: an MSE over the reconstruction is a fixed,
        unlearned function of the latent and is not a likelihood ratio.
        """
        recon, _ = self(x)
        err = F.mse_loss(recon, x, reduction="none")
        err = err.flatten(1).mean(1)
        if reduction == "none":
            return err
        return getattr(torch, reduction)(err)


def margin_loss(
    err_noise: torch.Tensor,
    err_signal: torch.Tensor,
    margin: float = 3.0,
) -> torch.Tensor:
    """Stage-2 weak supervision: push signal reconstruction error above noise error.

    `relu(margin + mean(err_noise) - err_signal)`, i.e. a hinge that is satisfied once
    a signal tile reconstructs at least `margin` worse than typical noise. Upstream
    uses m = 3.0 with weight lambda = 2.0 on top of the stage-1 MSE; both are prime
    HPO targets (Phase 5) since neither has an obvious a priori value.

    Note this is the *only* place signal labels enter the front end. Constraint C3
    requires stage 1 to stay label-free; keep it that way.
    """
    return F.relu(margin + err_noise.mean() - err_signal).mean()


def stage2_loss(
    recon_noise: torch.Tensor,
    x_noise: torch.Tensor,
    err_signal: torch.Tensor,
    margin: float = 3.0,
    lam: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Stage-1 MSE on noise plus `lam` times the margin term. Returns (loss, parts)."""
    err_noise = F.mse_loss(recon_noise, x_noise, reduction="none").flatten(1).mean(1)
    mse = err_noise.mean()
    hinge = margin_loss(err_noise, err_signal, margin)
    return mse + lam * hinge, {"mse": float(mse.detach()), "margin": float(hinge.detach())}
