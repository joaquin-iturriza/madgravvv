"""The supervised arms: the glitch classifier and the HM/LM specialists.

Both are the same four-block convnet, differing only in input channels and in which
frequency band the tile was cropped to. Module names and shapes match the vendored
checkpoints (`lr_cascade/p1v42/arm_deploy_seed*.pt`, `search_mode/{hm,lm}_native_seed0.pt`)
so those weights load, which is the precondition for validating a reimplementation
before changing anything.

Measured parameter counts:

    glitch arm (1 input channel, `blocks`/`head`)      105,953
    HM specialist (2 channels, `b`/`h`)                106,097
    LM specialist (2 channels, `b`/`h`)                106,097

The 144-parameter difference is exactly the extra input channel in the first conv
(16 * 1 * 3 * 3). That is the scale C2 is enforced at: a "same size" replacement means
within a few percent of ~1.06e5, not "about a hundred thousand".
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _blocks(channels: list[int]) -> nn.ModuleList:
    """[Conv 3x3 pad 1 -> BatchNorm -> ReLU -> MaxPool 2x2] per channel step."""
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            for i in range(len(channels) - 1)
        ]
    )


def _head(width: int = 128, hidden: int = 64, dropout: float = 0.3) -> nn.Sequential:
    """Global average pool to a single vector, then a 2-layer MLP to one logit.

    Indices 3 and 5 carry the only parameters, which is why the checkpoints show
    `head.3` / `h.3` and `head.5` / `h.5` and nothing between.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(width, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class GlitchArm(nn.Module):
    """Single-detector glitch classifier. Upstream ensembles five seeds of this."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.blocks = _blocks([in_channels, 16, 32, 64, 128])
        self.head = _head(dropout=dropout)

    def features(self, x):
        """Feature map before the head — what Grad-CAM localisation currently reads.

        Note the resolution this leaves: four 2x2 poolings on a 128-wide time axis
        give 8 time steps, i.e. ~60 ms of localisation on signals that may last
        ~100 ms. That is the case for replacing Grad-CAM with an explicit
        localisation head (plan section 9.2).
        """
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class SpecialistCNN(nn.Module):
    """HM/LM two-channel specialist (H1 and L1 tiles stacked), cropped at t0.

    HM reads 20-140 Hz, LM reads 50-500 Hz. Upstream combines them as
    `max(HM, LM) >= 0.5`, which throws away the probability values and is why the
    FAR carries a trials factor of 4. Replacing that with a single calibrated
    statistic is plan section 9.1 — an arithmetic FAR improvement at fixed model.

    Submodule names are `b`/`h` (not `blocks`/`head`) to match the vendored
    `hm_native_seed0.pt` / `lm_native_seed0.pt` state dicts.
    """

    def __init__(self, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.b = _blocks([in_channels, 16, 32, 64, 128])
        self.h = _head(dropout=dropout)

    def forward(self, x):
        for bl in self.b:
            x = bl(x)
        return self.h(x).squeeze(-1)


class SeedEnsemble(nn.Module):
    """Mean-of-logits ensemble over independently seeded members.

    Deep ensembles are the plan's recommended uncertainty route (Phase 8): they match
    or beat variational BNNs for calibration at a fraction of the complexity, and the
    five-seed infrastructure already exists for the glitch arm. `logit_spread` is the
    epistemic term to hand to the likelihood ratio alongside the mean.

    An ensemble is N times the parameters of one member, so it is not a C2-compliant
    single-component replacement; report it as an ensemble, against the upstream
    five-seed ensemble, not against a single arm.
    """

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        if not members:
            raise ValueError("SeedEnsemble needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)

    @torch.no_grad()
    def logit_spread(self, x):
        """(mean, std) over members — the distributional summary the LR should consume."""
        logits = torch.stack([m(x) for m in self.members])
        return logits.mean(0), logits.std(0, unbiased=False)
