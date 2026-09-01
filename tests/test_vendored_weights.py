"""The reimplementations must accept the distributed weights, byte for byte.

This is the precondition for section 3.4 of the plan: every comparison is against a
reimplemented baseline, so an unexplained gap between the reimplementation and the
frozen weights poisons everything downstream. A `strict=True` load is the cheapest
possible check that the topology is right — it catches a renamed submodule, a wrong
channel count, a transposed convolution and a missing head, all of which would
otherwise train to a plausible loss and mean nothing.

Skipped without torch or without `.reference/MADGRAV` (`bash scripts/vendor_reference.sh`).
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REFERENCE = Path(".reference/MADGRAV")
CASES = [
    ("cae", "assets/models/baseline_cae_weaksup_best.pt", 251_394),
    ("glitch_arm", "lr_cascade/p1v42/arm_deploy_seed0.pt", 105_953),
    ("specialist_hm", "search_mode/hm_native_seed0.pt", 106_097),
    ("specialist_lm", "search_mode/lm_native_seed0.pt", 106_097),
]


def build(name):
    from madgrav_ml.models.arms import GlitchArm, SpecialistCNN
    from madgrav_ml.models.cae import BaselineCAE

    return {
        "cae": BaselineCAE,
        "glitch_arm": GlitchArm,
        "specialist_hm": SpecialistCNN,
        "specialist_lm": SpecialistCNN,
    }[name]()


@pytest.mark.parametrize("name,rel,expected", CASES)
def test_vendored_weights_load_strictly(name, rel, expected):
    from madgrav_ml.models.param_budget import count_parameters

    path = REFERENCE / rel
    if not path.exists():
        pytest.skip(f"{path} not vendored; run scripts/vendor_reference.sh")
    model = build(name)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    assert count_parameters(model) == expected


def test_cae_forward_shapes():
    from madgrav_ml.models.cae import BaselineCAE

    m = BaselineCAE().eval()
    x = torch.rand(2, 1, 256, 128)
    recon, logit = m(x)
    assert recon.shape == x.shape
    assert logit.shape == (2,)
    assert m.reconstruction_error(x, reduction="none").shape == (2,)
