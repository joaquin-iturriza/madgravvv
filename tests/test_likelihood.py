"""The likelihood-ratio cascade, against upstream's own two-line definition and the
coefficients distributed with the package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from madgrav_ml.eval import likelihood as L

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / ".reference/MADGRAV/data/o3a_frozen_lr_off200.npz"

pytestmark = pytest.mark.skipif(not FROZEN.exists(),
                                reason="run scripts/vendor_reference.sh")


def upstream_gate(g, s):
    return np.clip(g, -6.0, 6.0) * np.clip(np.asarray(s) / 3.0, 0, 1)


def upstream_feats(sH, sL, coh, cH, cL, gH, gL):
    return np.column_stack([sH, sL, coh, cH, cL, upstream_gate(gH, sH),
                            upstream_gate(gL, sL)])


def upstream_loglr(mu, sd, beta, F):
    return beta[0] + ((np.asarray(F, float) - mu) / sd) @ beta[1:]


def draw(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return dict(sH=rng.normal(3, 3, n), sL=rng.normal(4, 2, n),
                coh=rng.uniform(0, 0.6, n), cH=rng.normal(195, 25, n),
                cL=rng.normal(207, 23, n), gH=rng.normal(-1, 3, n),
                gL=rng.normal(-2, 3, n))


def test_features_match_upstream():
    d = draw()
    assert np.allclose(L.features(d["sH"], d["sL"], d["coh"], d["cH"], d["cL"],
                                  d["gH"], d["gL"]),
                       upstream_feats(**d))


def test_loglr_matches_upstream():
    d = draw()
    f = L.features(d["sH"], d["sL"], d["coh"], d["cH"], d["cL"], d["gH"], d["gL"])
    frozen = L.load_frozen(FROZEN)
    for g in (0, 1):
        mu, sd, beta = frozen[g]
        assert np.allclose(L.log_likelihood_ratio(f, mu, sd, beta),
                           upstream_loglr(mu, sd, beta, f))


def test_frozen_model_has_the_shape_the_features_need():
    frozen = L.load_frozen(FROZEN)
    for g in (0, 1):
        mu, sd, beta = frozen[g]
        assert len(mu) == len(sd) == 7
        assert len(beta) == 8
        assert np.all(sd > 0)
    assert frozen["floor"] == pytest.approx(4.5)


def test_coherence_coefficient_is_non_negative_in_both_folds():
    """Upstream constrains it (`bnds[3] = (0, None)`), so a signal can never be made
    more likely by being LESS coherent. Worth asserting on the shipped numbers: it is a
    physics prior baked into the artifact, not something the fit would find alone."""
    frozen = L.load_frozen(FROZEN)
    for g in (0, 1):
        assert frozen[g][2][3] >= 0.0


def test_arm_gate_ramps_with_loudness_and_clips():
    assert L.arm_gate(np.array([5.0]), np.array([0.0]))[0] == 0.0
    assert L.arm_gate(np.array([5.0]), np.array([3.0]))[0] == pytest.approx(5.0)
    assert L.arm_gate(np.array([5.0]), np.array([30.0]))[0] == pytest.approx(5.0)
    # clipped at +-GCLIP however loud the trigger
    assert L.arm_gate(np.array([100.0]), np.array([30.0]))[0] == pytest.approx(6.0)
    assert L.arm_gate(np.array([-100.0]), np.array([30.0]))[0] == pytest.approx(-6.0)


def test_held_out_scoring_uses_the_other_fold():
    frozen = L.load_frozen(FROZEN)
    d = draw(n=6)
    f = L.features(d["sH"], d["sL"], d["coh"], d["cH"], d["cL"], d["gH"], d["gL"])
    fold = np.array([0, 0, 0, 1, 1, 1])
    got = L.score_held_out(f, fold, frozen)
    assert np.allclose(got[:3], L.log_likelihood_ratio(f[:3], *frozen[1]))
    assert np.allclose(got[3:], L.log_likelihood_ratio(f[3:], *frozen[0]))
    # the two folds really do give different answers, so the choice is not cosmetic
    assert not np.allclose(L.log_likelihood_ratio(f, *frozen[0]),
                           L.log_likelihood_ratio(f, *frozen[1]))
