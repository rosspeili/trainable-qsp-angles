"""Reproducibility and baseline accuracy tests."""

import jax.numpy as jnp
import pytest

from qsp_jax.polynomial import CHEB_COEFFS_D5, target_poly
from qsp_jax.train import TrainConfig, train


def test_same_seed_same_final_mse():
    cfg = TrainConfig(steps=12, seed=7, log_every=100)
    r1 = train(cfg, verbose=False)
    r2 = train(cfg, verbose=False)
    assert r1.loss_final == pytest.approx(r2.loss_final, rel=0, abs=1e-12)
    assert r1.phases_final == pytest.approx(r2.phases_final, rel=0, abs=1e-12)


def test_degree5_target_matches_legacy_coefficients():
    xs = jnp.linspace(-0.9, 0.9, 50)
    c1, c3, c5 = CHEB_COEFFS_D5
    legacy = c1 * xs + c3 * (4 * xs**3 - 3 * xs) + c5 * (16 * xs**5 - 20 * xs**3 + 5 * xs)
    assert jnp.allclose(target_poly(xs, degree=5), legacy, atol=1e-6)


@pytest.mark.slow
def test_full_train_achieves_sub_mill_mse():
    result = train(TrainConfig(steps=500, seed=0), verbose=False)
    assert result.metrics["train_mse"] < 1e-3
    assert result.metrics["holdout_mse"] < 2e-3


@pytest.mark.slow
def test_holdout_mse_on_dense_grid():
    """Hold-out (512 pts) exceeds train MSE (64 pts); both stay below protocol bounds."""
    result = train(TrainConfig(steps=500, seed=0), verbose=False)
    assert result.metrics["train_mse"] < 1e-3
    assert result.metrics["holdout_mse"] > result.metrics["train_mse"]
    assert result.metrics["holdout_mse"] < 2e-3
