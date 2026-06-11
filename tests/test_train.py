"""Smoke tests for the shared training module."""

from qsp_jax.train import TrainConfig, train


def test_train_reduces_loss_in_few_steps():
    """Short training run must decrease loss from initialization."""
    result = train(TrainConfig(steps=8, seed=0, log_every=4))
    assert result.loss_final < result.loss_init
    assert len(result.phases_final) == result.config.degree + 1
    assert "train_mse" in result.metrics
