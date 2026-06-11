"""Evaluation metrics for trained or analytic QSP phase angles."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qsp_jax.circuit import qsp_circuit
from qsp_jax.polynomial import target_poly


def mse(phases: jnp.ndarray, xs: jnp.ndarray) -> float:
    """Mean squared error between circuit output and target polynomial."""
    circuit_vals = jax.vmap(lambda x: qsp_circuit(phases, x))(xs)
    target_vals = target_poly(xs)
    return float(jnp.mean((circuit_vals - target_vals) ** 2))


def max_pointwise_error(phases: jnp.ndarray, xs: jnp.ndarray) -> float:
    """Maximum absolute error on the given signal grid."""
    circuit_vals = jax.vmap(lambda x: qsp_circuit(phases, x))(xs)
    target_vals = target_poly(xs)
    return float(jnp.max(jnp.abs(circuit_vals - target_vals)))


def evaluate_phases(phases: jnp.ndarray, xs_train: jnp.ndarray, xs_holdout: jnp.ndarray) -> dict:
    """Compute train and hold-out metrics for a phase vector."""
    return {
        "train_mse": mse(phases, xs_train),
        "holdout_mse": mse(phases, xs_holdout),
        "train_max_error": max_pointwise_error(phases, xs_train),
        "holdout_max_error": max_pointwise_error(phases, xs_holdout),
    }
