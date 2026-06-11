"""Evaluation metrics for trained or analytic QSP phase angles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qsp_jax.circuit import qsp_circuit
from qsp_jax.polynomial import target_poly


def mse(phases: jnp.ndarray, xs: jnp.ndarray, degree: int = 5) -> float:
    """Mean squared error between circuit output and target polynomial."""
    circuit_vals = jax.vmap(lambda x: qsp_circuit(phases, x))(xs)
    target_vals = target_poly(xs, degree=degree)
    return float(jnp.mean((circuit_vals - target_vals) ** 2))


def max_pointwise_error(phases: jnp.ndarray, xs: jnp.ndarray, degree: int = 5) -> float:
    """Maximum absolute error on the given signal grid."""
    circuit_vals = jax.vmap(lambda x: qsp_circuit(phases, x))(xs)
    target_vals = target_poly(xs, degree=degree)
    return float(jnp.max(jnp.abs(circuit_vals - target_vals)))


def evaluate_phases(
    phases: jnp.ndarray,
    xs_train: jnp.ndarray,
    xs_holdout: jnp.ndarray,
    degree: int = 5,
) -> dict[str, float]:
    """Compute train and hold-out metrics for a phase vector."""
    return {
        "train_mse": mse(phases, xs_train, degree=degree),
        "holdout_mse": mse(phases, xs_holdout, degree=degree),
        "train_max_error": max_pointwise_error(phases, xs_train, degree=degree),
        "holdout_max_error": max_pointwise_error(phases, xs_holdout, degree=degree),
    }


def evaluate_phases_mapped(
    solver_phases: np.ndarray,
    xs_train: jnp.ndarray,
    xs_holdout: jnp.ndarray,
    *,
    degree: int,
    source: str,
) -> tuple[dict[str, float], dict[str, float], jnp.ndarray]:
    """
    Evaluate raw and convention-mapped phases on the flat training circuit.

    Returns (metrics_mapped, metrics_unmapped, flat_phases_jax).
    """
    from qsp_jax.convention import map_to_flat

    raw = jnp.array(solver_phases)
    flat = jnp.array(map_to_flat(solver_phases, source=source))
    return (
        evaluate_phases(flat, xs_train, xs_holdout, degree=degree),
        evaluate_phases(raw, xs_train, xs_holdout, degree=degree),
        flat,
    )
