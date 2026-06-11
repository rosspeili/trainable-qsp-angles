"""
tests/test_circuit.py — Unit tests for qsp_jax.circuit.

Tests are fast (CPU, small circuits, <= 10 gradient steps where needed).
They validate:
  - Output shape and boundedness
  - JAX differentiability (non-zero gradient, correct shape)
  - Target polynomial properties (range, odd symmetry, zero at origin)
  - Loss function behavior (positive, decreases under gradient updates)
"""

import jax
import jax.numpy as jnp
import numpy as np

from qsp_jax.circuit import loss_fn, qsp_circuit, target_poly

KEY = jax.random.PRNGKey(0)
XS = jnp.linspace(-0.9, 0.9, 32)


# ---------------------------------------------------------------------------
# qsp_circuit
# ---------------------------------------------------------------------------


def test_qsp_circuit_returns_scalar():
    """Circuit must return a scalar for a single signal value."""
    phases = jnp.zeros(6)
    result = qsp_circuit(phases, 0.5)
    assert jnp.asarray(result).shape == (), "Expected scalar output"


def test_qsp_circuit_output_bounded():
    """Circuit output must lie in [-1, 1] (quantum expectation value)."""
    phases = jax.random.uniform(KEY, shape=(6,), minval=-jnp.pi, maxval=jnp.pi)
    for x in jnp.linspace(-1.0, 1.0, 10):
        val = float(qsp_circuit(phases, x))
        assert -1.0 - 1e-5 <= val <= 1.0 + 1e-5, f"Out-of-bounds output {val:.4f} at x={x:.2f}"


def test_qsp_circuit_gradient_shape():
    """Gradient of circuit output w.r.t. phases must have correct shape."""
    phases = jnp.array([0.3, -0.5, 0.8, -0.2, 0.1, -0.6])
    grad = jax.grad(lambda p: qsp_circuit(p, 0.4))(phases)
    assert grad.shape == phases.shape, f"Gradient shape {grad.shape} != phases shape {phases.shape}"


def test_qsp_circuit_gradient_nonzero():
    """Gradient must be non-zero at a point away from trivial symmetry."""
    phases = jnp.array([0.3, -0.5, 0.8, -0.2, 0.1, -0.6])
    grad = jax.grad(lambda p: qsp_circuit(p, 0.4))(phases)
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient"
    assert float(jnp.linalg.norm(grad)) > 1e-6, "Gradient is effectively zero"


# ---------------------------------------------------------------------------
# target_poly
# ---------------------------------------------------------------------------


def test_target_poly_range():
    """Target polynomial must stay within [-1, 1] on its domain."""
    xs = jnp.linspace(-1.0, 1.0, 200)
    vals = target_poly(xs)
    assert jnp.all(vals >= -1.0 - 1e-6), "Target poly below -1"
    assert jnp.all(vals <= 1.0 + 1e-6), "Target poly above 1"


def test_target_poly_odd():
    """Target polynomial must be odd: p(-x) == -p(x)."""
    xs = jnp.linspace(0.0, 1.0, 50)
    max_asymmetry = float(jnp.max(jnp.abs(target_poly(-xs) + target_poly(xs))))
    assert max_asymmetry < 1e-6, f"Polynomial is not odd: max asymmetry = {max_asymmetry:.2e}"


def test_target_poly_at_zero():
    """Odd polynomial must satisfy p(0) == 0."""
    val = float(target_poly(0.0))
    assert abs(val) < 1e-9, f"target_poly(0) = {val}, expected 0"


# ---------------------------------------------------------------------------
# loss_fn
# ---------------------------------------------------------------------------


def test_loss_fn_returns_scalar():
    """Loss must return a non-negative scalar."""
    phases = jax.random.uniform(KEY, shape=(6,))
    loss = loss_fn(phases, XS)
    assert jnp.asarray(loss).shape == (), "Expected scalar loss"
    assert float(loss) >= 0.0, "Loss must be non-negative"


def test_loss_fn_positive_for_random_phases():
    """Loss must be positive for a random phase initialization."""
    phases = jax.random.uniform(KEY, shape=(6,), minval=-jnp.pi, maxval=jnp.pi)
    assert float(loss_fn(phases, XS)) > 1e-6, "Loss unexpectedly zero for random phases"


def test_loss_fn_decreases_after_gradient_steps():
    """Loss must decrease after 10 gradient descent steps from a non-trivial init."""
    phases = jnp.array([0.3, -0.5, 0.8, -0.2, 0.1, -0.6])
    lr = 0.05
    loss_init = float(loss_fn(phases, XS))

    for _ in range(10):
        grad = jax.grad(loss_fn)(phases, XS)
        phases = phases - lr * grad

    loss_final = float(loss_fn(phases, XS))
    assert loss_final < loss_init, (
        f"Loss did not decrease: init={loss_init:.6f}, final={loss_final:.6f}"
    )
