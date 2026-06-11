"""Tests for JAX traceability requirements in QSP circuits."""

import jax
import jax.numpy as jnp
import pennylane as qp

from qsp_jax.circuit import qsp_circuit


def test_flat_circuit_preserves_phase_gradients():
    """Live JAX phase arrays must receive non-zero gradients."""
    phases = jnp.array([0.3, -0.5, 0.8, -0.2, 0.1, -0.6])
    grad = jax.grad(lambda p: qsp_circuit(p, 0.4))(phases)
    assert grad.shape == phases.shape
    assert float(jnp.linalg.norm(grad)) > 1e-6


def test_stop_gradient_on_phases_breaks_learning_signal():
    """
    Detaching phases from the JAX graph yields zero gradients.

    This mirrors the failure mode when operator objects or concrete NumPy
    values are captured at circuit-build time (e.g. misused QSVT patterns).
    """
    phases = jnp.array([0.3, -0.5, 0.8, -0.2, 0.1, -0.6])

    def frozen_circuit(phases, x):
        frozen = jax.lax.stop_gradient(phases)
        return qsp_circuit(frozen, x)

    grad = jax.grad(lambda p: frozen_circuit(p, 0.4))(phases)
    assert float(jnp.linalg.norm(grad)) < 1e-10


def test_concrete_rotation_angles_break_parameter_gradients():
    """Replacing a phase tracer with a Python float removes its gradient component."""

    dev = qp.device("default.qubit", wires=1)
    concrete_angle = 0.42

    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def broken_qnode(phases, x):
        qp.RZ(-2.0 * concrete_angle, wires=0)
        qp.Hadamard(wires=0)
        qp.RZ(-2.0 * jnp.arccos(x), wires=0)
        qp.Hadamard(wires=0)
        qp.RZ(-2.0 * phases[1], wires=0)
        return qp.expval(qp.PauliX(0))

    phases = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    grad = jax.grad(lambda p: broken_qnode(p, 0.3))(phases)
    assert float(jnp.linalg.norm(grad[0])) < 1e-10
    assert float(jnp.linalg.norm(grad[1:])) > 1e-6
