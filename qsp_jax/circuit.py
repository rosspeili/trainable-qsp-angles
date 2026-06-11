"""
circuit.py — QSP circuit, target polynomial, and loss function.

The QSP sequence is implemented as a flat circuit (manual interleaving of signal
oracle and phase rotations). This preserves the JAX computation graph through
the phase angles, making the circuit fully differentiable via jax.grad.

This file uses PennyLane as the **reference quantum frontend** only. The flat
sequence and loss are framework-agnostic; equivalent circuits can be built in
Qiskit, Cirq, TensorFlow Quantum, etc. See docs/FRAMEWORKS.md.

Note: high-level QSVT template APIs (e.g. PennyLane qp.QSVT) may capture
concrete values at construction time and break JAX tracing. Inline primitives
avoid this across SDKs when parameters stay in the autodiff graph.

Reference:
    Martyn et al., "A Grand Unification of Quantum Algorithms",
    PRX Quantum 2 (2021). https://arxiv.org/abs/2105.02859
"""

import jax
import jax.numpy as jnp
import pennylane as qp

from qsp_jax.polynomial import target_poly

__all__ = ["qsp_circuit", "target_poly", "loss_fn"]

# float64 must be enabled before any JAX computation is traced.
# Setting it here at import time ensures all QNodes and gradient
# computations in this module use 64-bit precision.
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

_dev = qp.device("default.qubit", wires=1)

# ---------------------------------------------------------------------------
# QSP circuit
# ---------------------------------------------------------------------------


@qp.qnode(_dev, interface="jax", diff_method="backprop")
def _qsp_qnode(phases, x):
    """
    Flat QSP sequence on one qubit.

    Signal oracle W(x): the standard QSP signal unitary
        W(x) = H @ RZ(-2*arccos(x)) @ H
    which encodes x in the top-left matrix element:
        <0|W(x)|0> = x  (for x in [-1, 1])

    Phase rotations: RZ(-2*phi_k) on wires=0.

    Sequence (degree d = len(phases) - 1):
        RZ(-2*phi_0)
        [W(x)  RZ(-2*phi_k)]  for k = 1, ..., d

    The expected value <X> encodes a polynomial in x of degree d.
    IMPORTANT: x must be in [-1, 1] for arccos to be real-valued.
    """
    n = phases.shape[0]
    # Initial phase rotation
    qp.RZ(-2.0 * phases[0], wires=0)
    for k in range(1, n):
        # Signal oracle W(x) = H R_Z(-2*arccos(x)) H
        qp.Hadamard(wires=0)
        qp.RZ(-2.0 * jnp.arccos(x), wires=0)
        qp.Hadamard(wires=0)
        # Phase rotation
        qp.RZ(-2.0 * phases[k], wires=0)
    return qp.expval(qp.PauliX(0))
    

def qsp_circuit(phases, x):
    """
    Evaluate the QSP polynomial at signal value x given phase angles.

    Parameters
    ----------
    phases : jnp.ndarray, shape (d+1,)
        QSP phase angles in radians. Use d+1 angles for degree-d polynomial.
    x : float
        Signal value in (-1, 1). arccos is not real-valued at the endpoints.

    Returns
    -------
    float
        Polynomial value P(x) encoded in the circuit expectation value.
    """
    return _qsp_qnode(phases, x)


# Re-exported from qsp_jax.polynomial for backward compatibility.


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def loss_fn(phases, xs, degree: int = 5):
    """
    Mean squared error between the QSP circuit output and the target polynomial.

    Parameters
    ----------
    phases : jnp.ndarray, shape (d+1,)
        QSP phase angles in radians.
    xs : jnp.ndarray, shape (N,)
        Grid of signal values in [-1, 1].
    degree : int
        Target polynomial degree (must match len(phases) - 1).

    Returns
    -------
    float
        Scalar MSE loss.
    """
    circuit_vals = jax.vmap(lambda x: qsp_circuit(phases, x))(xs)
    target_vals = target_poly(xs, degree=degree)
    return jnp.mean((circuit_vals - target_vals) ** 2)
