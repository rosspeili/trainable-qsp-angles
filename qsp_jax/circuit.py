"""
circuit.py — QSP circuit, target polynomial, and loss function.

The QSP sequence is implemented as a flat PennyLane circuit (manual interleaving
of signal oracle and phase rotations). This preserves the JAX computation graph
through the phase angles, making the circuit fully differentiable via jax.grad.

Note: qp.QSVT takes pre-built operator *objects* as arguments. Those objects
capture concrete values at construction time, breaking JAX tracing. The manual
flat-circuit approach avoids this and is the correct pattern for JAX-based
variational optimization of QSP phase angles.

Reference:
    Martyn et al., "A Grand Unification of Quantum Algorithms",
    PRX Quantum 2 (2021). https://arxiv.org/abs/2105.02859
"""

import jax
import jax.numpy as jnp
import pennylane as qp

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


# ---------------------------------------------------------------------------
# Target polynomial
# ---------------------------------------------------------------------------


def target_poly(x):
    """
    Target polynomial: degree-5 odd Chebyshev approximation of sin(x).

    Uses T_1, T_3, T_5 (Chebyshev polynomials of the first kind):
        T1(x) = x
        T3(x) = 4x^3 - 3x
        T5(x) = 16x^5 - 20x^3 + 5x

    Coefficients fit sin(x) on [-1, 1] via Chebyshev expansion.

    Parameters
    ----------
    x : float or array-like
        Signal value(s) in [-1, 1].

    Returns
    -------
    float or array-like
        Polynomial value(s), bounded in [-1, 1].
    """
    c1 = 0.9775
    c3 = -0.1564
    c5 = 0.0158

    T1 = x
    T3 = 4.0 * x**3 - 3.0 * x
    T5 = 16.0 * x**5 - 20.0 * x**3 + 5.0 * x

    return c1 * T1 + c3 * T3 + c5 * T5


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def loss_fn(phases, xs):
    """
    Mean squared error between the QSP circuit output and the target polynomial.

    Parameters
    ----------
    phases : jnp.ndarray, shape (d+1,)
        QSP phase angles in radians.
    xs : jnp.ndarray, shape (N,)
        Grid of signal values in [-1, 1].

    Returns
    -------
    float
        Scalar MSE loss.
    """
    circuit_vals = jax.vmap(lambda x: qsp_circuit(phases, x))(xs)
    target_vals = target_poly(xs)
    return jnp.mean((circuit_vals - target_vals) ** 2)
