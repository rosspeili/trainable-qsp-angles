"""Target polynomial definitions and QSP-admissible monomial coefficients."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Degree-5 odd Chebyshev coefficients (T1, T3, T5) for sin(x) on [-1, 1].
CHEB_COEFFS = (0.9775, -0.1564, 0.0158)

# Equivalent monomial coefficients [a0, a1, ..., a5] for poly_to_angles.
MONOMIAL_COEFFS = (0.0, 1.5257, 0.0, -0.9416, 0.0, 0.2528)


def target_poly(x):
    """
    Degree-5 odd Chebyshev approximation of sin(x).

    Parameters
    ----------
    x : array-like
        Signal value(s) in [-1, 1].

    Returns
    -------
    float or array-like
        Polynomial value(s), bounded in [-1, 1].
    """
    c1, c3, c5 = CHEB_COEFFS

    T1 = x
    T3 = 4.0 * x**3 - 3.0 * x
    T5 = 16.0 * x**5 - 20.0 * x**3 + 5.0 * x

    return c1 * T1 + c3 * T3 + c5 * T5


def monomial_coeffs_numpy() -> np.ndarray:
    """Low-to-high monomial coefficients for PennyLane ``poly_to_angles``."""
    return np.array(MONOMIAL_COEFFS, dtype=np.float64)
