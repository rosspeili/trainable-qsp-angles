"""Target polynomial definitions and QSP-admissible monomial coefficients."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as cheb

# Degree-5 odd Chebyshev coefficients (T1, T3, T5) for sin(x) on [-1, 1].
CHEB_COEFFS_D5 = (0.9775, -0.1564, 0.0158)

# Equivalent monomial coefficients [a0, a1, ..., a5] for poly_to_angles.
MONOMIAL_COEFFS_D5 = (0.0, 1.5257, 0.0, -0.9416, 0.0, 0.2528)

DEFAULT_DEGREE = 5


def _chebyshev_odd_term(k: int, x):
    """Chebyshev polynomial T_k(x) for odd k via recurrence-friendly closed forms."""
    if k == 1:
        return x
    if k == 3:
        return 4.0 * x**3 - 3.0 * x
    if k == 5:
        return 16.0 * x**5 - 20.0 * x**3 + 5.0 * x
    # General odd k via numpy Chebyshev evaluation
    return cheb.chebval(x, [0.0] * k + [1.0])


def chebyshev_sin_monomial_coeffs(degree: int) -> np.ndarray:
    """
    Monomial coefficients [a0, ..., a_degree] for odd Chebyshev approximation of sin(x).

    Parameters
    ----------
    degree : int
        Odd polynomial degree (>= 1).
    """
    if degree < 1 or degree % 2 == 0:
        raise ValueError(f"degree must be a positive odd integer, got {degree}")

    if degree == DEFAULT_DEGREE:
        return np.array(MONOMIAL_COEFFS_D5, dtype=np.float64)

    xs = np.linspace(-1.0, 1.0, 2000)
    ys = np.sin(xs)
    cheb_coeffs = cheb.chebfit(xs, ys, degree)
    mono = cheb.Chebyshev(cheb_coeffs).convert(kind=np.polynomial.Polynomial)
    return np.array(mono.coef, dtype=np.float64)


def target_poly(x, degree: int = DEFAULT_DEGREE):
    """
    Odd Chebyshev approximation of sin(x) at the requested degree.

    Parameters
    ----------
    x : array-like
        Signal value(s) in [-1, 1].
    degree : int
        Polynomial degree (must be odd).
    """
    if degree == DEFAULT_DEGREE:
        c1, c3, c5 = CHEB_COEFFS_D5
        return c1 * _chebyshev_odd_term(1, x) + c3 * _chebyshev_odd_term(3, x) + c5 * _chebyshev_odd_term(5, x)

    coeffs = chebyshev_sin_monomial_coeffs(degree)
    xs = jnp.asarray(x)
    out = jnp.zeros_like(xs, dtype=jnp.float64)
    for k, coeff in enumerate(coeffs):
        out = out + coeff * xs**k
    return out


def monomial_coeffs_numpy(degree: int = DEFAULT_DEGREE) -> np.ndarray:
    """Low-to-high monomial coefficients for PennyLane ``poly_to_angles``."""
    return chebyshev_sin_monomial_coeffs(degree)
