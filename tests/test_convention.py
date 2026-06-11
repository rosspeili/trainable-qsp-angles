"""Tests for QSP phase convention mapping."""

import numpy as np
import pytest
import pennylane as qml

from qsp_jax.chao_baseline import analytic_phases_chao
from qsp_jax.convention import chao_to_flat, map_to_flat, pennylane_to_flat, pennylane_to_pyqsp
from qsp_jax.metrics import mse
from qsp_jax.polynomial import monomial_coeffs_numpy
import jax.numpy as jnp


def test_chao_to_flat_reduces_mse_vs_unmapped():
    phases, _, _, _ = analytic_phases_chao(5, method="laurent")
    xs = jnp.linspace(-0.95, 0.95, 64)
    unmapped = float(mse(jnp.array(phases), xs, degree=5))
    mapped = float(mse(jnp.array(chao_to_flat(np.array(phases))), xs, degree=5))
    assert unmapped > 0.5
    assert mapped < unmapped / 10.0
    assert mapped < 0.02


def test_pennylane_to_flat_matches_chao_chain():
    poly = monomial_coeffs_numpy(5)
    pl = np.array(qml.poly_to_angles(poly, routine="QSP", angle_solver="iterative"))
    chao, _, _, _ = analytic_phases_chao(5, method="laurent")
    from_pl = pennylane_to_pyqsp(pl)
    assert np.max(np.abs(from_pl - np.array(chao))) < 2e-4
    assert np.allclose(pennylane_to_flat(pl), chao_to_flat(np.array(chao)), atol=1e-3)


def test_map_to_flat_dispatch():
    phases = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(map_to_flat(phases, "flat"), phases)
    with pytest.raises(ValueError):
        map_to_flat(phases, "unknown")
