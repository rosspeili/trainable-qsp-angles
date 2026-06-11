"""Tests for Chao / pyqsp analytic baseline (standalone from PennyLane)."""

import numpy as np
import pytest

from qsp_jax.chao_baseline import (
    analytic_phases_chao,
    evaluate_chao_analytic,
    verify_pyqsp_reconstruction,
)
from qsp_jax.polynomial import CHEB_COEFFS_D5, chebyshev_sin_cheb_coeffs
from qsp_jax.train import TrainConfig


def test_chebyshev_sin_cheb_coeffs_degree5():
    coeffs = chebyshev_sin_cheb_coeffs(5)
    assert coeffs.shape == (6,)
    assert coeffs[0] == pytest.approx(0.0)
    assert coeffs[1] == pytest.approx(CHEB_COEFFS_D5[0])
    assert coeffs[3] == pytest.approx(CHEB_COEFFS_D5[1])
    assert coeffs[5] == pytest.approx(CHEB_COEFFS_D5[2])


def test_analytic_phases_chao_laurent_shape():
    phases, solver, elapsed, recon_err = analytic_phases_chao(5, method="laurent")
    assert phases.shape == (6,)
    assert "pyqsp_laurent" in solver
    assert elapsed >= 0.0
    assert recon_err < 1e-3


def test_analytic_phases_chao_sym_qsp_shape():
    phases, solver, elapsed, recon_err = analytic_phases_chao(5, method="sym_qsp")
    assert phases.shape == (6,)
    assert "pyqsp_sym_qsp" in solver
    assert elapsed >= 0.0
    assert recon_err < 1e-3


def test_evaluate_chao_analytic_returns_metrics():
    result = evaluate_chao_analytic(TrainConfig(degree=5, seed=0))
    assert len(result.angles) == 6
    assert len(result.angles_solver) == 6
    assert result.pyqsp_reconstruction_max_error < 1e-3
    assert result.metrics["train_mse"] < 0.02
    assert result.metrics_unmapped["train_mse"] > 0.5


def test_verify_pyqsp_reconstruction_matches_solver():
    cheb = chebyshev_sin_cheb_coeffs(5)
    phases, _, _, recon_err = analytic_phases_chao(5, method="laurent")
    manual_err = verify_pyqsp_reconstruction(np.array(phases), cheb, method="laurent")
    assert manual_err == pytest.approx(recon_err, rel=0, abs=1e-12)
