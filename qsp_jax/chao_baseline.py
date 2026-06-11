"""Standalone analytic QSP angles via pyqsp (Chao et al. Laurent completion).

This module is SDK-independent: it uses the ``pyqsp`` reference implementation
of the root-finding / Laurent completion method from Chao et al.
(2020, arXiv:2003.02831). PennyLane ``poly_to_angles`` remains in
``qsp_jax.baseline`` for side-by-side comparison.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from qsp_jax.metrics import evaluate_phases_mapped
from qsp_jax.polynomial import chebyshev_sin_cheb_coeffs
from qsp_jax.train import TrainConfig, signal_grid

ChaoMethod = Literal["laurent", "sym_qsp"]
ChaoSignalOperator = Literal["Wx", "Wz"]

CHAO_CONVENTION_NOTE = (
    "``angles_solver`` are pyqsp Wx/x phases; ``angles`` and ``metrics`` apply "
    "``phi_flat = pi/2 - phi_chao`` (docs/CONVENTIONS.md). "
    "``metrics_unmapped`` is raw flat-circuit MSE without mapping. "
    "``pyqsp_reconstruction_max_error`` is native solver verification."
)


@dataclass
class ChaoAnalyticResult:
    """Chao / pyqsp solver output, flat-circuit metrics, and native verification."""

    angles: list[float]
    angles_solver: list[float]
    solver: str
    method: str
    signal_operator: str
    degree: int
    solve_time_s: float
    metrics: dict[str, float]
    metrics_unmapped: dict[str, float]
    pyqsp_reconstruction_max_error: float
    convention_note: str = CHAO_CONVENTION_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_phases(raw: Any, method: ChaoMethod) -> np.ndarray:
    """Extract a 1-D phase vector from pyqsp return values."""
    if method == "sym_qsp":
        if not isinstance(raw, tuple) or len(raw) < 1:
            raise ValueError(f"sym_qsp expected (full_phases, ...) tuple, got {type(raw)}")
        phases = np.asarray(raw[0], dtype=np.float64)
    else:
        phases = np.asarray(raw, dtype=np.float64)
    if phases.ndim != 1:
        raise ValueError(f"Expected 1-D phase vector, got shape {phases.shape}")
    return phases


def verify_pyqsp_reconstruction(
    phases: np.ndarray,
    cheb_coeffs: np.ndarray,
    *,
    signal_operator: ChaoSignalOperator = "Wx",
    measurement: str | None = None,
    method: ChaoMethod = "laurent",
    n_points: int = 128,
    tolerance: float = 1e-3,
    raise_on_failure: bool = False,
) -> float:
    """
    Max absolute error between pyqsp response and target Chebyshev polynomial.

    Uses pyqsp's native circuit convention (not our flat ``qsp_circuit``).
    The solver may apply capitalization (``eps``, ``suc``); residual error
    at the 1e-4 level is normal when comparing to uncapitalized coefficients.
    """
    from pyqsp.response import ComputeQSPResponse

    if measurement is None:
        measurement = "x" if signal_operator == "Wx" else "z"

    adat = np.linspace(-1.0, 1.0, n_points)
    response = ComputeQSPResponse(
        adat,
        phases,
        signal_operator=signal_operator,
        measurement=measurement,
        sym_qsp=(method == "sym_qsp"),
    )["pdat"]
    expected = Chebyshev(cheb_coeffs)(adat)
    if method == "sym_qsp":
        achieved = np.imag(response)
    else:
        achieved = np.real(response)
    max_err = float(np.max(np.abs(achieved - expected)))
    if raise_on_failure and max_err > tolerance:
        raise ValueError(
            f"pyqsp reconstruction error {max_err:.3e} exceeds tolerance {tolerance:.3e}"
        )
    return max_err


def analytic_phases_chao(
    degree: int = 5,
    *,
    method: ChaoMethod = "laurent",
    signal_operator: ChaoSignalOperator = "Wx",
    measurement: str | None = None,
    tolerance: float = 1e-6,
    eps: float = 1e-4,
    suc: float = 1.0 - 1e-4,
) -> tuple[jnp.ndarray, str, float, float]:
    """
    Compute QSP phase angles with pyqsp (Chao Laurent completion or sym_qsp).

    Returns
    -------
    phases, solver_label, solve_time_s, pyqsp_reconstruction_max_error
    """
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases

    cheb_coeffs = chebyshev_sin_cheb_coeffs(degree)
    if measurement is None:
        measurement = "x" if signal_operator == "Wx" else "z"

    solver_label = f"pyqsp_{method}_{signal_operator}_{measurement}"
    kwargs: dict[str, Any] = {
        "signal_operator": signal_operator,
        "measurement": measurement,
        "method": method,
        "tolerance": tolerance,
        "eps": eps,
        "suc": suc,
    }
    if method == "sym_qsp":
        kwargs["chebyshev_basis"] = True
    else:
        kwargs["chebyshev_basis"] = False

    t0 = time.perf_counter()
    raw = QuantumSignalProcessingPhases(cheb_coeffs, **kwargs)
    elapsed = time.perf_counter() - t0

    phases_np = _normalize_phases(raw, method)
    recon_err = verify_pyqsp_reconstruction(
        phases_np,
        cheb_coeffs,
        signal_operator=signal_operator,
        measurement=measurement,
        method=method,
    )
    return jnp.array(phases_np), solver_label, elapsed, recon_err


def evaluate_chao_analytic(
    config: TrainConfig | None = None,
    *,
    method: ChaoMethod = "laurent",
    signal_operator: ChaoSignalOperator = "Wx",
    measurement: str | None = None,
) -> ChaoAnalyticResult:
    """Evaluate Chao / pyqsp angles on the same grids as gradient training."""
    cfg = config or TrainConfig()
    xs_train = signal_grid(cfg.n_signal_points, cfg.grid_min, cfg.grid_max)
    xs_holdout = signal_grid(cfg.holdout_points, cfg.holdout_min, cfg.holdout_max)

    phases, solver_used, solve_time, recon_err = analytic_phases_chao(
        cfg.degree,
        method=method,
        signal_operator=signal_operator,
        measurement=measurement,
    )
    metrics, metrics_unmapped, phases_flat = evaluate_phases_mapped(
        jnp.asarray(phases),
        xs_train,
        xs_holdout,
        degree=cfg.degree,
        source="chao",
    )

    return ChaoAnalyticResult(
        angles=[float(x) for x in phases_flat],
        angles_solver=[float(x) for x in phases],
        solver=solver_used,
        method=method,
        signal_operator=signal_operator,
        degree=cfg.degree,
        solve_time_s=solve_time,
        metrics=metrics,
        metrics_unmapped=metrics_unmapped,
        pyqsp_reconstruction_max_error=recon_err,
    )
