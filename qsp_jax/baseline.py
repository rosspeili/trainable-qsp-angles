"""Analytic QSP phase angles — baseline solvers (PennyLane wrapper + notes on alternatives)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qml

from qsp_jax.metrics import evaluate_phases
from qsp_jax.polynomial import monomial_coeffs_numpy
from qsp_jax.train import SCHEMA_VERSION, TrainConfig, signal_grid


@dataclass
class AnalyticResult:
    """Analytic solver output and circuit metrics."""

    angles: list[float]
    solver: str
    degree: int
    solve_time_s: float
    metrics: dict[str, float]
    convention_note: str = (
        "PennyLane QSP angles may use a different convention than the flat "
        "<X> circuit; high MSE here does not necessarily imply solver failure."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analytic_phases(degree: int = 5, angle_solver: str = "iterative") -> tuple[jnp.ndarray, str, float]:
    """
    Compute QSP phase angles for the Chebyshev-sin target at ``degree``.

    Uses PennyLane ``poly_to_angles`` as an in-repo convenience wrapper.
    For SDK-independent angles, a standalone Chao et al. implementation is
    preferable (see docs/FRAMEWORKS.md).

    Notes
    -----
    PennyLane's ``root-finding`` solver can fail on some targets; ``iterative``
    is the reliable default in PennyLane 0.44+.
    """
    poly = monomial_coeffs_numpy(degree)
    t0 = time.perf_counter()
    angles = qml.poly_to_angles(poly, routine="QSP", angle_solver=angle_solver)
    elapsed = time.perf_counter() - t0
    return jnp.array(angles), angle_solver, elapsed


def evaluate_analytic(
    config: TrainConfig | None = None,
    angle_solver: str = "iterative",
) -> AnalyticResult:
    """Evaluate analytic angles on the same grids as gradient training."""
    cfg = config or TrainConfig()
    xs_train = signal_grid(cfg.n_signal_points, cfg.grid_min, cfg.grid_max)
    xs_holdout = signal_grid(cfg.holdout_points, cfg.holdout_min, cfg.holdout_max)

    phases, solver_used, solve_time = analytic_phases(cfg.degree, angle_solver=angle_solver)
    metrics = evaluate_phases(phases, xs_train, xs_holdout, degree=cfg.degree)

    return AnalyticResult(
        angles=[float(x) for x in phases],
        solver=solver_used,
        degree=cfg.degree,
        solve_time_s=solve_time,
        metrics=metrics,
    )
