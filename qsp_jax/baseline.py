"""Analytic QSP phase angles via PennyLane ``poly_to_angles``."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qml

from qsp_jax.metrics import evaluate_phases
from qsp_jax.polynomial import monomial_coeffs_numpy
from qsp_jax.train import TrainConfig, signal_grid


@dataclass
class AnalyticResult:
    """Analytic solver output and circuit metrics."""

    angles: list[float]
    solver: str
    solve_time_s: float
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analytic_phases(angle_solver: str = "iterative") -> tuple[jnp.ndarray, str, float]:
    """
    Compute QSP phase angles for the default target polynomial.

    Notes
    -----
    PennyLane's ``root-finding`` solver can fail on this target; ``iterative``
    is the reliable default in PennyLane 0.44+.
    """
    poly = monomial_coeffs_numpy()
    t0 = time.perf_counter()
    angles = qml.poly_to_angles(poly, routine="QSP", angle_solver=angle_solver)
    elapsed = time.perf_counter() - t0
    return jnp.array(angles), angle_solver, elapsed


def evaluate_analytic(config: TrainConfig | None = None, angle_solver: str = "iterative") -> AnalyticResult:
    """Evaluate analytic angles on the same grids as gradient training."""
    cfg = config or TrainConfig()
    xs_train = signal_grid(cfg.n_signal_points, cfg.grid_min, cfg.grid_max)
    xs_holdout = signal_grid(cfg.holdout_points, cfg.holdout_min, cfg.holdout_max)

    phases, solver_used, solve_time = analytic_phases(angle_solver=angle_solver)
    metrics = evaluate_phases(phases, xs_train, xs_holdout)

    return AnalyticResult(
        angles=[float(x) for x in phases],
        solver=solver_used,
        solve_time_s=solve_time,
        metrics=metrics,
    )
