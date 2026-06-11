"""Analytic QSP phase angles — baseline solvers (PennyLane wrapper + notes on alternatives).

For a standalone Chao et al. / pyqsp baseline (no PennyLane), see ``qsp_jax.chao_baseline``.
Both backends are kept for side-by-side comparison experiments.

Convention mapping to the flat training circuit: ``qsp_jax.convention`` and
``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import pennylane as qml

from qsp_jax.chao_baseline import analytic_phases_chao
from qsp_jax.convention import chao_to_flat
from qsp_jax.metrics import evaluate_phases, evaluate_phases_mapped
from qsp_jax.polynomial import monomial_coeffs_numpy
from qsp_jax.train import TrainConfig, signal_grid

CONVENTION_NOTE = (
    "``angles_solver`` holds native PennyLane ``poly_to_angles`` output. "
    "``metrics`` (mapped) use the shared Chebyshev→pyqsp→flat map "
    "(``chao_to_flat``) because direct PL→flat inversion is only reliable at "
    "low degree; see docs/CONVENTIONS.md. ``metrics_unmapped`` is raw PL on "
    "the flat circuit (audit)."
)


@dataclass
class AnalyticResult:
    """Analytic solver output and circuit metrics."""

    angles: list[float]
    angles_solver: list[float]
    solver: str
    degree: int
    solve_time_s: float
    metrics: dict[str, float]
    metrics_unmapped: dict[str, float]
    convention_note: str = CONVENTION_NOTE

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

    phases_solver, solver_used, solve_time = analytic_phases(cfg.degree, angle_solver=angle_solver)
    _, metrics_unmapped, _ = evaluate_phases_mapped(
        jnp.asarray(phases_solver),
        xs_train,
        xs_holdout,
        degree=cfg.degree,
        source="pennylane",
    )
    chao_phases, _, _, _ = analytic_phases_chao(cfg.degree)
    phases_flat = jnp.array(chao_to_flat(jnp.asarray(chao_phases)))
    metrics = evaluate_phases(phases_flat, xs_train, xs_holdout, degree=cfg.degree)

    return AnalyticResult(
        angles=[float(x) for x in phases_flat],
        angles_solver=[float(x) for x in phases_solver],
        solver=solver_used,
        degree=cfg.degree,
        solve_time_s=solve_time,
        metrics=metrics,
        metrics_unmapped=metrics_unmapped,
    )
