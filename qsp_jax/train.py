"""Gradient-based training of QSP phase angles."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax

from qsp_jax.circuit import loss_fn

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class TrainConfig:
    """Default training protocol aligned with demo.ipynb and manuscript."""

    degree: int = 5
    steps: int = 500
    learning_rate: float = 0.05
    n_signal_points: int = 64
    grid_min: float = -0.95
    grid_max: float = 0.95
    init_min: float = -0.5
    init_max: float = 0.5
    holdout_points: int = 512
    holdout_min: float = -0.99
    holdout_max: float = 0.99
    seed: int = 0
    log_every: int = 100


@dataclass
class TrainResult:
    """Training outcome and metrics."""

    config: TrainConfig
    phases_init: list[float]
    phases_final: list[float]
    loss_init: float
    loss_final: float
    loss_history: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    train_time_s: float = 0.0
    gradient_norm_final: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


def signal_grid(n_points: int, x_min: float, x_max: float) -> jnp.ndarray:
    """Uniform signal grid strictly inside (-1, 1) when endpoints allow."""
    return jnp.linspace(x_min, x_max, n_points)


def init_phases(key: jax.Array, n_phases: int, low: float, high: float) -> jnp.ndarray:
    """Uniform random phase initialization."""
    return jax.random.uniform(key, shape=(n_phases,), minval=low, maxval=high)


def train(config: TrainConfig | None = None, verbose: bool = True) -> TrainResult:
    """
    Run Adam optimization on QSP phase angles.

    Returns
    -------
    TrainResult
        Final phases, loss trace, and train/hold-out metrics.
    """
    from qsp_jax.metrics import evaluate_phases

    cfg = config or TrainConfig()
    n_phases = cfg.degree + 1

    key = jax.random.PRNGKey(cfg.seed)
    xs_train = signal_grid(cfg.n_signal_points, cfg.grid_min, cfg.grid_max)
    xs_holdout = signal_grid(cfg.holdout_points, cfg.holdout_min, cfg.holdout_max)

    phases = init_phases(key, n_phases, cfg.init_min, cfg.init_max)
    phases_init = [float(x) for x in phases]
    loss_init = float(loss_fn(phases, xs_train, degree=cfg.degree))

    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(phases)

    @jax.jit
    def step(phases, opt_state):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, xs_train, degree=cfg.degree))(phases)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        phases_new = optax.apply_updates(phases, updates)
        return phases_new, opt_state_new, loss, grads

    loss_history: list[float] = []
    t0 = time.perf_counter()
    final_grads = jnp.zeros_like(phases)

    for step_idx in range(cfg.steps):
        phases, opt_state, loss, final_grads = step(phases, opt_state)
        loss_history.append(float(loss))
        if verbose and (step_idx % cfg.log_every == 0 or step_idx == cfg.steps - 1):
            print(f"Step {step_idx:4d} | loss = {float(loss):.6f}")

    train_time = time.perf_counter() - t0
    metrics = evaluate_phases(phases, xs_train, xs_holdout, degree=cfg.degree)

    return TrainResult(
        config=cfg,
        phases_init=phases_init,
        phases_final=[float(x) for x in phases],
        loss_init=loss_init,
        loss_final=float(loss_history[-1]),
        loss_history=loss_history,
        metrics=metrics,
        train_time_s=train_time,
        gradient_norm_final=float(jnp.linalg.norm(final_grads)),
    )
