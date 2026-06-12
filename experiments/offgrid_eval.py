#!/usr/bin/env python3
"""Off-grid random max-error evaluation (v1.1 item 4).

Samples random signal points outside the fixed training grid and reports
max pointwise |circuit - target| for learned vs mapped analytic phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.config import load_protocol_config, train_config_from_protocol
from qsp_jax.baseline import evaluate_analytic
from qsp_jax.chao_baseline import evaluate_chao_analytic
from qsp_jax.io import utc_stamp, write_json
from qsp_jax.metrics import max_pointwise_error
from qsp_jax.train import TrainConfig, signal_grid, train


def random_offgrid_points(
    n_points: int,
    *,
    grid_min: float,
    grid_max: float,
    train_xs: np.ndarray,
    rng: np.random.Generator,
    train_tol: float = 1e-6,
) -> jnp.ndarray:
    """Uniform random samples in [grid_min, grid_max], excluding train-grid hits."""
    xs: list[float] = []
    while len(xs) < n_points:
        batch = rng.uniform(grid_min, grid_max, size=max(n_points - len(xs), 64))
        for x in batch:
            if np.min(np.abs(train_xs - x)) > train_tol:
                xs.append(float(x))
            if len(xs) >= n_points:
                break
    return jnp.array(xs[:n_points])


def load_or_train_phases(
    cfg: TrainConfig,
    baseline_json: Path | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, str]:
    """Return learned, PL mapped, Chao mapped phases; note data source."""
    if baseline_json and baseline_json.exists():
        payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        learned = jnp.array(payload["learned_phases"])
        pl = jnp.array(payload["pennylane_analytic_angles"])
        chao = jnp.array(payload["chao_analytic_angles"])
        return learned, pl, chao, str(baseline_json)

    learned_result = train(cfg, verbose=False)
    pl_result = evaluate_analytic(cfg)
    chao_result = evaluate_chao_analytic(cfg)
    return (
        learned_result.phases_final,
        jnp.array(pl_result.angles),
        jnp.array(chao_result.angles),
        "fresh_train_and_baselines",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Training / baseline seed")
    parser.add_argument("--offgrid-seed", type=int, default=42, help="RNG seed for random test points")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("results/paper/baseline_comparison_d5.json"),
        help="Reuse phases from existing baseline snapshot (recommended)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    args = parser.parse_args()

    protocol = load_protocol_config(args.config)
    cfg = train_config_from_protocol(
        {"degree": args.degree, "seed": args.seed},
        path=args.config,
    )
    train_xs = np.array(signal_grid(cfg.n_signal_points, cfg.grid_min, cfg.grid_max))
    rng = np.random.default_rng(args.offgrid_seed)
    xs_off = random_offgrid_points(
        args.n_points,
        grid_min=cfg.grid_min,
        grid_max=cfg.grid_max,
        train_xs=train_xs,
        rng=rng,
    )

    learned, pl_mapped, chao_mapped, source = load_or_train_phases(
        cfg,
        args.baseline_json if args.degree == 5 and args.seed == 0 else None,
    )

    rows = []
    for label, phases in (
        ("gradient_adam", learned),
        ("analytic_poly_to_angles_mapped", pl_mapped),
        ("analytic_chao_laurent_mapped", chao_mapped),
    ):
        max_err = max_pointwise_error(phases, xs_off, degree=cfg.degree)
        rows.append({"method": label, "offgrid_max_error": max_err})
        print(f"{label:34s} offgrid_max={max_err:.6f}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_type": "offgrid_random_eval",
        "timestamp_utc": utc_stamp(),
        "target_id": "T1",
        "degree": cfg.degree,
        "train_seed": args.seed,
        "offgrid_seed": args.offgrid_seed,
        "n_offgrid_points": args.n_points,
        "domain": [cfg.grid_min, cfg.grid_max],
        "train_grid_points": int(cfg.n_signal_points),
        "phase_source": source,
        "rows": rows,
    }
    out_path = out_dir / f"offgrid_random_d{cfg.degree}_seed{args.seed}.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
