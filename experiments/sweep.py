#!/usr/bin/env python3
"""Batch experiments: multi-seed replication, degree scaling, and ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from experiments.config import load_protocol_config
from qsp_jax.baseline import evaluate_analytic
from qsp_jax.io import utc_stamp, write_json
from qsp_jax.train import TrainConfig, train


def _run_dir(base: Path, name: str) -> Path:
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def multi_seed_t1(degree: int, seeds: list[int], output_dir: Path, steps: int) -> dict[str, Any]:
    """Train one seed at a time; return aggregate statistics."""
    run_dir = _run_dir(output_dir, f"t1_degree{degree}")
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n=== seed {seed} (degree {degree}) ===")
        cfg = TrainConfig(degree=degree, seed=seed, steps=steps)
        result = train(cfg, verbose=False)
        payload = {
            "run_type": "gradient_train",
            "timestamp_utc": utc_stamp(),
            "target_id": "T1",
            **result.to_dict(),
        }
        out_path = run_dir / f"train_seed{seed}.json"
        write_json(out_path, payload)
        runs.append(payload)
        print(f"  train_mse={result.metrics['train_mse']:.6e}")

    train_mses = [r["metrics"]["train_mse"] for r in runs]
    holdout_mses = [r["metrics"]["holdout_mse"] for r in runs]
    summary = {
        "run_type": "sweep_summary",
        "timestamp_utc": utc_stamp(),
        "target_id": "T1",
        "degree": degree,
        "n_seeds": len(seeds),
        "train_mse_mean": statistics.mean(train_mses),
        "train_mse_median": statistics.median(train_mses),
        "train_mse_stdev": statistics.pstdev(train_mses) if len(train_mses) > 1 else 0.0,
        "holdout_mse_mean": statistics.mean(holdout_mses),
        "holdout_mse_median": statistics.median(holdout_mses),
        "run_files": [str(run_dir / f"train_seed{s}.json") for s in seeds],
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def scaling_study(degrees: list[int], seed: int, output_dir: Path, steps: int) -> dict[str, Any]:
    """One seed per degree for a scaling cliff snapshot."""
    run_dir = _run_dir(output_dir, "scaling")
    rows: list[dict[str, Any]] = []

    for degree in degrees:
        print(f"\n=== scaling degree {degree} (seed {seed}) ===")
        cfg = TrainConfig(degree=degree, seed=seed, steps=steps)
        result = train(cfg, verbose=False)
        analytic = evaluate_analytic(cfg)
        row = {
            "degree": degree,
            "seed": seed,
            "train_mse": result.metrics["train_mse"],
            "holdout_mse": result.metrics["holdout_mse"],
            "train_time_s": result.train_time_s,
            "analytic_train_mse_flat_circuit": analytic.metrics["train_mse"],
            "analytic_solve_time_s": analytic.solve_time_s,
            "gradient_norm_final": result.gradient_norm_final,
            "success_train_mse_lt_1e-3": result.metrics["train_mse"] < 1e-3,
        }
        rows.append(row)
        write_json(run_dir / f"degree{degree}_seed{seed}.json", {
            "run_type": "gradient_train",
            "timestamp_utc": utc_stamp(),
            "target_id": "T1",
            **result.to_dict(),
            "analytic_baseline": analytic.to_dict(),
        })

    csv_path = run_dir / "scaling_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"run_type": "sweep_summary", "timestamp_utc": utc_stamp(), "rows": rows}
    write_json(run_dir / "summary.json", summary)
    return summary


def ablation_study(output_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Vary lr, grid density, and init range at degree 5."""
    run_dir = _run_dir(output_dir, "ablation")
    ablation = protocol["ablation"]
    rows: list[dict[str, Any]] = []

    for lr in ablation["learning_rates"]:
        for grid_points in ablation["grid_points"]:
            for init_min, init_max in ablation["init_ranges"]:
                cfg = TrainConfig(
                    degree=5,
                    seed=0,
                    steps=protocol["steps"],
                    learning_rate=lr,
                    n_signal_points=grid_points,
                    init_min=init_min,
                    init_max=init_max,
                )
                result = train(cfg, verbose=False)
                row = {
                    "learning_rate": lr,
                    "n_signal_points": grid_points,
                    "init_min": init_min,
                    "init_max": init_max,
                    "train_mse": result.metrics["train_mse"],
                    "holdout_mse": result.metrics["holdout_mse"],
                    "train_time_s": result.train_time_s,
                }
                rows.append(row)
                tag = f"lr{lr}_grid{grid_points}_init{init_min}_{init_max}"
                write_json(run_dir / f"{tag}.json", {
                    "run_type": "gradient_train",
                    "timestamp_utc": utc_stamp(),
                    "target_id": "T1",
                    **result.to_dict(),
                })
                print(f"  {tag}: train_mse={row['train_mse']:.6e}")

    csv_path = run_dir / "ablation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"run_type": "sweep_summary", "timestamp_utc": utc_stamp(), "rows": rows}
    write_json(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=("multi-seed", "scaling", "ablation", "all"),
        help="Which Phase-2 study to run",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Use 3 seeds / fewer degrees for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol_config(args.config)
    steps = protocol["steps"]
    seeds = protocol["seeds"][:3] if args.quick else protocol["seeds"]
    degrees = protocol["degrees"][:3] if args.quick else protocol["degrees"]

    if args.study in ("multi-seed", "all"):
        multi_seed_t1(protocol["degree"], seeds, args.output_dir, steps)
    if args.study in ("scaling", "all"):
        scaling_study(degrees, seed=0, output_dir=args.output_dir, steps=steps)
    if args.study in ("ablation", "all"):
        ablation_study(args.output_dir, protocol)


if __name__ == "__main__":
    main()
