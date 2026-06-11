#!/usr/bin/env python3
"""Aggregate experiment JSON into comparison tables and paper-ready curve data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.config import load_protocol_config
from qsp_jax.baseline import evaluate_analytic
from qsp_jax.io import utc_stamp, write_json
from qsp_jax.train import TrainConfig, train


def export_baseline_comparison(output_dir: Path, degree: int = 5, seed: int = 0) -> Path:
    """Run learned vs analytic snapshot and write comparison CSV/JSON."""
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(degree=degree, seed=seed)
    learned = train(cfg, verbose=False)
    analytic = evaluate_analytic(cfg)

    row = {
        "method": "gradient_adam",
        "degree": degree,
        "seed": seed,
        "train_mse": learned.metrics["train_mse"],
        "holdout_mse": learned.metrics["holdout_mse"],
        "train_max_error": learned.metrics["train_max_error"],
        "holdout_max_error": learned.metrics["holdout_max_error"],
        "wall_clock_s": learned.train_time_s,
        "gradient_norm_final": learned.gradient_norm_final,
    }
    analytic_row = {
        "method": "analytic_poly_to_angles",
        "backend": "pennylane_iterative",
        "degree": degree,
        "seed": seed,
        "train_mse": analytic.metrics["train_mse"],
        "holdout_mse": analytic.metrics["holdout_mse"],
        "train_max_error": analytic.metrics["train_max_error"],
        "holdout_max_error": analytic.metrics["holdout_max_error"],
        "wall_clock_s": analytic.solve_time_s,
        "gradient_norm_final": "",
        "convention_note": analytic.convention_note,
    }

    csv_path = paper_dir / f"baseline_comparison_d{degree}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()) + ["convention_note"])
        writer.writeheader()
        writer.writerow({**row, "convention_note": ""})
        writer.writerow(analytic_row)

    payload = {
        "run_type": "comparison_table",
        "timestamp_utc": utc_stamp(),
        "rows": [row, analytic_row],
        "learned_phases": learned.phases_final,
        "analytic_angles": analytic.angles,
    }
    json_path = paper_dir / f"baseline_comparison_d{degree}.json"
    write_json(json_path, payload)

    # Loss curve for manuscript (seed 0, degree 5)
    curve_path = paper_dir / f"loss_curve_d{degree}_seed{seed}.json"
    write_json(curve_path, {
        "run_type": "loss_curve",
        "timestamp_utc": utc_stamp(),
        "degree": degree,
        "seed": seed,
        "steps": list(range(len(learned.loss_history))),
        "loss": learned.loss_history,
    })

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {curve_path}")
    return csv_path


def summarize_multi_seed(summary_path: Path) -> None:
    """Print human-readable stats from a sweep summary JSON."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("baseline", "print-summary"),
        help="Summarization command",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "baseline":
        export_baseline_comparison(args.output_dir, degree=args.degree, seed=args.seed)
    elif args.command == "print-summary":
        path = args.summary or args.output_dir / "t1_degree5" / "summary.json"
        summarize_multi_seed(path)


if __name__ == "__main__":
    main()
