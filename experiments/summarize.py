#!/usr/bin/env python3
"""Aggregate experiment JSON into comparison tables and paper-ready curve data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.config import load_protocol_config
from qsp_jax.baseline import evaluate_analytic
from qsp_jax.chao_baseline import evaluate_chao_analytic
from qsp_jax.io import utc_stamp, write_json
from qsp_jax.train import TrainConfig, train

COMPARISON_FIELDS = [
    "method",
    "backend",
    "degree",
    "seed",
    "train_mse",
    "train_mse_unmapped",
    "holdout_mse",
    "train_max_error",
    "holdout_max_error",
    "wall_clock_s",
    "gradient_norm_final",
    "pyqsp_reconstruction_max_error",
    "convention_note",
]


def _comparison_row(
    *,
    method: str,
    backend: str,
    degree: int,
    seed: int,
    metrics: dict[str, float],
    wall_clock_s: float,
    metrics_unmapped: dict[str, float] | None = None,
    gradient_norm_final: str | float = "",
    pyqsp_reconstruction_max_error: str | float = "",
    convention_note: str = "",
) -> dict[str, Any]:
    unmapped = metrics_unmapped or {}
    return {
        "method": method,
        "backend": backend,
        "degree": degree,
        "seed": seed,
        "train_mse": metrics["train_mse"],
        "train_mse_unmapped": unmapped.get("train_mse", ""),
        "holdout_mse": metrics["holdout_mse"],
        "train_max_error": metrics["train_max_error"],
        "holdout_max_error": metrics["holdout_max_error"],
        "wall_clock_s": wall_clock_s,
        "gradient_norm_final": gradient_norm_final,
        "pyqsp_reconstruction_max_error": pyqsp_reconstruction_max_error,
        "convention_note": convention_note,
    }


def export_baseline_comparison(output_dir: Path, degree: int = 5, seed: int = 0) -> Path:
    """Run learned vs analytic snapshots and write comparison CSV/JSON."""
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(degree=degree, seed=seed)
    learned = train(cfg, verbose=False)
    pennylane = evaluate_analytic(cfg)
    chao = evaluate_chao_analytic(cfg)

    rows = [
        _comparison_row(
            method="gradient_adam",
            backend="jax_flat_circuit",
            degree=degree,
            seed=seed,
            metrics=learned.metrics,
            wall_clock_s=learned.train_time_s,
            gradient_norm_final=learned.gradient_norm_final,
        ),
        _comparison_row(
            method="analytic_poly_to_angles",
            backend="pennylane_iterative",
            degree=degree,
            seed=seed,
            metrics=pennylane.metrics,
            metrics_unmapped=pennylane.metrics_unmapped,
            wall_clock_s=pennylane.solve_time_s,
            convention_note=pennylane.convention_note,
        ),
        _comparison_row(
            method="analytic_chao_laurent",
            backend=f"pyqsp_laurent_{chao.signal_operator}",
            degree=degree,
            seed=seed,
            metrics=chao.metrics,
            metrics_unmapped=chao.metrics_unmapped,
            wall_clock_s=chao.solve_time_s,
            pyqsp_reconstruction_max_error=chao.pyqsp_reconstruction_max_error,
            convention_note=chao.convention_note,
        ),
    ]

    csv_path = paper_dir / f"baseline_comparison_d{degree}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "run_type": "comparison_table",
        "timestamp_utc": utc_stamp(),
        "rows": rows,
        "learned_phases": learned.phases_final,
        "pennylane_analytic_angles": pennylane.angles,
        "pennylane_analytic_angles_solver": pennylane.angles_solver,
        "chao_analytic_angles": chao.angles,
        "chao_analytic_angles_solver": chao.angles_solver,
        "analytic_angles": pennylane.angles,
    }
    json_path = paper_dir / f"baseline_comparison_d{degree}.json"
    write_json(json_path, payload)

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
