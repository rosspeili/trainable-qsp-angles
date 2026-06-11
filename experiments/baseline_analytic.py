#!/usr/bin/env python3
"""CLI: analytic QSP angles (PennyLane wrapper by default; see docs/FRAMEWORKS.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import train_config_from_protocol
from qsp_jax.baseline import evaluate_analytic
from qsp_jax.io import utc_stamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument(
        "--angle-solver",
        choices=("iterative", "iterative-optax", "root-finding"),
        default="iterative",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"degree": args.degree} if args.degree is not None else None
    config = train_config_from_protocol(overrides, path=args.config)
    print(f"Analytic angles (degree={config.degree}, solver={args.angle_solver})")
    result = evaluate_analytic(config, angle_solver=args.angle_solver)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"analytic_d{config.degree}_{args.angle_solver}_{utc_stamp()}.json"
    write_json(out_path, {
        "run_type": "analytic_baseline",
        "timestamp_utc": utc_stamp(),
        "target_id": "T1",
        "config": config.__dict__,
        **result.to_dict(),
    })
    print(f"Wrote {out_path}")
    print(f"Analytic train MSE (flat circuit): {result.metrics['train_mse']:.6e}")
    print(f"Solve time: {result.solve_time_s:.4f}s")
    print(f"Note: {result.convention_note}")


if __name__ == "__main__":
    main()
