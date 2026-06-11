#!/usr/bin/env python3
"""CLI: analytic QSP angles via PennyLane and circuit evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qsp_jax.baseline import evaluate_analytic
from qsp_jax.train import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angle-solver",
        choices=("iterative", "iterative-optax", "root-finding"),
        default="iterative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    print(f"Analytic angles (solver={args.angle_solver})")
    result = evaluate_analytic(config, angle_solver=args.angle_solver)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"analytic_{args.angle_solver}_{stamp}.json"
    payload = {
        "run_type": "analytic_baseline",
        "timestamp_utc": stamp,
        "config": config.__dict__,
        **result.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Analytic train MSE (flat circuit): {result.metrics['train_mse']:.6e}")
    print(f"Solve time: {result.solve_time_s:.4f}s")


if __name__ == "__main__":
    main()
