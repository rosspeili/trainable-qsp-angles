#!/usr/bin/env python3
"""CLI: gradient-based QSP phase training with JSON results export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qsp_jax.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for run JSON (default: results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        seed=args.seed,
        steps=args.steps,
        degree=args.degree,
        learning_rate=args.learning_rate,
    )

    print(f"Training degree-{config.degree} target (seed={config.seed}, steps={config.steps})")
    result = train(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"train_d{config.degree}_seed{config.seed}_{stamp}.json"
    payload = {
        "run_type": "gradient_train",
        "timestamp_utc": stamp,
        **result.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Final train MSE: {result.metrics['train_mse']:.6e}")


if __name__ == "__main__":
    main()
