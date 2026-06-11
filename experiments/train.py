#!/usr/bin/env python3
"""CLI: gradient-based QSP phase training with JSON results export."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import train_config_from_protocol
from qsp_jax.io import utc_stamp, write_json
from qsp_jax.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Protocol JSON (default: experiments/configs/default.json)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        k: v
        for k, v in {
            "seed": args.seed,
            "steps": args.steps,
            "degree": args.degree,
            "learning_rate": args.learning_rate,
        }.items()
        if v is not None
    }
    config = train_config_from_protocol(overrides, path=args.config)

    print(f"Training degree-{config.degree} target (seed={config.seed}, steps={config.steps})")
    result = train(config)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_d{config.degree}_seed{config.seed}_{utc_stamp()}.json"
    write_json(out_path, {
        "run_type": "gradient_train",
        "timestamp_utc": utc_stamp(),
        "target_id": "T1",
        **result.to_dict(),
    })
    print(f"Wrote {out_path}")
    print(f"Final train MSE: {result.metrics['train_mse']:.6e}")


if __name__ == "__main__":
    main()
