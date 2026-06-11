#!/usr/bin/env python3
"""CLI: analytic QSP angles — PennyLane and/or Chao (pyqsp) baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import train_config_from_protocol
from qsp_jax.baseline import evaluate_analytic
from qsp_jax.chao_baseline import evaluate_chao_analytic
from qsp_jax.io import utc_stamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("pennylane", "chao", "all"),
        default="all",
        help="Analytic solver backend (default: run both for comparison)",
    )
    parser.add_argument(
        "--angle-solver",
        choices=("iterative", "iterative-optax", "root-finding"),
        default="iterative",
        help="PennyLane poly_to_angles solver (pennylane backend only)",
    )
    parser.add_argument(
        "--chao-method",
        choices=("laurent", "sym_qsp"),
        default="laurent",
        help="pyqsp method (chao backend only)",
    )
    parser.add_argument(
        "--chao-signal-operator",
        choices=("Wx", "Wz"),
        default="Wx",
        help="pyqsp signal convention (chao backend only)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def _write_result(path: Path, payload: dict) -> None:
    write_json(path, payload)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    overrides = {"degree": args.degree} if args.degree is not None else None
    config = train_config_from_protocol(overrides, path=args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()

    if args.backend in ("pennylane", "all"):
        print(
            f"PennyLane analytic (degree={config.degree}, solver={args.angle_solver})"
        )
        pl_result = evaluate_analytic(config, angle_solver=args.angle_solver)
        pl_path = args.output_dir / f"analytic_pennylane_d{config.degree}_{args.angle_solver}_{stamp}.json"
        _write_result(pl_path, {
            "run_type": "analytic_baseline",
            "timestamp_utc": stamp,
            "target_id": "T1",
            "backend": "pennylane",
            "config": config.__dict__,
            **pl_result.to_dict(),
        })
        print(f"  Flat-circuit train MSE: {pl_result.metrics['train_mse']:.6e}")
        print(f"  Solve time: {pl_result.solve_time_s:.4f}s")
        print(f"  Note: {pl_result.convention_note}")

    if args.backend in ("chao", "all"):
        print(
            f"Chao / pyqsp analytic (degree={config.degree}, "
            f"method={args.chao_method}, signal={args.chao_signal_operator})"
        )
        chao_result = evaluate_chao_analytic(
            config,
            method=args.chao_method,
            signal_operator=args.chao_signal_operator,
        )
        chao_path = args.output_dir / (
            f"analytic_chao_d{config.degree}_{args.chao_method}_{args.chao_signal_operator}_{stamp}.json"
        )
        _write_result(chao_path, {
            "run_type": "analytic_baseline",
            "timestamp_utc": stamp,
            "target_id": "T1",
            "backend": "chao_pyqsp",
            "config": config.__dict__,
            **chao_result.to_dict(),
        })
        print(f"  pyqsp reconstruction max error: {chao_result.pyqsp_reconstruction_max_error:.6e}")
        print(f"  Flat-circuit train MSE: {chao_result.metrics['train_mse']:.6e}")
        print(f"  Solve time: {chao_result.solve_time_s:.4f}s")
        print(f"  Note: {chao_result.convention_note}")


if __name__ == "__main__":
    main()
