#!/usr/bin/env python3
"""Compare pyqsp Laurent vs sym_qsp on native and flat-circuit metrics (v1.1 item 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.config import train_config_from_protocol
from qsp_jax.chao_baseline import evaluate_chao_analytic
from qsp_jax.io import utc_stamp, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    args = parser.parse_args()

    cfg = train_config_from_protocol({"degree": args.degree}, path=args.config)
    rows: list[dict] = []
    for method in ("laurent", "sym_qsp"):
        result = evaluate_chao_analytic(cfg, method=method)
        rows.append(
            {
                "method": method,
                "degree": cfg.degree,
                "train_mse_mapped": result.metrics["train_mse"],
                "train_mse_unmapped": result.metrics_unmapped["train_mse"],
                "holdout_mse_mapped": result.metrics["holdout_mse"],
                "pyqsp_reconstruction_max_error": result.pyqsp_reconstruction_max_error,
                "solve_time_s": result.solve_time_s,
            }
        )
        print(
            f"{method:8s} native={result.pyqsp_reconstruction_max_error:.3e} "
            f"mapped={result.metrics['train_mse']:.3e}"
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_type": "chao_method_comparison",
        "timestamp_utc": utc_stamp(),
        "target_id": "T1",
        "degree": cfg.degree,
        "rows": rows,
        "note": (
            "sym_qsp reaches machine-precision native recon but mapped flat-circuit MSE "
            "matches Laurent after phi_flat = pi/2 - phi_chao (docs/CONVENTIONS.md)."
        ),
    }
    out_path = out_dir / f"chao_method_comparison_d{cfg.degree}.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
