#!/usr/bin/env python3
"""Export pgfplots-ready data files from results/ for manuscript.tex."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from qsp_jax.circuit import qsp_circuit
from qsp_jax.polynomial import target_poly


def _write_loss_curve(src: Path, dst: Path, subsample: int = 5) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    steps = data["steps"]
    losses = data["loss"]
    with dst.open("w", encoding="utf-8") as handle:
        handle.write("step loss\n")
        for step, loss in zip(steps, losses):
            if step % subsample == 0 or step == steps[-1]:
                handle.write(f"{step} {loss:.12e}\n")


def _write_poly_fidelity(phases: list[float], degree: int, dst: Path, n_points: int = 161) -> None:
    xs = np.linspace(-0.95, 0.95, n_points)
    phases_j = jnp.array(phases)
    with dst.open("w", encoding="utf-8") as handle:
        handle.write("x target learned residual\n")
        for x in xs:
            xj = jnp.array(x)
            target = float(target_poly(xj, degree=degree))
            learned = float(qsp_circuit(phases_j, xj))
            handle.write(f"{x:.6f} {target:.10f} {learned:.10f} {learned - target:.10e}\n")


def _write_baseline(rows: list[dict], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as handle:
        handle.write("method train_mse train_mse_unmapped wall_clock_s\n")
        for row in rows:
            if row["method"] == "gradient_adam":
                label = "Gradient Adam"
                unmapped = ""
            elif "pennylane" in row["backend"]:
                label = "PennyLane analytic"
                unmapped = row.get("train_mse_unmapped", "")
            else:
                label = "Chao / pyqsp"
                unmapped = row.get("train_mse_unmapped", "")
            handle.write(
                f"{label} {row['train_mse']:.12e} {unmapped} {row['wall_clock_s']:.6f}\n"
            )


def _write_scaling(csv_path: Path, dst: Path) -> None:
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    with dst.open("w", encoding="utf-8") as handle:
        handle.write("degree learned analytic_mapped analytic_unmapped chao_mapped\n")
        for row in rows:
            handle.write(
                f"{row['degree']} {row['train_mse']} "
                f"{row['analytic_train_mse_flat_circuit']} {row['analytic_train_mse_unmapped']} "
                f"{row['chao_train_mse_flat_circuit']}\n"
            )


def _write_multiseed(run_dir: Path, dst: Path) -> None:
    mses: list[float] = []
    for path in sorted(run_dir.glob("train_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mses.append(float(payload["metrics"]["train_mse"]))
    with dst.open("w", encoding="utf-8") as handle:
        handle.write("index train_mse\n")
        for i, mse in enumerate(mses):
            handle.write(f"{i} {mse:.12e}\n")


def export_all(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper = results_dir / "paper"
    comparison = json.loads((paper / "baseline_comparison_d5.json").read_text(encoding="utf-8"))

    _write_loss_curve(paper / "loss_curve_d5_seed0.json", output_dir / "loss_curve_d5_seed0.dat")
    _write_poly_fidelity(comparison["learned_phases"], degree=5, dst=output_dir / "poly_fidelity_d5_seed0.dat")
    _write_baseline(comparison["rows"], output_dir / "baseline_comparison_d5.dat")
    _write_scaling(results_dir / "scaling" / "scaling_table.csv", output_dir / "scaling_table.dat")
    _write_multiseed(results_dir / "t1_degree5", output_dir / "multiseed_d5_train_mse.dat")

    meta = {
        "source_results": str(results_dir),
        "learned_phases_seed0": comparison["learned_phases"],
        "loss_init": json.loads((paper / "loss_curve_d5_seed0.json").read_text())["loss"][0],
        "loss_final": json.loads((paper / "loss_curve_d5_seed0.json").read_text())["loss"][-1],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote figure data to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript_data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_all(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
