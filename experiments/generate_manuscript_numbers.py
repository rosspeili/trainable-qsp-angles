#!/usr/bin/env python3
"""Write manuscript_numbers.tex with hardcoded pgfplots data from results/."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from qsp_jax.circuit import qsp_circuit
from qsp_jax.polynomial import target_poly

HIGHER_MULTI_SEED_DEGREES = (7, 15)

# LaTeX command names must be letters only (no digits after \MultiSeedD).
DEGREE_MACRO_PREFIX = {
    7: "MultiSeedDSeven",
    15: "MultiSeedDFifteen",
}


def latex_sci(value: float, sig: int = 1) -> str:
    """Format a positive float as LaTeX math (e.g. 6.3\\times10^{-5})."""
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    if abs(mantissa - 1.0) < 10 ** (-sig):
        return f"10^{{{exponent}}}"
    mantissa_str = f"{mantissa:.{sig}f}".rstrip("0").rstrip(".")
    return f"{mantissa_str}\\times10^{{{exponent}}}"


def loss_curve_block(results: Path) -> list[str]:
    curve = json.loads((results / "paper" / "loss_curve_d5_seed0.json").read_text(encoding="utf-8"))
    loss_coords = [
        f"({step},{loss:.6e})"
        for step, loss in zip(curve["steps"], curve["loss"])
        if step % 10 == 0 or step == curve["steps"][-1]
    ]
    return [
        "% Loss curve — results/paper/loss_curve_d5_seed0.json (experiments.summarize baseline)",
        f"\\providecommand{{\\LossCurveCoords}}{{{' '.join(loss_coords)}}}",
        "",
    ]


def multi_seed_coords_block(results: Path, degree: int, macro_name: str) -> list[str]:
    run_dir = results / f"t1_degree{degree}"
    paths = sorted(run_dir.glob("train_seed*.json"))
    if not paths:
        return []
    mses = [float(json.loads(path.read_text(encoding="utf-8"))["metrics"]["train_mse"]) for path in paths]
    seed_coords = " ".join(f"({i},{m:.6e})" for i, m in enumerate(mses))
    return [
        f"% Multi-seed d={degree} — results/t1_degree{degree}/train_seed*.json",
        f"\\providecommand{{\\{macro_name}}}{{{seed_coords}}}",
        "",
    ]


def multi_seed_stats_block(results: Path, degree: int, prefix: str) -> list[str]:
    summary_path = results / f"t1_degree{degree}" / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [
        f"% Multi-seed stats d={degree} — {summary_path.as_posix()}",
        f"\\providecommand{{\\{prefix}TrainMedian}}{{{latex_sci(summary['train_mse_median'])}}}",
        f"\\providecommand{{\\{prefix}TrainMedianNum}}{{{summary['train_mse_median']:.6e}}}",
        f"\\providecommand{{\\{prefix}TrainMean}}{{{latex_sci(summary['train_mse_mean'])}}}",
        f"\\providecommand{{\\{prefix}TrainStdev}}{{{latex_sci(summary['train_mse_stdev'])}}}",
        f"\\providecommand{{\\{prefix}HoldoutMedian}}{{{latex_sci(summary['holdout_mse_median'])}}}",
        f"\\providecommand{{\\{prefix}NSuccess}}{{{summary.get('n_success', summary['n_seeds'])}}}",
        f"\\providecommand{{\\{prefix}NSeeds}}{{{summary['n_seeds']}}}",
        "",
    ]
    mses = [
        float(json.loads(path.read_text(encoding="utf-8"))["metrics"]["train_mse"])
        for path in sorted((results / f"t1_degree{degree}").glob("train_seed*.json"))
    ]
    ymin = min(mses) * 0.7
    ymax = max(mses) * 1.3
    lines += [
        f"\\providecommand{{\\{prefix}YMin}}{{{ymin:.6e}}}",
        f"\\providecommand{{\\{prefix}YMax}}{{{ymax:.6e}}}",
        "",
    ]
    return lines


def scaling_block(results: Path) -> list[str]:
    with (results / "scaling" / "scaling_table.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    learn = " ".join(f"({row['degree']},{float(row['train_mse']):.6e})" for row in rows)
    anal = " ".join(
        f"({row['degree']},{float(row['analytic_train_mse_flat_circuit']):.6e})" for row in rows
    )
    return [
        "% Scaling — results/scaling/scaling_table.csv (experiments.sweep scaling)",
        f"\\providecommand{{\\ScalingLearnedCoords}}{{{learn}}}",
        f"\\providecommand{{\\ScalingAnalyticCoords}}{{{anal}}}",
        "",
    ]


def poly_fidelity_block(results: Path) -> list[str]:
    comp = json.loads((results / "paper" / "baseline_comparison_d5.json").read_text(encoding="utf-8"))
    phases = jnp.array(comp["learned_phases"])
    xs = np.linspace(-0.95, 0.95, 41)
    learn_pts: list[str] = []
    res_pts: list[str] = []
    for x in xs:
        xj = jnp.array(x)
        target = float(target_poly(xj, degree=5))
        learned = float(qsp_circuit(phases, xj))
        learn_pts.append(f"({x:.4f},{learned:.6f})")
        res_pts.append(f"({x:.4f},{learned - target:.6f})")
    return [
        "% Polynomial fidelity — results/paper/baseline_comparison_d5.json + qsp_jax.circuit",
        f"\\providecommand{{\\PolyLearnedCoords}}{{{' '.join(learn_pts)}}}",
        f"\\providecommand{{\\PolyResidualCoords}}{{{' '.join(res_pts)}}}",
        "",
    ]


def legacy_blocks(results: Path) -> list[str]:
    """Blocks that feed v1.0 figures; must remain stable when higher-degree sweeps are added."""
    lines = multi_seed_coords_block(results, degree=5, macro_name="MultiSeedCoords")
    return loss_curve_block(results) + lines + scaling_block(results) + poly_fidelity_block(results)


def offgrid_block(results: Path, degree: int = 5, seed: int = 0) -> list[str]:
    path = results / "paper" / f"offgrid_random_d{degree}_seed{seed}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_method = {row["method"]: row["offgrid_max_error"] for row in payload["rows"]}
    return [
        f"% Off-grid random max error — {path.as_posix()}",
        f"\\providecommand{{\\OffgridNPoints}}{{{payload['n_offgrid_points']}}}",
        f"\\providecommand{{\\OffgridLearnedMax}}{{{latex_sci(by_method['gradient_adam'])}}}",
        f"\\providecommand{{\\OffgridPennyLaneMax}}{{{latex_sci(by_method['analytic_poly_to_angles_mapped'])}}}",
        f"\\providecommand{{\\OffgridChaoMax}}{{{latex_sci(by_method['analytic_chao_laurent_mapped'])}}}",
        "",
    ]


def higher_degree_blocks(results: Path) -> list[str]:
    lines: list[str] = ["% --- v1.1 higher-degree multi-seed (additive; does not alter legacy macros) ---", ""]
    for degree in HIGHER_MULTI_SEED_DEGREES:
        prefix = DEGREE_MACRO_PREFIX[degree]
        coords = multi_seed_coords_block(results, degree=degree, macro_name=f"{prefix}Coords")
        stats = multi_seed_stats_block(results, degree=degree, prefix=prefix)
        if coords or stats:
            lines.extend(coords)
            lines.extend(stats)
    lines.extend(offgrid_block(results))
    return lines


def generate_lines(results: Path) -> list[str]:
    header = [
        "% Auto-generated by experiments/generate_manuscript_numbers.py",
        "% Re-generate after re-running experiments; values are \\input into manuscript.tex.",
        "",
    ]
    return header + legacy_blocks(results) + higher_degree_blocks(results)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    out_path = root / "manuscript_numbers.tex"
    out_path.write_text("\n".join(generate_lines(results)), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
