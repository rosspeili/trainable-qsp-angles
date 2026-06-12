"""Guard legacy manuscript figure data against accidental regeneration drift."""

from __future__ import annotations

from pathlib import Path

from experiments.generate_manuscript_numbers import generate_lines, legacy_blocks

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT_NUMBERS = ROOT / "manuscript_numbers.tex"


def _macro_line(lines: list[str], command: str) -> str:
    needle = f"\\providecommand{{\\{command}}}"
    for line in lines:
        if line.startswith(needle):
            return line
    raise AssertionError(f"Missing macro {command}")


def test_legacy_blocks_match_committed_manuscript_numbers():
    """Legacy figure macros must not change unless paper/scaling/baseline sources are re-run."""
    if not (RESULTS / "paper" / "loss_curve_d5_seed0.json").exists():
        return
    committed = MANUSCRIPT_NUMBERS.read_text(encoding="utf-8").splitlines()
    regenerated = legacy_blocks(RESULTS)

    for command in (
        "LossCurveCoords",
        "MultiSeedCoords",
        "ScalingLearnedCoords",
        "ScalingAnalyticCoords",
        "PolyLearnedCoords",
        "PolyResidualCoords",
    ):
        committed_line = next(l for l in committed if l.startswith(f"\\providecommand{{\\{command}}}"))
        assert _macro_line(regenerated, command) == committed_line, command


def test_higher_degree_macros_present_after_full_generate():
    """When d=7/d=15 sweeps exist, generator emits additive macros only."""
    if not (RESULTS / "t1_degree7" / "summary.json").exists():
        return
    lines = generate_lines(RESULTS)
    text = "\n".join(lines)
    assert "\\MultiSeedDSevenCoords" in text
    assert "\\MultiSeedDFifteenCoords" in text
    assert "\\MultiSeedDSevenTrainMedian" in text
    assert "\\OffgridLearnedMax" in text
    test_legacy_blocks_match_committed_manuscript_numbers()
