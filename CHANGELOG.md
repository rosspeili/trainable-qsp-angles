# Changelog

All notable changes to this repository are documented here. For **failures, fixes, rationale, and comparison audits**, see [`docs/AUDIT_TRAIL.md`](docs/AUDIT_TRAIL.md) and the append-only log [`docs/audit/LOG.jsonl`](docs/audit/LOG.jsonl).

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries reference audit IDs where applicable.

---

## [1.1.0] — 2026-06-12

Zenodo version **1.1** ([Concept DOI 10.5281/zenodo.20645402](https://doi.org/10.5281/zenodo.20645402); version-specific: [10.5281/zenodo.20666387](https://doi.org/10.5281/zenodo.20666387)). Manuscript PDF: `Peilivanidis_2026_trainable-qsp-angles_manuscript_v1.1.pdf`.

### Added
- Multi-seed sweeps at $d=7$ and $d=15$ (`sweep multi-seed --degrees 7,15`); paper §4.3; additive `manuscript_numbers.tex` macros; golden test locks legacy figure data (AUD-2026-06-12-024).
- Chao `sym_qsp` audit — machine-precision native recon, same mapped flat MSE as Laurent (`experiments/compare_chao_methods.py`; AUD-2026-06-12-025).
- Off-grid random max-error eval (`experiments/offgrid_eval.py`); paper §4.6 + table (AUD-2026-06-12-026).
- Discussion §5.3 barren plateaus / trainability; `mcclean2018barren` in `references.bib` (AUD-2026-06-12-027).
- Convention mapping: `qsp_jax/convention.py`, `docs/CONVENTIONS.md`, `tests/test_convention.py` (AUD-018).
- `train_mse_unmapped` column in baseline comparison CSV for audit trail.

### Changed
- Manuscript reframed as **reproducible benchmark + JAX implementation note** (not a new QSP algorithm); subtitle, abstract, C1–C4, Discussion scope, Outlook (AUD-2026-06-12-023).
- LaTeX macro names: `\MultiSeedDSeven*` / `\MultiSeedDFifteen*` (digits illegal in command names).
- PennyLane mapped metrics use shared Chebyshev→pyqsp→flat bridge at all degrees (AUD-019).
- Author name (Vladimiros Peilivanidis), self-contained figures via `manuscript_numbers.tex`, DOI/ORCID badges; README and docs synced.
- Hyperparameter ablation subsection (§4.4); figure fixes (scaling, legend, baseline plot).
- `CITATION.cff`, `NOTICE`, and bibliography entry aligned with v1.1 subtitle.
- Primary Zenodo cite updated to **Concept DOI** [10.5281/zenodo.20645402](https://doi.org/10.5281/zenodo.20645402) (replaces v1.0-only `…20645403`).

### Fixed
- Convention mismatch (AUD-003, AUD-013): `phi_flat = pi/2 - phi_chao`.
- Stale paper CSV (AUD-016 → AUD-021).
- Direct PL→pyqsp inversion at high degree (AUD-019).

### Known residual
- Mapped analytic MSE (~10⁻²–10⁻³ at d≥7) exceeds gradient (~10⁻⁵–10⁻⁴) due to pyqsp capitalization / residual; documented in `docs/CONVENTIONS.md`.

---

## [1.0.0] — 2026-06-11

Initial Zenodo deposit and Phase 2 benchmark pipeline.

### Added
- Phase 2 experiments: `sweep.py` (multi-seed, scaling, ablation), `summarize.py`, `configs/default.json`.
- Notebooks: `01_baseline_comparison.ipynb`, `02_scaling_study.ipynb`.
- `docs/FRAMEWORKS.md` — SDK-agnostic framing (AUD-006).
- Tests: `test_jax_traceability.py`, `test_train.py`, `test_reproducibility.py`.
- CI: `.github/workflows/ci.yml` (Python 3.13, fast tests only).
- Manuscript with data-driven figures, baseline table, 30-seed/scaling plots, Limitations, Outlook (AUD-2026-06-11-022).

### Fixed
- Odd degrees only in scaling protocol (AUD-005).
- PennyLane default angle solver → `iterative` (AUD-002).

---

## [0.1.0] — 2026-06 Initial research repo

### Added
- Bootstrap from PennyLane demo into standalone repo (AUD-001).
- `qsp_jax/` package: flat circuit, train loop, PennyLane analytic baseline.
- `manuscript.tex`, `demo.ipynb`, tests, LICENSE/NOTICE.
- Flat QSP circuit replacing QSVT template for JAX traceability (AUD-004).

---

## Audit cross-reference

| Topic | Audit IDs |
|-------|-----------|
| Convention mismatch | AUD-003, AUD-013 |
| pyqsp / Chao integration | AUD-009–012, AUD-014–015 |
| Append-only baselines | AUD-012 |
| CI / tests | AUD-008, AUD-015 |
| v1.1 review items | AUD-2026-06-12-023 … 027 |
