# Changelog

All notable changes to this repository are documented here. For **failures, fixes, rationale, and comparison audits**, see [`docs/AUDIT_TRAIL.md`](docs/AUDIT_TRAIL.md) and the append-only log [`docs/audit/LOG.jsonl`](docs/audit/LOG.jsonl).

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries reference audit IDs where applicable.

---

## [Unreleased]

### Added
- **Convention mapping** (AUD-018): `qsp_jax/convention.py`, `docs/CONVENTIONS.md`, `tests/test_convention.py`.
- **`train_mse_unmapped`** column in baseline comparison CSV for audit trail.
- Phase 2 experiment results generated locally (AUD-020): 30-seed, scaling, ablation.

### Changed
- **v1.1 item 2 (AUD-2026-06-12-024):** Multi-seed sweeps at $d=7$ and $d=15$ (`sweep multi-seed --degrees 7,15`); additive `manuscript_numbers.tex` macros + paper §4.2.1; golden test locks legacy figure data.
- **v1.1 item 1 (AUD-2026-06-12-023):** Manuscript reframed as reproducible benchmark + JAX implementation note (not a new QSP algorithm); subtitle, abstract, contributions C1–C4, Discussion scope, Outlook; README aligned. `arpa_logo.png` is paper-only (local, gitignored).
- PennyLane mapped metrics use shared Chebyshev→pyqsp→flat bridge at all degrees (AUD-019).
- `RESEARCH_PLAN.md`: Phase 2 marked complete.
- **Manuscript Phase 3:** author name (Vladimiros Peilivanidis), shorter abstract, self-contained figures via `manuscript_numbers.tex`, DOI/ORCID badges; README and docs synced.
- Short hyperparameter ablation subsection in paper (§4.3) + README; figure fixes (scaling, legend, baseline plot).

### Fixed
- Convention mismatch (AUD-003, AUD-013): `phi_flat = pi/2 - phi_chao`.
- Stale paper CSV (AUD-016 → AUD-021).
- Direct PL→pyqsp inversion at high degree (AUD-019).

### Known residual
- Mapped analytic MSE (~10⁻²–10⁻³ at d≥7) exceeds gradient (~10⁻⁵–10⁻⁴) due to pyqsp capitalization / residual; documented in CONVENTIONS.md.

---

## [0.2.0] — 2026-06 Phase 2 pipeline

### Added
- Phase 2 experiments: `sweep.py` (multi-seed, scaling, ablation), `summarize.py`, `configs/default.json`.
- Notebooks: `01_baseline_comparison.ipynb`, `02_scaling_study.ipynb`.
- `docs/FRAMEWORKS.md` — SDK-agnostic framing (AUD-006).
- Tests: `test_jax_traceability.py`, `test_train.py`, `test_reproducibility.py`.
- CI: `.github/workflows/ci.yml` (Python 3.13, fast tests only).

### Fixed
- Odd degrees only in scaling protocol (AUD-005).
- PennyLane default angle solver → `iterative` (AUD-002).

---

## [0.1.0] — 2026-06 Initial research repo

### Added
- Bootstrap from PennyLane demo into standalone repo (AUD-001).
- `qsp_jax/` package: flat circuit, train loop, PennyLane analytic baseline.
- `manuscript.tex`, `RESEARCH_PLAN.md`, `demo.ipynb`, tests, LICENSE/NOTICE.
- Flat QSP circuit replacing QSVT template for JAX traceability (AUD-004).

---

## Audit cross-reference

| Topic | Audit IDs |
|-------|-----------|
| Convention mismatch | AUD-003, AUD-013 |
| pyqsp / Chao integration | AUD-009–012, AUD-014–015 |
| Append-only baselines | AUD-012 |
| CI / tests | AUD-008, AUD-015 |
| Open: refresh paper CSV | AUD-016 |
