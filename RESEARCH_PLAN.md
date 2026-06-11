# Research Analysis and Experimental Roadmap

**Project:** Trainable QSP Angles  
**Author:** Vladimiros Peilivanidis — ARPA Hellenic Logical Systems  
**Email:** r1@arpacorp.net · **ORCID:** [0009-0003-0121-465X](https://orcid.org/0009-0003-0121-465X) · **DOI:** [10.5281/zenodo.20645403](https://doi.org/10.5281/zenodo.20645403)  
**Repository:** [github.com/rosspeili/trainable-qsp-angles](https://github.com/rosspeili/trainable-qsp-angles)  
**Status:** Phase 3 complete (manuscript + self-contained figures); ready for Zenodo PDF upload.

---

## 1. Executive Summary

The core idea — **learning QSP phase angles by gradient descent on a differentiable flat circuit** — is theoretically sound and practically useful. The reference code uses **PennyLane + JAX** (historical demo origin, `poly_to_angles` baseline, JAX QNode interface); the **method itself is SDK-agnostic** — see `docs/FRAMEWORKS.md`.

However, **as a research contribution**, the work is currently an excellent **engineering note / tutorial**, not a novel QML result. Gradient-based optimization of differentiable quantum circuits has been possible since PennyLane (2018) and JAX-based QML stacks. The flat-circuit pattern is a necessary implementation fix, not a scientific discovery.

**The path forward** is not to oversell gradient learning itself, but to build **evidence-backed claims** around what is genuinely under-documented and under-measured:

1. When gradient training **beats, matches, or fails against** analytic QSP solvers (accuracy, runtime, stability).
2. How both methods **scale** with polynomial degree and condition number.
3. Whether learned phases **satisfy QSP reconstruction** beyond a sparse training grid.
4. Robustness across **seeds, hyperparameters, and target families**.

This document turns external review feedback into a concrete, reproducible plan stored alongside code and paper sources.

**Audit trail:** Every significant failure, fix, comparison, and design decision is recorded in [`docs/AUDIT_TRAIL.md`](docs/AUDIT_TRAIL.md) and append-only [`docs/audit/LOG.jsonl`](docs/audit/LOG.jsonl). See [`CHANGELOG.md`](CHANGELOG.md) for release summaries.

---

## 2. Current Assets (What We Have)

| Asset | Location | Strength |
|-------|----------|----------|
| Flat JAX-traceable QSP circuit | `qsp_jax/circuit.py` | Correct, minimal, tested |
| Target polynomial + MSE loss | `qsp_jax/circuit.py` | Standard benchmark |
| Unit tests (differentiability, loss decrease) | `tests/test_circuit.py` | Good smoke tests |
| Interactive demo | `demo.ipynb` | Reproducible training loop |
| Manuscript draft | `manuscript.tex` | Clear writing, good structure |
| Bibliography | `references.bib` | Appropriate references |
| Decision guide (Table 3) | `manuscript.tex` | Useful practitioner framing |

**Reproducibility baseline:** Git + requirements + tests + notebook. Strong foundation.

---

## 3. Gap Analysis (What Is Missing)

### 3.1 Scientific rigor gaps (from review + independent read)

| Gap | Severity | Why it matters |
|-----|----------|----------------|
| No analytic baseline comparison | **Critical** | ~~Cannot claim gradient training is viable without comparing to Chao et al. / `poly_to_angles`~~ **Addressed:** PennyLane + Chao (`qsp_jax/chao_baseline.py`) in comparison table |
| Single degree (d=5), single target, single seed | **Partial** | Multi-seed + scaling + ablation done; still one target family (T1) |
| No scaling study (d = 5, 10, 20, 50, …) | **Partial** | d ∈ {5,7,9,15,21}, seed 0; scaling figure in paper |
| No ablation (lr, grid density, init range, optimizer) | **Addressed** | 18-config sweep in `results/ablation/`; summarized in paper §4.3 |
| No QSP parity / reconstruction verification | **Partial** | Dense-grid poly fidelity figure + hold-out MSE; no phase-unwrapping audit |
| No dense hold-out evaluation (e.g. 10⁴ Chebyshev points) | **Partial** | 512-point hold-out in protocol; hold-out ≫ train MSE documented |
| No runtime / wall-clock benchmarks | **Addressed** | Table 3 wall-clock column |
| No gradient norm / landscape diagnostics | **Open** | Phase 4 / future work |
| Philosophical section unsupported by data | **Addressed** | Condensed Outlook |
| Placeholder DOI, postdated metadata | **Addressed** | Zenodo DOI 10.5281/zenodo.20645403 on title page |
| Title implies general method; experiment is one case study | **Addressed** | Subtitle: reproducible empirical study; Limitations name T1 scope |

### 3.2 Wording / positioning gaps

- Abstract and contributions (**C2**) imply a new paradigm; safer framing: **"empirical study + implementation pattern"**.
- "Grand unification" / "peculiar object" language is disproportionate for current evidence.
- Paper links still pointed at `qsp-pennylane-demo`; now canonical repo is `trainable-qsp-angles`.

### 3.3 Repository gaps (for self-contained science)

| Missing artifact | Purpose |
|------------------|---------|
| `experiments/` scripts with fixed configs | Batch reproducibility |
| `results/` versioned summary tables (CSV/JSON) | Paper numbers from code, not hand-typed |
| Analytic solver wrapper | Baseline phases via PennyLane `poly_to_angles` (convenience); Chao et al. standalone solver optional |
| Multi-seed runner | Mean ± std over seeds |
| Scaling sweep notebook | Degree vs success rate / MSE / time |
| CI (pytest on push) | Regression guard |
| `CITATION.cff` | Machine-readable metadata |
| Pre-generated figures for README (optional) | Docs without running full train |

---

## 4. Reframed Research Questions

These are answerable with code in this repo and would materially strengthen the paper.

### RQ1 — Implementation (already partially answered)
> Does a flat gate sequence preserve non-zero JAX gradients through QSP phase angles where high-level QSVT templates (e.g. PennyLane `qml.QSVT`) do not?

**Deliverable:** Minimal failing/passing test pair; documented in paper §3.3. Optional: replicate in Qiskit/Cirq parameterized circuits.

### RQ2 — Equivalence at low degree
> For admissible targets at degree d ≤ 10, can Adam reach analytic-solver accuracy (MSE and max error)?

**Deliverable:** Table: analytic vs learned phases, multiple targets.

### RQ3 — Scaling and failure modes
> At increasing degree d, where does gradient training succeed vs stagnate vs diverge, compared to analytic solvers?

**Deliverable:** Phase diagram (degree × init) with success criterion.

### RQ4 — Statistical reliability
> What is the distribution of final MSE over ≥30 random seeds per (d, target)?

**Deliverable:** Box plots; report median and IQR, not single runs.

### RQ5 — QSP validity beyond training grid
> Do learned phases produce the expected parity and boundedness on dense Chebyshev nodes in [-1, 1]?

**Deliverable:** Hold-out max error; optional phase-unwrapping consistency checks.

### RQ6 — Cost trade-off
> For fixed accuracy ε, what is wall-clock time: analytic solve + eval vs gradient training?

**Deliverable:** Log-log plot vs degree.

---

## 5. Experimental Protocol (Proposed)

### 5.1 Targets (expand beyond sin Chebyshev)

| ID | Target | Degree | Parity | Notes |
|----|--------|--------|--------|-------|
| T1 | Chebyshev sin (current) | 5 | odd | Baseline replication |
| T2 | Exact monomial x^d (d odd) | 3, 5, 7 | odd | Simplest admissible |
| T3 | Step / sign approx (odd) | 9, 15 | odd | Harder landscape |
| T4 | Low-pass filter (even) | 4, 6 | even | Requires even-degree QSP variant |
| T5 | Random bounded Chebyshev series | d | mixed | Sample coefficients s.t. \|P\|≤1 |

### 5.2 Baselines

1. **PennyLane `poly_to_angles`** (current convenience wrapper) — fast sanity check in-repo.
2. **Chao et al. QSP solver** — reference for machine-precision angles independent of any QML SDK ([arXiv:2003.02831](https://arxiv.org/abs/2003.02831)).
3. **Optional cross-SDK circuit eval** — same phases evaluated in Qiskit/Cirq for convention checks (`docs/FRAMEWORKS.md`).
4. **Random search / CMA-ES** (optional) — non-gradient baseline for ill-conditioned cases.

### 5.3 Training protocol (fixed for comparability)

Document in `experiments/configs/default.json`:

```json
{
  "optimizer": "adam",
  "learning_rate": 0.05,
  "steps": 500,
  "n_signal_points": 64,
  "grid_min": -0.95,
  "grid_max": 0.95,
  "init_min": -0.5,
  "init_max": 0.5,
  "seeds": [0, 1, "... ", 29],
  "degrees": [5, 7, 9, 15, 21],
  "success_mse_threshold": 0.001,
  "holdout_points": 512
}
```

### 5.4 Metrics (report all)

- Train MSE (grid), hold-out max error, hold-out MSE
- \|φ_learned − φ_analytic\| (when analytic exists)
- Wall-clock: solve time, train time, eval time
- Gradient norm at convergence
- Success rate over seeds
- Phase angle magnitude max (numerical stability proxy)

### 5.5 Hypotheses (falsifiable)

- **H1:** At d ≤ 10, gradient training reaches analytic accuracy within 10× MSE for ≥90% of seeds (T1, T2).
- **H2:** Analytic solvers are faster at low d; gradient training becomes competitive when target is implicit (no closed form) — demonstrate on synthetic implicit loss.
- **H3:** Failure rate increases superlinearly with d for fixed Adam budget (scaling cliff).
- **H4:** Hold-out error ≤ 2× train MSE when QSP admissibility holds (sanity check).

Each hypothesis gets a pre-registered config file before running sweeps.

---

## 6. Repository Roadmap (Phased)

### Phase 0 — Foundation ✅ (this commit)

- [x] Standalone repo structure from demo
- [x] LICENSE + NOTICE attribution
- [x] README aligned with research focus
- [x] This plan document

### Phase 1 — Reproducibility hardening ✅

- [x] `experiments/train.py` — CLI training with seed, degree, steps from JSON protocol
- [x] `experiments/baseline_analytic.py` — analytic phase computation + eval (PennyLane + Chao backends)
- [x] `qsp_jax/chao_baseline.py` — standalone Chao / pyqsp Laurent completion (alongside PennyLane)
- [x] `results/schema.json` for run metadata
- [x] Extend tests: JAX traceability regression; hold-out / reproducibility tests
- [x] GitHub Actions: `pytest tests/` (fast suite; `-m "not slow"`)
- [x] Update `demo.ipynb` to import shared training module
- [x] Fix manuscript author/email/repo links; soften overclaims in abstract

### Phase 2 — Core experiments ✅

- [x] Multi-seed sweep T1 (`experiments/sweep.py multi-seed`) — 30 seeds, `results/t1_degree5/summary.json`
- [x] Baseline comparison table (`experiments/summarize.py baseline`) — 3 rows + mapped/unmapped columns
- [x] Degree scaling d ∈ {5, 7, 9, 15, 21} (`experiments/sweep.py scaling`)
- [x] Ablation: lr, grid size, init range (`experiments/sweep.py ablation`)
- [x] Notebooks: `notebooks/01_baseline_comparison.ipynb`, `02_scaling_study.ipynb`
- [x] Paper curve export: `results/paper/loss_curve_d5_seed0.json`
- [x] Convention mapping: `qsp_jax/convention.py`, `docs/CONVENTIONS.md`
- [x] Regenerate manuscript figures from exported JSON/CSV (`experiments/generate_manuscript_numbers.py`, `manuscript_numbers.tex`)
- [x] Results section: baselines, 30-seed stats, scaling (manuscript draft updated)
- [x] Limitations + short Outlook (replaces Food for Thought)
- [x] Zenodo DOI: 10.5281/zenodo.20645403
- [x] Final author proofread / title polish (V. Peilivanidis, DOI/ORCID badges, self-contained figures)

### Phase 3 — Paper upgrade ✅

- [x] Retitle: *Learning QSP Phase Angles via Gradient Descent* + subtitle *A Reproducible Empirical Study with JAX-Traceable Circuits*
- [x] New Results section: baselines, scaling, statistics, polynomial fidelity
- [x] Move "Food for Thought" to short Outlook (clearly labeled speculative)
- [x] Add Limitations section (explicit)
- [x] Zenodo DOI on title page (self-publish; no arXiv)
- [x] Self-contained figures via `manuscript_numbers.tex` (no compile-time `.dat` dependency)
- [x] Ablation results in paper (§4.3 + Table 4; full data in `results/ablation/`)
- [x] Target venue: Zenodo + GitHub (self-publish; no arXiv)

### Phase 4 — Optional extensions (research frontier)

- [ ] End-to-end VQA stub: QSP phases inside larger loss (demonstrate implicit-target motivation)
- [ ] Curriculum over degree (warm-start φ from d−1)
- [ ] Second-order optimizers (L-BFGS on small d)
- [ ] QSVT block-encoding extension (multi-qubit flat pattern)
- [ ] **Cross-framework validation** (Qiskit / Cirq) — see `docs/FRAMEWORKS.md`; Chao/pyqsp baseline implemented in `qsp_jax/chao_baseline.py`
- [ ] PennyLane-Lightning device for faster Phase 2 sweeps (same API, different backend)

---

## 7. Manuscript Edit Checklist

| Section | Action |
|---------|--------|
| Title | Done — empirical study subtitle; matches README |
| Abstract | Done — shorter; leads with learning vs analytic; JAX traceability |
| Contributions C1–C4 | Done |
| Author | Done — Vladimiros Peilivanidis; r1@arpacorp.net; ORCID; DOI badge |
| §Results | Done — hardcoded figures + ablation table + source paths in captions |
| §Discussion | Done — Limitations + Outlook |
| Metadata | Done — Zenodo DOI 10.5281/zenodo.20645403 |
| Code availability | Done — trainable-qsp-angles + regenerate script |

---

## 8. Test Plan (Expanded)

Current tests (`tests/test_circuit.py`) cover:

- Scalar output, boundedness, gradient shape/nonzero
- Target poly range, odd symmetry, p(0)=0
- Loss scalar, positive for random init, decreases in 10 steps

**Add:**

| Test | File | Asserts |
|------|------|---------|
| `test_qsvt_breaks_jax_grad` | `tests/test_jax_traceability.py` | QSVT path gives zero grad (documented failure) |
| `test_flat_circuit_matches_analytic_d5` | `tests/test_baselines.py` | Learned or analytic phases achieve MSE < 1e-3 (slow; mark `@pytest.mark.slow`) |
| `test_holdout_error_bounded` | `tests/test_baselines.py` | Hold-out max err < threshold after full train |
| `test_reproducible_seed` | `tests/test_reproducibility.py` | Same seed → same final MSE |

---

## 9. What Would Make This "Undeniable"

Not hype — **evidence density**:

1. **Every number in the paper traceable to a committed results file** (hash in manuscript footnote).
2. **Analytic baseline on equal footing** — if gradient wins nowhere at low d, say so honestly and pivot claim to implicit-target / no-closed-form scenarios.
3. **Scaling cliff chart** — even negative results are publishable if well-measured.
4. **Open failure logs** — seeds that fail, with gradient norms and phase trajectories. Tracked in `docs/audit/LOG.jsonl` and `docs/AUDIT_TRAIL.md`.
5. **Independent re-run instructions** — one command reproduces Table X.

That package supports either a solid **software + methods paper** or a stronger arXiv preprint — without pretending gradient descent on QSP is unknown physics.

---

## 10. Next Actions (post Phase 3)

1. Final PDF compile and proofread; upload to Zenodo (DOI 10.5281/zenodo.20645403).
2. Attach `results/` tarball to Zenodo release (gitignored locally; reproducible via `experiments.sweep all`).
3. Tag GitHub release (e.g. `v1.0.0`) linked to Zenodo deposit.
4. Optional: refresh README PNGs from current runs; regenerate `training_results.png`.
5. Phase 4 items only if extending claims (implicit target, cross-SDK, curriculum).

---

## 11. Review Verdict (Recorded for Alignment)

| Criterion | Assessment |
|-----------|------------|
| Scientifically sound | Yes, with explicit Limitations |
| Conceptually useful | Yes |
| QML research novelty | No (implementation + empirical study) |
| Practitioner value | Medium–high (JAX + QSP pattern) |
| Top-tier venue ready | No — scope is empirical / software note |
| Zenodo + GitHub release ready | **Yes** — Phase 2/3 complete |

**Strategy:** Embrace the honest scope, then **expand evidence** until claims and experiments match. The JAX traceability contribution plus rigorous baselines and scaling data is a defensible, self-contained story — more so than claiming a new quantum ML paradigm.

---

*Last updated: June 2026 — sync with repository commits as phases complete.*
