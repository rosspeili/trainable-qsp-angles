# Research Analysis and Experimental Roadmap

**Project:** Trainable QSP Angles  
**Author:** Ross Peili (Vladimiros Peilivanidis) — ARPA Hellenic Logical Systems  
**Repository:** [github.com/rosspeili/trainable-qsp-angles](https://github.com/rosspeili/trainable-qsp-angles)  
**Status:** Initial replication from PennyLane demo → research-grade artifact (in progress)

---

## 1. Executive Summary

The core idea — **learning QSP phase angles by gradient descent on a differentiable flat circuit** — is theoretically sound and practically useful. The current manuscript and demo correctly implement the method, diagnose the PennyLane `qml.QSVT` + JAX tracing incompatibility, and reproduce a degree-5 Chebyshev target with plausible accuracy.

However, **as a research contribution**, the work is currently an excellent **engineering note / tutorial**, not a novel QML result. Gradient-based optimization of differentiable quantum circuits has been possible since PennyLane (2018) and JAX-based QML stacks. The flat-circuit pattern is a necessary implementation fix, not a scientific discovery.

**The path forward** is not to oversell gradient learning itself, but to build **evidence-backed claims** around what is genuinely under-documented and under-measured:

1. When gradient training **beats, matches, or fails against** analytic QSP solvers (accuracy, runtime, stability).
2. How both methods **scale** with polynomial degree and condition number.
3. Whether learned phases **satisfy QSP reconstruction** beyond a sparse training grid.
4. Robustness across **seeds, hyperparameters, and target families**.

This document turns external review feedback into a concrete, reproducible plan stored alongside code and paper sources.

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
| No analytic baseline comparison | **Critical** | Cannot claim gradient training is viable without comparing to Chao et al. / `poly_to_angles` |
| Single degree (d=5), single target, single seed | **Critical** | No evidence of generality or statistical reliability |
| No scaling study (d = 5, 10, 20, 50, …) | **Critical** | QSP theory and practice hinge on degree; both solvers fail differently at scale |
| No ablation (lr, grid density, init range, optimizer) | **High** | Hyperparameters may dominate reported MSE |
| No QSP parity / reconstruction verification | **High** | Matching MSE on 64 points ≠ valid QSP phase sequence |
| No dense hold-out evaluation (e.g. 10⁴ Chebyshev points) | **High** | Grid overfitting is possible |
| No runtime / wall-clock benchmarks | **Medium** | Practitioners need cost trade-offs |
| No gradient norm / landscape diagnostics | **Medium** | Explains convergence and failure modes |
| Philosophical section unsupported by data | **Low** (for research) | Fine as outlook; weak for claims |
| Placeholder DOI, postdated metadata | **Medium** (presentation) | Fix before any public arXiv submission |
| Title implies general method; experiment is one case study | **Medium** | Reframe title or expand experiments |

### 3.2 Wording / positioning gaps

- Abstract and contributions (**C2**) imply a new paradigm; safer framing: **"empirical study + implementation pattern"**.
- "Grand unification" / "peculiar object" language is disproportionate for current evidence.
- Paper links still pointed at `qsp-pennylane-demo`; now canonical repo is `trainable-qsp-angles`.

### 3.3 Repository gaps (for self-contained science)

| Missing artifact | Purpose |
|------------------|---------|
| `experiments/` scripts with fixed configs | Batch reproducibility |
| `results/` versioned summary tables (CSV/JSON) | Paper numbers from code, not hand-typed |
| Analytic solver wrapper | Baseline phases from Chao / PennyLane |
| Multi-seed runner | Mean ± std over seeds |
| Scaling sweep notebook | Degree vs success rate / MSE / time |
| CI (pytest on push) | Regression guard |
| `CITATION.cff` | Machine-readable metadata |
| Pre-generated figures for README (optional) | Docs without running full train |

---

## 4. Reframed Research Questions

These are answerable with code in this repo and would materially strengthen the paper.

### RQ1 — Implementation (already partially answered)
> Does a flat PennyLane gate sequence preserve non-zero JAX gradients through QSP phase angles where `qml.QSVT` does not?

**Deliverable:** Minimal failing/passing test pair; documented in paper §3.3.

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

1. **PennyLane `qp.poly_to_angles`** (when applicable) — fast sanity check.
2. **Chao et al. QSP solver** — reference for machine-precision angles ([arXiv:2003.02831](https://arxiv.org/abs/2003.02831)); implement or vendor minimal Python port.
3. **Random search / CMA-ES** (optional) — non-gradient baseline for ill-conditioned cases.

### 5.3 Training protocol (fixed for comparability)

Document in `experiments/configs/default.yaml`:

```yaml
optimizer: adam
learning_rate: 0.05
steps: 500
grid_points: 64
grid_range: [-0.95, 0.95]
init_uniform: [-0.5, 0.5]
jax_x64: true
seeds: [0, 1, 2, ..., 29]
degrees: [5, 7, 10, 15, 20]
success_mse_threshold: 1.0e-3
holdout_points: 512
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
- [x] `.gitignore` excludes local `qsp-pennylane-demo/` reference
- [x] LICENSE + NOTICE attribution
- [x] README aligned with research focus
- [x] This plan document

### Phase 1 — Reproducibility hardening (1–2 weeks)

- [ ] `experiments/train.py` — CLI training with seed, degree, steps from YAML
- [ ] `experiments/baseline_analytic.py` — analytic phase computation + eval
- [ ] `results/` JSON schema for run metadata
- [ ] Extend tests: QSVT-breaks-JAX regression test; hold-out eval helper
- [ ] GitHub Actions: `pytest tests/`
- [ ] Update `demo.ipynb` to import shared training module (no duplicated logic)
- [ ] Fix manuscript author/email/repo links; soften overclaims in abstract

### Phase 2 — Core experiments (2–4 weeks)

- [ ] Multi-seed sweep T1 at d=5 (reproduce paper numbers from script)
- [ ] Baseline comparison table (analytic vs learned)
- [ ] Degree scaling d ∈ {5, 7, 10, 15, 20}
- [ ] Ablation: lr, grid size, init range
- [ ] Notebooks: `notebooks/01_baseline_comparison.ipynb`, `02_scaling_study.ipynb`
- [ ] Regenerate paper figures from `results/` (pgfplots data export)

### Phase 3 — Paper upgrade (2–3 weeks)

- [ ] Retitle (suggested): *"Gradient-Based Learning of QSP Phase Angles: A Reproducible Study with JAX-Traceable Circuits"*
- [ ] New Results section: baselines, scaling, statistics
- [ ] Move "Food for Thought" to short Outlook (clearly labeled speculative)
- [ ] Add Limitations section (explicit)
- [ ] Remove placeholder DOI until arXiv ID exists
- [ ] Target venue: SoftwareX, Quantum Sci. Technol. technical note, or arXiv + community report

### Phase 4 — Optional extensions (research frontier)

- [ ] End-to-end VQA stub: QSP phases inside larger loss (demonstrate RQ2 motivation)
- [ ] Curriculum over degree (warm-start φ from d−1)
- [ ] Second-order optimizers (L-BFGS on small d)
- [ ] QSVT block-encoding extension (multi-qubit flat pattern)

---

## 7. Manuscript Edit Checklist

| Section | Action |
|---------|--------|
| Title | Add "A Case Study" or "An Empirical Study" unless Phase 2 experiments complete |
| Abstract | Lead with JAX traceability fix; gradient learning as demonstrated method, not new paradigm |
| Contributions C1–C4 | Keep C1 (JAX fix); reword C2 as reproducible benchmark; C3 keep; C4 update repo URL |
| Author | Ross Peili; vpeilivanidis@gmail.com; ARPA Hellenic Logical Systems |
| §Results | Replace hand-plotted coordinates with script-generated data; add error bars |
| §Discussion | Add Limitations subsection from §3.1 gaps |
| §Food for Thought | Rename §Outlook; shorten claims |
| Metadata | Remove fake DOI; date = actual submission date |
| Code availability | Point to `trainable-qsp-angles` only |

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
4. **Open failure logs** — seeds that fail, with gradient norms and phase trajectories.
5. **Independent re-run instructions** — one command reproduces Table X.

That package supports either a solid **software + methods paper** or a stronger arXiv preprint — without pretending gradient descent on QSP is unknown physics.

---

## 10. Immediate Next Actions (Priority Order)

1. Run `pytest tests/ -v` after every change (CI next).
2. Implement `experiments/baseline_analytic.py` using PennyLane polynomial-to-angles.
3. Run 30-seed replication of current d=5 benchmark; store under `results/t1_degree5/`.
4. Draft baseline comparison table for manuscript §4.
5. Add JAX/QSVT regression test (short, deterministic).
6. Revise abstract + contributions in `manuscript.tex` (wording only, no new data yet).

---

## 11. Review Verdict (Recorded for Alignment)

| Criterion | Assessment |
|-----------|------------|
| Scientifically sound | Yes, with limited scope |
| Conceptually useful | Yes |
| QML research novelty | No (as of demo scope) |
| Practitioner value | Medium–high (JAX + QSP pattern) |
| Top-tier venue ready | No — needs Phase 2 experiments |
| Tutorial / software venue ready | Yes, after Phase 1 hardening |

**Strategy:** Embrace the honest scope, then **expand evidence** until claims and experiments match. The JAX traceability contribution plus rigorous baselines and scaling data is a defensible, self-contained story — more so than claiming a new quantum ML paradigm.

---

*Last updated: June 2026 — sync with repository commits as phases complete.*
