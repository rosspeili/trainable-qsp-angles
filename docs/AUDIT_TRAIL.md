# Audit Trail — Trainable QSP Angles

**Purpose:** Record what worked, what failed, why, how it was fixed, and the rationale behind comparisons, tests, and design choices. This is the human-readable companion to the machine log at [`audit/LOG.jsonl`](audit/LOG.jsonl).

**Maintainers:** Append to `LOG.jsonl` after significant events (`py -3.13 -m experiments.audit append ...`). Update this document when a failure category or open issue changes status.

**Related:** [`CHANGELOG.md`](../CHANGELOG.md) (version summaries) · [`results/`](../results/) (raw run data, gitignored)

---

## Quick status (2026-06-12)

| Area | Status | Notes |
|------|--------|-------|
| Flat JAX QSP circuit | ✅ Works | Sub-mill MSE at d=5 after 500 Adam steps |
| PennyLane `poly_to_angles` baseline | ⚠️ Partial | Solves target; **high MSE on flat circuit** (convention) |
| Chao / pyqsp Laurent baseline | ⚠️ Partial | Native recon ~1.3×10⁻⁴; **high flat-circuit MSE** (convention) |
| Convention alignment | ✅ Mapped | `phi_flat = pi/2 - phi_chao`; PL mapped via shared bridge (docs/CONVENTIONS.md) |
| Phase 2 sweeps (30-seed) | ✅ Done | `results/t1_degree5/summary.json` (30 seeds) |
| Paper CSV (3-row baseline) | ✅ Done | `results/paper/baseline_comparison_d5.csv` |
| Manuscript v1.1 reframing | ✅ Done | Benchmark + implementation note scope (AUD-2026-06-12-023) |
| Multi-seed d=7, d=15 | ✅ Done | `results/t1_degree7/`, `results/t1_degree15/` (30/30 success each) |
| Chao sym_qsp audit (item 3) | ✅ Done | Native ≲10⁻¹⁵; mapped MSE unchanged vs Laurent |
| Off-grid random eval (item 4) | ✅ Done | Learned max 2.9×10⁻² vs analytic ~1.8×10⁻¹ |
| Barren plateaus discussion (item 5) | ✅ Done | §5.3 + McClean/Cerezo cites |

---

## 1. Failures register (labeled)

Each entry: **ID** · **Status** · **Label**

### AUD-003 · fixed · `convention-mismatch` · PennyLane analytic vs flat circuit

| | |
|---|---|
| **What failed** | Raw PennyLane phases: train_mse ~ **0.59** on flat circuit |
| **Root cause** | Native PL QSP convention ≠ flat `RZ(-2φ)` circuit |
| **Fix** | Mapped metrics via shared `chao_to_flat(pyqsp(...))`; `train_mse_unmapped` retained |
| **Rationale** | Same Chebyshev target; pyqsp bridge is reproducible at all degrees |
| **Evidence** | `results/paper/baseline_comparison_d5.csv`, `docs/CONVENTIONS.md` |

### AUD-009 · fixed · `pyqsp` · CompletionError on monomial input

| | |
|---|---|
| **What failed** | `QuantumSignalProcessingPhases(monomial_coeffs_numpy(5))` → `CompletionError` |
| **Why** | pyqsp expects **Chebyshev** coefficients or Laurent form, not PennyLane monomial list |
| **Fix** | Added `chebyshev_sin_cheb_coeffs()`; pass Chebyshev array; use `method='laurent'` |
| **Rationale** | Align with pyqsp `angle_sequence.py` and `test_response.py` |
| **Evidence** | `qsp_jax/polynomial.py`, `qsp_jax/chao_baseline.py` |

### AUD-010 · fixed · `sym_qsp` · Missing `chebyshev_basis=True`

| | |
|---|---|
| **What failed** | `method='sym_qsp'` without flag → `ValueError` |
| **Fix** | Set `chebyshev_basis=True` for sym_qsp branch |
| **Rationale** | sym_qsp is optional; default Chao path remains **laurent** |

### AUD-011 · fixed · `pyqsp` · Wx/z completion model wrong for T1

| | |
|---|---|
| **What failed** | `signal_operator='Wx', measurement='z'` → completion failed |
| **Fix** | Default **Wx/x** (or Wz/z); avoid Wx/z for sin target |
| **Rationale** | Wrong P-type monomial model for this target family |

### AUD-013 · fixed · `convention-mismatch` · Chao native vs flat circuit

| | |
|---|---|
| **What failed (unmapped)** | Raw pyqsp phases: flat train_mse ~ **0.95** |
| **Fix** | `phi_flat = pi/2 - phi_chao` in `qsp_jax/convention.py` → mapped ~ **4.7×10⁻³** at d=5 |
| **Native verification** | `pyqsp_reconstruction_max_error` ~ **1.3×10⁻⁴** |
| **Note** | Mapped analytic MSE > gradient (~10⁻⁴) due to pyqsp capitalization residual |
| **Evidence** | `qsp_jax/convention.py`, `results/paper/baseline_comparison_d5.csv` |

### AUD-014 · fixed · `sym_qsp` verification test

| | |
|---|---|
| **What failed** | `test_analytic_phases_chao_sym_qsp_shape`: recon_err ≈ 1.39 |
| **Why** | Verified real part of response; sym_qsp uses **imag** of matrix element |
| **Fix** | `verify_pyqsp_reconstruction(..., method='sym_qsp', sym_qsp=True)` on `np.imag` |
| **Evidence** | `tests/test_chao_baseline.py` |

### AUD-002 · fixed · PennyLane `root-finding` solver

| | |
|---|---|
| **What failed** | `angle_solver='root-finding'` raises on T1 |
| **Fix** | Default **`iterative`**; root-finding documented as unreliable here |

### AUD-004 · fixed · QSVT + JAX tracing

| | |
|---|---|
| **What failed** | `qml.QSVT` path → zero / broken gradients |
| **Fix** | Flat inline gates in `qsp_jax/circuit.py`; regression test added |
| **Rationale** | Core contribution requires phases in autodiff graph |

### AUD-005 · fixed · Even degrees in scaling

| | |
|---|---|
| **What failed** | Even d invalid for odd sin Chebyshev target |
| **Fix** | Protocol odd degrees only: `[5, 7, 9, 15, 21]` |

---

## 2. Comparison audit (degree 5, seed 0, T1)

**Mapped** flat-circuit MSE (primary). See `train_mse_unmapped` in CSV for raw audit.

| Method | train_mse (mapped) | train_mse_unmapped | holdout_mse | wall_clock_s |
|--------|-------------------|--------------------|-------------|--------------|
| Gradient Adam | 9.62×10⁻⁵ | — | 1.66×10⁻³ | ~1.45 |
| PennyLane iterative | 4.73×10⁻³ | 5.89×10⁻¹ | 9.0×10⁻³ | ~1.94 |
| Chao pyqsp Laurent | 4.73×10⁻³ | 9.51×10⁻¹ | 9.0×10⁻³ | ~0.007 |

Chao native recon: **1.32×10⁻⁴**. Gradient still best on flat circuit at d=5.

### 30-seed summary (T1, d=5)

| Stat | train_mse |
|------|-----------|
| mean | 1.63×10⁻⁴ |
| median | 6.30×10⁻⁵ |
| stdev | 1.76×10⁻⁴ |

Source: `results/t1_degree5/summary.json`

### Scaling (seed 0, mapped analytic ~ chao)

| d | learned train_mse | analytic mapped | success (<10⁻³) |
|---|-------------------|-----------------|-----------------|
| 5 | 9.6×10⁻⁵ | 4.7×10⁻³ | yes |
| 7 | 1.8×10⁻⁵ | 8.6×10⁻² | yes |
| 9 | 2.4×10⁻⁵ | 2.9×10⁻² | yes |
| 15 | 6.3×10⁻⁴ | 8.6×10⁻² | yes |
| 21 | 1.7×10⁻⁵ | 2.9×10⁻² | yes |

Source: `results/scaling/scaling_table.csv`

### AUD-023 · fixed · `paper` · Manuscript v1.1 benchmark reframing

| | |
|---|---|
| **What changed** | Subtitle, abstract, §1.3 scope, C1–C4, Discussion opener, Outlook reframed as benchmark + implementation note (not new QSP algorithm) |
| **Why** | Friend review: claims should match what the repo actually delivers (measurement + reproducibility, not a new solver) |
| **Logo** | `arpa_logo.png` on title page (top-left); **local only**, gitignored, not on GitHub |
| **Evidence** | `manuscript.tex`, `README.md`, `CHANGELOG.md` |

### AUD-024 · fixed · `experiment` · Multi-seed at d=7 and d=15

| | |
|---|---|
| **What** | 30-seed gradient training at $d=7$ and $d=15$ (default protocol); all seeds below $10^{-3}$ train MSE |
| **How** | `py -3.13 -m experiments.sweep multi-seed --degrees 7,15` → new dirs only; legacy `t1_degree5/`, `scaling/`, `paper/` untouched |
| **Paper** | §4.2.1 table + two scatter figures; `tests/test_manuscript_numbers.py` guards legacy macros |
| **Evidence** | `results/t1_degree7/summary.json`, `results/t1_degree15/summary.json`, `manuscript_numbers.tex` |

### AUD-025 · fixed · `chao` · sym_qsp vs Laurent audit (v1.1 item 3)

| | |
|---|---|
| **Question** | Can sym_qsp / machine precision close the mapped analytic MSE gap vs gradient? |
| **Result** | sym_qsp native recon ≲ 10⁻¹⁵ at d=5; mapped flat train MSE ≈ 4.73×10⁻³ (same as Laurent ≈ 4.73×10⁻³) |
| **Conclusion** | Gap is convention/capitalization residual, not solver error; paper reports Laurent in main table |
| **Evidence** | `results/paper/chao_method_comparison_d5.json`, `experiments/compare_chao_methods.py` |

### AUD-026 · fixed · `experiment` · Off-grid random max error (v1.1 item 4)

| | |
|---|---|
| **What** | 1024 uniform random $x\in[-0.95,0.95]$ excluding train-grid hits; max $\|P-\langle X\rangle\|$ at $d=5$, seed~0 |
| **Result** | Learned max $2.9\times10^{-2}$; mapped analytic $\approx1.8\times10^{-1}$ |
| **How** | `py -3.13 -m experiments.offgrid_eval`; phases from `baseline_comparison_d5.json` (no retrain) |
| **Evidence** | `results/paper/offgrid_random_d5_seed0.json`, `manuscript_numbers.tex` `\Offgrid*` macros |

### AUD-027 · fixed · `paper` · Barren plateaus discussion (v1.1 item 5)

| | |
|---|---|
| **What** | New Discussion §5.3 on barren plateaus vs our low-parameter QSP benchmark |
| **Cites** | McClean et al. 2018 (`mcclean2018barren`); Cerezo et al. 2021 (already in bib) |
| **Evidence** | Multi-seed success to $d=15$; nonzero final $\|\nabla\mathcal{L}\|$ in scaling CSV |

---

## 3. Test audit

| Suite | Count | CI | Asserts |
|-------|-------|-----|---------|
| `test_circuit.py` | 10 fast | ✅ | Shape, bounds, grad nonzero, loss decreases |
| `test_jax_traceability.py` | fast | ✅ | QSVT grad failure documented |
| `test_train.py` | fast | ✅ | Smoke train, JSON export |
| `test_reproducibility.py` | 2 fast + 2 slow | fast only in CI | Same seed → same MSE; sub-mill after 500 steps |
| `test_chao_baseline.py` | 5 fast | ✅ | Chebyshev coeffs, laurent/sym_qsp, native recon < 1e-3 |

**Command:** `py -3.13 -m pytest tests/ -v -m "not slow"`

---

## 4. Decision log (rationale)

| ID | Decision | Rationale |
|----|----------|-----------|
| AUD-006 | Framework-agnostic framing | Research is trainable QSP, not PennyLane-specific |
| AUD-012 | Append baselines, never replace | User policy; enables multi-column paper tables |
| AUD-012 | pyqsp for Chao, not vendored C++ | Maintained package, matches literature reference |
| AUD-007 | `py -3.13 -m experiments.*` | Reliable imports on Windows; avoids script path issues |
| AUD-008 | `@pytest.mark.slow` for 500-step train | CI speed vs convergence coverage |

---

## 5. Open issues (tracked)

| ID | Issue | Next action |
|----|-------|-------------|
| — | Zenodo v1.1 release | Recompile PDF; upload results zip + paper after user review |

---

## 6. How to append (template)

```bash
py -3.13 -m experiments.audit append \
  --category failure \
  --status open \
  --title "Short title" \
  --what "Observed behavior" \
  --why "Hypothesis or root cause" \
  --action "Fix or workaround" \
  --rationale "Why this approach" \
  --evidence "path/to/result.json" \
  --labels "tag1,tag2" \
  --related-ids "AUD-2026-06-11-003"
```

For failed experiment runs, always attach:

1. Command used  
2. Output path under `results/`  
3. Label (`convention-mismatch`, `solver`, `ci`, etc.)  
4. Link to related AUD IDs if continuing a thread  

---

## 7. Chronological index (machine log)

Full append-only log: [`audit/LOG.jsonl`](audit/LOG.jsonl) (22 entries as of 2026-06-12).

```bash
py -3.13 -m experiments.audit list --last 20
py -3.13 -m experiments.audit list --label convention-mismatch
```

---

*Last updated: 2026-06-12 — sync with `docs/audit/LOG.jsonl` when adding entries.*
