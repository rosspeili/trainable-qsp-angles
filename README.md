<div align="center">

## Gradient-Based Learning of Quantum Signal Processing Phase Angles

<img src="TRAINING_QSP_PHASE_ANGLES.png" alt="Training QSP Phase Angles via Gradient Descent" width="640">

**A reproducible benchmark and JAX implementation note**

[![DOI](https://img.shields.io/badge/DOI-20645402-ddd6fe?style=flat-square)](https://doi.org/10.5281/zenodo.20645402)
[![Version](https://img.shields.io/badge/Version-1.1.0-c4b5fd?style=flat-square)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-efcefa?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13%2B-bae6fd?style=flat-square)
![JAX](https://img.shields.io/badge/JAX-0.4.25%2B-a5f3fc?style=flat-square)
![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-e9d5ff?style=flat-square)

</div>

**Train QSP phase angles with Adam** instead of analytic solvers — on a flat, JAX-traceable circuit (QSP · QSVT · variational / differentiable quantum programming). We measure **when gradient learning beats mapped analytic baselines** (PennyLane `poly_to_angles`, Chao/pyqsp) on a shared flat-circuit MSE, and release code, sweeps, convention docs, and a reproducible Chebyshev benchmark.

**For:** variational / QML pipelines with implicit targets · solver comparisons · anyone embedding differentiable QSP blocks · reproducibility audits. **PennyLane** is the reference frontend only; the flat-circuit pattern ports to Qiskit, Cirq, TFQ, etc. ([`docs/FRAMEWORKS.md`](docs/FRAMEWORKS.md)).

**Outcome (headline):** learned train MSE $\approx10^{-4}$ vs mapped analytic $\approx10^{-3}$ on the same circuit at $d=5$ — with a documented phase-map protocol, not a claim of universal superiority.

---

## Paper

**Read or download:** [Zenodo — Version 1.1 (PDF + results ZIP)](https://doi.org/10.5281/zenodo.20645402)

LaTeX sources for audit and reuse: [`manuscript.tex`](manuscript.tex), [`manuscript_numbers.tex`](manuscript_numbers.tex), [`references.bib`](references.bib).

---

## Problem & scope

Classical QSP computes phase angles analytically from a target polynomial. This repo **benchmarks the alternative**: optimize angles with automatic differentiation and Adam, compare fairly to analytic solvers on one metric, and document pitfalls (e.g. high-level QSVT templates that break JAX grads through `qml.QSVT`).

**This is not a new QSP algorithm** — a reproducible measurement protocol, open implementation, and convention mapping.

**When to use which**

| Situation | Approach |
|-----------|----------|
| Known target, flat-circuit MSE (this benchmark) | **Gradient training** |
| Fast classical angles in native solver convention | **Chao / pyqsp** or PennyLane |
| Implicit loss (e.g. energy in a VQA) | **End-to-end gradient** |
| Novel polynomial, no closed form | **Gradient training** |

**Contributions**

- JAX-traceable flat QSP circuit + QSVT-template workaround
- Degree-5 Chebyshev `sin(x)` benchmark: multi-seed ($d=5,7,15$), scaling, ablation, off-grid probe
- Phase-convention protocol — [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) (`φ_flat = π/2 − φ_chao`)
- Experiment pipeline, tests, audit trail, manuscript

**Code layout:** start in [`demo.ipynb`](demo.ipynb) · library in [`qsp_jax/circuit.py`](qsp_jax/circuit.py) · CLI in [`experiments/`](experiments/)

---

## Quick start

```bash
git clone https://github.com/rosspeili/trainable-qsp-angles
cd trainable-qsp-angles
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pytest tests/ -v
```

### Try it (~5 min)

Interactive training and plots — no sweeps required:

```bash
py -3.13 -m jupyter notebook demo.ipynb
```

[View on nbviewer](https://nbviewer.org/github/rosspeili/trainable-qsp-angles/blob/main/demo.ipynb)

### Reproduce the paper

Download the **results ZIP** from [Zenodo](https://doi.org/10.5281/zenodo.20645402), or regenerate locally (defaults in `experiments/configs/default.json`):

```bash
py -3.13 -m experiments.train --seed 0 --steps 500
py -3.13 -m experiments.baseline_analytic
py -3.13 -m experiments.sweep multi-seed
py -3.13 -m experiments.sweep scaling
py -3.13 -m experiments.summarize baseline
```

Full CLI: [`docs/REPRODUCING.md`](docs/REPRODUCING.md). Analysis notebooks (after sweeps):

| Notebook | nbviewer |
|----------|----------|
| [`notebooks/01_baseline_comparison.ipynb`](notebooks/01_baseline_comparison.ipynb) | [open](https://nbviewer.org/github/rosspeili/trainable-qsp-angles/blob/main/notebooks/01_baseline_comparison.ipynb) |
| [`notebooks/02_scaling_study.ipynb`](notebooks/02_scaling_study.ipynb) | [open](https://nbviewer.org/github/rosspeili/trainable-qsp-angles/blob/main/notebooks/02_scaling_study.ipynb) |

Local runs write to `results/` (gitignored; see `results/schema.json`). Committed paper figure values: `manuscript_numbers.tex`.

---

## Results (headline)

Default target: degree-5 Chebyshev approximation of `sin(x)` on $[-1,1]$.

![Target polynomial vs. sin(x)](target_polynomial.png)

*Chebyshev target (solid) vs. `sin(x)` (dashed). Training minimizes MSE to the polynomial, not to `sin(x)` directly.*

Mapped analytic rows use the **same flat circuit** after `φ_flat = π/2 − φ_chao` ([`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)); PennyLane and pyqsp baselines are compared on equal footing.

| Metric (seed 0, default protocol) | Value |
|-----------------------------------|-------|
| Train MSE (gradient Adam) | $9.6 \times 10^{-5}$ |
| Train MSE (PennyLane / pyqsp, mapped) | $4.7 \times 10^{-3}$ |
| Multi-seed median ($n=30$, $d=5$) | $6.3 \times 10^{-5}$ |
| Multi-seed success ($d=7$, $d=15$) | 30/30 each, below $10^{-3}$ |
| Off-grid max error (learned vs mapped analytic) | $2.9 \times 10^{-2}$ vs $1.8 \times 10^{-1}$ |

![Training loss and learned polynomial](training_results.png)

*Demo snapshot (seed 0). Full figures and tables: [Zenodo PDF](https://doi.org/10.5281/zenodo.20645402).*

---

## Documentation

| Doc | What it covers |
|-----|----------------|
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Full experiment CLI and sweep commands |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | Phase maps (pyqsp / PennyLane → flat circuit) |
| [`docs/FRAMEWORKS.md`](docs/FRAMEWORKS.md) | PennyLane vs Qiskit / Cirq / TFQ / analytic solvers |
| [`docs/AUDIT_TRAIL.md`](docs/AUDIT_TRAIL.md) | Failures, fixes, rationale |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |
| [`CITATION.cff`](CITATION.cff) | Machine-readable cite metadata |

---

## Citation

> Peilivanidis, V. (2026). *Learning Quantum Signal Processing Phase Angles via Gradient Descent: A Reproducible Benchmark and JAX Implementation Note* (Version 1.1). https://doi.org/10.5281/zenodo.20645402

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Attribution required for code, data, figures, or manuscript reuse.

---

## Related work

Low & Chuang (2017) · Gilyén et al. (2019) · Martyn et al. (2021) · Chao et al. (2020) · Haah (2019) · Cerezo et al. (2021) · Bergholm et al. (2022) — full list in [`references.bib`](references.bib).

Evolved from the earlier [PennyLane community demo](https://github.com/rosspeili/qsp-pennylane-demo).
