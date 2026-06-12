<div align="center">

## Gradient-Based Learning of Quantum Signal Processing Phase Angles

<img src="TRAINING_QSP_PHASE_ANGLES.png" alt="Training QSP Phase Angles via Gradient Descent" width="640">

**A reproducible benchmark and JAX implementation note**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20645402-ddd6fe?style=flat-square)](https://doi.org/10.5281/zenodo.20645402)
[![Version](https://img.shields.io/badge/Version-1.1.0-c4b5fd?style=flat-square)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-efcefa?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13%2B-bae6fd?style=flat-square)
![JAX](https://img.shields.io/badge/JAX-0.4.25%2B-a5f3fc?style=flat-square)
![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-e9d5ff?style=flat-square)

</div>

Research code for learning Quantum Signal Processing (QSP) phase angles by gradient descent on a flat, JAX-traceable circuit. PennyLane is the reference frontend in `qsp_jax/`; the method is framework-agnostic.

---

## Paper

**Read or download:** [Zenodo — Version 1.1 (PDF + results)](https://doi.org/10.5281/zenodo.20645402)

LaTeX sources for audit and reuse: [`manuscript.tex`](manuscript.tex), [`manuscript_numbers.tex`](manuscript_numbers.tex), [`references.bib`](references.bib).

---

## What this is

Classical QSP computes phase angles analytically from a target polynomial. This project **benchmarks the alternative**: train angles with automatic differentiation and Adam on a flat circuit, and compare against mapped analytic baselines (PennyLane, Chao/pyqsp) on a shared metric.

It is **not** a new QSP algorithm — a reproducible measurement protocol, convention mapping, and open implementation.

**Contributions**

- JAX-traceable flat QSP circuit (QSVT-template pitfall documented)
- Degree-5 Chebyshev `sin(x)` benchmark with multi-seed stats ($d=5$, $7$, $15$), scaling, ablation, off-grid probe
- Phase-convention protocol for fair baseline comparison — [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)
- Full experiment pipeline, tests, audit trail, and manuscript

---

## Quick start

```bash
git clone https://github.com/rosspeili/trainable-qsp-angles
cd trainable-qsp-angles
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pytest tests/ -v
py -3.13 -m jupyter notebook demo.ipynb
```

**Reproduce experiments** (defaults in `experiments/configs/default.json`):

```bash
py -3.13 -m experiments.train --seed 0 --steps 500
py -3.13 -m experiments.baseline_analytic
py -3.13 -m experiments.sweep multi-seed
py -3.13 -m experiments.sweep scaling
py -3.13 -m experiments.summarize baseline
```

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the full experiment CLI. Notebooks: `notebooks/01_baseline_comparison.ipynb`, `notebooks/02_scaling_study.ipynb`.

Run outputs go to `results/` (gitignored; schema in `results/schema.json`). Precomputed paper coordinates are committed in `manuscript_numbers.tex`.

---

## Results (headline)

Default target: degree-5 Chebyshev approximation of `sin(x)` on $[-1,1]$.

![Target polynomial vs. sin(x)](target_polynomial.png)

*Chebyshev target (solid) vs. `sin(x)` (dashed). Training minimizes MSE to the polynomial.*

| Metric (seed 0, default protocol) | Value |
|-----------------------------------|-------|
| Train MSE (learned) | $9.6 \times 10^{-5}$ |
| Train MSE (mapped analytic) | $4.7 \times 10^{-3}$ |
| Multi-seed median ($n=30$, $d=5$) | $6.3 \times 10^{-5}$ |
| Multi-seed success ($d=7$, $d=15$) | 30/30 each, below $10^{-3}$ |
| Off-grid max error (learned vs analytic) | $2.9 \times 10^{-2}$ vs $1.8 \times 10^{-1}$ |

![Training loss and learned polynomial](training_results.png)

*Demo snapshot (seed 0). Full figures and tables are in the [Zenodo PDF](https://doi.org/10.5281/zenodo.20645402).*

---

## Documentation

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) · [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) · [`docs/FRAMEWORKS.md`](docs/FRAMEWORKS.md) · [`docs/AUDIT_TRAIL.md`](docs/AUDIT_TRAIL.md) · [`CHANGELOG.md`](CHANGELOG.md) · [`CITATION.cff`](CITATION.cff)

---

## Citation

> Peilivanidis, V. (2026). *Learning Quantum Signal Processing Phase Angles via Gradient Descent: A Reproducible Benchmark and JAX Implementation Note* (Version 1.1). https://doi.org/10.5281/zenodo.20645402

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Attribution required for code, data, figures, or manuscript reuse.

---

## Related work

Low & Chuang (2017) · Gilyén et al. (2019) · Martyn et al. (2021) · Chao et al. (2020) · Haah (2019) · Cerezo et al. (2021) · Bergholm et al. (2022) — full list in [`references.bib`](references.bib).

Evolved from the earlier [PennyLane community demo](https://github.com/rosspeili/qsp-pennylane-demo).
