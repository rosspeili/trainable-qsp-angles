# Trainable QSP Angles

Research code and manuscript for **learning Quantum Signal Processing (QSP) phase angles via gradient descent** — a framework-agnostic method (flat differentiable circuit + JAX + Optax), with a **PennyLane reference implementation** in `qsp_jax/`.

This repository is the canonical home for the paper, reproducible experiments, notebooks, and tests. It evolved from the earlier [PennyLane community demo](https://github.com/rosspeili/qsp-pennylane-demo) but is maintained here as a standalone research project.

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-efcefa?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13%2B-bae6fd?style=flat-square)
![JAX](https://img.shields.io/badge/JAX-0.4.25%2B-a5f3fc?style=flat-square)
![PennyLane (ref.)](https://img.shields.io/badge/PennyLane-ref._impl.-e9d5ff?style=flat-square)

</div>

---

## Author

**Ross Peili** (Vladimiros Peilivanidis)  
ARPA Hellenic Logical Systems, Thessaloniki, Greece  
[vpeilivanidis@gmail.com](mailto:vpeilivanidis@gmail.com)

---

## What This Is

Quantum Signal Processing encodes polynomial transformations of a scalar signal into a single-qubit circuit via phase-shifted oracle calls. The classical approach computes phase angles analytically from a target polynomial. This project explores the complementary route: **train phase angles from random initialization** using automatic differentiation and gradient-based optimization.

Primary contributions (current state):

- A **JAX-traceable flat QSP circuit** (avoiding high-level QSVT templates that capture concrete values and break gradients — documented first in our PennyLane reference code)
- A **reproducible degree-5 benchmark** (Chebyshev approximation of `sin(x)`)
- The accompanying **manuscript** (`manuscript.tex`) and **research roadmap** (`RESEARCH_PLAN.md`)

See `docs/FRAMEWORKS.md` for when PennyLane is the right tool here vs. Qiskit, Cirq, TensorFlow Quantum, OpenFermion, or standalone analytic solvers.

---

## Quick Start

```bash
git clone https://github.com/rosspeili/trainable-qsp-angles
cd trainable-qsp-angles
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pytest tests/ -v
py -3.13 -m jupyter notebook demo.ipynb
```

Regenerate figures with `demo.ipynb`, or use the committed assets below.

### Reproducible experiments

Protocol defaults: `experiments/configs/default.json`

```bash
# Single training run → results/*.json
py -3.13 -m experiments.train --seed 0 --steps 500

# Analytic baseline (included convenience wrapper: PennyLane poly_to_angles; see docs/FRAMEWORKS.md for alternatives)
py -3.13 -m experiments.baseline_analytic

# Phase 2 sweeps (use --quick for smoke tests)
py -3.13 -m experiments.sweep multi-seed
py -3.13 -m experiments.sweep scaling
py -3.13 -m experiments.sweep ablation

# Comparison table + loss curve for paper/notebooks
py -3.13 -m experiments.summarize baseline
```

Analysis notebooks (after sweeps): `notebooks/01_baseline_comparison.ipynb`, `notebooks/02_scaling_study.ipynb`.

JSON outputs go to `results/` (see `results/schema.json`).

---

## Repository Layout

```
trainable-qsp-angles/
├── docs/
│   └── FRAMEWORKS.md       # Stack choices; PennyLane as ref. impl., not exclusive
├── manuscript.tex          # Paper source
├── references.bib          # Bibliography
├── RESEARCH_PLAN.md        # Analysis, gaps, and experimental roadmap
├── experiments/
│   ├── configs/default.json
│   ├── train.py
│   ├── baseline_analytic.py
│   ├── sweep.py
│   └── summarize.py
├── notebooks/
│   ├── 01_baseline_comparison.ipynb
│   └── 02_scaling_study.ipynb
├── results/                # Generated run outputs (gitignored except .gitkeep)
├── demo.ipynb              # Interactive training demo
├── target_polynomial.png   # Benchmark figure (also in demo.ipynb)
├── training_results.png
├── qsp_jax/
│   ├── __init__.py
│   └── circuit.py          # Flat QSP circuit, target poly, loss
├── tests/
│   └── test_circuit.py     # Unit tests
├── requirements.txt
├── LICENSE                 # Apache 2.0
└── NOTICE                  # Attribution requirements
```

---

## Key Concepts

- **Signal oracle**: `W(x) = H @ RZ(-2*arccos(x)) @ H`, encoding `x ∈ (-1, 1)` in the top-left matrix element
- **QSP sequence**: Flat alternating circuit — one phase rotation `RZ(-2*phi_k)` per signal query `W(x)`
- **Polynomial encoding**: The expectation value `<X>` encodes a degree-d polynomial in `x` determined by the phase angles
- **Training**: Adam (Optax) minimizes MSE between circuit output and target polynomial via `jax.grad`
- **Implementation note**: `qsp_jax/circuit.py` uses PennyLane primitives as the reference frontend; the same flat pattern applies in Qiskit, Cirq, TFQ, etc. (see `docs/FRAMEWORKS.md`)
- **JAX note**: Use inline rotations, not opaque template objects that freeze parameters at build time

---

## Target Polynomial (Default Benchmark)

The default target is a **degree-5 Chebyshev approximation of `sin(x)`** on `[-1, 1]` — an odd polynomial bounded in `[-1, 1]`, consistent with QSP conventions for odd-degree transformations. It has a maximum deviation of ~0.174 from the true `sin(x)`.

![Target polynomial vs. sin(x)](target_polynomial.png)

After 500 Adam steps (lr=0.05, 64-point grid), typical results:

- **Final MSE**: ~4.82×10⁻⁴
- **Max pointwise error**: ~9.83×10⁻² (vs. target polynomial, not vs. `sin(x)` directly)

Loss drops from ~1.44 (random initialization) to ~6.4×10⁻⁴, converging quickly within the first 100 steps.

![Training loss and learned polynomial](training_results.png)

---

## Attribution

This work is **free and open source** under [Apache 2.0](LICENSE).

If you use this repository **in whole or in part** — code, snippets, notebooks, figures, data, or the LaTeX manuscript — you **must attribute**:

> Ross Peili (Vladimiros Peilivanidis), ARPA Hellenic Logical Systems

See [NOTICE](NOTICE) for the full attribution text and suggested citation.

---

## Related Work

- Martyn et al., [A Grand Unification of Quantum Algorithms](https://arxiv.org/abs/2105.02859), PRX Quantum 2021
- Gilyen et al., [Quantum singular value transformation](https://arxiv.org/abs/1806.01838), STOC 2019
- Chao et al., [Finding Angles for QSP with Machine Precision](https://arxiv.org/abs/2003.02831), 2020

---

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
