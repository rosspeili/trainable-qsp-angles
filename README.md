# Trainable QSP Angles

Research code and manuscript for **learning Quantum Signal Processing (QSP) phase angles via gradient descent** — a differentiable, flat-circuit implementation using PennyLane, JAX, and Optax.

This repository is the canonical home for the paper, reproducible experiments, notebooks, and tests. It evolved from the earlier [PennyLane community demo](https://github.com/rosspeili/qsp-pennylane-demo) but is maintained here as a standalone research project.

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-efcefa?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13%2B-bae6fd?style=flat-square)
![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-e9d5ff?style=flat-square)
![JAX](https://img.shields.io/badge/JAX-0.4.25%2B-a5f3fc?style=flat-square)

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

- A **JAX-traceable flat QSP circuit** (avoiding `qml.QSVT` construction-time capture that breaks gradients)
- A **reproducible degree-5 benchmark** (Chebyshev approximation of `sin(x)`)
- The accompanying **manuscript** (`manuscript.tex`) and **research roadmap** (`RESEARCH_PLAN.md`)

See `RESEARCH_PLAN.md` for the full gap analysis, experimental plan, and path from demo to publication-quality work.

---

## Quick Start

```bash
git clone https://github.com/rosspeili/trainable-qsp-angles
cd trainable-qsp-angles
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pytest tests/ -v
py -3.13 -m jupyter notebook demo.ipynb
```

Figures referenced in the README are generated when you run `demo.ipynb` (`target_polynomial.png`, `training_results.png`).

### Reproducible experiments

```bash
py -3.13 experiments/train.py --seed 0 --steps 500
py -3.13 experiments/baseline_analytic.py
```

JSON outputs are written to `results/`.

---

## Repository Layout

```
trainable-qsp-angles/
├── manuscript.tex          # Paper source
├── references.bib          # Bibliography
├── RESEARCH_PLAN.md        # Analysis, gaps, and experimental roadmap
├── experiments/
│   ├── train.py            # CLI: gradient training → results/*.json
│   └── baseline_analytic.py
├── results/                # Generated run outputs (gitignored except .gitkeep)
├── demo.ipynb              # Interactive training demo
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
- **JAX note**: Implemented with inline `qp.RZ` + `qp.Hadamard` gates, not `qp.QSVT`, to preserve traceability

---

## Target Polynomial (Default Benchmark)

Degree-5 odd Chebyshev approximation of `sin(x)` on `[-1, 1]`. After 500 Adam steps (lr=0.05, 64-point grid), typical results:

- **Final MSE**: ~4.82×10⁻⁴
- **Max pointwise error**: ~9.83×10⁻² (vs. target polynomial, not vs. `sin(x)` directly)

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
