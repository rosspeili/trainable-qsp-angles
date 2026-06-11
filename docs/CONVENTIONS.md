# QSP Phase Conventions

This document defines how analytic solver output relates to the **flat training
circuit** in `qsp_jax/circuit.py`.

## Training circuit (reference)

| Element | Implementation |
|---------|------------------|
| Initial / interleaved rotations | `RZ(-2 * phi_k)` |
| Signal oracle | `W(x) = H @ RZ(-2 arccos(x)) @ H` |
| Readout | `⟨X⟩` |
| Loss | MSE vs Chebyshev sin target on `x ∈ [-0.95, 0.95]` |

Gradient Adam optimizes phases **directly in this convention**.

## pyqsp / Chao (Wx/x)

pyqsp `QuantumSignalProcessingPhases(..., method='laurent')` returns phases whose
native response matches the target Chebyshev polynomial in pyqsp's Wx/x model
(verified by `pyqsp_reconstruction_max_error`).

**Mapping to flat circuit** (degree-5 T1 validated; odd degrees use same rule):

```python
phi_flat = pi/2 - phi_chao
```

Implemented in `qsp_jax/convention.py` as `chao_to_flat()`.

Without mapping, flat-circuit MSE is ~0.95 (misleading). With mapping, ~5×10⁻³
(capitalization / residual mapping error), not gradient-optimal ~10⁻⁴.

## PennyLane `poly_to_angles`

PennyLane returns angles in its native QSP convention. A closed-form
PL→pyqsp inversion with endpoint ``± pi/4`` shifts matches pyqsp at **degree 5**
but diverges at higher degree (PennyLane sparse angle structure).

**Reporting policy for PennyLane baseline:**

- ``angles_solver`` — raw PennyLane output
- ``metrics_unmapped`` — those angles on the flat circuit (audit; often ~0.55 MSE)
- ``metrics`` / ``angles`` — **shared** ``chao_to_flat(pyqsp(...))`` from the same
  Chebyshev sin target, identical to the Chao row mapped metrics

Both solvers target the same polynomial; mapped flat metrics use pyqsp as the
convention bridge. Compare solve time and native ``metrics_unmapped`` for PL vs Chao.

Direct helper ``pennylane_to_pyqsp()`` remains for d=5 validation tests only.

## Reporting policy (experiments)

Comparison tables include:

| Column | Meaning |
|--------|---------|
| `train_mse` | Flat circuit, **mapped** phases (apples-to-apples with gradient) |
| `train_mse_unmapped` | Flat circuit, raw solver output (audit / failure mode) |
| `pyqsp_reconstruction_max_error` | Native Chao accuracy (Chao row only) |

See `results/paper/baseline_comparison_d*.csv` after `experiments.summarize baseline`.

## Open refinement

Sub-mill mapped MSE may require joint optimization or exact capitalization matching
pyqsp `eps`/`suc`. Tracked in audit log if pursued.
