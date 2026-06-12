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

### sym_qsp (machine-precision native solver)

`method='sym_qsp'` with `chebyshev_basis=True` reaches near machine precision
in pyqsp's native Wx/x convention (`pyqsp_reconstruction_max_error` ≲ 10⁻¹⁵ at
d=5). After the same `chao_to_flat()` mapping, flat-circuit train MSE remains
≈4.7×10⁻³—indistinguishable from Laurent. **Headline tables keep Laurent**
(faster); sym_qsp is documented as a native-accuracy audit.

Reproduce: `py -3.13 -m experiments.compare_chao_methods --degree 5`
→ `results/paper/chao_method_comparison_d5.json`.

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
