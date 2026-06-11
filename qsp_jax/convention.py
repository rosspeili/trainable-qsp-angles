"""QSP phase convention transforms between analytic solvers and the flat training circuit.

The flat circuit in ``qsp_jax.circuit`` uses ``RZ(-2 phi)`` phase gates,
``W(x) = H RZ(-2 arccos x) H``, and ``<X>`` readout (Martyn et al. / training
convention).

pyqsp (Chao et al.) returns phases in the Wx/x convention (diagonal
``exp(i phi)`` rotations). Empirically, for odd Chebyshev targets:

    phi_flat = pi/2 - phi_chao

PennyLane ``poly_to_angles`` angles relate to the same pyqsp phases with
endpoint ``pi/4`` shifts on odd-degree sequences (see ``pennylane_to_pyqsp``).

References: docs/CONVENTIONS.md, docs/audit/LOG.jsonl (AUD-019+).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "chao_to_flat",
    "pennylane_to_pyqsp",
    "pennylane_to_flat",
    "map_to_flat",
]


def chao_to_flat(phases: np.ndarray) -> np.ndarray:
    """Map pyqsp / Chao Wx/x phases to flat ``qsp_circuit`` phases."""
    phases = np.asarray(phases, dtype=np.float64)
    return np.pi / 2.0 - phases


def pennylane_to_pyqsp(phases: np.ndarray) -> np.ndarray:
    """
    Recover pyqsp-equivalent phases from PennyLane ``poly_to_angles`` output.

    For degree d=5 validated; endpoints pick up ``± pi/4`` relative to inner
    negation symmetry ``phi_pl[k] ≈ -phi_chao[k]`` for k=1..d-1.
    """
    phases = np.asarray(phases, dtype=np.float64)
    n = phases.shape[0]
    if n < 2:
        raise ValueError(f"Expected at least 2 phases, got {n}")

    chao = np.empty_like(phases)
    chao[0] = -np.pi / 4.0 - phases[0]
    if n > 2:
        chao[1 : n - 1] = -phases[1 : n - 1]
    chao[n - 1] = np.pi / 4.0 - phases[n - 1]
    return chao


def pennylane_to_flat(phases: np.ndarray) -> np.ndarray:
    """Map PennyLane QSP angles to flat circuit phases."""
    return chao_to_flat(pennylane_to_pyqsp(phases))


def map_to_flat(phases: np.ndarray, source: str = "chao") -> np.ndarray:
    """Dispatch convention mapping by solver source label."""
    phases = np.asarray(phases, dtype=np.float64)
    if source in ("flat", "identity", "training"):
        return phases.copy()
    if source in ("chao", "pyqsp"):
        return chao_to_flat(phases)
    if source in ("pennylane", "pl"):
        return pennylane_to_flat(phases)
    raise ValueError(f"Unknown phase source {source!r}; use chao, pennylane, or flat")
