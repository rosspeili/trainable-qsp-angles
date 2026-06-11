"""
qsp_jax — QSP circuit construction and loss utilities for JAX-based phase angle training.

Importing this package enables JAX float64 precision (set in circuit.py).
"""

from qsp_jax.circuit import loss_fn, qsp_circuit, target_poly

__all__ = ["qsp_circuit", "target_poly", "loss_fn"]
