# Quantum Software Stack — Framework Notes

This project studies **trainable QSP phase angles** using gradient descent on a **flat, fully differentiable circuit**. That method is **not tied to a single quantum SDK**. The mathematics (QSP sequence, MSE loss, Adam on phases) is framework-agnostic.

**PennyLane is the current reference implementation** in this repository because:

- It was the starting point for the original community demo.
- It provides a convenient **`poly_to_angles` analytic baseline** for comparison experiments.
- Its **JAX interface** matches our autodiff + `jax.vmap` training loop with minimal glue code.
- It can **target many backends** (see [PennyLane plugins](https://pennylane.ai/plugins/)) including Qiskit, Cirq, and Braket simulators/devices when deployment needs change.

We do **not** claim PennyLane is the only or best choice for every task below.

---

## What actually must hold (any framework)

Regardless of library, a gradient-training setup needs:

1. **Live phase parameters** in the autodiff graph (no concrete capture at circuit-build time).
2. **Flat primitive gates** (or an API that expands templates without breaking tracing).
3. A **signal oracle** \(W(x) = H R_Z(-2\arccos x) H\) and phase rotations \(R_Z(-2\phi_k)\) in the agreed QSP convention.
4. A **scalar readout** (\(\langle X\rangle\) in this repo) comparable to the target polynomial on \([-1,1]\).

If those hold, the same loss landscape and training protocol apply.

---

## When another library may be a better fit

| Need | Reason to consider | Examples |
|------|-------------------|----------|
| **IBM Quantum hardware / Qiskit ecosystem** | Native transpilation, runtime, and job submission | [Qiskit](https://github.com/Qiskit/qiskit) + `qiskit-aer`; PennyLane’s Qiskit plugin if you want to keep this codebase |
| **Google Cirq workflows / cirq-google** | Circuits-first style, Google devices | [Cirq](https://github.com/quantumlib/Cirq); PennyLane Cirq plugin |
| **TensorFlow / Keras pipelines** | Existing TF training graphs, TF Data, TPU adjacency | [TensorFlow Quantum](https://github.com/tensorflow/quantum) (Cirq-backed); different AD path than JAX |
| **PyTorch-native VQAs** | `torch.autograd` end-to-end with classical co-training | Custom simulator or `torch`-compatible backends; re-implement flat QSP in Torch (not in repo yet) |
| **Chemistry / fermionic Hamiltonians → block encoding** | OpenFermion focuses on fermionic operators and chemistry models, useful upstream of QSVT | [OpenFermion](https://github.com/quantumlib/OpenFermion) + downstream QSP/QSVT (PennyLane, Qiskit, or custom) |
| **Machine-precision analytic angles only** | No quantum SDK required for the classical solve | [Chao et al.](https://arxiv.org/abs/2003.02831) via **`pyqsp`** in `qsp_jax/chao_baseline.py`; PennyLane `poly_to_angles` is a separate convenience wrapper |
| **Maximum sim speed (CPU/GPU)** | Device-specific kernels | PennyLane-Lightning, Qiskit Aer MPS/GPU, or dedicated tensor-network sims |
| **Minimal dependencies / pedagogy** | Teach QSP without a full QML stack | NumPy + explicit unitaries (slow but transparent); validate against this repo’s results |

**Rule of thumb:** stay on PennyLane here for **reproducing paper numbers and experiments**. Reach for another stack when **deployment, ecosystem, or performance** constraints dominate—not because QSP theory requires it.

---

## PennyLane + other frameworks (not either/or)

PennyLane is often described as a **differentiable front-end** that can delegate execution:

- **Same training code, different device:** swap `default.qubit` for a Qiskit, Cirq, or Lightning device when you need fidelity to a target platform or faster simulation.
- **Analytic angles elsewhere, train here:** phases from Chao et al. (`qsp_jax/chao_baseline.py`) or PennyLane `poly_to_angles` can be evaluated in our flat circuit for apples-to-apples MSE — convention alignment matters more than which library produced the angles.

The documented **`qml.QSVT` + JAX tracing issue** is a **template-capture pitfall** observed in PennyLane. Analogous bugs can appear in any system that builds operator objects from concrete NumPy values before autodiff sees them. The fix—**inline primitives with tracer-valued parameters**—transfers directly to Qiskit (`QuantumCircuit` with `ParameterExpression`), Cirq (`sympy`/parameterized gates), or TFQ.

---

## Roadmap: optional cross-framework checks (not required for core claims)

| Check | Purpose | Effort |
|-------|---------|--------|
| Rebuild flat QSP in **Qiskit** (`Parameter` phases) + JAX via `qiskit-jax` or manual unitary | Confirm learned phases match PennyLane sim | Medium |
| Same circuit in **Cirq** | Google-stack validation | Medium |
| **Standalone Chao solver** baseline (no `poly_to_angles`) | Fair analytic comparison independent of PennyLane | **Implemented:** `qsp_jax/chao_baseline.py` (`pyqsp` Laurent completion); compare alongside PennyLane in `experiments/summarize.py baseline` |
| **PennyLane-Lightning** device | Faster sweeps for Phase 2 statistics | Low |

These are **evidence multipliers**, not prerequisites. The research question is about **gradient learning of QSP angles**, not about PennyLane advocacy.

---

## References (external)

- PennyLane plugins: https://pennylane.ai/plugins  
- Qiskit: https://qiskit.org  
- Cirq: https://quantumai.google/cirq  
- TensorFlow Quantum: https://www.tensorflow.org/quantum  
- OpenFermion: https://quantumai.google/openfermion  
