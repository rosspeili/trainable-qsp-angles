# Reproducing experiments

Protocol defaults: `experiments/configs/default.json`

```bash
# Single training run → results/*.json
py -3.13 -m experiments.train --seed 0 --steps 500

# Analytic baselines (see docs/FRAMEWORKS.md)
py -3.13 -m experiments.baseline_analytic
py -3.13 -m experiments.baseline_analytic --backend pennylane
py -3.13 -m experiments.baseline_analytic --backend chao --chao-method laurent
py -3.13 -m experiments.compare_chao_methods
py -3.13 -m experiments.offgrid_eval

# Sweeps (use --quick for smoke tests)
py -3.13 -m experiments.sweep multi-seed
py -3.13 -m experiments.sweep multi-seed --degrees 7,15
py -3.13 -m experiments.sweep scaling
py -3.13 -m experiments.sweep ablation

# Paper tables and loss curve
py -3.13 -m experiments.summarize baseline

# Regenerate manuscript_numbers.tex after new sweeps
py -3.13 -m experiments.generate_manuscript_numbers
```

Outputs: `results/` (gitignored). Committed paper coordinates: `manuscript_numbers.tex`.

Audit log CLI: [`docs/audit/README.md`](audit/README.md).
