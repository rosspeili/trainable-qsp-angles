# Audit Log — How to Use

This folder holds the **machine-readable audit trail** for the project. Human-readable narrative lives in [`../AUDIT_TRAIL.md`](../AUDIT_TRAIL.md). Version summaries live in [`../../CHANGELOG.md`](../../CHANGELOG.md).

## Files

| File | Purpose |
|------|---------|
| `LOG.jsonl` | Append-only JSON Lines log (one event per line). **Committed to git.** |
| `../AUDIT_TRAIL.md` | Curated narrative: failures, fixes, decisions, comparisons, tests |

## When to append an entry

Add a log entry after any of:

- A **failed** experiment, test, solver call, or CI run (label `status: failed` or `partial`)
- A **fix** or workaround (label `status: fixed`)
- An **architectural or scientific decision** (label `status: decided`)
- A **comparison run** whose numbers matter for the paper (label `status: success`)
- A **regression** or re-opened issue (link `related_ids`)

## Entry schema

Each line in `LOG.jsonl` is a JSON object:

```json
{
  "id": "AUD-2026-06-11-012",
  "timestamp_utc": "20260611T140000Z",
  "category": "failure",
  "status": "failed",
  "title": "Short label",
  "what": "What happened",
  "why": "Root cause or hypothesis",
  "action": "What we did or plan to do",
  "rationale": "Why that action was chosen",
  "evidence": ["results/foo.json", "tests/test_x.py"],
  "labels": ["pyqsp", "convention-mismatch"],
  "related_ids": ["AUD-2026-06-11-003"]
}
```

**Categories:** `failure`, `fix`, `decision`, `comparison`, `test`, `experiment`, `docs`, `infra`

**Statuses:** `failed`, `partial`, `fixed`, `decided`, `success`, `open`, `wontfix`

## CLI (append from terminal)

```bash
# Interactive-style one-liner
py -3.13 -m experiments.audit append \
  --category failure --status open \
  --title "Example failure" \
  --what "Solver raised X" \
  --why "Wrong polynomial basis" \
  --action "Switch to Chebyshev coeffs" \
  --labels pyqsp,polynomial

# List recent entries
py -3.13 -m experiments.audit list --last 10

# Print entries matching a label
py -3.13 -m experiments.audit list --label convention-mismatch
```

## Python API

```python
from qsp_jax.audit import append_audit_entry

append_audit_entry(
    category="comparison",
    status="success",
    title="Chao baseline smoke run",
    what="pyqsp Laurent d=5 verified in native convention",
    why="Needed SDK-independent analytic baseline",
    action="Added qsp_jax/chao_baseline.py; kept PennyLane path",
    rationale="Append, do not replace baselines for fair comparison",
    evidence=["results/analytic_chao_d5_laurent_Wx_20260611T134709Z.json"],
    labels=["chao", "pyqsp", "baseline"],
)
```

## Relationship to `results/`

| Location | Role |
|----------|------|
| `results/` | Raw experiment JSON/CSV (gitignored except `schema.json`) |
| `docs/audit/LOG.jsonl` | **Interpreted** events: what failed, why, what we changed |
| `docs/AUDIT_TRAIL.md` | Synthesized story for humans and manuscript alignment |

Always reference `results/` file paths in `evidence` when a run produced data.
