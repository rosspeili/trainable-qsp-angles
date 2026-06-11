"""Tests for append-only audit trail helpers."""

import json
from pathlib import Path

from qsp_jax.audit import append_audit_entry, read_audit_entries


def test_append_and_read_audit_entry(tmp_path):
    log_path = tmp_path / "LOG.jsonl"

    entry = append_audit_entry(
        category="test",
        status="success",
        title="unit test entry",
        what="append/read round trip",
        labels=["pytest"],
        log_path=log_path,
    )

    assert log_path.exists()
    assert entry.id.startswith("AUD-")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["title"] == "unit test entry"
    assert payload["labels"] == ["pytest"]

    entries = read_audit_entries(log_path=log_path, label="pytest")
    assert len(entries) == 1
