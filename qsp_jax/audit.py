"""Append-only project audit trail (JSONL + helpers)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / "docs" / "audit" / "LOG.jsonl"


@dataclass
class AuditEntry:
    """One auditable event (success, failure, fix, decision, etc.)."""

    category: str
    status: str
    title: str
    what: str
    why: str = ""
    action: str = ""
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    id: str = ""
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _next_entry_id(log_path: Path = AUDIT_LOG_PATH) -> str:
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pattern = re.compile(rf"AUD-{re.escape(date_prefix)}-(\d{{3}})")
    max_seq = 0
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            match = pattern.fullmatch(str(payload.get("id", "")))
            if match:
                max_seq = max(max_seq, int(match.group(1)))
    return f"AUD-{date_prefix}-{max_seq + 1:03d}"


def append_audit_entry(
    *,
    category: str,
    status: str,
    title: str,
    what: str,
    why: str = "",
    action: str = "",
    rationale: str = "",
    evidence: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    related_ids: Iterable[str] | None = None,
    log_path: Path | None = None,
) -> AuditEntry:
    """Append one audit record to ``docs/audit/LOG.jsonl`` and return it."""
    path = log_path or AUDIT_LOG_PATH
    entry = AuditEntry(
        id=_next_entry_id(path),
        timestamp_utc=utc_stamp(),
        category=category,
        status=status,
        title=title,
        what=what,
        why=why,
        action=action,
        rationale=rationale,
        evidence=list(evidence or []),
        labels=list(labels or []),
        related_ids=list(related_ids or []),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return entry


def read_audit_entries(
    *,
    log_path: Path | None = None,
    last: int | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """Read audit entries, optionally filtered by label or truncated to last N."""
    path = log_path or AUDIT_LOG_PATH
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if label is not None and label not in payload.get("labels", []):
            continue
        entries.append(payload)

    if last is not None:
        return entries[-last:]
    return entries
