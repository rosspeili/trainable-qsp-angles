"""JSON I/O helpers for experiment runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qsp_jax.train import SCHEMA_VERSION


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA_VERSION, **payload}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path
