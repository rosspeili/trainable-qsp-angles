"""Load experiment protocol defaults and merge CLI overrides."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qsp_jax.train import TrainConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.json"


def load_protocol_config(path: Path | None = None) -> dict[str, Any]:
    """Load JSON protocol file (defaults for sweeps and single runs)."""
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def train_config_from_protocol(
    overrides: dict[str, Any] | None = None,
    path: Path | None = None,
) -> TrainConfig:
    """Build ``TrainConfig`` from protocol JSON plus optional overrides."""
    protocol = load_protocol_config(path)
    train_fields = {field for field in TrainConfig.__dataclass_fields__}
    merged = {key: protocol[key] for key in protocol if key in train_fields}
    if overrides:
        merged.update({k: v for k, v in overrides.items() if k in train_fields})
    return TrainConfig(**merged)
