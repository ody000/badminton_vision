"""Load config.yaml and merge with optional override dict.

Usage:
    cfg = load_config()                        # loads config.yaml
    cfg = load_config("config.yaml", {"fps": 60.0, "device": "cuda"})
    print(cfg.fps)  # 60.0
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import yaml


def load_config(path: str = "config.yaml", overrides: dict | None = None) -> SimpleNamespace:
    """Load YAML config file and merge with override dict.

    Args:
        path: Path to YAML config file. Resolved relative to CWD.
        overrides: Flat dict whose values take precedence over YAML values.
                   Keys may use the same dot-free names as config.yaml.

    Returns:
        SimpleNamespace where every top-level config key is an attribute.
    """
    config_path = os.path.abspath(path)

    if not os.path.exists(config_path):
        # Try to find config.yaml relative to this file's parent tree.
        here = os.path.dirname(os.path.abspath(__file__))
        for ancestor in [here, os.path.dirname(here)]:
            candidate = os.path.join(ancestor, "config.yaml")
            if os.path.exists(candidate):
                config_path = candidate
                break

    data: dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
    else:
        print(f"[CONFIG] Warning: config file not found at {config_path}; using defaults only.")

    if overrides:
        for k, v in overrides.items():
            data[k] = v

    return SimpleNamespace(**data)
