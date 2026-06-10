"""Configuration loading and path resolution.

All other modules import :data:`CONFIG` and the resolved :data:`PATHS` from here so
that paths and parameters live in exactly one place (``config/config.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_file: Path = CONFIG_FILE) -> dict:
    """Load the YAML configuration file into a dictionary."""
    with open(config_file, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class Paths:
    """Absolute paths resolved from the config, relative to the project root."""

    raw_data: Path
    processed_data: Path
    tables_dir: Path
    figures_dir: Path

    def ensure_dirs(self) -> None:
        """Create all output directories (and data dirs) if missing."""
        self.raw_data.parent.mkdir(parents=True, exist_ok=True)
        self.processed_data.parent.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


def resolve_paths(config: dict) -> Paths:
    """Turn the relative paths in the config into absolute :class:`Paths`."""
    paths = config["paths"]
    return Paths(
        raw_data=PROJECT_ROOT / paths["raw_data"],
        processed_data=PROJECT_ROOT / paths["processed_data"],
        tables_dir=PROJECT_ROOT / paths["tables_dir"],
        figures_dir=PROJECT_ROOT / paths["figures_dir"],
    )


CONFIG = load_config()
PATHS = resolve_paths(CONFIG)
