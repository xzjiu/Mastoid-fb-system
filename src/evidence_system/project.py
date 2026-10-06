"""Repository and runtime path resolution without machine-specific defaults."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("MASTOID_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
EXAMPLE_CONFIG_DIR = CONFIG_DIR / "examples"
RULES_DIR = CONFIG_DIR / "rules"
SCHEMA_DIR = PROJECT_ROOT / "schemas"


def _configured_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else fallback


REGISTRY_PATH = _configured_path(
    "MASTOID_REGISTRY_PATH", EXAMPLE_CONFIG_DIR / "runs.example.csv"
)
SOURCE_PATHS_LOCAL = _configured_path(
    "MASTOID_SOURCE_CONFIG", CONFIG_DIR / "source_paths.local.json"
)
SOURCE_PATHS_EXAMPLE = EXAMPLE_CONFIG_DIR / "source_paths.example.json"
ANATOMY_PROFILES_PATH = _configured_path(
    "MASTOID_ANATOMY_PROFILES", EXAMPLE_CONFIG_DIR / "anatomy_profiles.example.json"
)
OUTPUT_ROOT = _configured_path("MASTOID_OUTPUT_ROOT", PROJECT_ROOT / "results")


def resolve_project_path(value: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def first_existing(*paths: Path) -> Path:
    """Return the first existing candidate, or the first candidate for diagnostics."""

    if not paths:
        raise ValueError("At least one path candidate is required")
    return next((path for path in paths if path.exists()), paths[0])
