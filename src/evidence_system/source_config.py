"""Runtime source-root configuration supplied by file, environment, or CLI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from .project import PROJECT_ROOT, SOURCE_PATHS_EXAMPLE, SOURCE_PATHS_LOCAL
from .registry import RunRecord


@dataclass(frozen=True)
class SourceRoots:
    data_root: Path
    edt_root: Path
    phase_annotation_file: Path


@dataclass(frozen=True)
class ResolvedRunSources:
    raw_run_dir: Path
    edt_dir: Path
    phase_annotation_file: Path
    golden_reference_path: Path


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_source_roots(
    path: Path | None = None, *, project_root: Path = PROJECT_ROOT
) -> SourceRoots:
    configured = path
    if configured is None:
        env_value = os.environ.get("MASTOID_SOURCE_CONFIG")
        configured = Path(env_value) if env_value else (
            SOURCE_PATHS_LOCAL if SOURCE_PATHS_LOCAL.is_file() else SOURCE_PATHS_EXAMPLE
        )
    configured = configured.expanduser().resolve()
    if not configured.is_file():
        raise FileNotFoundError(
            f"Source configuration not found: {configured}. "
            "Pass --source-config or set MASTOID_SOURCE_CONFIG."
        )
    payload = json.loads(configured.read_text(encoding="utf-8"))
    data_root = payload.get("data_root")
    required = {
        "data_root": data_root,
        "edt_root": payload.get("edt_root"),
        "phase_annotation_file": payload.get("phase_annotation_file"),
    }
    missing = [key for key, value in required.items() if not str(value or "").strip()]
    if missing:
        raise ValueError(f"Missing source path keys: {', '.join(missing)}")
    base = project_root
    return SourceRoots(
        data_root=_resolve(str(data_root), base),
        edt_root=_resolve(str(required["edt_root"]), base),
        phase_annotation_file=_resolve(str(required["phase_annotation_file"]), base),
    )


def resolve_run_sources(
    record: RunRecord,
    roots: SourceRoots,
    *,
    raw_run_dir: Path | None = None,
    edt_dir: Path | None = None,
    phase_annotation_file: Path | None = None,
) -> ResolvedRunSources:
    raw = raw_run_dir or (
        roots.data_root / Path(record.raw_run_relpath)
        if record.raw_run_relpath else roots.data_root
    )
    edt = edt_dir or (
        roots.edt_root / record.edt_profile if record.edt_profile else roots.edt_root
    )
    return ResolvedRunSources(
        raw_run_dir=raw.expanduser().resolve(),
        edt_dir=edt.expanduser().resolve(),
        phase_annotation_file=(
            phase_annotation_file or roots.phase_annotation_file
        ).expanduser().resolve(),
        golden_reference_path=record.golden_reference_path,
    )
