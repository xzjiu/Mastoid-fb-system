"""Transactional SQLite persistence for canonical evidence artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from ..events import EventBuild, event_to_dict
from ..frame_pipeline.core_frames import CoreFrameBuild, _git_commit
from ..frame_pipeline.strokes import StrokeBuild
from ..project import PROJECT_ROOT
from ..registry import RunRecord
from ..relations import RelationBuild, relation_to_dict
from ..source_config import ResolvedRunSources


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
STORE_VERSION = "evidence_store_v1"


def _rule_text(project_root: Path, name: str) -> str:
    candidates = (
        project_root / "config" / "rules" / name,
        project_root / "config" / name,
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Required rule configuration was not found: {name}"
        )
    return path.read_text(encoding="utf-8")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_value(value: Any) -> str:
    return _hash_bytes(_json(value).encode("utf-8"))


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(global_run_id: str, source_type: str, name: str) -> str:
    return f"{global_run_id}:source_file:{source_type}:{name}"


def _artifact_rows(
    record: RunRecord,
    artifact_root: Path,
    *,
    created_at: str,
    git_commit: str,
    config_hash: str,
) -> list[dict[str, Any]]:
    candidates = (
        ("stroke_build", artifact_root / "05_strokes" / "strokes.csv"),
        ("event_inventory", artifact_root / "06_events" / "events.jsonl"),
        ("relation_inventory", artifact_root / "07_relations" / "relations.jsonl"),
        ("deterministic_report", artifact_root / "08_deterministic_report" / "deterministic_report.json"),
        ("replay_comparison", artifact_root / "09_replay" / "batch_replay_comparison.json"),
    )
    rows = []
    for artifact_type, path in candidates:
        if not path.is_file():
            continue
        rows.append(
            {
                "artifact_id": f"{record.global_run_id}:artifact:{artifact_type}",
                "global_run_id": record.global_run_id,
                "artifact_type": artifact_type,
                "path": str(path.resolve()),
                "sha256": _hash_file(path),
                "generator_version": STORE_VERSION,
                "git_commit": git_commit,
                "config_hash": config_hash,
                "created_at": created_at,
            }
        )
    return rows


def _insert_many(
    connection: sqlite3.Connection,
    sql: str,
    rows: Iterable[Sequence[Any]],
) -> None:
    connection.executemany(sql, rows)


def populate_evidence_store(
    db_path: Path,
    *,
    record: RunRecord,
    sources: ResolvedRunSources,
    source_audit: Mapping[str, Any],
    core: CoreFrameBuild,
    strokes: StrokeBuild,
    events: EventBuild,
    relations: RelationBuild,
    artifact_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build a new database atomically; existing databases are never updated in place."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = db_path.with_name(f"{db_path.name}.building")
    if temporary.exists():
        temporary.unlink()
    created_at = datetime.now(tz=timezone.utc).isoformat()
    git_commit = _git_commit(project_root)
    config_hash = _hash_value(
        {
            "event_rules": _rule_text(project_root, "event_rules.yaml"),
            "relation_rules": _rule_text(project_root, "relation_rules.yaml"),
        }
    )
    manifest_hash = _hash_value(
        {key: value for key, value in source_audit.items() if key != "generated_at"}
    )

    connection = sqlite3.connect(temporary)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        with connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.global_run_id,
                    record.dataset_id,
                    record.run_id,
                    record.participant_id,
                    record.trial_id,
                    float(record.training_year) if record.training_year else None,
                    manifest_hash,
                    created_at,
                ),
            )

            hdf5_source_ids: dict[str, str] = {}
            source_file_rows = []
            for chunk in source_audit["hdf5"]["chunks"]:
                name = str(chunk["file_name"])
                source_id = _source_id(record.global_run_id, "hdf5", name)
                hdf5_source_ids[name] = source_id
                source_file_rows.append(
                    (
                        source_id, record.global_run_id, "raw_hdf5", str(chunk["path"]),
                        chunk.get("sha256"), int(chunk["size_bytes"]), chunk.get("modified_at"), 1,
                    )
                )
            phase_source_id = _source_id(record.global_run_id, "phase", "human_annotation")
            phase_audit = source_audit["phase_annotation"]
            phase_path = Path(str(phase_audit["path"]))
            phase_stat = phase_path.stat() if phase_path.is_file() else None
            source_file_rows.append(
                (
                    phase_source_id, record.global_run_id, "human_phase_annotation",
                    str(phase_path), phase_audit.get("sha256"),
                    phase_stat.st_size if phase_stat else None,
                    datetime.fromtimestamp(phase_stat.st_mtime, tz=timezone.utc).isoformat() if phase_stat else None,
                    1,
                )
            )
            for header in source_audit["edt"]["headers"]:
                name = str(header["file_name"])
                path = sources.edt_dir / name
                source_file_rows.append(
                    (
                        _source_id(record.global_run_id, "edt", name), record.global_run_id,
                        "anatomy_edt", str(path), None, int(header["size_bytes"]),
                        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.is_file() else None,
                        1,
                    )
                )
            _insert_many(
                connection,
                "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                source_file_rows,
            )

            artifacts = _artifact_rows(
                record, artifact_root, created_at=created_at,
                git_commit=git_commit, config_hash=config_hash,
            )
            stroke_artifact_id = f"{record.global_run_id}:artifact:stroke_build"
            if not any(row["artifact_id"] == stroke_artifact_id for row in artifacts):
                artifacts.append(
                    {
                        "artifact_id": stroke_artifact_id,
                        "global_run_id": record.global_run_id,
                        "artifact_type": "stroke_build",
                        "path": f"in_memory://{record.global_run_id}/stroke_build",
                        "sha256": _hash_value({"strokes": strokes.strokes, "metrics": strokes.metrics}),
                        "generator_version": STORE_VERSION,
                        "git_commit": git_commit,
                        "config_hash": config_hash,
                        "created_at": created_at,
                    }
                )
            _insert_many(
                connection,
                "INSERT INTO generated_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [tuple(row.values()) for row in artifacts],
            )

            golden = sources.golden_reference_path
            connection.execute(
                "INSERT INTO validation_references VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"{record.global_run_id}:validation:legacy_graph",
                    record.global_run_id, "legacy_graph", str(golden),
                    _hash_file(golden), "validation_only; never used to populate canonical evidence",
                ),
            )
            _insert_many(
                connection,
                "INSERT INTO evidence_sources VALUES (?, ?, ?)",
                [
                    ("source:human_video_timepoint_annotation", "human_video_timepoint_annotation", "Manual phase interval annotation"),
                    ("source:simulator_removed_voxel_edt", "simulator_removed_voxel_edt", "Removed-voxel anatomy EDT evidence"),
                    ("source:simulator_stroke_table", "simulator_stroke_table", "Deterministic stroke interval and metric evidence"),
                    ("source:derived_frame_visibility_geometry", "derived_frame_visibility_geometry", "Automatic RGB/depth visibility candidate evidence"),
                ],
            )

            _insert_many(
                connection,
                "INSERT INTO phases VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["phase_id"], record.global_run_id, row["phase"],
                        {"antrum": "antrum_entry", "incus": "deep_landmark"}.get(row["phase"], row["phase"]),
                        float(row["annotation_start_s"]), float(row["annotation_end_s"]), phase_source_id,
                    )
                    for row in core.phase_intervals
                ],
            )
            phase_ids = {str(row["phase"]): str(row["phase_id"]) for row in core.phase_intervals}

            _insert_many(
                connection,
                "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["frame_evidence_id"], record.global_run_id,
                        int(row["frame_idx_global"]), float(row["t_video_s"]),
                        phase_ids.get(str(row["phase"])), hdf5_source_ids[str(row["source_hdf5"])],
                        int(row["source_frame_idx"]), _json(row),
                    )
                    for row in strokes.frames
                ],
            )
            metric_by_id = {int(row["stroke_id"]): row for row in strokes.metrics}
            _insert_many(
                connection,
                "INSERT INTO strokes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["stroke_evidence_id"], record.global_run_id, str(row["stroke_id"]),
                        float(row["start_time_video"]), float(row["end_time_video"]),
                        None, stroke_artifact_id, int(row["stroke_id"]),
                        _json({"interval": row, "metrics": metric_by_id[int(row["stroke_id"])]}),
                    )
                    for row in strokes.strokes
                ],
            )

            _insert_many(
                connection,
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        event.event_id, record.global_run_id, event.event_type,
                        event.t_start_s, event.t_end_s, phase_ids.get(str(event.phase)),
                        event.anatomy, event.evidence_status.value,
                        event.lifecycle_status.value, event.rule_version,
                        _json(event.attributes), created_at,
                    )
                    for event in events.events
                ],
            )
            _insert_many(
                connection,
                "INSERT INTO event_evidence VALUES (?, ?, ?)",
                [
                    (
                        event.event_id, evidence_id,
                        "frame" if ":frame:" in evidence_id else "stroke" if ":stroke:" in evidence_id else "unknown",
                    )
                    for event in events.events for evidence_id in event.source_evidence_ids
                ],
            )
            _insert_many(
                connection,
                "INSERT INTO motion_contexts VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["motion_context_id"], record.global_run_id, row["event_id"],
                        float(row["t_start_s"]), float(row["t_end_s"]), _json(row),
                    )
                    for row in events.motion_contexts
                ],
            )
            entity_rows = [
                (node["id"], "anatomy", str(node.get("label") or node["id"]))
                for node in relations.graph["nodes"] if node["type"] == "anatomy"
            ]
            _insert_many(connection, "INSERT INTO entities VALUES (?, ?, ?)", entity_rows)
            _insert_many(
                connection,
                "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        relation.relation_id, record.global_run_id,
                        relation.source_id, relation.relation_type, relation.target_id,
                        relation.construction_method, int(relation.provisional),
                        relation.rule_version, _json(relation.attributes),
                    )
                    for relation in relations.relations
                ],
            )

        foreign_key_violations = [dict(row) for row in connection.execute("PRAGMA foreign_key_check")]
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if foreign_key_violations or integrity != "ok":
            raise RuntimeError(
                f"Evidence store integrity failure: integrity={integrity}, foreign_keys={foreign_key_violations}"
            )
        table_names = (
            "runs", "source_files", "generated_artifacts", "validation_references",
            "evidence_sources", "entities", "phases", "frames", "strokes", "events",
            "event_evidence", "motion_contexts", "relations",
        )
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in table_names
        }
        manifest = {
            "schema": "evidence_store_manifest_v1",
            "store_version": STORE_VERSION,
            "generated_at": created_at,
            "database_path": str(db_path.resolve()),
            "global_run_id": record.global_run_id,
            "source_manifest_hash": manifest_hash,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "table_counts": counts,
            "integrity_check": integrity,
            "foreign_key_violation_count": len(foreign_key_violations),
            "input_roles": {
                "canonical_runtime_inputs": ["raw_hdf5", "anatomy_edt", "human_phase_annotation"],
                "validation_only": ["legacy_graph"],
            },
        }
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass

    if db_path.exists():
        db_path.unlink()
    temporary.replace(db_path)
    manifest["database_sha256"] = _hash_file(db_path)
    return manifest


def write_store_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence_store_manifest.json").write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
