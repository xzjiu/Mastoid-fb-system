"""Deterministic source-to-evidence build without validation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import build_events, write_event_build
from .frame_pipeline import (
    build_strokes,
    compare_legacy_core_frames,
    compare_legacy_strokes,
    compare_legacy_tip_edt,
    compare_legacy_visibility,
    compare_legacy_voxel_edt,
    enrich_removed_voxels_with_edt,
    enrich_tip_edt,
    enrich_visibility,
    extract_core_frames,
    write_core_frame_build,
    write_edt_build,
    write_stroke_build,
    write_tip_edt_build,
    write_visibility_build,
)
from .project import PROJECT_ROOT
from .registry import RunRecord
from .relations import build_relations, compare_legacy_graph, write_relation_build
from .review_video import build_review_video
from .source_audit import build_source_audit, write_source_audit
from .source_config import ResolvedRunSources
from .storage import populate_evidence_store, write_store_manifest


def build_evidence(
    *,
    record: RunRecord,
    sources: ResolvedRunSources,
    result_dir: Path,
    project_root: Path = PROJECT_ROOT,
    build_video: bool = False,
    video_fps: float = 10.0,
) -> dict[str, Any]:
    """Build canonical evidence and a read-only SQLite store for one run."""

    result_dir = result_dir.expanduser().resolve()
    stages = result_dir / "stages"
    audit = build_source_audit(record, sources, hash_hdf5=False)
    if not audit["ready_for_frame_extraction"]:
        raise RuntimeError(
            "Source audit failed: " + ", ".join(audit["blocking_items"])
        )
    write_source_audit(stages / "00_source_audit", audit)

    core = extract_core_frames(record, sources, project_root=project_root)
    core_regression = compare_legacy_core_frames(
        core, sources.raw_run_dir / "frame_level_spatial_all_frames_with_voxel.csv"
    )
    write_core_frame_build(stages / "01_core_frames", core, core_regression)

    edt = enrich_removed_voxels_with_edt(
        core, record, sources, project_root=project_root
    )
    edt_regression = compare_legacy_voxel_edt(
        edt, sources.raw_run_dir / "frame_level_spatial_all_frames_voxel_edt.csv"
    )
    write_edt_build(stages / "02_edt_enrichment", edt, edt_regression)

    tip = enrich_tip_edt(edt, record, sources, project_root=project_root)
    tip_regression = compare_legacy_tip_edt(
        tip, sources.raw_run_dir / "frame_level_spatial_all_frames_voxel_edt.csv"
    )
    write_tip_edt_build(stages / "03_tip_edt", tip, tip_regression)

    visibility = enrich_visibility(
        tip, record, sources, project_root=project_root
    )
    visibility_regression = compare_legacy_visibility(
        visibility,
        sources.raw_run_dir / "frame_level_spatial_all_frames_visibility.csv",
    )
    write_visibility_build(
        stages / "04_visibility", visibility, visibility_regression
    )

    strokes = build_strokes(visibility, record, project_root=project_root)
    stroke_regression = compare_legacy_strokes(strokes, sources.raw_run_dir)
    write_stroke_build(stages / "05_strokes", strokes, stroke_regression)

    events = build_events(strokes, core.phase_intervals, record)
    write_event_build(stages / "06_events", events)
    relations = build_relations(events, record)
    relation_regression = compare_legacy_graph(
        relations, record.golden_reference_path
    )
    write_relation_build(
        stages / "07_relations", relations, relation_regression
    )

    database_path = result_dir / "evidence.sqlite3"
    store_manifest = populate_evidence_store(
        database_path,
        record=record,
        sources=sources,
        source_audit=audit,
        core=core,
        strokes=strokes,
        events=events,
        relations=relations,
        artifact_root=stages,
        project_root=project_root,
    )
    write_store_manifest(result_dir, store_manifest)

    video_manifest = None
    if build_video:
        video_manifest = build_review_video(
            record, sources, result_dir, fps=video_fps
        )

    return {
        "schema": "evidence_build_summary_v1",
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "result_dir": str(result_dir),
        "evidence_database": str(database_path),
        "frame_count": len(strokes.frames),
        "removed_voxel_count": len(strokes.removed_voxels),
        "stroke_count": len(strokes.strokes),
        "event_count": len(events.events),
        "relation_count": len(relations.relations),
        "store_integrity": store_manifest["integrity_check"],
        "review_video": (
            str(result_dir / str(video_manifest["video_file"]))
            if video_manifest else None
        ),
    }
