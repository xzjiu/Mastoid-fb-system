"""Deterministic frame/stroke-to-event construction.

This module deliberately owns issue labels and temporal grouping. Downstream
LLMs therefore receive pre-identified evidence episodes and must not be credited
with detecting these events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from ..frame_pipeline.strokes import StrokeBuild
from ..frame_pipeline.edt_enrichment import (
    SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM,
    SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME,
    SAFETY_COACHING_VOLUME_MIN_EDT_MM,
)
from ..models import EvidenceEvent, EvidenceStatus
from ..registry import RunRecord


RULE_VERSION = "event_rules_v1_canonical_phase"
OVERSHOOT_GAP_S = 10.0
BOUNDARY_GAP_S = 10.0
BOUNDARY_NEAR_MM = 5.0
VISIBILITY_GAP_S = 1.0
PERFORMANCE_BOUT_GAP_S = 2.0
SAFETY_COACHING_SEGMENT_GAP_S = 1.0

BOUNDARY_COLUMNS = {
    "EAC": "voxel_min_EAC_mm",
    "SinusDura": "voxel_min_SigSinus_mm",
    "TMJ": "voxel_min_TMJ_mm",
}
SENSITIVE_COLUMNS = {
    "BonyLabyrinth": "voxel_min_BonyLabyrinth_mm",
    "FacialNerve": "voxel_min_FacialNerve_mm",
    "Chorda": "voxel_min_Chorda_mm",
    "CochlearNerve": "voxel_min_CochlearNerve_mm",
    "IAC": "voxel_min_IAC_mm",
    "ICA": "voxel_min_ICA_mm",
    "SigSinus": "voxel_min_SigSinus_mm",
    "SupVestNerve": "voxel_min_SupVestNerve_mm",
    "InfVestNerve": "voxel_min_InfVestNerve_mm",
    "Stapes": "voxel_min_Stapes_mm",
}
OCCLUSION_STATES = {"not_visible", "body_occluded", "out_of_view"}
UNDER_LEDGE_LEVEL = {"none": 0, "mild": 1, "clear": 2}


@dataclass
class EventBuild:
    events: list[EvidenceEvent]
    phases: list[dict[str, Any]]
    motion_contexts: list[dict[str, Any]]
    manifest: dict[str, Any]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _split(value: Any) -> list[str]:
    text = str(value or "").replace(";", ",").replace("|", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _phase_at(t_s: float, phases: Sequence[Mapping[str, Any]]) -> str:
    for phase in phases:
        if phase["phase"] == "pre_drilling":
            continue
        start = float(phase["annotation_start_s"])
        end = float(phase["annotation_end_s"])
        if start <= t_s < end or math.isclose(t_s, end, abs_tol=1e-9):
            return str(phase["phase"])
    return "outside_annotated_phase"


def _chunks(
    rows: Iterable[dict[str, Any]],
    *,
    key_fields: Sequence[str],
    gap_s: float,
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    result: list[list[dict[str, Any]]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        current: list[dict[str, Any]] = []
        previous: float | None = None
        for row in sorted(groups[key], key=lambda item: (item["t_video_s"], item["frame_idx_global"])):
            now = float(row["t_video_s"])
            if current and previous is not None and now - previous > gap_s:
                result.append(current)
                current = []
            current.append(row)
            previous = now
        if current:
            result.append(current)
    return result


def _frame_event(
    chunk: Sequence[Mapping[str, Any]],
    *,
    record: RunRecord,
    event_type: str,
    counter: int,
    evidence_status: EvidenceStatus,
    anatomy: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> EvidenceEvent:
    times = [float(row["t_video_s"]) for row in chunk]
    evidence_ids = tuple(dict.fromkeys(str(row["frame_evidence_id"]) for row in chunk))
    return EvidenceEvent(
        event_id=f"{record.global_run_id}:event:{event_type}:{counter:04d}",
        dataset_id=record.dataset_id,
        run_id=record.run_id,
        event_type=event_type,
        t_start_s=min(times),
        t_end_s=max(times),
        evidence_status=evidence_status,
        source_evidence_ids=evidence_ids,
        rule_version=RULE_VERSION,
        phase=str(chunk[0]["phase"]),
        anatomy=anatomy,
        attributes=dict(attributes or {}),
    )


def _overshoot_attributes(
    chunk: Sequence[Mapping[str, Any]],
    *,
    record: RunRecord,
    anatomy: str,
    counter: int,
) -> dict[str, Any]:
    """Build raw-event evidence plus the stricter trainee-coaching gate."""

    distance_column = SENSITIVE_COLUMNS.get(anatomy, "")
    distances = [_finite(row.get(distance_column)) for row in chunk]
    distances = [value for value in distances if value is not None]
    coaching_rows = [
        row for row in chunk
        if anatomy in _split(row.get("voxel_safety_signal_structures"))
    ]
    coaching_chunks = _chunks(
        coaching_rows,
        key_fields=("anatomy", "phase"),
        gap_s=SAFETY_COACHING_SEGMENT_GAP_S,
    )
    count_column = f"voxel_inside_{anatomy}_count"
    coaching_segments = []
    for segment_index, segment in enumerate(coaching_chunks, 1):
        segment_distances = [
            _finite(row.get(distance_column)) for row in segment
        ]
        segment_distances = [
            value for value in segment_distances if value is not None
        ]
        def strength(row: Mapping[str, Any]) -> tuple[float, float, float]:
            inside_count = int(row.get(count_column) or 0)
            edt = _finite(row.get(distance_column))
            depth_ratio = max(0.0, -(edt or 0.0)) / abs(
                SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM
            )
            volume_ratio = (
                inside_count / SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME
            )
            return max(depth_ratio, volume_ratio), volume_ratio, depth_ratio

        anchor = max(segment, key=strength)
        anchor_count = int(anchor.get(count_column) or 0)
        anchor_edt = _finite(anchor.get(distance_column))
        volume_triggered = (
            anchor_count >= SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME
            and anchor_edt is not None
            and anchor_edt <= SAFETY_COACHING_VOLUME_MIN_EDT_MM
        )
        depth_triggered = (
            anchor_edt is not None
            and anchor_edt < SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM
        )
        trigger_branch = (
            "deep_and_volume" if depth_triggered and volume_triggered else
            "deep" if depth_triggered else "volume_with_penetration"
        )
        coaching_segments.append({
            "segment_id": (
                f"{record.global_run_id}:safety_segment:"
                f"{counter:04d}:{segment_index:02d}"
            ),
            "t_start_s": min(float(row["t_video_s"]) for row in segment),
            "t_end_s": max(float(row["t_video_s"]) for row in segment),
            "phase": segment[0].get("phase"),
            "anatomy": anatomy,
            "source_frame_count": len({
                int(row["frame_idx_global"]) for row in segment
            }),
            "max_inside_voxel_count_per_frame": max(
                int(row.get(count_column) or 0) for row in segment
            ),
            "min_edt_mm": min(segment_distances) if segment_distances else None,
            "anchor_t_s": float(anchor["t_video_s"]),
            "anchor_frame_idx_global": int(anchor["frame_idx_global"]),
            "anchor_inside_voxel_count": anchor_count,
            "anchor_min_edt_mm": anchor_edt,
            "trigger_branch": trigger_branch,
            "strength_score": round(strength(anchor)[0], 4),
        })
    return {
        "label": "voxel_edt_le_zero",
        "n_source_frames": len({row["frame_idx_global"] for row in chunk}),
        "source_frame_start": min(int(row["frame_idx_global"]) for row in chunk),
        "source_frame_end": max(int(row["frame_idx_global"]) for row in chunk),
        "removed_voxels_sum": sum(int(row["frame_removed_voxels"]) for row in chunk),
        "min_edt_mm": min(distances) if distances else None,
        "coaching_eligible": bool(coaching_segments),
        "coaching_source_frame_count": len({
            int(row["frame_idx_global"]) for row in coaching_rows
        }),
        "coaching_segments": coaching_segments,
        "coaching_gate": {
            "logic": "volume_and_minimum_penetration_or_deep_single_voxel",
            "min_inside_voxels_per_structure_per_frame": (
                SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME
            ),
            "volume_branch_max_edt_mm": SAFETY_COACHING_VOLUME_MIN_EDT_MM,
            "deep_edt_threshold_mm": SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM,
            "segment_gap_s": SAFETY_COACHING_SEGMENT_GAP_S,
            "status": "provisional_pilot_threshold_pending_expert_validation",
        },
        "claim_scope": "simulator_evidence_only",
    }


def _build_overshoot_events(frames: Sequence[Mapping[str, Any]], record: RunRecord) -> list[EvidenceEvent]:
    observations: list[dict[str, Any]] = []
    for frame in frames:
        if not bool(frame.get("voxel_overshoot")):
            continue
        for anatomy in _split(frame.get("voxel_overshoot_structures")):
            row = dict(frame)
            row["anatomy"] = anatomy
            observations.append(row)
    events = []
    for counter, chunk in enumerate(
        _chunks(observations, key_fields=("anatomy", "phase"), gap_s=OVERSHOOT_GAP_S), 1
    ):
        anatomy = str(chunk[0]["anatomy"])
        events.append(
            _frame_event(
                chunk,
                record=record,
                event_type="simulator_overshoot",
                counter=counter,
                evidence_status=EvidenceStatus.SIMULATOR_SUPPORTED,
                anatomy=anatomy,
                attributes=_overshoot_attributes(
                    chunk,
                    record=record,
                    anatomy=anatomy,
                    counter=counter,
                ),
            )
        )
    return events


def _build_boundary_events(frames: Sequence[Mapping[str, Any]], record: RunRecord) -> list[EvidenceEvent]:
    observations: list[dict[str, Any]] = []
    for frame in frames:
        if int(frame.get("frame_removed_voxels") or 0) <= 0:
            continue
        for boundary, column in BOUNDARY_COLUMNS.items():
            distance = _finite(frame.get(column))
            if distance is not None and distance <= BOUNDARY_NEAR_MM:
                row = dict(frame)
                row["boundary"] = boundary
                row["distance_mm"] = distance
                observations.append(row)
    events = []
    for counter, chunk in enumerate(
        _chunks(observations, key_fields=("boundary", "phase"), gap_s=BOUNDARY_GAP_S), 1
    ):
        boundary = str(chunk[0]["boundary"])
        minimum = min(float(row["distance_mm"]) for row in chunk)
        if minimum <= 0:
            band = "breached"
        elif minimum <= 1:
            band = "skeletonized"
        elif minimum <= 3:
            band = "approaching"
        else:
            band = "near_working_region"
        events.append(
            _frame_event(
                chunk,
                record=record,
                event_type="boundary_work",
                counter=counter,
                evidence_status=EvidenceStatus.SIMULATOR_SUPPORTED,
                anatomy=boundary,
                attributes={
                    "boundary": boundary,
                    "label": band,
                    "n_source_frames": len({row["frame_idx_global"] for row in chunk}),
                    "source_frame_start": min(int(row["frame_idx_global"]) for row in chunk),
                    "source_frame_end": max(int(row["frame_idx_global"]) for row in chunk),
                    "removed_voxels_sum": sum(int(row["frame_removed_voxels"]) for row in chunk),
                    "min_edt_mm": minimum,
                    "threshold_mm": BOUNDARY_NEAR_MM,
                    "claim_scope": "simulator_evidence_only",
                },
            )
        )
    return events


def _build_visibility_events(frames: Sequence[Mapping[str, Any]], record: RunRecord) -> list[EvidenceEvent]:
    definitions = (
        (
            "under_ledge_candidate",
            [dict(row) for row in frames if str(row.get("under_ledge") or "none") in {"mild", "clear"}],
            ("phase",),
        ),
        (
            "tip_occlusion",
            [dict(row) for row in frames if str(row.get("tip_visibility") or "") in OCCLUSION_STATES],
            ("phase", "tip_visibility"),
        ),
    )
    events: list[EvidenceEvent] = []
    for event_type, observations, keys in definitions:
        for counter, chunk in enumerate(_chunks(observations, key_fields=keys, gap_s=VISIBILITY_GAP_S), 1):
            attributes: dict[str, Any] = {
                "n_source_frames": len({row["frame_idx_global"] for row in chunk}),
                "source_frame_start": min(int(row["frame_idx_global"]) for row in chunk),
                "source_frame_end": max(int(row["frame_idx_global"]) for row in chunk),
                "review_status": "candidate_unvalidated",
                "claim_scope": "automatic_visibility_candidate_only",
            }
            if event_type == "under_ledge_candidate":
                attributes["candidate_level"] = max(
                    (str(row["under_ledge"]) for row in chunk),
                    key=lambda value: UNDER_LEDGE_LEVEL.get(value, 0),
                )
            else:
                attributes["visibility_state"] = str(chunk[0]["tip_visibility"])
            events.append(
                _frame_event(
                    chunk,
                    record=record,
                    event_type=event_type,
                    counter=counter,
                    evidence_status=EvidenceStatus.CANDIDATE,
                    attributes=attributes,
                )
            )
    return events


def _metric_by_stroke(strokes: StrokeBuild) -> dict[int, dict[str, Any]]:
    return {int(row["stroke_id"]): dict(row) for row in strokes.metrics}


def _summary(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return median(finite) if finite else None


def _build_performance_bouts(
    strokes: StrokeBuild, phases: Sequence[Mapping[str, Any]], record: RunRecord
) -> list[EvidenceEvent]:
    metrics = _metric_by_stroke(strokes)
    records = []
    for row in strokes.strokes:
        item = dict(row)
        item.update(metrics[int(row["stroke_id"])])
        midpoint = (float(row["start_time_video"]) + float(row["end_time_video"])) / 2
        item["phase"] = _phase_at(midpoint, phases)
        records.append(item)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_end: float | None = None
    previous_phase: str | None = None
    for row in sorted(records, key=lambda item: (item["start_time_video"], item["stroke_id"])):
        start = float(row["start_time_video"])
        phase = str(row["phase"])
        gap = start - previous_end if previous_end is not None else None
        if current and (phase != previous_phase or (gap is not None and gap > PERFORMANCE_BOUT_GAP_S)):
            chunks.append(current)
            current = []
        current.append(row)
        previous_end = float(row["end_time_video"])
        previous_phase = phase
    if current:
        chunks.append(current)

    events: list[EvidenceEvent] = []
    for counter, chunk in enumerate(chunks, 1):
        start = min(float(row["start_time_video"]) for row in chunk)
        end = max(float(row["end_time_video"]) for row in chunk)
        active_time = sum(float(row["duration_s"]) for row in chunk)
        removed = sum(int(row["removed_voxel_count"]) for row in chunk)
        evidence_ids = tuple(str(row["stroke_evidence_id"]) for row in chunk)
        events.append(
            EvidenceEvent(
                event_id=f"{record.global_run_id}:event:performance_bout:{counter:04d}",
                dataset_id=record.dataset_id,
                run_id=record.run_id,
                event_type="performance_bout",
                t_start_s=start,
                t_end_s=end,
                evidence_status=EvidenceStatus.SIMULATOR_SUPPORTED,
                source_evidence_ids=evidence_ids,
                rule_version=RULE_VERSION,
                phase=str(chunk[0]["phase"]),
                attributes={
                    "label": "descriptive_metrics_only",
                    "n_source_strokes": len(chunk),
                    "active_stroke_time_s": active_time,
                    "stroke_rate_per_active_s": len(chunk) / active_time if active_time > 0 else None,
                    "removed_voxels_per_active_s": removed / active_time if active_time > 0 else None,
                    "stroke_velocity_median": _summary(row.get("legacy_velocity_m_s") for row in chunk),
                    "stroke_jerk_median": _summary(row.get("legacy_jerk") for row in chunk),
                    "stroke_curvature_median": _summary(row.get("legacy_curvature") for row in chunk),
                    "efficiency_judgment": "not_calibrated",
                    "claim_scope": "descriptive_motion_productivity_evidence",
                },
            )
        )
    return events


def _build_motion_contexts(events: Sequence[EvidenceEvent], strokes: StrokeBuild) -> list[dict[str, Any]]:
    metric_lookup = _metric_by_stroke(strokes)
    contexts = []
    for event in events:
        if event.event_type not in {"simulator_overshoot", "boundary_work"}:
            continue
        overlapping = [
            row for row in strokes.strokes
            if float(row["start_time_video"]) <= event.t_end_s
            and float(row["end_time_video"]) >= event.t_start_s
        ]
        if not overlapping:
            continue
        values = [metric_lookup[int(row["stroke_id"])] for row in overlapping]
        contexts.append(
            {
                "motion_context_id": f"{event.event_id}:motion_context",
                "event_id": event.event_id,
                "phase": event.phase,
                "t_start_s": event.t_start_s,
                "t_end_s": event.t_end_s,
                "n_source_strokes": len(overlapping),
                "source_evidence_ids": [str(row["stroke_evidence_id"]) for row in overlapping],
                "stroke_velocity_median": _summary(row.get("legacy_velocity_m_s") for row in values),
                "stroke_jerk_median": _summary(row.get("legacy_jerk") for row in values),
                "stroke_curvature_median": _summary(row.get("legacy_curvature") for row in values),
                "role": "post_hoc_motion_context",
                "used_for_event_boundaries": False,
            }
        )
    return contexts


def build_events(
    strokes: StrokeBuild,
    phase_intervals: Sequence[Mapping[str, Any]],
    record: RunRecord,
) -> EventBuild:
    if not strokes.manifest.get("ready_for_eventization"):
        raise ValueError("Stroke build is not ready for eventization")
    phases = [dict(row) for row in phase_intervals if row["phase"] != "pre_drilling"]
    events = [
        *_build_overshoot_events(strokes.frames, record),
        *_build_boundary_events(strokes.frames, record),
        *_build_performance_bouts(strokes, phases, record),
        *_build_visibility_events(strokes.frames, record),
    ]
    events.sort(key=lambda event: (event.t_start_s, event.event_type, event.event_id))
    contexts = _build_motion_contexts(events, strokes)
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    coaching_safety_events = [
        event for event in events
        if event.event_type == "simulator_overshoot"
        and bool(event.attributes.get("coaching_eligible"))
    ]
    manifest = {
        "schema": "deterministic_event_build_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "rule_version": RULE_VERSION,
        "event_counts": counts,
        "motion_context_count": len(contexts),
        "safety_coaching_eligible_event_count": len(coaching_safety_events),
        "safety_coaching_segment_count": sum(
            len(event.attributes.get("coaching_segments") or [])
            for event in coaching_safety_events
        ),
        "construction_disclosure": {
            "issue_detection": "deterministic_rules",
            "issue_typing": "deterministic_rules",
            "temporal_grouping": "deterministic_rules",
            "start_end_localization": "deterministic_rules",
            "llm_used": False,
        },
        "thresholds": {
            "overshoot_gap_s": OVERSHOOT_GAP_S,
            "safety_coaching_min_inside_voxels_per_structure_per_frame": (
                SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME
            ),
            "safety_coaching_volume_branch_max_edt_mm": (
                SAFETY_COACHING_VOLUME_MIN_EDT_MM
            ),
            "safety_coaching_deep_edt_threshold_mm": (
                SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM
            ),
            "safety_coaching_segment_gap_s": SAFETY_COACHING_SEGMENT_GAP_S,
            "boundary_near_mm": BOUNDARY_NEAR_MM,
            "boundary_gap_s": BOUNDARY_GAP_S,
            "visibility_gap_s": VISIBILITY_GAP_S,
            "performance_bout_gap_s": PERFORMANCE_BOUT_GAP_S,
        },
        "threshold_origin": "legacy_pilot_reproduction; not clinically validated",
        "phase_semantics": "exact human annotation intervals retained; not clipped to last drilling frame",
        "interpretation": {
            "simulator_overshoot": "simulator-supported; does not establish clinical injury",
            "boundary_work": "working-region proximity descriptor, not an error label",
            "visibility": "automatic candidate; not human-confirmed ground truth",
            "performance_bout": "descriptive only; efficiency is not calibrated",
        },
    }
    return EventBuild(events, phases, contexts, manifest)


def event_to_dict(event: EvidenceEvent) -> dict[str, Any]:
    value = asdict(event)
    value["evidence_status"] = event.evidence_status.value
    value["lifecycle_status"] = event.lifecycle_status.value
    value["source_evidence_ids"] = list(event.source_evidence_ids)
    return value


def write_event_build(output_dir: Path, build: EventBuild) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for event in build.events:
            handle.write(json.dumps(event_to_dict(event), ensure_ascii=False) + "\n")
    (output_dir / "phase_intervals.json").write_text(
        json.dumps(build.phases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "motion_contexts.json").write_text(
        json.dumps(build.motion_contexts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "event_manifest.json").write_text(
        json.dumps(build.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

