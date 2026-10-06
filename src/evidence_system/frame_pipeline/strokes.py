"""Deterministic stroke segmentation and evidence aggregation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .core_frames import _git_commit, _require_numeric_stack
from .tip_edt import TIP_STRUCTURES
from .visibility import VisibilityBuild
from ..registry import RunRecord


@dataclass
class StrokeBuild:
    frames: list[dict[str, Any]]
    strokes: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    removed_voxels: list[dict[str, Any]]
    manifest: dict[str, Any]


def detect_stroke_ends(
    drill_positions: Any,
    drilling_times: Any,
    *,
    neighbor_k: int = 3,
) -> Any:
    _, np = _require_numeric_stack()
    positions = np.asarray(drill_positions, dtype=float)
    times = np.asarray(drilling_times, dtype=float)
    if len(positions) < 2 * neighbor_k + 1:
        raise ValueError("Too few drilling frames for stroke segmentation")
    k_cosines: list[float] = []
    for pivot in range(len(positions)):
        if pivot - neighbor_k < 0 or pivot + neighbor_k >= len(positions):
            continue
        before = positions[pivot] - positions[pivot - neighbor_k]
        after = positions[pivot] - positions[pivot + neighbor_k]
        denominator = np.linalg.norm(before) * np.linalg.norm(after)
        cosine = np.dot(before, after) / denominator
        theta = np.arccos(cosine) * 180.0 / np.pi
        k_cosines.append(float(180.0 - theta))
    mean = float(np.mean(k_cosines))
    standard_deviation = float(np.std(k_cosines))
    padded = [mean] * neighbor_k + k_cosines + [mean] * neighbor_k
    stroke_ends = np.asarray(
        [1 if value > mean + standard_deviation else 0 for value in padded],
        dtype=int,
    )
    differences = np.diff(positions, axis=0)
    speeds = np.linalg.norm(differences, axis=1) / np.diff(times)
    speeds = np.insert(speeds, 0, 0.0)
    pivots = np.where(stroke_ends == 1)[0]
    index = 0
    while index < len(pivots) - 1:
        if pivots[index] + 1 == pivots[index + 1]:
            start = index
            while index < len(pivots) - 1 and pivots[index] + 1 == pivots[index + 1]:
                index += 1
            end = index
            selected = int(np.argmin(speeds[pivots[start : end + 1]]))
            for candidate in range(start, end + 1):
                if candidate != start + selected:
                    stroke_ends[pivots[candidate]] = 0
        index += 1
    stroke_ends[-1] = 1
    return stroke_ends


def _legacy_kinematics(positions: Any, times: Any, stroke_ends: Any) -> dict[str, Any]:
    _, np = _require_numeric_stack()
    from scipy import integrate

    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    stroke_ends = np.asarray(stroke_ends, dtype=int)

    end_indices = np.where(stroke_ends == 1)[0]
    legacy_length_boundaries = np.insert(end_indices, 0, 0)
    lengths = []
    for index in range(int(stroke_ends.sum())):
        segment = positions[
            legacy_length_boundaries[index] : legacy_length_boundaries[index + 1]
        ]
        lengths.append(float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()))

    starts = np.insert(end_indices + 1, 0, 0)[:-1]
    velocities, accelerations, jerks, curvatures = [], [], [], []
    for index, start in enumerate(starts):
        end_exclusive = int(starts[index + 1]) if index + 1 < len(starts) else len(times)
        segment = positions[int(start) : end_exclusive]
        segment_times = times[int(start) : end_exclusive]
        duration = float(np.ptp(segment_times))
        vx = np.gradient(segment[:, 0], segment_times)
        vy = np.gradient(segment[:, 1], segment_times)
        vz = np.gradient(segment[:, 2], segment_times)
        path = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
        velocities.append(path / duration)
        velocity_vectors = np.column_stack((vx, vy, vz))
        accelerations.append(
            float(np.linalg.norm(np.diff(velocity_vectors, axis=0), axis=1).sum()) / duration
        )
        ax = np.gradient(vx, segment_times)
        ay = np.gradient(vy, segment_times)
        az = np.gradient(vz, segment_times)
        acceleration_vectors = np.column_stack((ax, ay, az))
        jerks.append(
            float(np.linalg.norm(np.diff(acceleration_vectors, axis=0), axis=1).sum())
            / duration
        )
        curvature_values = []
        curvature_times = list(segment_times)
        for sample in range(len(segment_times)):
            first = np.asarray([vx[sample], vy[sample], vz[sample]])
            second = np.asarray([ax[sample], ay[sample], az[sample]])
            if np.linalg.norm(first) == 0:
                curvature_times.pop(sample)
                continue
            curvature_values.append(
                float(np.linalg.norm(np.cross(first, second)) / np.linalg.norm(first) ** 3)
            )
        curvatures.append(
            float(integrate.simpson(curvature_values, x=curvature_times))
            / float(np.ptp(curvature_times))
        )
    return {
        "Length": np.asarray(lengths),
        "Velocity": np.asarray(velocities),
        "Acceleration": np.asarray(accelerations),
        "Jerk": np.asarray(jerks),
        "Curvature": np.asarray(curvatures),
    }


def build_strokes(
    visibility: VisibilityBuild,
    record: RunRecord,
    *,
    project_root: Path,
) -> StrokeBuild:
    _, np = _require_numeric_stack()
    frames = [dict(row) for row in visibility.frames]
    drilling_global_indices = np.asarray(
        [index for index, row in enumerate(frames) if row["is_drilling_frame"]],
        dtype=int,
    )
    positions = np.asarray(
        [
            [frames[index]["drill_x"], frames[index]["drill_y"], frames[index]["drill_z"]]
            for index in drilling_global_indices
        ],
        dtype=float,
    )
    times_abs = np.asarray([frames[index]["t_abs"] for index in drilling_global_indices])
    times_video = np.asarray([frames[index]["t_video_s"] for index in drilling_global_indices])
    stroke_ends = detect_stroke_ends(positions, times_abs)
    end_indices = np.where(stroke_ends == 1)[0]
    start_indices = np.insert(end_indices[:-1] + 1, 0, 0)
    boundary_end_abs = times_abs[end_indices]
    boundary_start_abs = np.insert(boundary_end_abs[:-1], 0, np.min(times_abs))
    legacy_metrics = _legacy_kinematics(positions, times_abs, stroke_ends)

    for row in frames:
        row["stroke_id"] = None
        row["stroke_end_flag"] = False
    for stroke_id, (start, end) in enumerate(zip(start_indices, end_indices)):
        globals_for_stroke = drilling_global_indices[int(start) : int(end) + 1]
        for global_index in globals_for_stroke:
            frames[int(global_index)]["stroke_id"] = stroke_id
        frames[int(drilling_global_indices[int(end)])]["stroke_end_flag"] = True

    stroke_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    risk_names = [name for name in TIP_STRUCTURES if name != "Bone"]
    for stroke_id, (start, end, boundary_start, boundary_end) in enumerate(
        zip(start_indices, end_indices, boundary_start_abs, boundary_end_abs)
    ):
        start, end = int(start), int(end)
        global_indices = drilling_global_indices[start : end + 1]
        segment_frames = [frames[int(index)] for index in global_indices]
        start_abs, end_abs = float(times_abs[start]), float(times_abs[end])
        start_video, end_video = float(times_video[start]), float(times_video[end])
        stroke_rows.append(
            {
                "stroke_evidence_id": f"{record.global_run_id}:stroke:{stroke_id:04d}",
                "global_run_id": record.global_run_id,
                "stroke_id": stroke_id,
                "start_idx": start,
                "end_idx": end,
                "start_frame_idx_global": int(global_indices[0]),
                "end_frame_idx_global": int(global_indices[-1]),
                "start_time_abs": start_abs,
                "end_time_abs": end_abs,
                "start_time_video": start_video,
                "end_time_video": end_video,
                "boundary_start_abs": float(boundary_start),
                "boundary_end_abs": float(boundary_end),
                "boundary_start_video": float(boundary_start - frames[0]["t_abs"]),
                "boundary_end_video": float(boundary_end - frames[0]["t_abs"]),
                "duration_s": end_abs - start_abs,
                "n_frames": end - start + 1,
            }
        )
        phase_counts: dict[str, int] = {}
        for row in segment_frames:
            phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
        dominant_phase = max(phase_counts, key=phase_counts.get)
        segment_positions = positions[start : end + 1]
        direct_path = float(np.linalg.norm(np.diff(segment_positions, axis=0), axis=1).sum())
        duration = end_abs - start_abs
        finite_tip_risks = [
            float(row[f"dist_{name}_mm"])
            for row in segment_frames
            for name in risk_names
            if row.get(f"dist_{name}_mm") is not None
        ]
        finite_voxel_risks = [
            float(row["voxel_min_sensitive_mm"])
            for row in segment_frames
            if row.get("voxel_min_sensitive_mm") is not None
        ]
        metric_rows.append(
            {
                "stroke_evidence_id": f"{record.global_run_id}:stroke:{stroke_id:04d}",
                "global_run_id": record.global_run_id,
                "stroke_id": stroke_id,
                "stroke_end_time_abs": end_abs,
                "stroke_end_time_video": end_video,
                "legacy_length_m": float(legacy_metrics["Length"][stroke_id]),
                "legacy_velocity_m_s": float(legacy_metrics["Velocity"][stroke_id]),
                "legacy_acceleration": float(legacy_metrics["Acceleration"][stroke_id]),
                "legacy_jerk": float(legacy_metrics["Jerk"][stroke_id]),
                "legacy_curvature": float(legacy_metrics["Curvature"][stroke_id]),
                "direct_path_length_m": direct_path,
                "direct_mean_speed_m_s": direct_path / duration if duration > 0 else None,
                "removed_voxel_count": sum(int(row["frame_removed_voxels"]) for row in segment_frames),
                "removed_volume_mm3": sum(float(row["frame_removed_mm3"]) for row in segment_frames),
                "dominant_phase": dominant_phase,
                "phase_counts": json.dumps(phase_counts, sort_keys=True),
                "visible_frame_fraction": sum(bool(row["tip_visible"]) for row in segment_frames)
                / len(segment_frames),
                "frozen_frame_fraction": sum(bool(row["pose_frozen"]) for row in segment_frames)
                / len(segment_frames),
                "under_ledge_candidate_frame_count": sum(
                    row["under_ledge"] in {"mild", "clear"} for row in segment_frames
                ),
                "voxel_overshoot_frame_count": sum(bool(row["voxel_overshoot"]) for row in segment_frames),
                "min_tip_sensitive_mm": min(finite_tip_risks) if finite_tip_risks else None,
                "min_removed_voxel_sensitive_mm": min(finite_voxel_risks) if finite_voxel_risks else None,
            }
        )

    manifest = {
        "schema": "stroke_evidence_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "upstream_schema": visibility.manifest["schema"],
        "frame_count": len(frames),
        "drilling_frame_count": len(drilling_global_indices),
        "stroke_count": len(stroke_rows),
        "segmentation": {
            "method": "legacy_k_cosine_v1",
            "neighbor_k": 3,
            "pivot_threshold": "mean_plus_one_standard_deviation",
            "consecutive_pivot_tiebreak": "minimum_speed",
        },
        "invariants": {
            "upstream_ready": bool(visibility.manifest["ready_for_strokes"]),
            "all_drilling_frames_assigned": sum(row["stroke_id"] is not None for row in frames)
            == len(drilling_global_indices),
            "non_drilling_frames_unassigned": all(
                row["stroke_id"] is None for row in frames if not row["is_drilling_frame"]
            ),
            "last_drilling_frame_closes_stroke": bool(stroke_ends[-1]),
            "stroke_metric_count_matches": len(metric_rows) == len(stroke_rows),
        },
    }
    manifest["ready_for_eventization"] = all(manifest["invariants"].values())
    return StrokeBuild(frames, stroke_rows, metric_rows, visibility.removed_voxels, manifest)


def _compare_rows(
    new_rows: Sequence[dict[str, Any]],
    legacy_path: Path,
    mapping: dict[str, str],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    if not legacy_path.is_file():
        return {"available": False, "legacy_path": str(legacy_path)}
    with legacy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        legacy = list(csv.DictReader(handle))
    if len(legacy) != len(new_rows):
        return {
            "available": True, "row_count_match": False,
            "new_row_count": len(new_rows), "legacy_row_count": len(legacy), "all_match": False,
        }
    mismatches = {new_name: 0 for new_name in mapping}
    max_error = {new_name: 0.0 for new_name in mapping}
    for new, old in zip(new_rows, legacy):
        for new_name, old_name in mapping.items():
            left, right = float(new[new_name]), float(old[old_name])
            error = abs(left - right)
            max_error[new_name] = max(max_error[new_name], error)
            if error > tolerance:
                mismatches[new_name] += 1
    return {
        "available": True,
        "legacy_path": str(legacy_path),
        "role": "validation_only",
        "row_count_match": True,
        "mismatches": mismatches,
        "max_absolute_errors": max_error,
        "all_match": not any(mismatches.values()),
    }


def compare_legacy_strokes(build: StrokeBuild, raw_run_dir: Path) -> dict[str, Any]:
    table_mapping = {
        "stroke_id": "stroke_id", "start_idx": "start_idx", "end_idx": "end_idx",
        "start_time_abs": "start_time_abs", "end_time_abs": "end_time_abs",
        "start_time_video": "start_time_video", "end_time_video": "end_time_video",
        "boundary_start_abs": "boundary_start_abs", "boundary_end_abs": "boundary_end_abs",
        "boundary_start_video": "boundary_start_video", "boundary_end_video": "boundary_end_video",
        "duration_s": "duration_s", "n_frames": "n_frames",
    }
    metric_mapping = {
        "stroke_id": "stroke_id",
        "stroke_end_time_abs": "stroke_end_time_abs",
        "stroke_end_time_video": "Video Time",
        "legacy_length_m": "Length",
        "legacy_velocity_m_s": "Velocity",
        "legacy_acceleration": "Acceleration",
        "legacy_jerk": "Jerk",
        "legacy_curvature": "Curvature",
    }
    table = _compare_rows(build.strokes, raw_run_dir / "stroke_table_raw.csv", table_mapping)
    metrics = _compare_rows(
        build.metrics,
        raw_run_dir / "timed_stroke_metrics_in_memory.csv",
        metric_mapping,
        tolerance=1e-8,
    )
    return {"stroke_table": table, "legacy_kinematics": metrics}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_stroke_build(output_dir: Path, build: StrokeBuild, regression: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frame_observations_complete.csv", build.frames)
    _write_csv(output_dir / "strokes.csv", build.strokes)
    _write_csv(output_dir / "stroke_metrics.csv", build.metrics)
    (output_dir / "stroke_manifest.json").write_text(
        json.dumps(build.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "legacy_stroke_regression.json").write_text(
        json.dumps(regression, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Stroke Audit: {build.manifest['run_id']}", "",
        f"- Drilling frames: {build.manifest['drilling_frame_count']}",
        f"- Strokes: {build.manifest['stroke_count']}",
        f"- Stroke-table exact regression: **{regression['stroke_table'].get('all_match')}**",
        f"- Legacy-kinematics regression: **{regression['legacy_kinematics'].get('all_match')}**",
        f"- Ready for eventization: **{build.manifest['ready_for_eventization']}**",
    ]
    (output_dir / "STROKE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

