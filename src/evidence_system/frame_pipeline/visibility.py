"""RGB/depth tip-visibility and under-ledge candidate enrichment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .core_frames import _git_commit, _require_numeric_stack
from .tip_edt import TipEdtBuild
from ..registry import RunRecord
from ..source_config import ResolvedRunSources


SEARCH_RADIUS = 70
DARK_THRESHOLD = 50
MIN_TIP_AREA = 5
MAX_TIP_AREA = 15000
MAX_ASPECT = 2.6
MIN_TIP_EXTENT = 0.25
BODY_AREA = 25000
BODY_CENTROID_MAX = 90
DEPTH_MARGIN = 0.005
LEDGE_MIN_TIP_AREA = 30
BONE_R = 180
BONE_G = 180
BONE_B = 150
MILD_OVERHANG = 0.10
CLEAR_OVERHANG = 0.25
MILD_PIXELS = 5
CLEAR_PIXELS = 8

VISIBILITY_COLUMNS = [
    "tip_visibility",
    "tip_visible",
    "pose_frozen",
    "tip_proj_u",
    "tip_proj_v",
    "tip_u",
    "tip_v",
    "tip_area_px",
    "tip_depth",
    "bone_edge_frac",
    "overhang_frac",
    "bone_nearer_px",
    "under_ledge",
]


@dataclass
class VisibilityBuild:
    frames: list[dict[str, Any]]
    removed_voxels: list[dict[str, Any]]
    manifest: dict[str, Any]


def _require_vision_stack() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("Visibility enrichment requires opencv-python and scipy") from exc
    return cv2, np, Rotation


def project_tip(drill_pose: Any, camera_pose: Any, extrinsic: Any, intrinsic: Any):
    _, np, Rotation = _require_vision_stack()
    tip_world = drill_pose[:3]
    camera_rotation = Rotation.from_quat(camera_pose[3:])
    local = camera_rotation.inv().apply(tip_world - camera_pose[:3])
    camera_point = extrinsic[:3, :3].astype(float) @ local
    if camera_point[2] <= 0:
        return None
    u = intrinsic[0, 0] * camera_point[0] / camera_point[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_point[1] / camera_point[2] + intrinsic[1, 2]
    return int(round(u)), int(round(v))


def find_tip_blob(rgb: Any, projection: tuple[int, int]):
    cv2, np, _ = _require_vision_stack()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    labels_count, _, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    projected_u, projected_v = projection
    best_center, best_area, best_distance = None, 0, np.inf
    body_occluded = False
    for label in range(1, labels_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= BODY_AREA:
            center_x, center_y = centroids[label]
            if np.hypot(center_x - projected_u, center_y - projected_v) < BODY_CENTROID_MAX:
                body_occluded = True
    for label in range(1, labels_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_TIP_AREA or area > MAX_TIP_AREA:
            continue
        center_x, center_y = centroids[label]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(width, height) / (min(width, height) + 1e-6)
        extent = area / (width * height + 1e-6)
        if aspect > MAX_ASPECT or extent < MIN_TIP_EXTENT:
            continue
        distance = np.hypot(center_x - projected_u, center_y - projected_v)
        if distance < best_distance and distance < SEARCH_RADIUS:
            best_center = int(center_x), int(center_y)
            best_area = int(area)
            best_distance = distance
    return best_center, best_area, best_distance, body_occluded


def classify_frame(
    rgb: Any,
    drill_pose: Any,
    camera_pose: Any,
    extrinsic: Any,
    intrinsic: Any,
    previous_drill_pose: Any | None,
) -> dict[str, Any]:
    _, np, _ = _require_vision_stack()
    frozen = previous_drill_pose is not None and np.abs(
        drill_pose[:3] - previous_drill_pose[:3]
    ).sum() < 1e-9
    projection = project_tip(drill_pose, camera_pose, extrinsic, intrinsic)
    if projection is None:
        return {
            "category": "no_drill", "projection": None, "tip_center": None,
            "tip_area": 0, "frozen": bool(frozen),
        }
    height, width = rgb.shape[:2]
    u, v = projection
    if u < -10 or u > width + 10 or v < -10 or v > height + 10:
        return {
            "category": "out_of_view", "projection": projection, "tip_center": None,
            "tip_area": 0, "frozen": bool(frozen),
        }
    center, area, _, body_occluded = find_tip_blob(rgb, projection)
    category = "body_occluded" if body_occluded else ("visible" if center else "not_visible")
    return {
        "category": category,
        "projection": projection,
        "tip_center": center,
        "tip_area": area,
        "frozen": bool(frozen),
    }


def overhang_metrics(rgb: Any, depth: Any, tip_center: tuple[int, int]):
    cv2, np, _ = _require_vision_stack()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    _, labels, _, _ = cv2.connectedComponentsWithStats(dark, 8)
    tip_label = labels[tip_center[1], tip_center[0]]
    if tip_label == 0:
        return None
    tip_mask = labels == tip_label
    if int(tip_mask.sum()) < LEDGE_MIN_TIP_AREA:
        return None
    mask_u8 = tip_mask.astype(np.uint8)
    edge = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8)) - mask_u8
    edge_y, edge_x = np.where(edge > 0)
    if not len(edge_y):
        return None
    blue, green, red = cv2.split(rgb)
    bone = (red > BONE_R) & (green > BONE_G) & (blue > BONE_B)
    tip_depth = float(np.nanmedian(depth[tip_mask]))
    bone_edge_mask = bone[edge_y, edge_x]
    edge_depth = depth[edge_y, edge_x]
    bone_finite = bone_edge_mask & np.isfinite(edge_depth)
    bone_pixels = int(bone_finite.sum())
    nearer = int((bone_finite & (edge_depth < tip_depth - DEPTH_MARGIN)).sum())
    return {
        "tip_depth": round(tip_depth, 4),
        "bone_edge_frac": round(int(bone_edge_mask.sum()) / len(edge_y), 3),
        "overhang_frac": round(nearer / bone_pixels, 3) if bone_pixels else 0.0,
        "bone_nearer_px": nearer,
    }


def under_ledge_severity(overhang_fraction: float, nearer_pixels: int) -> str:
    if overhang_fraction >= CLEAR_OVERHANG and nearer_pixels >= CLEAR_PIXELS:
        return "clear"
    if overhang_fraction >= MILD_OVERHANG and nearer_pixels >= MILD_PIXELS:
        return "mild"
    return "none"


def enrich_visibility(
    tip: TipEdtBuild,
    record: RunRecord,
    sources: ResolvedRunSources,
    *,
    project_root: Path,
) -> VisibilityBuild:
    h5py, _ = _require_numeric_stack()
    _, np, _ = _require_vision_stack()
    rows = [dict(row) for row in tip.frames]
    # Reuse the exact frame-bearing chunk set accepted by core extraction.
    # Recorder shutdown can leave a valid metadata-only terminal HDF5.
    source_names = list(dict.fromkeys(str(row["source_hdf5"]) for row in rows))
    hdf5_paths = [sources.raw_run_dir / name for name in source_names]
    first_removal = np.inf
    for path in hdf5_paths:
        with h5py.File(path, "r") as handle:
            timestamps = handle.get("voxels_removed/voxel_time_stamp")
            if timestamps is not None and len(timestamps):
                first_removal = min(first_removal, float(np.min(timestamps[()])))
    first_removal_time = float(first_removal) if np.isfinite(first_removal) else None

    previous_pose = None
    global_index = 0
    for path in hdf5_paths:
        with h5py.File(path, "r") as handle:
            times = handle["data/time"][()]
            intrinsic = handle["metadata/camera_intrinsic"][()]
            extrinsic = handle["metadata/camera_extrinsic"][()]
            drill_poses = handle["data/pose_mastoidectomy_drill"][()]
            camera_poses = handle["data/pose_main_camera"][()]
            rgb_frames = handle["data/l_img"][()]
            depth_frames = handle["data/depth"][()]
            for source_index in range(len(times)):
                if global_index >= len(rows):
                    raise ValueError("Raw HDF5 contains more visibility frames than core frames")
                row = rows[global_index]
                if row["source_hdf5"] != path.name or row["source_frame_idx"] != source_index:
                    raise ValueError(f"Visibility/core alignment mismatch at frame {global_index}")
                defaults = {
                    "tip_visibility": "pre_drill",
                    "tip_visible": False,
                    "pose_frozen": False,
                    "tip_proj_u": -1,
                    "tip_proj_v": -1,
                    "tip_u": -1,
                    "tip_v": -1,
                    "tip_area_px": 0,
                    "tip_depth": None,
                    "bone_edge_frac": None,
                    "overhang_frac": None,
                    "bone_nearer_px": -1,
                    "under_ledge": "",
                    "visibility_status": "automatic_candidate",
                }
                row.update(defaults)
                current_pose = drill_poses[source_index]
                if first_removal_time is not None and times[source_index] >= first_removal_time:
                    rgb = rgb_frames[source_index]
                    result = classify_frame(
                        rgb,
                        current_pose,
                        camera_poses[source_index],
                        extrinsic,
                        intrinsic,
                        previous_pose,
                    )
                    row["tip_visibility"] = result["category"]
                    row["tip_visible"] = result["category"] == "visible"
                    row["pose_frozen"] = result["frozen"]
                    if result["projection"] is not None:
                        row["tip_proj_u"], row["tip_proj_v"] = result["projection"]
                    if result["tip_center"] is not None:
                        row["tip_u"], row["tip_v"] = result["tip_center"]
                    row["tip_area_px"] = int(result["tip_area"])
                    if (
                        result["category"] == "visible"
                        and not result["frozen"]
                        and result["tip_center"] is not None
                    ):
                        depth = depth_frames[source_index].astype(np.float32)
                        metrics = overhang_metrics(rgb, depth, result["tip_center"])
                        if metrics is None:
                            row["under_ledge"] = "none"
                        else:
                            row.update(metrics)
                            row["under_ledge"] = under_ledge_severity(
                                metrics["overhang_frac"], metrics["bone_nearer_px"]
                            )
                previous_pose = current_pose
                global_index += 1
    if global_index != len(rows):
        raise ValueError(f"Visibility/core frame count mismatch: {global_index}/{len(rows)}")

    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["tip_visibility"]] = category_counts.get(row["tip_visibility"], 0) + 1
        severity_counts[row["under_ledge"]] = severity_counts.get(row["under_ledge"], 0) + 1
    manifest = {
        "schema": "tip_visibility_enrichment_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "upstream_schema": tip.manifest["schema"],
        "frame_count": len(rows),
        "source_hdf5_file_count": len(hdf5_paths),
        "source_hdf5_files": source_names,
        "first_removal_time_abs": first_removal_time,
        "category_counts": category_counts,
        "under_ledge_counts": severity_counts,
        "frozen_frame_count": sum(bool(row["pose_frozen"]) for row in rows),
        "manual_override_count": 0,
        "under_ledge_evidence_status": "automatic_depth_candidate_not_human_confirmed",
        "thresholds": {
            "search_radius_px": SEARCH_RADIUS,
            "dark_threshold": DARK_THRESHOLD,
            "depth_margin": DEPTH_MARGIN,
            "mild_overhang_fraction": MILD_OVERHANG,
            "clear_overhang_fraction": CLEAR_OVERHANG,
            "mild_pixels": MILD_PIXELS,
            "clear_pixels": CLEAR_PIXELS,
        },
        "invariants": {
            "upstream_ready": bool(tip.manifest["ready_for_visibility"]),
            "frame_count_preserved": len(rows) == len(tip.frames),
            "all_frames_have_source_alignment": global_index == len(rows),
        },
    }
    manifest["ready_for_strokes"] = all(manifest["invariants"].values())
    return VisibilityBuild(rows, tip.removed_voxels, manifest)


def compare_legacy_visibility(build: VisibilityBuild, legacy_path: Path) -> dict[str, Any]:
    if not legacy_path.is_file():
        return {"available": False, "legacy_path": str(legacy_path)}
    with legacy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        legacy = list(csv.DictReader(handle))
    if len(legacy) != len(build.frames):
        return {"available": True, "row_count_match": False, "all_match": False}
    string_columns = ["tip_visibility", "under_ledge"]
    bool_columns = ["tip_visible", "pose_frozen"]
    numeric_columns = [column for column in VISIBILITY_COLUMNS if column not in string_columns + bool_columns]
    mismatches = {column: 0 for column in VISIBILITY_COLUMNS}
    max_error = {column: 0.0 for column in numeric_columns}
    differing_frames: set[int] = set()

    def normalized_string(value: Any) -> str:
        return "" if value is None else str(value).strip()

    def boolean(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "t", "yes"}

    def number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    for new, old in zip(build.frames, legacy):
        for column in string_columns:
            if normalized_string(new.get(column)) != normalized_string(old.get(column)):
                mismatches[column] += 1
                differing_frames.add(int(new["frame_idx_global"]))
        for column in bool_columns:
            if bool(new.get(column)) != boolean(old.get(column)):
                mismatches[column] += 1
                differing_frames.add(int(new["frame_idx_global"]))
        for column in numeric_columns:
            left, right = number(new.get(column)), number(old.get(column))
            if left is None or right is None:
                mismatch, error = left is not None or right is not None, 0.0
            else:
                error = abs(left - right)
                mismatch = error > 1e-9
            max_error[column] = max(max_error[column], error)
            if mismatch:
                mismatches[column] += 1
                differing_frames.add(int(new["frame_idx_global"]))
    return {
        "available": True,
        "legacy_path": str(legacy_path),
        "role": "validation_only",
        "row_count_match": True,
        "mismatches": mismatches,
        "max_absolute_errors": max_error,
        "differing_frame_indices": sorted(differing_frames),
        "differing_frame_count": len(differing_frames),
        "all_match": not any(mismatches.values()),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_visibility_build(output_dir: Path, build: VisibilityBuild, regression: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frame_observations_visibility.csv", build.frames)
    (output_dir / "visibility_manifest.json").write_text(
        json.dumps(build.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "legacy_visibility_regression.json").write_text(
        json.dumps(regression, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Visibility Audit: {build.manifest['run_id']}", "",
        f"- Category counts: {build.manifest['category_counts']}",
        f"- Under-ledge candidate counts: {build.manifest['under_ledge_counts']}",
        f"- Frozen frames: {build.manifest['frozen_frame_count']}",
        f"- Manual overrides: {build.manifest['manual_override_count']}",
        f"- Exact legacy regression: **{regression.get('all_match')}**",
        f"- Differing frames: {regression.get('differing_frame_count')}", "",
        "Under-ledge values are automatic depth candidates, not human-confirmed labels.",
    ]
    (output_dir / "VISIBILITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

