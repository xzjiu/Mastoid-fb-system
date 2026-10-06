"""Drill-tip to anatomy EDT enrichment from raw drill and volume poses."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .core_frames import _git_commit, _require_numeric_stack
from .edt_enrichment import EdtFrameBuild, load_anatomy_profile, read_edt_file
from ..registry import RunRecord
from ..source_config import ResolvedRunSources


TIP_STRUCTURES = {
    "Bone": "Bone",
    "Malleus": "Malleus",
    "Incus": "Incus",
    "Stapes": "Stapes",
    "BonyLabyrinth": "Bony_Labyrinth",
    "IAC": "IAC",
    "SuperiorVestNerve": "Superior_Vestibular_Nerve",
    "InferiorVestNerve": "Inferior_Vestibular_Nerve",
    "CochlearNerve": "Cochlear_Nerve",
    "FacialNerve": "Facial_Nerve",
    "Chorda": "Chorda_Tympani",
    "ICA": "ICA",
    "SigSinus": "Sinus_+_Dura",
    "VestAqueduct": "Vestibular_Aqueduct",
    "TMJ": "TMJ",
    "EAC": "EAC",
}


@dataclass
class TipEdtBuild:
    frames: list[dict[str, Any]]
    removed_voxels: list[dict[str, Any]]
    manifest: dict[str, Any]


def _inverse_quaternion_rotate(vectors: Any, quaternions_xyzw: Any) -> Any:
    """Apply inverse unit-quaternion rotations to row-aligned vectors."""

    _, np = _require_numeric_stack()
    vectors = np.asarray(vectors, dtype=float)
    quaternion = np.asarray(quaternions_xyzw, dtype=float)
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Volume pose contains an invalid quaternion")
    quaternion = quaternion / norms[:, None]
    inverse_vector = -quaternion[:, :3]
    scalar = quaternion[:, 3:4]
    return vectors + 2.0 * np.cross(
        inverse_vector,
        np.cross(inverse_vector, vectors) + scalar * vectors,
    )


def _plugin_indices(local_sim: Any, dimensions: Any, resolution: tuple[int, int, int]):
    _, np = _require_numeric_stack()
    local_sim = np.asarray(local_sim, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    rx, ry, rz = resolution
    inside = np.all(np.abs(local_sim) < 0.5 * dimensions, axis=1)
    indices = np.full((len(local_sim), 3), -1, dtype=int)
    x, y, z = local_sim[:, 0], local_sim[:, 1], local_sim[:, 2]
    indices[:, 0] = np.rint((x + 0.5 * dimensions[0]) / dimensions[0] * rx).astype(int)
    indices[:, 1] = -np.rint((y - 0.5 * dimensions[1]) / dimensions[1] * ry).astype(int)
    indices[:, 2] = np.rint((z + 0.5 * dimensions[2]) / dimensions[2] * rz).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, rx - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, ry - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, rz - 1)
    return indices, inside


def enrich_tip_edt(
    edt: EdtFrameBuild,
    record: RunRecord,
    sources: ResolvedRunSources,
    *,
    project_root: Path,
) -> TipEdtBuild:
    _, np = _require_numeric_stack()
    profile = load_anatomy_profile(project_root, record.anatomy_key)
    frames = edt.frames
    drill_position = np.asarray(
        [[row["drill_x"], row["drill_y"], row["drill_z"]] for row in frames],
        dtype=float,
    )
    volume_position = np.asarray(
        [[row["volume_x"], row["volume_y"], row["volume_z"]] for row in frames],
        dtype=float,
    )
    volume_quaternion = np.asarray(
        [
            [row["volume_qx"], row["volume_qy"], row["volume_qz"], row["volume_qw"]]
            for row in frames
        ],
        dtype=float,
    )
    local_m = _inverse_quaternion_rotate(
        drill_position - volume_position, volume_quaternion
    )
    meters_per_sim = float(profile["meters_per_simulator_unit"])
    local_sim = local_m / meters_per_sim
    dimensions = profile["volume_dimensions_sim"]
    scale_mm = float(profile["edt_distance_scale_mm"])
    burr_mm = np.asarray([row["burr_mm"] for row in frames], dtype=float)

    first_path = sources.edt_dir / f"{next(iter(TIP_STRUCTURES.values()))}.edt"
    first_grid, resolution = read_edt_file(first_path)
    del first_grid
    indices, inside = _plugin_indices(local_sim, dimensions, resolution)
    ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
    distances: dict[str, Any] = {}
    for short_name, stem in TIP_STRUCTURES.items():
        grid, current_resolution = read_edt_file(sources.edt_dir / f"{stem}.edt")
        if current_resolution != resolution:
            raise ValueError(f"Tip EDT resolution mismatch for {stem}")
        values = np.full(len(frames), np.nan, dtype=float)
        values[inside] = (
            grid[iz[inside], iy[inside], ix[inside]].astype(float) * scale_mm
            - burr_mm[inside] * 0.5
        )
        distances[short_name] = values
        del grid

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(frames):
        row = dict(source)
        row["p_local_sim_x"] = float(local_sim[index, 0])
        row["p_local_sim_y"] = float(local_sim[index, 1])
        row["p_local_sim_z"] = float(local_sim[index, 2])
        for short_name in TIP_STRUCTURES:
            value = float(distances[short_name][index])
            row[f"idx_{short_name}_x"] = int(ix[index]) if inside[index] else None
            row[f"idx_{short_name}_y"] = int(iy[index]) if inside[index] else None
            row[f"idx_{short_name}_z"] = int(iz[index]) if inside[index] else None
            row[f"dist_{short_name}_mm"] = value if math.isfinite(value) else None
        rows.append(row)

    outside_count = int(np.sum(~inside))
    manifest = {
        "schema": "tip_edt_enrichment_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "upstream_schema": edt.manifest["schema"],
        "frame_count": len(rows),
        "edt_profile": record.edt_profile,
        "structures": list(TIP_STRUCTURES),
        "grid_resolution_xyz": list(resolution),
        "meters_per_simulator_unit": meters_per_sim,
        "volume_dimensions_sim": dimensions,
        "tip_outside_volume_frame_count": outside_count,
        "tip_inside_volume_frame_count": int(np.sum(inside)),
        "distance_definition": "EDT_raw * edt_distance_scale_mm - burr_mm / 2",
        "invariants": {
            "upstream_ready": bool(edt.manifest["ready_for_visibility_enrichment"]),
            "frame_count_preserved": len(rows) == len(frames),
            "all_tip_grids_loaded": len(distances) == len(TIP_STRUCTURES),
            "finite_local_coordinates": bool(np.all(np.isfinite(local_sim))),
        },
    }
    manifest["ready_for_visibility"] = all(manifest["invariants"].values())
    return TipEdtBuild(rows, edt.removed_voxels, manifest)


def compare_legacy_tip_edt(build: TipEdtBuild, legacy_path: Path) -> dict[str, Any]:
    if not legacy_path.is_file():
        return {"available": False, "legacy_path": str(legacy_path)}
    with legacy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        legacy = list(csv.DictReader(handle))
    if len(legacy) != len(build.frames):
        return {"available": True, "row_count_match": False, "all_match": False}
    numeric_columns = ["p_local_sim_x", "p_local_sim_y", "p_local_sim_z"]
    for name in TIP_STRUCTURES:
        numeric_columns.extend(
            [f"idx_{name}_x", f"idx_{name}_y", f"idx_{name}_z", f"dist_{name}_mm"]
        )
    mismatches = {column: 0 for column in numeric_columns}
    max_error = {column: 0.0 for column in numeric_columns}

    def number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    differing_frames: set[int] = set()
    for new, old in zip(build.frames, legacy):
        for column in numeric_columns:
            left, right = number(new.get(column)), number(old.get(column))
            if left is None or right is None:
                mismatch = left is not None or right is not None
                error = 0.0
            else:
                error = abs(left - right)
                mismatch = error > 1e-9
            max_error[column] = max(max_error[column], error)
            if mismatch:
                mismatches[column] += 1
                differing_frames.add(int(new["frame_idx_global"]))
    coordinate_columns = {"p_local_sim_x", "p_local_sim_y", "p_local_sim_z"}
    edt_columns = set(numeric_columns) - coordinate_columns
    edt_fields_exact = not any(mismatches[column] for column in edt_columns)
    coordinate_max_error = max(max_error[column] for column in coordinate_columns)
    coordinate_within_tolerance = coordinate_max_error <= 2e-7
    return {
        "available": True,
        "legacy_path": str(legacy_path),
        "role": "validation_only",
        "row_count_match": True,
        "mismatches": mismatches,
        "max_absolute_errors": max_error,
        "differing_frame_indices": sorted(differing_frames),
        "differing_frame_count": len(differing_frames),
        "edt_indices_and_distances_exact": edt_fields_exact,
        "local_coordinate_max_absolute_error": coordinate_max_error,
        "local_coordinates_within_2e_7": coordinate_within_tolerance,
        "legacy_equivalent": edt_fields_exact and coordinate_within_tolerance,
        "all_match": not any(mismatches.values()),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tip_edt_build(output_dir: Path, build: TipEdtBuild, regression: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frame_observations_tip_edt.csv", build.frames)
    (output_dir / "tip_edt_manifest.json").write_text(
        json.dumps(build.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "legacy_tip_edt_regression.json").write_text(
        json.dumps(regression, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Tip EDT Audit: {build.manifest['run_id']}",
        "",
        f"- Frames: {build.manifest['frame_count']}",
        f"- Tip inside volume: {build.manifest['tip_inside_volume_frame_count']}",
        f"- Tip outside volume: {build.manifest['tip_outside_volume_frame_count']}",
        f"- Exact legacy regression: **{regression.get('all_match')}**",
        f"- EDT indices and distances exact: **{regression.get('edt_indices_and_distances_exact')}**",
        f"- Local coordinates within 2e-7: **{regression.get('local_coordinates_within_2e_7')}**",
        f"- Legacy-equivalent result: **{regression.get('legacy_equivalent')}**",
        f"- Frames with local-coordinate differences above 1e-9: {regression.get('differing_frame_count')}",
        "",
        "The legacy table is validation-only and was opened after construction.",
    ]
    (output_dir / "TIP_EDT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

