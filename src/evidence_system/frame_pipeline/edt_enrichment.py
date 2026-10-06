"""Add voxel-level anatomy EDT evidence without reading legacy frame tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from .core_frames import CoreFrameBuild, _git_commit, _require_numeric_stack
from ..registry import RunRecord
from ..source_config import ResolvedRunSources


SENSITIVE_STRUCTURES = {
    "Bony_Labyrinth": "BonyLabyrinth",
    "Facial_Nerve": "FacialNerve",
    "Chorda_Tympani": "Chorda",
    "Cochlear_Nerve": "CochlearNerve",
    "IAC": "IAC",
    "ICA": "ICA",
    "Sinus_+_Dura": "SigSinus",
    "Superior_Vestibular_Nerve": "SupVestNerve",
    "Inferior_Vestibular_Nerve": "InfVestNerve",
    "Stapes": "Stapes",
}

BOUNDARY_STRUCTURES = {"EAC": "EAC", "TMJ": "TMJ"}

# Provisional pilot-study promotion gate. The legacy EDT<=0 signal is retained
# for source fidelity, while trainee-facing safety coaching requires either a
# spatially substantial incursion in one frame or a deeper single voxel.
SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME = 10
SAFETY_COACHING_VOLUME_MIN_EDT_MM = -0.1
SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM = -1.0


def passes_safety_coaching_gate(
    inside_voxel_count: int, min_edt_mm: float | None
) -> bool:
    """Return whether one structure/frame passes the provisional coaching gate."""

    if min_edt_mm is None or not math.isfinite(float(min_edt_mm)):
        return False
    minimum = float(min_edt_mm)
    return (
        (
            int(inside_voxel_count)
            >= SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME
            and minimum <= SAFETY_COACHING_VOLUME_MIN_EDT_MM
        )
        or minimum < SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM
    )


@dataclass
class EdtFrameBuild:
    frames: list[dict[str, Any]]
    removed_voxels: list[dict[str, Any]]
    manifest: dict[str, Any]


def load_anatomy_profile(project_root: Path, anatomy_key: str) -> dict[str, Any]:
    configured = os.environ.get("MASTOID_ANATOMY_PROFILES")
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root / "config" / "anatomy_profiles.json",
        project_root / "config" / "examples" / "anatomy_profiles.example.json",
    ]
    path = next(
        (candidate.resolve() for candidate in candidates if candidate and candidate.is_file()),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "Anatomy profiles were not found. Set MASTOID_ANATOMY_PROFILES "
            "or provide config/anatomy_profiles.json."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        profile = dict(payload["profiles"][anatomy_key])
    except KeyError as exc:
        raise KeyError(f"No versioned anatomy profile for {anatomy_key}") from exc
    profile["config_path"] = str(path)
    return profile


def read_edt_file(path: Path) -> tuple[Any, tuple[int, int, int]]:
    _, np = _require_numeric_stack()
    with path.open("rb") as handle:
        def next_token() -> str:
            char = handle.read(1)
            while char and char in b" \t\r\n":
                char = handle.read(1)
            if not char:
                raise EOFError(f"Unexpected EOF in EDT header: {path.name}")
            buffer = bytearray()
            while char and char not in b" \t\r\n":
                buffer.extend(char)
                char = handle.read(1)
            return buffer.decode("ascii", errors="strict")

        magic = next_token()
        if not magic.startswith("G"):
            raise ValueError(f"Bad EDT magic in {path.name}: {magic}")
        dimension = int(magic[1:])
        type_dimension = int(next_token())
        type_name = next_token()
        if type_dimension != 1 or type_name != "FLOAT":
            raise ValueError(f"Unsupported EDT type in {path.name}: {type_dimension} {type_name}")
        rx, ry, rz = int(next_token()), int(next_token()), int(next_token())
        for _ in range((dimension + 1) ** 2):
            float(next_token())
        total = rx * ry * rz
        data = np.fromfile(handle, dtype=np.float32, count=total)
        if len(data) != total:
            raise ValueError(f"EDT payload length mismatch in {path.name}")
    return data.reshape((rz, ry, rx)), (rx, ry, rz)


def _selected_grid_specs(edt_dir: Path) -> dict[str, dict[str, Any]]:
    grids: dict[str, dict[str, Any]] = {}
    for role, structures in (
        ("sensitive", SENSITIVE_STRUCTURES),
        ("boundary", BOUNDARY_STRUCTURES),
    ):
        for stem, short_name in structures.items():
            path = edt_dir / f"{stem}.edt"
            if not path.is_file():
                raise FileNotFoundError(path)
            grids[short_name] = {
                "stem": stem,
                "role": role,
                "path": path,
            }
    return grids


def enrich_removed_voxels_with_edt(
    core: CoreFrameBuild,
    record: RunRecord,
    sources: ResolvedRunSources,
    *,
    project_root: Path,
) -> EdtFrameBuild:
    _, np = _require_numeric_stack()
    profile = load_anatomy_profile(project_root, record.anatomy_key)
    if profile["edt_profile"] != record.edt_profile:
        raise ValueError("Versioned anatomy profile and run registry disagree")
    grids = _selected_grid_specs(sources.edt_dir)
    first_grid, first_resolution = read_edt_file(next(iter(grids.values()))["path"])
    del first_grid
    rx, ry, rz = first_resolution

    x = np.asarray([row["x"] for row in core.removed_voxels], dtype=int)
    y = (ry - 1) - np.asarray([row["y"] for row in core.removed_voxels], dtype=int)
    z = np.asarray([row["z"] for row in core.removed_voxels], dtype=int)
    assigned_mask = np.asarray(
        [row["frame_idx_global"] is not None for row in core.removed_voxels],
        dtype=bool,
    )
    frame_index = np.asarray(
        [
            row["frame_idx_global"] if row["frame_idx_global"] is not None else 0
            for row in core.removed_voxels
        ],
        dtype=int,
    )
    in_bounds = (x >= 0) & (x < rx) & (y >= 0) & (y < ry) & (z >= 0) & (z < rz)
    # Preserve the EDT grid's float32 arithmetic used by the original plugin.
    scale_mm = np.float32(profile["edt_distance_scale_mm"])
    n_frames = len(core.frames)
    frame_minima: dict[str, Any] = {}
    raw_by_structure: dict[str, Any] = {}
    overshoot_by_structure: dict[str, Any] = {}
    inside_count_by_structure: dict[str, Any] = {}

    for short_name, item in grids.items():
        grid, resolution = read_edt_file(item["path"])
        if resolution != first_resolution:
            raise ValueError(
                f"EDT resolution mismatch in {item['path'].name}: {resolution}"
            )
        raw = np.full(len(x), np.nan, dtype=np.float32)
        raw[in_bounds] = grid[z[in_bounds], y[in_bounds], x[in_bounds]]
        del grid
        raw_by_structure[short_name] = raw
        distances = raw * scale_mm
        minima = np.full(n_frames, np.inf, dtype=float)
        valid = in_bounds & assigned_mask & np.isfinite(distances)
        np.minimum.at(minima, frame_index[valid], distances[valid])
        minima[np.isinf(minima)] = np.nan
        frame_minima[short_name] = minima
        if item["role"] == "sensitive":
            inside = valid & (raw <= 0)
            frame_inside_count = np.zeros(n_frames, dtype=np.int64)
            np.add.at(frame_inside_count, frame_index[inside], 1)
            inside_count_by_structure[short_name] = frame_inside_count
            overshoot_by_structure[short_name] = frame_inside_count > 0

    sensitive_names = list(SENSITIVE_STRUCTURES.values())

    removed_rows: list[dict[str, Any]] = []
    for index, source in enumerate(core.removed_voxels):
        row = dict(source)
        row["edt_x"] = int(x[index])
        row["edt_y"] = int(y[index])
        row["edt_z"] = int(z[index])
        row["edt_in_bounds"] = bool(in_bounds[index])
        inside = []
        for short_name, item in grids.items():
            value = raw_by_structure[short_name][index]
            row[f"edt_{short_name}_mm"] = (
                float(value * scale_mm) if math.isfinite(float(value)) else None
            )
            if item["role"] == "sensitive" and math.isfinite(float(value)) and value <= 0:
                inside.append(short_name)
        row["inside_sensitive_structures"] = ",".join(inside)
        removed_rows.append(row)

    frame_rows: list[dict[str, Any]] = []
    for index, source in enumerate(core.frames):
        row = dict(source)
        finite_sensitive = []
        overshoot = []
        safety_signal = []
        for short_name, item in grids.items():
            minimum = frame_minima[short_name][index]
            rounded = round(float(minimum), 4) if math.isfinite(float(minimum)) else None
            row[f"voxel_min_{short_name}_mm"] = rounded
            if item["role"] == "sensitive" and rounded is not None:
                inside_count = int(inside_count_by_structure[short_name][index])
                row[f"voxel_inside_{short_name}_count"] = inside_count
                finite_sensitive.append(float(minimum))
                if overshoot_by_structure[short_name][index]:
                    overshoot.append(short_name)
                    if passes_safety_coaching_gate(inside_count, float(minimum)):
                        safety_signal.append(short_name)
            elif item["role"] == "sensitive":
                row[f"voxel_inside_{short_name}_count"] = 0
        row["voxel_min_sensitive_mm"] = (
            round(min(finite_sensitive), 4) if finite_sensitive else None
        )
        row["voxel_overshoot"] = bool(overshoot)
        row["voxel_overshoot_structures"] = ",".join(overshoot)
        row["voxel_safety_signal"] = bool(safety_signal)
        row["voxel_safety_signal_structures"] = ",".join(safety_signal)
        frame_rows.append(row)

    out_of_bounds = int(np.sum(~in_bounds))
    unassigned = int(np.sum(~assigned_mask))
    overshoot_frames = sum(bool(row["voxel_overshoot"]) for row in frame_rows)
    safety_signal_frames = sum(bool(row["voxel_safety_signal"]) for row in frame_rows)
    manifest = {
        "schema": "voxel_edt_enrichment_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "upstream_schema": core.manifest["schema"],
        "upstream_frame_count": len(core.frames),
        "upstream_removed_voxel_count": len(core.removed_voxels),
        "anatomy_key": record.anatomy_key,
        "edt_profile": record.edt_profile,
        "edt_dir": str(sources.edt_dir),
        "anatomy_profile": profile,
        "grid_resolution_xyz": [rx, ry, rz],
        "sensitive_structures": sensitive_names,
        "boundary_structures": list(BOUNDARY_STRUCTURES.values()),
        "removed_voxels_out_of_bounds": out_of_bounds,
        "unassigned_removed_voxel_count": unassigned,
        "unassigned_removed_voxel_policy": (
            "preserved_in_removed_voxel_edt_but_excluded_from_frame_aggregation"
        ),
        "voxel_overshoot_frame_count": overshoot_frames,
        "voxel_safety_signal_frame_count": safety_signal_frames,
        "safety_coaching_gate": {
            "logic": (
                "volume_and_minimum_penetration_or_deep_single_voxel"
            ),
            "min_inside_voxels_per_structure_per_frame": (
                SAFETY_COACHING_MIN_INSIDE_VOXELS_PER_FRAME
            ),
            "volume_branch_max_edt_mm": SAFETY_COACHING_VOLUME_MIN_EDT_MM,
            "deep_edt_threshold_mm": SAFETY_COACHING_DEEP_EDT_THRESHOLD_MM,
            "status": "provisional_pilot_threshold_pending_expert_validation",
        },
        "invariants": {
            "upstream_ready": bool(core.manifest["ready_for_edt_enrichment"]),
            "all_selected_grids_loaded": len(grids)
            == len(SENSITIVE_STRUCTURES) + len(BOUNDARY_STRUCTURES),
            "all_removed_voxels_in_bounds": out_of_bounds == 0,
            "frame_count_preserved": len(frame_rows) == len(core.frames),
            "removed_voxel_count_preserved": len(removed_rows) == len(core.removed_voxels),
            "unassigned_voxels_excluded_from_frame_aggregation": True,
        },
    }
    manifest["ready_for_visibility_enrichment"] = all(manifest["invariants"].values())
    return EdtFrameBuild(frame_rows, removed_rows, manifest)


def compare_legacy_voxel_edt(build: EdtFrameBuild, legacy_path: Path) -> dict[str, Any]:
    if not legacy_path.is_file():
        return {"available": False, "legacy_path": str(legacy_path)}
    with legacy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        legacy = list(csv.DictReader(handle))
    if len(legacy) != len(build.frames):
        return {
            "available": True,
            "legacy_path": str(legacy_path),
            "row_count_match": False,
            "all_match": False,
        }
    columns = [f"voxel_min_{name}_mm" for name in SENSITIVE_STRUCTURES.values()]
    columns += ["voxel_min_sensitive_mm", "voxel_min_EAC_mm", "voxel_min_TMJ_mm"]
    mismatches = {column: 0 for column in columns}
    mismatches.update({"voxel_overshoot": 0, "voxel_overshoot_structures": 0})
    max_error = {column: 0.0 for column in columns}
    differing_frames: set[int] = set()

    def numeric_or_none(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    for new, old in zip(build.frames, legacy):
        for column in columns:
            new_value = numeric_or_none(new.get(column))
            old_value = numeric_or_none(old.get(column))
            if new_value is None or old_value is None:
                if new_value is not None or old_value is not None:
                    mismatches[column] += 1
                    differing_frames.add(int(new["frame_idx_global"]))
            else:
                error = abs(new_value - old_value)
                max_error[column] = max(max_error[column], error)
                if error > 1e-9:
                    mismatches[column] += 1
                    differing_frames.add(int(new["frame_idx_global"]))
        old_bool = str(old.get("voxel_overshoot", "")).strip().lower() in {
            "1", "true", "t", "yes"
        }
        if old_bool != bool(new["voxel_overshoot"]):
            mismatches["voxel_overshoot"] += 1
            differing_frames.add(int(new["frame_idx_global"]))
        if str(old.get("voxel_overshoot_structures", "")).strip() != str(
            new["voxel_overshoot_structures"]
        ):
            mismatches["voxel_overshoot_structures"] += 1
            differing_frames.add(int(new["frame_idx_global"]))
    overshoot_exact = (
        mismatches["voxel_overshoot"] == 0
        and mismatches["voxel_overshoot_structures"] == 0
    )
    return {
        "available": True,
        "legacy_path": str(legacy_path),
        "role": "validation_only",
        "row_count_match": True,
        "compared_columns": columns + ["voxel_overshoot", "voxel_overshoot_structures"],
        "mismatches": mismatches,
        "max_absolute_errors": max_error,
        "differing_frame_indices": sorted(differing_frames),
        "differing_frame_count": len(differing_frames),
        "overshoot_semantics_exact": overshoot_exact,
        "difference_interpretation": (
            "The legacy EDT stage re-read absolute timestamps from an intermediate "
            "CSV with pandas. Direct HDF5 timestamps place boundary voxels in a "
            "small set of frames differently; simulator overshoot labels remain identical."
            if differing_frames and overshoot_exact
            else None
        ),
        "all_match": not any(mismatches.values()),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_edt_build(output_dir: Path, build: EdtFrameBuild, regression: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frame_observations_edt.csv", build.frames)
    _write_csv(output_dir / "removed_voxel_edt.csv", build.removed_voxels)
    (output_dir / "edt_manifest.json").write_text(
        json.dumps(build.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "legacy_voxel_edt_regression.json").write_text(
        json.dumps(regression, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Voxel EDT Audit: {build.manifest['run_id']}",
        "",
        f"- Frames: {len(build.frames)}",
        f"- Removed voxels: {len(build.removed_voxels)}",
        f"- Out-of-bounds removed voxels: {build.manifest['removed_voxels_out_of_bounds']}",
        f"- Simulator overshoot frames: {build.manifest['voxel_overshoot_frame_count']}",
        f"- Safety-coaching qualified frames: {build.manifest['voxel_safety_signal_frame_count']}",
        f"- Ready for visibility enrichment: **{build.manifest['ready_for_visibility_enrichment']}**",
        "",
        "## Validation-only legacy comparison",
        "",
        f"- Exact distance parity: **{regression.get('all_match')}**",
        f"- Overshoot labels and structures exact: **{regression.get('overshoot_semantics_exact')}**",
        f"- Frames with any EDT-distance difference: {regression.get('differing_frame_count')}",
        f"- Differing frame indices: {regression.get('differing_frame_indices')}",
        "",
        str(regression.get("difference_interpretation") or "No interpreted difference."),
        "",
        "The legacy EDT table is a regression target, not an enrichment input.",
    ]
    (output_dir / "VOXEL_EDT_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
