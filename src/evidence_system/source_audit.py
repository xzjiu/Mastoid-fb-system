"""Read-only audit of one canonical raw simulator run and its references."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .registry import RunRecord
from .source_config import ResolvedRunSources


DERIVED_SUFFIXES = {".csv", ".json", ".jsonl"}
RAW_MEDIA_SUFFIXES = {".mp4", ".npy", ".intrinsics", ".pldata"}
REQUIRED_EDT_STEMS = {
    "Bone",
    "Malleus",
    "Incus",
    "Bony_Labyrinth",
    "Facial_Nerve",
    "Chorda_Tympani",
    "Cochlear_Nerve",
    "IAC",
    "ICA",
    "Sinus_+_Dura",
    "Superior_Vestibular_Nerve",
    "Inferior_Vestibular_Nerve",
    "Stapes",
    "EAC",
    "TMJ",
    "Vestibular_Aqueduct",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def inspect_hdf5(path: Path, *, hash_files: bool) -> dict[str, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "source-audit requires h5py. Run it in the assessment environment."
        ) from exc

    datasets: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "file_name": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha256_file(path) if hash_files else None,
        "readable": False,
        "chunk_role": "unknown",
        "error": None,
        "root_groups": [],
        "datasets": datasets,
        "frame_count": 0,
        "image_frame_count": 0,
        "depth_frame_count": 0,
        "drill_pose_count": 0,
        "time_start": None,
        "time_end": None,
        "time_monotonic": None,
        "removed_timestamp_count": 0,
        "removed_voxel_row_count": 0,
        "removed_timestamp_index_min": None,
        "removed_timestamp_index_max": None,
        "removed_timestamp_indices_valid": True,
        "voxel_volume": None,
    }
    try:
        with h5py.File(path, "r") as handle:
            result["root_groups"] = sorted(handle.keys())

            def visit(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset):
                    datasets.append(
                        {
                            "path": name,
                            "shape": list(item.shape),
                            "dtype": str(item.dtype),
                        }
                    )

            handle.visititems(visit)

            time_dataset = handle.get("data/time")
            if time_dataset is not None:
                result["chunk_role"] = "frame_data"
                times = np.asarray(time_dataset[()]).reshape(-1)
                numeric_times = np.asarray(times, dtype=float)
                finite = numeric_times[np.isfinite(numeric_times)]
                result["frame_count"] = int(len(times))
                if len(finite):
                    result["time_start"] = float(finite[0])
                    result["time_end"] = float(finite[-1])
                    result["time_monotonic"] = bool(np.all(np.diff(finite) >= 0))
            else:
                frame_like_paths = {
                    "data/l_img", "data/depth", "data/pose_mastoidectomy_drill",
                    "data/pose_mastoidectomy_volume", "data/pose_main_camera",
                    "voxels_removed/voxel_removed",
                }
                present_paths = {row["path"] for row in datasets}
                result["chunk_role"] = (
                    "invalid_missing_time"
                    if present_paths & frame_like_paths else "metadata_only"
                )

            for dataset_path, field in (
                ("data/l_img", "image_frame_count"),
                ("data/depth", "depth_frame_count"),
                ("data/pose_mastoidectomy_drill", "drill_pose_count"),
                ("voxels_removed/voxel_time_stamp", "removed_timestamp_count"),
                ("voxels_removed/voxel_removed", "removed_voxel_row_count"),
            ):
                dataset = handle.get(dataset_path)
                if dataset is not None and dataset.shape:
                    result[field] = int(dataset.shape[0])

            removed = handle.get("voxels_removed/voxel_removed")
            if removed is not None and len(removed):
                timestamp_indices = np.asarray(removed[:, 0], dtype=int)
                index_min = int(timestamp_indices.min())
                index_max = int(timestamp_indices.max())
                result["removed_timestamp_index_min"] = index_min
                result["removed_timestamp_index_max"] = index_max
                result["removed_timestamp_indices_valid"] = bool(
                    index_min >= 0
                    and index_max < int(result["removed_timestamp_count"])
                )

            voxel_volume = handle.get("metadata/voxel_volume")
            if voxel_volume is not None:
                value = voxel_volume[()]
                if hasattr(value, "item"):
                    value = value.item()
                result["voxel_volume"] = _finite_float(value)
            result["readable"] = True
    except Exception as exc:  # audit must report a bad chunk rather than hide it
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_edt_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        def next_token() -> str:
            char = handle.read(1)
            while char and char in b" \t\r\n":
                char = handle.read(1)
            buffer = bytearray()
            while char and char not in b" \t\r\n":
                buffer.extend(char)
                char = handle.read(1)
            return buffer.decode("ascii", errors="replace")

        magic = next_token()
        dimension = int(magic[1:]) if magic.startswith("G") else None
        type_dimension = next_token()
        type_name = next_token()
        resolution = [int(next_token()), int(next_token()), int(next_token())]
    return {
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "magic": magic,
        "dimension": dimension,
        "type_dimension": type_dimension,
        "type_name": type_name,
        "resolution": resolution,
    }


def run_id_from_video(video: str) -> str | None:
    match = re.search(r"_P(\d+)T(\d+)\.mp4$", str(video), flags=re.IGNORECASE)
    if not match:
        return None
    return f"P{int(match.group(1)):02d}_T{int(match.group(2)):02d}"


def inspect_phase_annotation(path: Path, record: RunRecord) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows
        if (
            str(row.get("run_id", "")).strip() == record.run_id
            or (
                record.phase_annotation_id
                and str(row.get("phase_annotation_id", "")).strip()
                == record.phase_annotation_id
            )
            or (
                record.exp_id
                and str(row.get("exp_id", "")).strip() == record.exp_id
            )
        )
    ]
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "exp_id": record.exp_id,
        "match_count": len(matches),
        "valid": False,
        "row": None,
        "ordered": False,
        "run_id_matches": False,
    }
    if len(matches) != 1:
        return result
    row = matches[0]
    antrum = _finite_float(
        row.get("antrum_start_sec") or row.get("antrum_start_s")
    )
    incus = _finite_float(
        row.get("antrum_end_incus_start_sec") or row.get("incus_start_s")
    )
    incus_end = _finite_float(row.get("incus_end_sec") or row.get("incus_end_s"))
    ordered = (
        antrum is not None
        and incus is not None
        and incus_end is not None
        and 0 <= antrum < incus < incus_end
    )
    explicit_run = str(row.get("run_id", "")).strip()
    parsed_run = explicit_run or run_id_from_video(str(row.get("video", "")))
    run_id_matches = parsed_run == record.run_id
    if not parsed_run:
        run_id_matches = bool(
            (
                record.phase_annotation_id
                and str(row.get("phase_annotation_id", "")).strip()
                == record.phase_annotation_id
            )
            or (
                record.exp_id
                and str(row.get("exp_id", "")).strip() == record.exp_id
            )
        )
    result.update(
        {
            "valid": bool(ordered and run_id_matches),
            "ordered": bool(ordered),
            "run_id_matches": run_id_matches,
            "row": {
                "video": row.get("video"),
                "participant": row.get("participant"),
                "training_year": row.get("training_year"),
                "antrum_start_sec": antrum,
                "incus_start_sec": incus,
                "incus_end_sec": incus_end,
                "annotation_status": row.get("annotation_status"),
                "notes": row.get("notes"),
            },
        }
    )
    return result


def classify_run_files(raw_run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    derived = []
    raw_media = []
    for path in sorted(raw_run_dir.glob("*"), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in DERIVED_SUFFIXES:
            derived.append({"name": path.name, "size_bytes": path.stat().st_size})
    media_dir = raw_run_dir / "000"
    if media_dir.is_dir():
        for path in sorted(media_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in RAW_MEDIA_SUFFIXES:
                raw_media.append({"name": path.name, "size_bytes": path.stat().st_size})
    return {"excluded_derived_files": derived, "raw_media_files": raw_media}


def build_source_audit(
    record: RunRecord,
    sources: ResolvedRunSources,
    *,
    hash_hdf5: bool = True,
) -> dict[str, Any]:
    if not sources.raw_run_dir.is_dir():
        raise FileNotFoundError(sources.raw_run_dir)
    if not sources.edt_dir.is_dir():
        raise FileNotFoundError(sources.edt_dir)
    if not sources.phase_annotation_file.is_file():
        raise FileNotFoundError(sources.phase_annotation_file)

    hdf5_paths = sorted(sources.raw_run_dir.glob("*.hdf5"), key=lambda path: path.name)
    hdf5 = [inspect_hdf5(path, hash_files=hash_hdf5) for path in hdf5_paths]
    readable = [item for item in hdf5 if item["readable"]]
    time_chunks = [
        item
        for item in readable
        if item["time_start"] is not None and item["time_end"] is not None
    ]
    continuity = []
    for previous, current in zip(time_chunks, time_chunks[1:]):
        gap = float(current["time_start"]) - float(previous["time_end"])
        continuity.append(
            {
                "previous": previous["file_name"],
                "current": current["file_name"],
                "gap_s": gap,
                "classification": "overlap" if gap < 0 else "forward",
            }
        )

    edt_files = sorted(sources.edt_dir.glob("*.edt"), key=lambda path: path.name)
    edt_headers = [read_edt_header(path) for path in edt_files]
    edt_stems = {path.stem for path in edt_files}
    missing_required_edt = sorted(REQUIRED_EDT_STEMS - edt_stems)
    phase = inspect_phase_annotation(sources.phase_annotation_file, record)
    classified = classify_run_files(sources.raw_run_dir)
    anatomy_mapping_confirmed = record.anatomy_mapping_status.casefold().startswith(
        ("confirmed", "synthetic", "configured")
    )

    checks = {
        "one_canonical_raw_directory": True,
        "hdf5_files_found": len(hdf5) > 0,
        "all_hdf5_readable": len(readable) == len(hdf5) and len(hdf5) > 0,
        "frame_data_chunks_found": any(
            item["chunk_role"] == "frame_data" for item in readable
        ),
        "all_hdf5_roles_supported": all(
            item["chunk_role"] in {"frame_data", "metadata_only"}
            for item in readable
        ),
        "all_hdf5_time_monotonic": all(
            item["time_monotonic"] in (True, None) for item in readable
        ),
        "all_frame_modalities_aligned": all(
            item["frame_count"]
            == item["image_frame_count"]
            == item["depth_frame_count"]
            == item["drill_pose_count"]
            for item in readable
            if item["frame_count"] > 0
        ),
        "all_removed_voxel_timestamp_indices_valid": all(
            item["removed_timestamp_indices_valid"] for item in readable
        ),
        "edt_profile_found": len(edt_files) > 0,
        "required_edt_files_present": not missing_required_edt,
        "phase_annotation_unique_and_valid": bool(phase["valid"]),
        "anatomy_mapping_confirmed": anatomy_mapping_confirmed,
        "old_derived_files_excluded": True,
    }
    ready = all(checks.values())
    return {
        "schema": "canonical_source_audit_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_id": record.dataset_id,
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "source_classification": {
            "primary_raw": str(sources.raw_run_dir),
            "auxiliary_edt": str(sources.edt_dir),
            "auxiliary_phase_annotation": str(sources.phase_annotation_file),
            "validation_only_golden": str(sources.golden_reference_path),
        },
        "registry": {
            "exp_id": record.exp_id,
            "anatomy_key": record.anatomy_key,
            "raw_anatomy_token": record.raw_anatomy_token,
            "anatomy_mapping_status": record.anatomy_mapping_status,
            "anatomy_mapping_basis": record.anatomy_mapping_basis,
            "edt_profile": record.edt_profile,
            "phase_annotation_id": record.phase_annotation_id,
        },
        "hdf5": {
            "file_count": len(hdf5),
            "readable_count": len(readable),
            "frame_count_total": sum(int(item["frame_count"]) for item in readable),
            "removed_timestamp_count_total": sum(
                int(item["removed_timestamp_count"]) for item in readable
            ),
            "removed_voxel_row_count_total": sum(
                int(item["removed_voxel_row_count"]) for item in readable
            ),
            "chunks": hdf5,
            "continuity": continuity,
        },
        "edt": {
            "profile": record.edt_profile,
            "file_count": len(edt_files),
            "missing_required_stems": missing_required_edt,
            "headers": edt_headers,
        },
        "phase_annotation": phase,
        "raw_media_files": classified["raw_media_files"],
        "excluded_derived_files": classified["excluded_derived_files"],
        "checks": checks,
        "ready_for_frame_extraction": ready,
        "blocking_items": [name for name, passed in checks.items() if not passed],
    }


def _write_inventory(path: Path, audit: dict[str, Any]) -> None:
    fields = [
        "file_name",
        "size_bytes",
        "sha256",
        "readable",
        "error",
        "frame_count",
        "image_frame_count",
        "depth_frame_count",
        "drill_pose_count",
        "time_start",
        "time_end",
        "time_monotonic",
        "removed_timestamp_count",
        "removed_voxel_row_count",
        "removed_timestamp_index_min",
        "removed_timestamp_index_max",
        "removed_timestamp_indices_valid",
        "voxel_volume",
        "root_groups",
        "dataset_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in audit["hdf5"]["chunks"]:
            row = {key: item.get(key) for key in fields}
            row["root_groups"] = "|".join(item.get("root_groups", []))
            row["dataset_count"] = len(item.get("datasets", []))
            writer.writerow(row)


def render_markdown(audit: dict[str, Any]) -> str:
    phase_row = audit["phase_annotation"].get("row") or {}
    lines = [
        f"# Canonical Source Audit: {audit['run_id']}",
        "",
        f"- Overall ready for frame extraction: **{audit['ready_for_frame_extraction']}**",
        f"- Primary raw directory: `{audit['source_classification']['primary_raw']}`",
        f"- HDF5 chunks: {audit['hdf5']['readable_count']}/{audit['hdf5']['file_count']} readable",
        f"- Total image-time frames: {audit['hdf5']['frame_count_total']}",
        f"- Removed-voxel rows: {audit['hdf5']['removed_voxel_row_count_total']}",
        f"- EDT profile: `{audit['edt']['profile']}` ({audit['edt']['file_count']} files)",
        "",
        "## Phase annotation",
        "",
        f"- Annotation ID: `{audit['registry']['phase_annotation_id']}`",
        f"- Video: `{phase_row.get('video')}`",
        f"- Antrum start: {phase_row.get('antrum_start_sec')} s",
        f"- Incus start: {phase_row.get('incus_start_sec')} s",
        f"- Incus end: {phase_row.get('incus_end_sec')} s",
        "",
        "## Source boundary",
        "",
        "- HDF5 and raw media are primary raw evidence.",
        "- EDT and human phase annotation are auxiliary references.",
        "- Existing CSV/JSON files in the run directory are excluded derived artifacts.",
        "- The legacy graph is validation-only and is not a pipeline input.",
        "",
        "## Checks",
        "",
    ]
    for name, passed in audit["checks"].items():
        lines.append(f"- {'PASS' if passed else 'BLOCK'}: `{name}`")
    if audit["blocking_items"]:
        lines.extend(["", "## Blocking items", ""])
        lines.extend(f"- `{item}`" for item in audit["blocking_items"])
    lines.extend(
        [
            "",
            "## Anatomy mapping requiring confirmation",
            "",
            f"- Registered anatomy key: `{audit['registry']['anatomy_key']}`",
            f"- Token in raw directory/video name: `{audit['registry']['raw_anatomy_token']}`",
            f"- Configured EDT profile: `{audit['registry']['edt_profile']}`",
            f"- Mapping status: `{audit['registry']['anatomy_mapping_status']}`",
            f"- Mapping basis: `{audit['registry']['anatomy_mapping_basis']}`",
            "",
            "The raw token and the internal anatomy key use different naming systems.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_source_audit(output_dir: Path, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_inventory(output_dir / "hdf5_inventory.csv", audit)
    (output_dir / "SOURCE_AUDIT.md").write_text(
        render_markdown(audit), encoding="utf-8"
    )
