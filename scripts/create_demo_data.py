"""Create the deterministic synthetic dataset referenced by config/examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evidence_system.source_audit import REQUIRED_EDT_STEMS  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "demo"
DEMO_SEED = 20261005
DEMO_RUN_ID = "DEMO_CASE_01"
DEMO_EDT_PROFILE = "demo_edt_16"
FRAME_COUNT = 24
GRID_RESOLUTION = 16
DEMO_SAFETY_EDT_STEM = "Sinus_+_Dura"
DEMO_SAFETY_EDT_RAW_VALUE = -3.0
DEMO_REMOVED_VOXEL_XYZ = (8, 8, 8)


def _numeric_stack() -> tuple[Any, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Demo generation requires the pipeline dependencies. "
            "Install the project with the 'pipeline' extra."
        ) from exc
    return h5py, np


def _write_hdf5(path: Path, *, seed: int) -> None:
    h5py, np = _numeric_stack()
    rng = np.random.default_rng(seed)
    times = np.arange(FRAME_COUNT, dtype=float) * 0.1

    drill_poses = np.zeros((FRAME_COUNT, 7), dtype=float)
    drill_poses[:, 0] = np.linspace(-0.01, 0.01, FRAME_COUNT)
    drill_poses[:, 6] = 1.0
    volume_poses = np.zeros((FRAME_COUNT, 7), dtype=float)
    volume_poses[:, 6] = 1.0
    camera_poses = np.zeros((FRAME_COUNT, 7), dtype=float)
    camera_poses[:, 2] = -0.1
    camera_poses[:, 6] = 1.0

    rgb = rng.integers(210, 231, size=(FRAME_COUNT, 64, 64, 3), dtype=np.uint8)
    rgb[:, 29:36, 29:36] = 0
    depth = (
        1.0
        + rng.normal(0.0, 0.002, size=(FRAME_COUNT, 64, 64))
    ).astype(np.float32)

    removal_times = np.arange(0.2, 1.81, 0.1, dtype=float)
    removed_coordinates = np.tile(
        np.asarray(DEMO_REMOVED_VOXEL_XYZ, dtype=np.int32),
        (len(removal_times), 1),
    )
    removed = np.column_stack(
        (np.arange(len(removal_times), dtype=np.int32), removed_coordinates)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.create_dataset("time", data=times, track_times=False)
        data.create_dataset("l_img", data=rgb, track_times=False)
        data.create_dataset("depth", data=depth, track_times=False)
        data.create_dataset(
            "pose_mastoidectomy_drill", data=drill_poses, track_times=False
        )
        data.create_dataset(
            "pose_mastoidectomy_volume", data=volume_poses, track_times=False
        )
        data.create_dataset("pose_main_camera", data=camera_poses, track_times=False)

        metadata = handle.create_group("metadata")
        metadata.create_dataset(
            "voxel_volume", data=np.asarray(0.008, dtype=float), track_times=False
        )
        metadata.create_dataset(
            "camera_intrinsic",
            data=np.asarray(
                [
                    [40.0, 0.0, 32.0],
                    [0.0, 40.0, 32.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            track_times=False,
        )
        metadata.create_dataset(
            "camera_extrinsic", data=np.eye(4), track_times=False
        )

        voxels = handle.create_group("voxels_removed")
        voxels.create_dataset(
            "voxel_time_stamp", data=removal_times, track_times=False
        )
        voxels.create_dataset("voxel_removed", data=removed, track_times=False)


def _write_edt(
    path: Path,
    value: float,
    *,
    safety_signal_at_removed_voxel: bool = False,
) -> None:
    _, np = _numeric_stack()
    transform = np.eye(4, dtype=float).reshape(-1)
    header = (
        "G3\n"
        "1 FLOAT\n"
        f"{GRID_RESOLUTION} {GRID_RESOLUTION} {GRID_RESOLUTION}\n"
        + " ".join(str(float(item)) for item in transform)
        + "\n"
    ).encode("ascii")
    grid = np.full(
        (GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION),
        value,
        dtype=np.float32,
    )
    if safety_signal_at_removed_voxel:
        x, source_y, z = DEMO_REMOVED_VOXEL_XYZ
        edt_y = (GRID_RESOLUTION - 1) - source_y
        grid[z, edt_y, x] = DEMO_SAFETY_EDT_RAW_VALUE
    payload = grid.tobytes(order="C")
    path.write_bytes(header + payload)


def _write_phase_annotation(path: Path) -> None:
    fields = (
        "run_id",
        "phase_annotation_id",
        "exp_id",
        "antrum_start_s",
        "incus_start_s",
        "incus_end_s",
        "annotation_status",
        "notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": DEMO_RUN_ID,
                "phase_annotation_id": "exp_id:9001",
                "exp_id": "9001",
                "antrum_start_s": "0.4",
                "incus_start_s": "1.0",
                "incus_end_s": "1.8",
                "annotation_status": "synthetic",
                "notes": "Deterministically generated demo fixture.",
            }
        )


def create_demo_data(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Write the synthetic source tree consumed by the example configuration."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(output_root)

    raw_run_dir = output_root / "raw" / DEMO_RUN_ID
    edt_dir = output_root / "edt" / DEMO_EDT_PROFILE
    annotation_path = output_root / "annotations" / "phase_timepoints.csv"
    raw_run_dir.mkdir(parents=True, exist_ok=True)
    edt_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = raw_run_dir / "chunk_000.hdf5"
    _write_hdf5(hdf5_path, seed=DEMO_SEED)
    for index, stem in enumerate(sorted(REQUIRED_EDT_STEMS)):
        _write_edt(
            edt_dir / f"{stem}.edt",
            10.0 + index / 100.0,
            safety_signal_at_removed_voxel=(stem == DEMO_SAFETY_EDT_STEM),
        )
    _write_phase_annotation(annotation_path)

    return {
        "schema": "synthetic_demo_data_v1",
        "seed": DEMO_SEED,
        "output_root": str(output_root),
        "run_id": DEMO_RUN_ID,
        "raw_run_dir": str(raw_run_dir),
        "hdf5_file": str(hdf5_path),
        "edt_dir": str(edt_dir),
        "edt_file_count": len(REQUIRED_EDT_STEMS),
        "phase_annotation_file": str(annotation_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Demo data root (default: repository data/demo).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(create_demo_data(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
