"""Build a browser-reviewable video aligned to canonical simulator time."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .registry import RunRecord
from .source_config import ResolvedRunSources


@dataclass(frozen=True)
class FrameReference:
    timestamp_s: float
    source_path: Path
    source_index: int


def _require_video_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2  # type: ignore
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Review-video generation requires the pipeline vision dependencies "
            "(h5py, numpy, and opencv-python)."
        ) from exc
    return cv2, h5py, np


def _index_source_frames(raw_run_dir: Path) -> list[FrameReference]:
    _, h5py, np = _require_video_dependencies()
    references: list[FrameReference] = []
    for path in sorted(raw_run_dir.glob("*.hdf5"), key=lambda item: item.name):
        with h5py.File(path, "r") as handle:
            time_dataset = handle.get("data/time")
            frame_dataset = handle.get("data/l_img")
            if time_dataset is None and frame_dataset is None:
                # Valid recorder terminal file containing metadata only.
                continue
            if time_dataset is None or frame_dataset is None:
                raise KeyError(
                    f"Incomplete frame-bearing HDF5 chunk: {path.name}"
                )
            times = np.asarray(time_dataset, dtype=float).reshape(-1)
            frame_count = int(frame_dataset.shape[0])
            if len(times) != frame_count:
                raise ValueError(
                    f"Timestamp/frame mismatch in {path.name}: {len(times)} != {frame_count}"
                )
            references.extend(
                FrameReference(float(timestamp), path, index)
                for index, timestamp in enumerate(times)
            )
    references.sort(key=lambda row: (row.timestamp_s, row.source_path.name, row.source_index))
    if not references:
        raise FileNotFoundError(f"No HDF5 RGB frames found under {raw_run_dir}")
    return references


def _nearest_source_indices(source_times: Any, target_times: Any, np: Any) -> Any:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right)


def build_review_video(
    record: RunRecord,
    sources: ResolvedRunSources,
    output_dir: Path,
    *,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Resample irregular HDF5 frames to a seek-stable WebM timeline.

    The raw ``data/l_img`` array is stored in BGR channel order and is written
    directly to OpenCV. This is deliberate: converting it as RGB swaps red and
    blue and produces the previously observed false-color video.
    """

    if fps <= 0 or fps > 60:
        raise ValueError("fps must be greater than 0 and no more than 60")
    cv2, h5py, np = _require_video_dependencies()
    references = _index_source_frames(sources.raw_run_dir)
    source_times_abs = np.asarray([row.timestamp_s for row in references], dtype=float)
    zero_time = float(source_times_abs[0])
    source_times = source_times_abs - zero_time
    duration_s = float(source_times[-1])
    output_count = int(math.ceil(duration_s * fps)) + 1
    target_times = np.arange(output_count, dtype=float) / float(fps)
    source_indices = _nearest_source_indices(source_times, target_times, np)
    selected_times = source_times[source_indices]
    alignment_error = np.abs(selected_times - target_times)

    first = references[int(source_indices[0])]
    with h5py.File(first.source_path, "r") as handle:
        sample = np.asarray(handle["data/l_img"][first.source_index])
    if sample.ndim != 3 or sample.shape[2] != 3 or sample.dtype != np.uint8:
        raise ValueError(f"Unsupported RGB frame shape/dtype: {sample.shape} {sample.dtype}")
    height, width = int(sample.shape[0]), int(sample.shape[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "review_video.webm"
    building_path = output_dir / "review_video.building.webm"
    if building_path.exists():
        building_path.unlink()
    writer = cv2.VideoWriter(
        str(building_path), cv2.VideoWriter_fourcc(*"VP80"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the browser-compatible VP8 encoder")

    current_path: Path | None = None
    handle: Any = None
    try:
        for selected in source_indices:
            reference = references[int(selected)]
            if reference.source_path != current_path:
                if handle is not None:
                    handle.close()
                handle = h5py.File(reference.source_path, "r")
                current_path = reference.source_path
            frame_bgr = np.asarray(handle["data/l_img"][reference.source_index])
            writer.write(frame_bgr)
    finally:
        writer.release()
        if handle is not None:
            handle.close()
    if not building_path.is_file() or building_path.stat().st_size == 0:
        raise RuntimeError("Review video encoder produced no output")
    building_path.replace(video_path)

    validation = validate_review_video(
        video_path, references, source_indices, output_count, fps
    )
    manifest = {
        "schema": "review_video_manifest_v1",
        "run_id": record.run_id,
        "video_file": video_path.name,
        "container": "webm",
        "codec": "VP8",
        "browser_mime_type": "video/webm",
        "fps": float(fps),
        "width": width,
        "height": height,
        "source_frame_count": len(references),
        "output_frame_count": output_count,
        "source_hdf5_count": len({row.source_path for row in references}),
        "duration_s": duration_s,
        "timeline_origin": "first_HDF5_data/time_timestamp",
        "resampling": "uniform_timeline_nearest_source_frame",
        "mean_alignment_error_s": float(np.mean(alignment_error)),
        "max_alignment_error_s": float(np.max(alignment_error)),
        "input_color_order": "BGR",
        "encoder_input_color_order": "BGR",
        "color_rule": "write data/l_img directly; do not swap red and blue",
        "validation": validation,
    }
    manifest_path = output_dir / "review_video_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_review_video(
    video_path: Path,
    references: list[FrameReference],
    source_indices: Any,
    expected_count: int,
    expected_fps: float,
) -> dict[str, Any]:
    """Check decode, timeline metadata, and red/blue channel orientation."""

    cv2, h5py, np = _require_video_dependencies()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Generated review video cannot be decoded: {video_path}")
    decoded_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    sample_positions = sorted({0, expected_count // 2, expected_count - 1})
    correct_errors: list[float] = []
    swapped_errors: list[float] = []
    try:
        for output_index in sample_positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, output_index)
            ok, decoded_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode validation frame {output_index}")
            reference = references[int(source_indices[output_index])]
            with h5py.File(reference.source_path, "r") as handle:
                source_bgr = np.asarray(handle["data/l_img"][reference.source_index])
            correct_errors.append(float(np.mean(np.abs(
                decoded_bgr.astype(np.float32) - source_bgr.astype(np.float32)
            ))))
            swapped = source_bgr[:, :, ::-1]
            swapped_errors.append(float(np.mean(np.abs(
                decoded_bgr.astype(np.float32) - swapped.astype(np.float32)
            ))))
    finally:
        capture.release()
    mean_correct = float(np.mean(correct_errors))
    mean_swapped = float(np.mean(swapped_errors))
    return {
        "decode_ok": True,
        "decoded_frame_count": decoded_count,
        "expected_frame_count": expected_count,
        "frame_count_within_codec_tolerance": abs(decoded_count - expected_count) <= 1,
        "decoded_fps": decoded_fps,
        "expected_fps": float(expected_fps),
        "fps_within_tolerance": abs(decoded_fps - expected_fps) <= 0.01,
        "sample_output_frame_indices": sample_positions,
        "mean_absolute_error_bgr": mean_correct,
        "mean_absolute_error_if_red_blue_swapped": mean_swapped,
        "color_order_check_passed": mean_correct < mean_swapped,
    }

