"""Public reviewer view for the isolated registration-aware Ray-v3 output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def build_ray_v3_view(
    run_id: str,
    result_dir: Path | None,
    ray_v2_dir: Path | None = None,
) -> dict[str, Any]:
    if result_dir is None:
        return _unavailable("Ray-v3 result directory was not configured.")
    summary_path = result_dir / "ray_v3_summary.json"
    episode_path = result_dir / "ray_v3_candidate_episodes.json"
    if not summary_path.is_file() or not episode_path.is_file():
        return _unavailable("Registration-aware Ray-v3 output is not available for this run.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_id") != run_id:
        return _unavailable("Ray-v3 output belongs to a different run.")
    episodes = json.loads(episode_path.read_text(encoding="utf-8"))
    frame_path = result_dir / "ray_v3_frame_scores.csv"
    frames_by_index: dict[int, dict[str, str]] = {}
    if frame_path.is_file():
        with frame_path.open("r", encoding="utf-8-sig", newline="") as handle:
            frames_by_index = {
                int(row["frame_idx_global"]): row for row in csv.DictReader(handle)
            }
    enriched_episodes = []
    for source_episode in episodes:
        episode = dict(source_episode)
        candidates = [
            frames_by_index[index]
            for index in (int(value) for value in episode.get("source_frame_indices") or [])
            if index in frames_by_index
        ]
        if candidates:
            anchor = max(
                candidates,
                key=lambda row: (
                    float(row.get("blocked_contact_fraction") or 0.0),
                    int(row.get("registered_contact_voxel_count") or 0),
                ),
            )
            blocked_fraction = float(
                anchor.get("blocked_contact_fraction") or 0.0
            )
            registered_count = int(
                anchor.get("registered_contact_voxel_count") or 0
            )
            episode.update({
                "anchor_t_s": float(anchor["t_video_s"]),
                "anchor_frame_idx_global": int(anchor["frame_idx_global"]),
                "anchor_blocked_contact_fraction": blocked_fraction,
                "anchor_registered_contact_voxel_count": registered_count,
                "strength_score": round(blocked_fraction * registered_count, 4),
            })
        enriched_episodes.append(episode)
    episodes = enriched_episodes
    high_support_episodes = [
        row for row in episodes if row.get("screening_tier") == "high_support_review"
    ]
    coaching_eligible = bool(summary.get("coaching_eligible", True))
    coaching_episodes = high_support_episodes if coaching_eligible else []
    coaching_example = None
    if coaching_episodes:
        episode = max(
            coaching_episodes,
            key=lambda row: (
                float(row.get("positive_frame_count") or 0)
                * float(row.get("mean_blocked_contact_fraction") or 0),
                -float(row["t_start_s"]),
            ),
        )
        coaching_example = {
            "title": "Improve exposure before drilling deeper",
            "t_start_s": float(episode["t_start_s"]),
            "t_end_s": float(episode["t_end_s"]),
            "phase": episode.get("phase"),
            "delivery_status": "automatic_screening_example",
            "selection_rule": (
                "positive_frame_count_x_mean_blocked_contact_fraction"
            ),
        }
    old_before_150 = None
    if ray_v2_dir is not None:
        frame_path = ray_v2_dir / "ray_v2_frame_scores.csv"
        if frame_path.is_file():
            with frame_path.open("r", encoding="utf-8-sig", newline="") as handle:
                old_before_150 = sum(
                    row.get("ray_state") == "contact_occlusion_candidate"
                    and float(row["t_video_s"]) < 150.0
                    for row in csv.DictReader(handle)
                )
    return {
        "available": True,
        "run_id": run_id,
        "anatomy_key": summary.get("anatomy_key"),
        "edt_profile": summary.get("edt_profile"),
        "method": "registration_aware_camera_to_contact_ray_v3",
        "registration_orientation": summary.get("registration_orientation"),
        "registration_orientation_selection": (
            summary.get("registration_orientation_selection") or {}
        ),
        "orientation_is_provisional": summary.get("registration_orientation") != "identity",
        "config": summary.get("config") or {},
        "registration_audit": summary.get("registration_audit") or {},
        "counts": {
            "frames_with_raw_contact": int(summary.get("frames_with_raw_contact") or 0),
            "frames_passing_registration_gate": int(summary.get("frames_passing_registration_gate") or 0),
            "candidate_frame_count": int(summary.get("candidate_frame_count") or 0),
            "candidate_episode_count": int(summary.get("candidate_episode_count") or 0),
            "high_support_candidate_episode_count": len(high_support_episodes),
            "coaching_eligible_episode_count": len(coaching_episodes),
            "before_150_s_candidate_frame_count": int(
                summary.get("before_150_s_candidate_frame_count") or 0
            ),
        },
        "comparison": {
            "ray_v2_before_150_s_candidate_frame_count": old_before_150,
        },
        "episodes": episodes,
        "coaching_episodes": coaching_episodes,
        "coaching_eligible": coaching_eligible,
        "coaching_suppression_reason": summary.get("coaching_suppression_reason"),
        "coaching_example": coaching_example,
        "claim_boundary": summary.get("claim_boundary"),
    }

