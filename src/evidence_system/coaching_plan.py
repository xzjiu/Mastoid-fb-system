"""Build concise, evidence-typed coaching items for the review UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_cohort_benchmark(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_bounds(
    phases: Sequence[Mapping[str, Any]], name: str
) -> tuple[float, float] | None:
    row = next(
        (item for item in phases if str(item.get("phase_name")) == name), None
    )
    if row is None:
        return None
    return float(row["t_start_s"]), float(row["t_end_s"])


def _merge_review_clips(
    episodes: Sequence[Mapping[str, Any]], max_gap_s: float = 5.0
) -> list[dict[str, Any]]:
    """Group nearby detector episodes for review while preserving provenance."""

    ordered = sorted(episodes, key=lambda row: float(row["t_start_s"]))
    clips: list[dict[str, Any]] = []
    for row in ordered:
        start = float(row["t_start_s"])
        end = float(row["t_end_s"])
        episode_id = str(row["episode_id"])
        if clips and start - float(clips[-1]["t_end_s"]) <= max_gap_s:
            clip = clips[-1]
            clip["t_end_s"] = max(float(clip["t_end_s"]), end)
            clip["source_episode_ids"].append(episode_id)
            clip["source_episode_count"] = len(clip["source_episode_ids"])
            phase = row.get("phase")
            if phase is not None and phase not in clip["phases"]:
                clip["phases"].append(phase)
            if float(row.get("strength_score") or 0.0) > float(
                clip.get("strength_score") or 0.0
            ):
                for key in (
                    "anchor_t_s", "anchor_frame_idx_global",
                    "anchor_blocked_contact_fraction",
                    "anchor_registered_contact_voxel_count", "strength_score",
                ):
                    clip[key] = row.get(key)
            continue
        clips.append({
            "review_clip_id": f"exposure_review_clip:{len(clips) + 1:03d}",
            "t_start_s": start,
            "t_end_s": end,
            "source_episode_ids": [episode_id],
            "source_episode_count": 1,
            "phases": [row.get("phase")] if row.get("phase") is not None else [],
            **{
                key: row.get(key)
                for key in (
                    "anchor_t_s", "anchor_frame_idx_global",
                    "anchor_blocked_contact_fraction",
                    "anchor_registered_contact_voxel_count", "strength_score",
                )
                if row.get(key) is not None
            },
        })
    return clips


def build_session_coaching_plan(
    events: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    under_ledge_screening: Mapping[str, Any],
    cohort_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a small prioritized plan; each item keeps its evidence type."""

    items: list[dict[str, Any]] = []
    raw_overshoots = sorted(
        (
            row for row in events
            if str(row.get("event_type")) == "simulator_overshoot"
        ),
        key=lambda row: float(row["t_start_s"]),
    )
    overshoots = [
        row for row in raw_overshoots
        if bool((row.get("attributes") or {}).get("coaching_eligible", True))
    ]
    if overshoots:
        first = overshoots[0]
        phase_label = str(first.get("phase") or "the procedure").replace("_", " ")
        anatomy_label = {
            "SigSinus": "sigmoid sinus",
            "FacialNerve": "facial nerve",
            "BonyLabyrinth": "bony labyrinth",
        }.get(str(first.get("anatomy")), "protected-structure")
        anatomy_counts: dict[str, int] = {}
        for row in overshoots:
            anatomy = str(row.get("anatomy") or "unassigned")
            anatomy_counts[anatomy] = anatomy_counts.get(anatomy, 0) + 1
        safety_segments: list[dict[str, Any]] = []
        for row in overshoots:
            event_segments = list(
                (row.get("attributes") or {}).get("coaching_segments") or []
            )
            if event_segments:
                safety_segments.extend(dict(segment) for segment in event_segments)
            else:
                safety_segments.append({
                    "segment_id": f"{row['event_id']}:fallback",
                    "t_start_s": float(row["t_start_s"]),
                    "t_end_s": float(row["t_end_s"]),
                    "phase": row.get("phase"),
                    "anatomy": row.get("anatomy"),
                })
        items.append({
            "feedback_id": "coaching:safety_contacts",
            "category": "safety",
            "priority": "high",
            "title": "Review a possible safety-boundary signal",
            "message": (
                f"Pause and review the {anatomy_label} boundary before drilling "
                "further."
            ),
            "observed_issue": (
                "The simulator identified a possible "
                f"{anatomy_label} boundary signal during {phase_label} drilling."
            ),
            "why_it_matters": (
                "A possible protected-structure signal deserves prompt visual "
                "review. The finding may not be apparent from the video alone."
            ),
            "recommended_action": (
                f"Use the linked moment and anatomy view to verify the {anatomy_label} "
                "boundary; if the signal is confirmed, re-establish the boundary "
                "before drilling further."
            ),
            "evidence_summary": (
                f"The simulator identified {len(safety_segments)} "
                "protected-structure review moment"
                f"{'s' if len(safety_segments) != 1 else ''}."
            ),
            "technical_facts": {
                "episode_count": len(overshoots),
                "raw_episode_count": len(raw_overshoots),
                "review_moment_count": len(safety_segments),
                "anatomy_counts": anatomy_counts,
                "first_event_id": first["event_id"],
                "coaching_gate": (
                    (first.get("attributes") or {}).get("coaching_gate")
                ),
            },
            "t_start_s": float(first["t_start_s"]),
            "t_end_s": float(first["t_end_s"]),
            "phase": first.get("phase"),
            "anatomy": first.get("anatomy"),
            "evidence_status": "simulator_supported",
            "source_type": "canonical_simulator_events",
            "source_ids": [str(row["event_id"]) for row in overshoots],
            "evidence_segments": safety_segments,
            "claim_boundary": (
                "This is a simulator signal for review. It does not by itself "
                "confirm visible boundary contact, clinical injury, or intent."
            ),
        })

    benchmark = dict(cohort_benchmark or {})
    antrum = ((benchmark.get("phases") or {}).get("antrum") or {})
    observed = antrum.get("observed") or {}
    q25 = antrum.get("experienced_q25") or {}
    q75 = antrum.get("experienced_q75") or {}
    bounds = _phase_bounds(phases, "antrum")
    shorter = (
        observed.get("mean_stroke_length") is not None
        and q25.get("mean_stroke_length") is not None
        and float(observed["mean_stroke_length"])
        < float(q25["mean_stroke_length"])
    )
    more_frequent = (
        observed.get("stroke_rate_per_s") is not None
        and q75.get("stroke_rate_per_s") is not None
        and float(observed["stroke_rate_per_s"])
        > float(q75["stroke_rate_per_s"])
    )
    lower_efficiency = (
        observed.get("voxel_removal_efficiency") is not None
        and q25.get("voxel_removal_efficiency") is not None
        and float(observed["voxel_removal_efficiency"])
        < float(q25["voxel_removal_efficiency"])
    )
    if bounds and shorter and more_frequent:
        items.append({
            "feedback_id": "coaching:antrum_stroke_pattern",
            "category": "technique",
            "priority": "medium",
            "title": "Use longer strokes in the antrum",
            "message": "In the antrum, use fewer, longer sweeps.",
            "observed_issue": (
                "Antrum strokes were shorter and more frequent than the "
                "experienced-user reference range."
            ),
            "why_it_matters": (
                "The same phase also showed less progress per unit of active "
                "drilling time than the experienced-user reference range."
                if lower_efficiency else
                "The pattern was more fragmented than the experienced-user "
                "reference range."
            ),
            "recommended_action": (
                "Use longer, continuous sweeps while keeping the working area visible."
            ),
            "evidence_summary": (
                "Antrum strokes were shorter and more frequent than the "
                "experienced-user range."
            ),
            "technical_facts": {
                "observed_mean_stroke_length": observed["mean_stroke_length"],
                "experienced_q25_mean_stroke_length": q25["mean_stroke_length"],
                "observed_stroke_rate_per_s": observed["stroke_rate_per_s"],
                "experienced_q75_stroke_rate_per_s": q75["stroke_rate_per_s"],
                "observed_voxel_removal_efficiency": observed.get(
                    "voxel_removal_efficiency"
                ),
                "experienced_q25_voxel_removal_efficiency": q25.get(
                    "voxel_removal_efficiency"
                ),
                "experienced_trial_n": antrum.get("experienced_trial_n"),
            },
            "t_start_s": bounds[0],
            "t_end_s": bounds[1],
            "phase": "antrum",
            "anatomy": None,
            "evidence_status": "descriptive_cohort_comparison",
            "source_type": "experienced_group_phase_benchmark",
            "source_ids": [],
            "claim_boundary": benchmark.get("claim_boundary"),
        })

    example = (
        under_ledge_screening.get("coaching_example")
        if under_ledge_screening.get("available") else None
    )
    coaching_segments = list(
        under_ledge_screening.get("coaching_episodes") or []
    )
    # Low-support episodes and runs that fail the registration-quality gate stay
    # visible in the technical evidence screen, but are not auto-promoted to a
    # trainee coaching problem.
    exposure_segments = coaching_segments
    screening_only = False
    if example is None and coaching_segments:
        episode = max(
            coaching_segments,
            key=lambda row: (
                float(row.get("positive_frame_count") or 0)
                * float(row.get("mean_blocked_contact_fraction") or 0),
                -float(row["t_start_s"]),
            ),
        )
        example = {
            "t_start_s": float(episode["t_start_s"]),
            "t_end_s": float(episode["t_end_s"]),
            "phase": episode.get("phase"),
            "selection_rule": (
                "positive_frame_count_x_mean_blocked_contact_fraction"
            ),
        }
    if example and exposure_segments:
        review_clip_gap_s = 5.0
        review_clips = _merge_review_clips(exposure_segments, review_clip_gap_s)
        total_duration_s = sum(
            max(0.0, float(row["t_end_s"]) - float(row["t_start_s"]))
            for row in exposure_segments
        )
        items.append({
            "feedback_id": "coaching:working_corridor",
            "category": "exposure",
            "priority": "medium",
            "title": "Widen before drilling deeper",
            "message": (
                "Widen the working corridor and restore a direct view before "
                "drilling deeper."
            ),
            "observed_issue": (
                "In the linked drilling moments, residual bone may have limited "
                "the direct view of the burr tip and contact point."
            ),
            "why_it_matters": (
                "Reduced visibility can make it harder to identify and maintain a "
                "protected boundary, increasing the risk of unintended bone removal. "
                "This general risk mechanism does not establish that visibility "
                "caused any recorded contact in this session."
            ),
            "recommended_action": (
                "Review the linked clips, then widen the exposure and confirm a "
                "clear view of the burr tip and contact point before drilling deeper."
                if screening_only else
                "Widen the exposure and confirm a clear view of the burr tip and "
                "contact point before drilling deeper."
            ),
            "evidence_summary": (
                "The recorded view and simulator anatomy support reviewing these "
                "locations."
            ),
            "technical_facts": {
                "selection_rule": example.get("selection_rule"),
                "candidate_episode_count": (
                    under_ledge_screening.get("counts") or {}
                ).get("candidate_episode_count"),
                "high_support_candidate_episode_count": len(exposure_segments),
                "candidate_total_duration_s": round(total_duration_s, 3),
                "review_clip_count": len(review_clips),
                "review_clip_merge_gap_s": review_clip_gap_s,
                "screening_only": screening_only,
            },
            "t_start_s": float(example["t_start_s"]),
            "t_end_s": float(example["t_end_s"]),
            "phase": "multiple phases",
            "anatomy": None,
            "evidence_status": "automatic_geometric_screening_candidate",
            "source_type": "under_ledge_ray_v3",
            "source_ids": [str(row["episode_id"]) for row in exposure_segments],
            "evidence_segments": exposure_segments,
            "review_clips": review_clips,
            "claim_boundary": (
                "These locations are prompts for visual review. They do not "
                "establish that visibility caused an error."
            ),
        })

    order = {"high": 0, "medium": 1, "low": 2}
    return {
        "schema": "concise_session_coaching_plan_v1",
        "style": "one_behavior_or_location_plus_one_direct_action",
        "items": sorted(
            items,
            key=lambda row: (
                order.get(str(row["priority"]), 9),
                float(row["t_start_s"]),
            ),
        ),
    }

