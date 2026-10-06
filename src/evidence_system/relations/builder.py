"""Build factual relations over deterministic events and their evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import EventBuild, event_to_dict
from ..models import EvidenceEvent, EvidenceRelation
from ..registry import RunRecord


RULE_VERSION = "relation_rules_v1_factual_only"
SOURCE_MANUAL_PHASE = "human_video_timepoint_annotation"
SOURCE_VOXEL_EDT = "simulator_removed_voxel_edt"
SOURCE_STROKES = "simulator_stroke_table"
SOURCE_VISIBILITY = "derived_frame_visibility_geometry"


@dataclass
class RelationBuild:
    relations: list[EvidenceRelation]
    graph: dict[str, Any]
    manifest: dict[str, Any]


def _overlap(left: EvidenceEvent, right: EvidenceEvent) -> bool:
    return max(left.t_start_s, right.t_start_s) <= min(left.t_end_s, right.t_end_s)


def _source_for(event_type: str) -> str:
    if event_type in {"simulator_overshoot", "boundary_work"}:
        return SOURCE_VOXEL_EDT
    if event_type == "performance_bout":
        return SOURCE_STROKES
    return SOURCE_VISIBILITY


class _RelationCollector:
    def __init__(self) -> None:
        self.values: list[EvidenceRelation] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add(
        self,
        source: str,
        relation_type: str,
        target: str,
        *,
        provisional: bool = False,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        key = (source, relation_type, target)
        if key in self._seen:
            return
        self._seen.add(key)
        self.values.append(
            EvidenceRelation(
                relation_id=f"relation:{len(self.values) + 1:05d}",
                source_id=source,
                relation_type=relation_type,
                target_id=target,
                construction_method="deterministic",
                rule_version=RULE_VERSION,
                provisional=provisional,
                attributes=dict(attributes or {}),
            )
        )


def build_relations(events: EventBuild, record: RunRecord) -> RelationBuild:
    run_node = f"run:{record.global_run_id}"
    nodes: list[dict[str, Any]] = [
        {
            "id": run_node,
            "type": "run",
            "attributes": {
                "dataset_id": record.dataset_id,
                "run_id": record.run_id,
                "global_run_id": record.global_run_id,
                "analysis_unit": "single_simulator_run",
            },
        }
    ]
    for source in (SOURCE_MANUAL_PHASE, SOURCE_VOXEL_EDT, SOURCE_STROKES, SOURCE_VISIBILITY):
        nodes.append({"id": f"source:{source}", "type": "evidence_source", "label": source})

    collector = _RelationCollector()
    phase_ids: dict[str, str] = {}
    for phase in events.phases:
        phase_id = str(phase["phase_id"])
        phase_ids[str(phase["phase"])] = phase_id
        nodes.append({"id": phase_id, "type": "phase", "attributes": dict(phase)})
        collector.add(run_node, "contains", phase_id)
        collector.add(phase_id, "derived_from", f"source:{SOURCE_MANUAL_PHASE}")

    anatomy_ids: set[str] = set()
    for event in events.events:
        value = event_to_dict(event)
        nodes.append({"id": event.event_id, "type": event.event_type, "attributes": value})
        collector.add(run_node, "contains", event.event_id)
        collector.add(event.event_id, "derived_from", f"source:{_source_for(event.event_type)}")
        if event.phase in phase_ids:
            collector.add(event.event_id, "occurs_during", phase_ids[str(event.phase)])
        entity = event.anatomy or event.attributes.get("boundary")
        if entity:
            entity_id = f"anatomy:{entity}"
            if entity_id not in anatomy_ids:
                nodes.append({"id": entity_id, "type": "anatomy", "label": entity})
                anatomy_ids.add(entity_id)
            collector.add(
                event.event_id,
                "simulator_overshoot_involves" if event.event_type == "simulator_overshoot" else "working_region_near",
                entity_id,
            )

    safety = [event for event in events.events if event.event_type == "simulator_overshoot"]
    boundary = [event for event in events.events if event.event_type == "boundary_work"]
    performance = [event for event in events.events if event.event_type == "performance_bout"]
    visibility = [
        event for event in events.events
        if event.event_type in {"under_ledge_candidate", "tip_occlusion"}
    ]
    for left in safety:
        for right in boundary:
            if _overlap(left, right):
                collector.add(left.event_id, "temporally_overlaps", right.event_id)
        for right in performance:
            if _overlap(left, right):
                collector.add(left.event_id, "temporally_overlaps_performance", right.event_id)
    for left in visibility:
        for right in performance:
            if _overlap(left, right):
                collector.add(left.event_id, "temporally_overlaps_performance", right.event_id)

    for context in events.motion_contexts:
        context_id = str(context["motion_context_id"])
        nodes.append({"id": context_id, "type": "motion_context", "attributes": dict(context)})
        collector.add(str(context["event_id"]), "has_motion_context", context_id)
        collector.add(context_id, "derived_from", f"source:{SOURCE_STROKES}")
        phase = str(context.get("phase") or "")
        if phase in phase_ids:
            collector.add(context_id, "occurs_during", phase_ids[phase])

    relations = collector.values
    graph = {
        "schema": "evidence_relation_graph_v1",
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "construction": {
            "event_rule_version": events.manifest["rule_version"],
            "relation_rule_version": RULE_VERSION,
            "causal_edges_allowed": False,
            "llm_used": False,
        },
        "nodes": nodes,
        "edges": [relation_to_dict(relation) for relation in relations],
    }
    manifest = {
        "schema": "deterministic_relation_build_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": record.run_id,
        "global_run_id": record.global_run_id,
        "rule_version": RULE_VERSION,
        "node_counts": dict(Counter(node["type"] for node in nodes)),
        "relation_counts": dict(Counter(relation.relation_type for relation in relations)),
        "factual_only": True,
        "forbidden_claims": [
            "caused_by",
            "indicates_insufficient_exposure",
            "caused_injury",
            "should_drill_deeper",
            "should_widen_corridor",
        ],
    }
    return RelationBuild(relations, graph, manifest)


def relation_to_dict(relation: EvidenceRelation) -> dict[str, Any]:
    return asdict(relation)


def _legacy_phase(value: str | None) -> str | None:
    return {"early_surface": "early_surface", "antrum_entry": "antrum", "deep_landmark": "incus"}.get(
        str(value), value
    )


def compare_legacy_graph(build: RelationBuild, legacy_path: Path) -> dict[str, Any]:
    """Validation-only comparison; the legacy graph is never a construction input."""
    if not legacy_path.is_file():
        return {"available": False, "legacy_path": str(legacy_path)}
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    new_events = [
        node for node in build.graph["nodes"]
        if node["type"] in {
            "simulator_overshoot", "boundary_work", "performance_bout",
            "under_ledge_candidate", "tip_occlusion",
        }
    ]
    old_events = [
        node for node in legacy["nodes"]
        if node["type"] in {
            "simulator_overshoot", "boundary_work", "performance_bout",
            "under_ledge_candidate", "tip_occlusion",
        }
    ]

    def signature(node: Mapping[str, Any], *, old: bool) -> tuple[Any, ...]:
        attrs = node.get("attributes") or {}
        if old:
            start, end = attrs.get("t_start"), attrs.get("t_end")
            phase = _legacy_phase(attrs.get("phase"))
        else:
            start, end = attrs.get("t_start_s"), attrs.get("t_end_s")
            phase = attrs.get("phase")
        extra = attrs.get("anatomy") or attrs.get("boundary")
        if not extra and isinstance(attrs.get("attributes"), Mapping):
            extra = attrs["attributes"].get("boundary")
        return (
            str(node["type"]), str(phase), str(extra or ""),
            round(float(start), 3), round(float(end), 3),
        )

    old_signatures = Counter(signature(node, old=True) for node in old_events)
    new_signatures = Counter(signature(node, old=False) for node in new_events)
    missing = list((old_signatures - new_signatures).elements())
    extra = list((new_signatures - old_signatures).elements())
    old_counts = Counter(node["type"] for node in legacy["nodes"])
    new_counts = Counter(node["type"] for node in build.graph["nodes"])
    old_relations = Counter(edge["relation"] for edge in legacy["edges"])
    new_relations = Counter(edge["relation_type"] for edge in build.graph["edges"])

    def merged_boundary_geometry(nodes: Sequence[Mapping[str, Any]], *, old: bool) -> list[tuple[str, float, float]]:
        grouped: dict[str, list[tuple[float, float]]] = {}
        for node in nodes:
            if node["type"] != "boundary_work":
                continue
            attrs = node.get("attributes") or {}
            nested = attrs.get("attributes") or {}
            boundary_name = str(attrs.get("boundary") or nested.get("boundary") or attrs.get("anatomy") or "")
            start_key, end_key = (("t_start", "t_end") if old else ("t_start_s", "t_end_s"))
            grouped.setdefault(boundary_name, []).append((float(attrs[start_key]), float(attrs[end_key])))
        result = []
        for boundary_name, intervals in sorted(grouped.items()):
            current_start: float | None = None
            current_end: float | None = None
            for start, end in sorted(intervals):
                if current_start is not None and current_end is not None and start - current_end > 10.0:
                    result.append((boundary_name, round(current_start, 3), round(current_end, 3)))
                    current_start, current_end = start, end
                else:
                    current_start = start if current_start is None else min(current_start, start)
                    current_end = end if current_end is None else max(current_end, end)
            if current_start is not None and current_end is not None:
                result.append((boundary_name, round(current_start, 3), round(current_end, 3)))
        return result

    old_boundary_geometry = merged_boundary_geometry(old_events, old=True)
    new_boundary_geometry = merged_boundary_geometry(new_events, old=False)
    return {
        "available": True,
        "legacy_path": str(legacy_path),
        "role": "validation_only",
        "event_signature_exact": not missing and not extra,
        "legacy_event_count": len(old_events),
        "new_event_count": len(new_events),
        "missing_legacy_event_signatures": missing,
        "extra_new_event_signatures": extra,
        "legacy_node_counts": dict(old_counts),
        "new_node_counts": dict(new_counts),
        "legacy_relation_counts": dict(old_relations),
        "new_relation_counts": dict(new_relations),
        "boundary_geometry_equivalent_ignoring_legacy_phase_clip": old_boundary_geometry == new_boundary_geometry,
        "legacy_boundary_geometry": old_boundary_geometry,
        "new_boundary_geometry": new_boundary_geometry,
        "known_phase_semantics_difference": (
            "New phase nodes retain exact human annotation bounds; the legacy graph used observed drilling coverage."
        ),
        "known_motion_semantics_difference": (
            "Overlap relations use full-precision timestamps; older comparison "
            "artifacts may have rounded event endpoints before overlap testing."
        ),
    }


def write_relation_build(
    output_dir: Path,
    build: RelationBuild,
    regression: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "relations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for relation in build.relations:
            handle.write(json.dumps(relation_to_dict(relation), ensure_ascii=False) + "\n")
    (output_dir / "evidence_graph.json").write_text(
        json.dumps(build.graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "relation_manifest.json").write_text(
        json.dumps(build.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "legacy_graph_regression.json").write_text(
        json.dumps(dict(regression), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
