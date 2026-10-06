"""Allowlisted, parameterized evidence retrieval API.

No method accepts SQL, a filesystem path, or an unconstrained table/column name.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, Mapping, Sequence


MAX_ROWS = 500

TOOL_DESCRIPTIONS = {
    "get_run_summary": ("Retrieve compact run, phase, and evidence counts.", ["run_id"]),
    "get_phase_summary": ("Retrieve compact evidence statistics for one manual phase.", ["run_id", "phase"]),
    "get_events": ("Filter canonical events by type, phase, status, or time overlap.", ["run_id"]),
    "get_event_details": ("Retrieve one event, its evidence IDs, relations, and motion context.", ["event_id"]),
    "get_event_relations": ("Retrieve the factual subgraph incident to one event.", ["event_id"]),
    "get_recurrent_patterns": ("Count repeated event-type/anatomy/phase combinations.", ["run_id"]),
    "get_motion_context": ("Retrieve post-hoc stroke summaries attached to an event.", ["event_id"]),
    "get_source_frames": ("Page through source frames supporting one event.", ["event_id"]),
    "get_source_strokes": ("Page through source strokes supporting one event.", ["event_id"]),
    "get_review_clip": ("Retrieve a generated review clip or explicit unavailable status.", ["event_id"]),
    "compare_phases": ("Compare compact statistics across one to four manual phases.", ["run_id", "phases"]),
    "get_expert_reference": ("Retrieve registered expert reference or explicit abstention.", ["run_id"]),
    "get_validated_coaching_rule": ("Retrieve only expert-validated coaching rules.", ["event_type"]),
}

TOOL_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_run_summary": {"run_id": "string"},
    "get_phase_summary": {"run_id": "string", "phase": "string"},
    "get_events": {
        "run_id": "string", "event_types": "string[]", "phase": "string",
        "evidence_status": "string", "t_start_s": "number", "t_end_s": "number",
        "limit": "integer",
    },
    "get_event_details": {"event_id": "string"},
    "get_event_relations": {"event_id": "string"},
    "get_recurrent_patterns": {"run_id": "string"},
    "get_motion_context": {"event_id": "string"},
    "get_source_frames": {
        "event_id": "string", "limit": "integer", "offset": "integer",
        "include_observation": "boolean",
    },
    "get_source_strokes": {"event_id": "string", "limit": "integer", "offset": "integer"},
    "get_review_clip": {"event_id": "string"},
    "compare_phases": {"run_id": "string", "phases": "string[]"},
    "get_expert_reference": {"run_id": "string"},
    "get_validated_coaching_rule": {"event_type": "string"},
}


def _decode(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


class EvidenceRetriever:
    def __init__(self, db_path: Path) -> None:
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        self.db_path = db_path
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "get_run_summary": self.get_run_summary,
            "get_phase_summary": self.get_phase_summary,
            "get_events": self.get_events,
            "get_event_details": self.get_event_details,
            "get_event_relations": self.get_event_relations,
            "get_recurrent_patterns": self.get_recurrent_patterns,
            "get_motion_context": self.get_motion_context,
            "get_source_frames": self.get_source_frames,
            "get_source_strokes": self.get_source_strokes,
            "get_review_clip": self.get_review_clip,
            "compare_phases": self.compare_phases,
            "get_expert_reference": self.get_expert_reference,
            "get_validated_coaching_rule": self.get_validated_coaching_rule,
        }

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name][0],
                "required_arguments": TOOL_DESCRIPTIONS[name][1],
                "allowed_arguments": TOOL_INPUT_SCHEMAS[name],
                "access": "read_only_parameterized",
                "returns_evidence_ids": True,
            }
            for name in self.allowed_tools
        ]

    def call(self, tool_name: str, **arguments: Any) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(
                f"Tool is not allowlisted: {tool_name}. Allowed: {', '.join(self.allowed_tools)}"
            )
        return tool(**arguments)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _limit(value: int) -> int:
        value = int(value)
        if value < 1 or value > MAX_ROWS:
            raise ValueError(f"limit must be between 1 and {MAX_ROWS}")
        return value

    def _global_run_id(self, connection: sqlite3.Connection, run_id: str) -> str:
        rows = connection.execute(
            "SELECT global_run_id FROM runs WHERE run_id = ? OR global_run_id = ?",
            (run_id, run_id),
        ).fetchall()
        if not rows:
            raise KeyError(f"Unknown run_id: {run_id}")
        if len(rows) != 1:
            raise KeyError(f"Ambiguous run_id; use global_run_id: {run_id}")
        return str(rows[0]["global_run_id"])

    @staticmethod
    def _event_row(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        event_id = str(row["event_id"])
        evidence = connection.execute(
            "SELECT evidence_id, evidence_type FROM event_evidence WHERE event_id = ? ORDER BY evidence_id",
            (event_id,),
        ).fetchall()
        return {
            "event_id": event_id,
            "event_type": row["event_type"],
            "t_start_s": row["t_start_s"],
            "t_end_s": row["t_end_s"],
            "phase": row["phase_name"],
            "anatomy": row["anatomy"],
            "evidence_status": row["evidence_status"],
            "lifecycle_status": row["lifecycle_status"],
            "rule_version": row["rule_version"],
            "attributes": _decode(row["attributes_json"], {}),
            "source_evidence_ids": [item["evidence_id"] for item in evidence],
            "source_evidence_types": dict(Counter(item["evidence_type"] for item in evidence)),
        }

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            global_run_id = self._global_run_id(connection, run_id)
            run = dict(connection.execute("SELECT * FROM runs WHERE global_run_id = ?", (global_run_id,)).fetchone())
            counts = {}
            for table in ("frames", "strokes", "events", "relations", "motion_contexts"):
                counts[table] = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE global_run_id = ?", (global_run_id,)
                    ).fetchone()[0]
                )
            event_counts = {
                row["event_type"]: int(row["n"])
                for row in connection.execute(
                    "SELECT event_type, COUNT(*) AS n FROM events WHERE global_run_id = ? GROUP BY event_type ORDER BY event_type",
                    (global_run_id,),
                )
            }
            status_counts = {
                row["evidence_status"]: int(row["n"])
                for row in connection.execute(
                    "SELECT evidence_status, COUNT(*) AS n FROM events WHERE global_run_id = ? GROUP BY evidence_status ORDER BY evidence_status",
                    (global_run_id,),
                )
            }
            phases = [
                {
                    **dict(row),
                    "segment_role": "pre_drilling_context" if row["phase_name"] == "pre_drilling" else "annotated_drilling_phase",
                }
                for row in connection.execute(
                    "SELECT phase_id, phase_name, t_start_s, t_end_s FROM phases WHERE global_run_id = ? ORDER BY t_start_s",
                    (global_run_id,),
                )
            ]
            return {
                "tool": "get_run_summary",
                "run": run,
                "counts": counts,
                "event_counts": event_counts,
                "evidence_status_counts": status_counts,
                "phases": phases,
                "annotated_drilling_phase_count": sum(
                    row["segment_role"] == "annotated_drilling_phase" for row in phases
                ),
                "pre_drilling_context_count": sum(
                    row["segment_role"] == "pre_drilling_context" for row in phases
                ),
                "evidence_ids": [global_run_id],
            }

    def get_events(
        self,
        run_id: str,
        *,
        event_types: Sequence[str] | None = None,
        phase: str | None = None,
        evidence_status: str | None = None,
        t_start_s: float | None = None,
        t_end_s: float | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        with self._connect() as connection:
            global_run_id = self._global_run_id(connection, run_id)
            clauses = ["e.global_run_id = ?"]
            parameters: list[Any] = [global_run_id]
            if event_types:
                values = [str(value) for value in event_types]
                clauses.append(f"e.event_type IN ({','.join('?' for _ in values)})")
                parameters.extend(values)
            if phase is not None:
                clauses.append("p.phase_name = ?")
                parameters.append(phase)
            if evidence_status is not None:
                clauses.append("e.evidence_status = ?")
                parameters.append(evidence_status)
            if t_start_s is not None:
                clauses.append("e.t_end_s >= ?")
                parameters.append(float(t_start_s))
            if t_end_s is not None:
                clauses.append("e.t_start_s <= ?")
                parameters.append(float(t_end_s))
            parameters.append(limit)
            rows = connection.execute(
                "SELECT e.*, p.phase_name FROM events e LEFT JOIN phases p ON p.phase_id = e.phase_id "
                f"WHERE {' AND '.join(clauses)} ORDER BY e.t_start_s, e.event_type, e.event_id LIMIT ?",
                parameters,
            ).fetchall()
            events = [self._event_row(connection, row) for row in rows]
            return {
                "tool": "get_events",
                "run_id": run_id,
                "filters": {
                    "event_types": list(event_types or []), "phase": phase,
                    "evidence_status": evidence_status, "t_start_s": t_start_s, "t_end_s": t_end_s,
                },
                "count": len(events),
                "events": events,
                "evidence_ids": [event["event_id"] for event in events],
            }

    def get_event_relations(self, event_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,)).fetchone()
            if exists is None:
                raise KeyError(f"Unknown event_id: {event_id}")
            outgoing = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM relations WHERE source_id = ? ORDER BY relation_type, target_id", (event_id,)
                )
            ]
            incoming = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM relations WHERE target_id = ? ORDER BY relation_type, source_id", (event_id,)
                )
            ]
            for row in [*outgoing, *incoming]:
                row["attributes"] = _decode(row.pop("attributes_json"), {})
                row["provisional"] = bool(row["provisional"])
            motion_ids = [
                row["target_id"] for row in outgoing
                if row["relation_type"] == "has_motion_context"
            ]
            context_relations = []
            for motion_id in motion_ids:
                values = connection.execute(
                    "SELECT * FROM relations WHERE source_id = ? ORDER BY relation_type, target_id",
                    (motion_id,),
                ).fetchall()
                for value in values:
                    item = dict(value)
                    item["attributes"] = _decode(item.pop("attributes_json"), {})
                    item["provisional"] = bool(item["provisional"])
                    context_relations.append(item)
            return {
                "tool": "get_event_relations",
                "event_id": event_id,
                "outgoing": outgoing,
                "incoming": incoming,
                "motion_context_outgoing": context_relations,
                "evidence_ids": [
                    event_id,
                    *[row["relation_id"] for row in [*outgoing, *incoming, *context_relations]],
                    *[row["target_id"] for row in outgoing],
                    *[row["source_id"] for row in incoming],
                    *[row["target_id"] for row in context_relations],
                ],
            }

    def get_motion_context(self, event_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM motion_contexts WHERE event_id = ? ORDER BY motion_context_id", (event_id,)
            ).fetchall()
            contexts = [_decode(row["context_json"], {}) for row in rows]
            return {
                "tool": "get_motion_context",
                "event_id": event_id,
                "count": len(contexts),
                "contexts": contexts,
                "evidence_ids": [
                    evidence_id for context in contexts
                    for evidence_id in context.get("source_evidence_ids", [])
                ],
            }

    def get_event_details(self, event_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT e.*, p.phase_name FROM events e LEFT JOIN phases p ON p.phase_id = e.phase_id WHERE e.event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown event_id: {event_id}")
            event = self._event_row(connection, row)
        relations = self.get_event_relations(event_id)
        motion = self.get_motion_context(event_id)
        return {
            "tool": "get_event_details",
            "event": event,
            "relations": {
                "outgoing": relations["outgoing"],
                "incoming": relations["incoming"],
                "motion_context_outgoing": relations["motion_context_outgoing"],
            },
            "motion_contexts": motion["contexts"],
            "traceability": {
                "frame_count": event["source_evidence_types"].get("frame", 0),
                "stroke_count": event["source_evidence_types"].get("stroke", 0),
            },
            "evidence_ids": list(dict.fromkeys([
                event_id,
                *event["source_evidence_ids"],
                *relations["evidence_ids"],
            ])),
        }

    def get_source_frames(
        self,
        event_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
        include_observation: bool = False,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        offset = max(0, int(offset))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT f.* FROM event_evidence ee JOIN frames f ON f.frame_evidence_id = ee.evidence_id "
                "WHERE ee.event_id = ? AND ee.evidence_type = 'frame' ORDER BY f.t_video_s, f.frame_idx LIMIT ? OFFSET ?",
                (event_id, limit, offset),
            ).fetchall()
            total = int(connection.execute(
                "SELECT COUNT(*) FROM event_evidence WHERE event_id = ? AND evidence_type = 'frame'", (event_id,)
            ).fetchone()[0])
            frames = []
            for row in rows:
                item = {
                        "frame_evidence_id": row["frame_evidence_id"],
                        "frame_idx": row["frame_idx"],
                        "t_video_s": row["t_video_s"],
                        "phase_id": row["phase_id"],
                        "source_file_id": row["source_file_id"],
                        "source_row": row["source_row"],
                    }
                if include_observation:
                    item["observation"] = _decode(row["observation_json"], {})
                frames.append(item)
            return {
                "tool": "get_source_frames", "event_id": event_id,
                "total": total, "offset": offset, "count": len(frames), "frames": frames,
                "evidence_ids": [row["frame_evidence_id"] for row in frames],
            }

    def get_source_strokes(self, event_id: str, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        limit = self._limit(limit)
        offset = max(0, int(offset))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT s.* FROM event_evidence ee JOIN strokes s ON s.stroke_evidence_id = ee.evidence_id "
                "WHERE ee.event_id = ? AND ee.evidence_type = 'stroke' ORDER BY s.t_start_s, s.stroke_id LIMIT ? OFFSET ?",
                (event_id, limit, offset),
            ).fetchall()
            strokes = [
                {
                    "stroke_evidence_id": row["stroke_evidence_id"],
                    "stroke_id": row["stroke_id"],
                    "t_start_s": row["t_start_s"], "t_end_s": row["t_end_s"],
                    "metrics": _decode(row["metrics_json"], {}),
                }
                for row in rows
            ]
            total = int(connection.execute(
                "SELECT COUNT(*) FROM event_evidence WHERE event_id = ? AND evidence_type = 'stroke'", (event_id,)
            ).fetchone()[0])
            return {
                "tool": "get_source_strokes", "event_id": event_id,
                "total": total, "offset": offset, "count": len(strokes), "strokes": strokes,
                "evidence_ids": [row["stroke_evidence_id"] for row in strokes],
            }

    def get_phase_summary(self, run_id: str, phase: str) -> dict[str, Any]:
        with self._connect() as connection:
            global_run_id = self._global_run_id(connection, run_id)
            phase_row = connection.execute(
                "SELECT * FROM phases WHERE global_run_id = ? AND phase_name = ?", (global_run_id, phase)
            ).fetchone()
            if phase_row is None:
                raise KeyError(f"Unknown phase for {run_id}: {phase}")
            frame_rows = connection.execute(
                "SELECT frame_evidence_id, observation_json FROM frames WHERE phase_id = ? ORDER BY frame_idx",
                (phase_row["phase_id"],),
            ).fetchall()
            observations = [_decode(row["observation_json"], {}) for row in frame_rows]
            event_rows = connection.execute(
                "SELECT event_type, COUNT(*) AS n FROM events WHERE phase_id = ? GROUP BY event_type ORDER BY event_type",
                (phase_row["phase_id"],),
            ).fetchall()
            relation_rows = connection.execute(
                "SELECT * FROM relations WHERE source_id = ? OR target_id = ? ORDER BY relation_id",
                (phase_row["phase_id"], phase_row["phase_id"]),
            ).fetchall()
            phase_relations = []
            for relation in relation_rows:
                item = dict(relation)
                item["attributes"] = _decode(item.pop("attributes_json"), {})
                item["provisional"] = bool(item["provisional"])
                phase_relations.append(item)
            return {
                "tool": "get_phase_summary",
                "run_id": run_id,
                "phase": dict(phase_row),
                "frame_count": len(frame_rows),
                "drilling_frame_count": sum(bool(row.get("is_drilling_frame")) for row in observations),
                "removed_volume_mm3": sum(float(row.get("frame_removed_mm3") or 0) for row in observations),
                "event_counts": {row["event_type"]: int(row["n"]) for row in event_rows},
                "relations": phase_relations,
                "evidence_ids": [phase_row["phase_id"]],
            }

    def compare_phases(self, run_id: str, phases: Sequence[str]) -> dict[str, Any]:
        if not phases or len(phases) > 4:
            raise ValueError("compare_phases requires one to four phase names")
        summaries = [self.get_phase_summary(run_id, phase) for phase in phases]
        return {
            "tool": "compare_phases", "run_id": run_id,
            "phases": [
                {
                    "phase_id": row["phase"]["phase_id"],
                    "phase": row["phase"]["phase_name"],
                    "frame_count": row["frame_count"],
                    "drilling_frame_count": row["drilling_frame_count"],
                    "removed_volume_mm3": row["removed_volume_mm3"],
                    "event_counts": row["event_counts"],
                }
                for row in summaries
            ],
            "evidence_ids": [evidence_id for row in summaries for evidence_id in row["evidence_ids"]],
        }

    def get_recurrent_patterns(self, run_id: str) -> dict[str, Any]:
        events = self.get_events(run_id, limit=MAX_ROWS)["events"]
        counts = Counter((row["event_type"], row.get("anatomy") or "none", row.get("phase") or "none") for row in events)
        patterns = [
            {"event_type": key[0], "anatomy": key[1], "phase": key[2], "count": count}
            for key, count in sorted(counts.items()) if count > 1
        ]
        return {
            "tool": "get_recurrent_patterns", "run_id": run_id,
            "definition": "same event_type, anatomy, and phase occurring more than once",
            "patterns": patterns,
            "evidence_ids": [row["event_id"] for row in events],
        }

    def get_review_clip(self, event_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM review_clips WHERE event_id = ?", (event_id,)).fetchone()
            return {
                "tool": "get_review_clip", "event_id": event_id,
                "available": row is not None,
                "clip": dict(row) if row is not None else None,
                "status": row["status"] if row is not None else "not_generated",
                "evidence_ids": [event_id],
            }

    def get_expert_reference(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            global_run_id = self._global_run_id(connection, run_id)
            return {
                "tool": "get_expert_reference", "run_id": run_id,
                "available": False,
                "reason": "No expert feedback reference is registered; legacy graphs are validation-only.",
                "evidence_ids": [global_run_id],
            }

    def get_validated_coaching_rule(self, event_type: str) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT rule_id, rule_json FROM knowledge_rules WHERE rule_status = 'expert_validated' ORDER BY rule_id"
            ).fetchall()
            matching = []
            for row in rows:
                value = _decode(row["rule_json"], {})
                if value.get("event_type") == event_type:
                    matching.append({"rule_id": row["rule_id"], "rule": value})
            return {
                "tool": "get_validated_coaching_rule", "event_type": event_type,
                "available": bool(matching), "rules": matching,
                "reason": None if matching else "No expert-validated coaching rule is available.",
                "evidence_ids": [row["rule_id"] for row in matching],
            }

