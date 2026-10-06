"""Small dependency-free domain contracts used by early milestones."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    SIMULATOR_SUPPORTED = "simulator_supported"
    HUMAN_CONFIRMED = "human_confirmed"
    HUMAN_REJECTED = "human_rejected"


class LifecycleStatus(str, Enum):
    OPEN = "open"
    UPDATED = "updated"
    CLOSED = "closed"


class RuleStatus(str, Enum):
    DRAFT = "draft"
    EXPERT_VALIDATED = "expert_validated"
    RETIRED = "retired"


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    dataset_id: str
    run_id: str
    event_type: str
    t_start_s: float
    t_end_s: float
    evidence_status: EvidenceStatus
    source_evidence_ids: tuple[str, ...]
    rule_version: str
    phase: str | None = None
    anatomy: str | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.CLOSED
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_start_s < 0 or self.t_end_s < self.t_start_s:
            raise ValueError("Event times must satisfy 0 <= start <= end")
        if not self.source_evidence_ids:
            raise ValueError("Every event requires at least one source evidence ID")
        candidate_event_types = {"tip_occlusion", "under_ledge_candidate"}
        if self.event_type in candidate_event_types and self.evidence_status not in {
            EvidenceStatus.CANDIDATE,
            EvidenceStatus.HUMAN_CONFIRMED,
            EvidenceStatus.HUMAN_REJECTED,
        }:
            raise ValueError("Candidate events cannot silently become simulator-confirmed facts")


@dataclass(frozen=True)
class EvidenceRelation:
    relation_id: str
    source_id: str
    relation_type: str
    target_id: str
    construction_method: str
    rule_version: str
    provisional: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

