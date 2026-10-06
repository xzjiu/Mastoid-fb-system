"""Local evidence-feedback web service with explicit read/write app modes."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .coaching_plan import build_session_coaching_plan, load_cohort_benchmark
from .ray_v3_view import build_ray_v3_view
from .retrieval import EvidenceRetriever


APP_MODES = {"recommendations", "expert-study"}
EXPERT_STUDY_SCHEMA = "expert_feedback_study_v1"
DEFAULT_INTERFACE_VERSION = "expert-feedback-v1"
EXPERT_AUDIO_MAX_BYTES = 8 * 1024 * 1024
EXPERT_AUDIO_MIME_TYPES = {
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}
EXPERT_STUDY_EVENT_TYPES = {
    "study_started", "study_resumed", "stimulus_opened", "video_play",
    "video_pause", "video_seek", "blind_submitted", "feedback_revealed",
    "revealed_submitted", "case_completed", "next_case_opened",
    "analysis_started", "analysis_completed", "feedback_item_opened",
    "feedback_clip_played", "feedback_signal_frame_opened",
    "support_details_opened", "experience_completed",
    "expert_moment_saved", "expert_moment_deleted", "audio_recording_started",
}
PARTICIPANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _official_ui_directory(mode: str) -> Path:
    if mode not in APP_MODES:
        raise ValueError(f"Unsupported app mode: {mode}")
    directory = {
        "recommendations": "recommendations",
        "expert-study": "expert_study",
    }[mode]
    return (Path(__file__).resolve().parents[2] / "apps" / directory).resolve()


def _validate_ui_directory(mode: str, ui_dir: Path) -> Path:
    expected = _official_ui_directory(mode)
    actual = ui_dir.expanduser().resolve()
    if actual != expected:
        raise ValueError(f"{mode} mode must use {expected}; got {actual}")
    return actual


def normalize_participant_id(value: str) -> str:
    participant_id = value.strip()
    if not PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
        raise ValueError(
            "Study ID must be 1 to 64 letters, numbers, periods, dashes, or underscores"
        )
    return participant_id


class ExpertStudyStore:
    """Atomic local study storage, separate from the read-only evidence database."""

    def __init__(self, path: Path, interface_version: str) -> None:
        self.path = path
        self.interface_version = interface_version
        self._lock = threading.Lock()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema": "local_expert_study_state_v1",
            "expert_study": {
                "schema": EXPERT_STUDY_SCHEMA,
                "sessions": {},
                "events": [],
                "expert_proposed_moments": {},
            },
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        study = dict(payload.get("expert_study") or {})
        return {
            "schema": "local_expert_study_state_v1",
            "expert_study": {
                "schema": EXPERT_STUDY_SCHEMA,
                "sessions": dict(study.get("sessions") or {}),
                "events": list(study.get("events") or []),
                "expert_proposed_moments": dict(
                    study.get("expert_proposed_moments") or {}
                ),
            },
        }

    def _write_unlocked(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=self.path.parent,
            prefix="study-state-",
            suffix=".tmp",
        ) as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.path)

    @staticmethod
    def _session_key(participant_id: str, run_id: str) -> str:
        return f"{participant_id}::{run_id}"

    def progress(self, participant_id: str, run_id: str) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        with self._lock:
            study = self._read_unlocked()["expert_study"]
            return dict(
                study["sessions"].get(self._session_key(participant_id, run_id))
                or {}
            )

    def start_session(
        self,
        *,
        participant_id: str,
        run_id: str,
        stimulus_ids: Sequence[str],
    ) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        key = self._session_key(participant_id, run_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            payload = self._read_unlocked()
            sessions = payload["expert_study"]["sessions"]
            existing = dict(sessions.get(key) or {})
            row = {
                **existing,
                "participant_id": participant_id,
                "run_id": run_id,
                "stimulus_ids": list(stimulus_ids),
                "started_at": existing.get("started_at") or now,
                "last_opened_at": now,
                "case_completed_at": existing.get("case_completed_at"),
                "interface_version_at_start": (
                    existing.get("interface_version_at_start")
                    or self.interface_version
                ),
                "interface_version_last_opened": self.interface_version,
            }
            sessions[key] = row
            self._write_unlocked(payload)
        return row

    def add_event(
        self,
        *,
        participant_id: str,
        run_id: str,
        event_type: str,
        stimulus_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        if event_type not in EXPERT_STUDY_EVENT_TYPES:
            raise ValueError(f"Unsupported expert-study event: {event_type}")
        encoded = json.dumps(dict(details or {}), ensure_ascii=False)
        if len(encoded) > 4000:
            raise ValueError("Expert-study event details are too large")
        row = {
            "event_id": str(uuid4()),
            "participant_id": participant_id,
            "run_id": run_id,
            "stimulus_id": stimulus_id,
            "event_type": event_type,
            "interface_version": self.interface_version,
            "details": dict(details or {}),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            payload = self._read_unlocked()
            payload["expert_study"]["events"].append(row)
            self._write_unlocked(payload)
        return row

    def moments(self, participant_id: str, run_id: str) -> list[dict[str, Any]]:
        participant_id = normalize_participant_id(participant_id)
        with self._lock:
            values = self._read_unlocked()["expert_study"][
                "expert_proposed_moments"
            ].values()
            return sorted(
                [
                    dict(row)
                    for row in values
                    if row.get("participant_id") == participant_id
                    and row.get("run_id") == run_id
                ],
                key=lambda row: (
                    float(row.get("t_video_s") or 0.0),
                    str(row.get("created_at") or ""),
                ),
            )

    def add_moment(
        self,
        *,
        participant_id: str,
        run_id: str,
        t_video_s: float,
        category: str,
        observation: str,
        coaching_message: str,
        system_coverage: str,
        importance: str,
        audio_bytes: bytes | None = None,
        audio_mime_type: str | None = None,
        audio_duration_s: float | None = None,
    ) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        if t_video_s < 0:
            raise ValueError("Coaching-moment time must be non-negative")
        if category not in {
            "exposure", "safety", "visualization", "efficiency", "other"
        }:
            raise ValueError("Select a supported coaching-moment category")
        if system_coverage not in {"missed", "partial", "addressed", "unsure"}:
            raise ValueError("Select how the automated feedback covered this moment")
        if importance not in {"low", "medium", "high"}:
            raise ValueError("Select the importance of this coaching moment")
        observation = observation.strip()[:4000]
        coaching_message = coaching_message.strip()[:4000]
        if not observation and not coaching_message and not audio_bytes:
            raise ValueError("Type an observation, coaching message, or record audio")
        if audio_duration_s is not None and not 0 <= audio_duration_s <= 300:
            raise ValueError("Audio recording must be five minutes or shorter")
        if audio_bytes is not None and len(audio_bytes) > EXPERT_AUDIO_MAX_BYTES:
            raise ValueError("Audio recording is larger than 8 MiB")

        normalized_mime = (audio_mime_type or "").split(";", 1)[0].strip().lower()
        if audio_bytes is not None and normalized_mime not in EXPERT_AUDIO_MIME_TYPES:
            raise ValueError("Unsupported audio recording format")

        moment_id = str(uuid4())
        audio: dict[str, Any] | None = None
        final_audio: Path | None = None
        if audio_bytes is not None:
            recordings = self.path.parent / "recordings"
            recordings.mkdir(parents=True, exist_ok=True)
            extension = EXPERT_AUDIO_MIME_TYPES[normalized_mime]
            final_audio = recordings / f"{moment_id}{extension}"
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=recordings,
                prefix="recording-",
                suffix=".tmp",
            ) as handle:
                handle.write(audio_bytes)
                temporary_audio = Path(handle.name)
            temporary_audio.replace(final_audio)
            audio = {
                "file_name": final_audio.name,
                "mime_type": normalized_mime,
                "duration_s": round(float(audio_duration_s or 0.0), 3),
                "size_bytes": len(audio_bytes),
            }

        row = {
            "moment_id": moment_id,
            "participant_id": participant_id,
            "run_id": run_id,
            "t_video_s": round(float(t_video_s), 3),
            "category": category,
            "observation": observation,
            "coaching_message": coaching_message,
            "system_coverage": system_coverage,
            "importance": importance,
            "audio": audio,
            "interface_version": self.interface_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with self._lock:
                payload = self._read_unlocked()
                payload["expert_study"]["expert_proposed_moments"][moment_id] = row
                self._write_unlocked(payload)
        except Exception:
            if final_audio is not None:
                final_audio.unlink(missing_ok=True)
            raise
        return row

    def delete_moment(
        self, participant_id: str, run_id: str, moment_id: str
    ) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        with self._lock:
            payload = self._read_unlocked()
            moments = payload["expert_study"]["expert_proposed_moments"]
            row = dict(moments.get(moment_id) or {})
            if (
                not row
                or row.get("participant_id") != participant_id
                or row.get("run_id") != run_id
            ):
                raise KeyError(moment_id)
            del moments[moment_id]
            self._write_unlocked(payload)
        filename = Path(str((row.get("audio") or {}).get("file_name") or "")).name
        if filename:
            (self.path.parent / "recordings" / filename).unlink(missing_ok=True)
        return row

    def audio(
        self, participant_id: str, run_id: str, moment_id: str
    ) -> tuple[Path, str]:
        participant_id = normalize_participant_id(participant_id)
        with self._lock:
            row = dict(
                self._read_unlocked()["expert_study"][
                    "expert_proposed_moments"
                ].get(moment_id)
                or {}
            )
        if (
            not row
            or row.get("participant_id") != participant_id
            or row.get("run_id") != run_id
        ):
            raise KeyError(moment_id)
        audio = row.get("audio") or {}
        filename = Path(str(audio.get("file_name") or "")).name
        path = self.path.parent / "recordings" / filename
        if not filename or not path.is_file():
            raise KeyError(moment_id)
        return path, str(audio.get("mime_type") or "application/octet-stream")

    def complete(
        self,
        *,
        participant_id: str,
        run_id: str,
        answers: Mapping[str, Any],
    ) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        key = self._session_key(participant_id, run_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            payload = self._read_unlocked()
            sessions = payload["expert_study"]["sessions"]
            session = dict(sessions.get(key) or {})
            if not session:
                raise ValueError(
                    "Start the expert-study experience before completing it"
                )
            session["experience_response"] = {
                "answers": dict(answers),
                "submitted_at": now,
            }
            session["case_completed_at"] = now
            session["interface_version_at_completion"] = self.interface_version
            sessions[key] = session
            self._write_unlocked(payload)
        return session


class FeedbackService:
    def __init__(
        self,
        *,
        mode: str,
        db_path: Path,
        run_id: str,
        video_path: Path | None,
        video_manifest_path: Path | None,
        ray_v3_dir: Path | None,
        cohort_benchmark_path: Path | None,
        available_runs: Sequence[Mapping[str, str]],
        display_name: str,
        interface_version: str,
        study_state_path: Path | None = None,
    ) -> None:
        if mode not in APP_MODES:
            raise ValueError(f"Unsupported app mode: {mode}")
        if mode == "recommendations" and study_state_path is not None:
            raise ValueError("Recommendations mode cannot be given a study-state path")
        if mode == "expert-study" and study_state_path is None:
            raise ValueError("Expert-study mode requires a study-state path")
        self.mode = mode
        self.retriever = EvidenceRetriever(db_path)
        self.run_id = run_id
        self.video_path = video_path
        self.video_manifest_path = video_manifest_path
        self.ray_v3_dir = ray_v3_dir
        self.cohort_benchmark_path = cohort_benchmark_path
        self.available_runs = [dict(row) for row in available_runs]
        self.display_name = display_name
        self.interface_version = interface_version
        self.study_store = (
            ExpertStudyStore(study_state_path, interface_version)
            if study_state_path is not None
            else None
        )

    def media_summary(self) -> dict[str, Any]:
        available = bool(self.video_path and self.video_path.is_file())
        manifest: dict[str, Any] = {}
        if available and self.video_manifest_path and self.video_manifest_path.is_file():
            manifest = json.loads(
                self.video_manifest_path.read_text(encoding="utf-8")
            )
        return {
            "available": available,
            "url": "/api/media/video" if available else None,
            "mime_type": manifest.get("browser_mime_type", "video/webm"),
            "duration_s": manifest.get("duration_s"),
            "fps": manifest.get("fps"),
            "color_order_validated": (manifest.get("validation") or {}).get(
                "color_order_check_passed"
            ),
        }

    def bootstrap(self) -> dict[str, Any]:
        summary = self.retriever.get_run_summary(self.run_id)
        events = self.retriever.get_events(self.run_id, limit=500)["events"]
        phases = summary["phases"]
        media = self.media_summary()
        duration = max(
            max((float(row["t_end_s"]) for row in phases), default=0.0),
            max((float(row["t_end_s"]) for row in events), default=0.0),
            float(media.get("duration_s") or 0.0),
        )
        ray_v3 = build_ray_v3_view(self.run_id, self.ray_v3_dir)
        benchmark = load_cohort_benchmark(self.cohort_benchmark_path)
        coaching_plan = build_session_coaching_plan(
            events, phases, ray_v3, benchmark
        )
        return {
            "schema": "automatic_recommendations_v1",
            "run": {
                "run_id": self.run_id,
                "display_name": self.display_name,
                "duration_s": duration,
                "counts": summary["counts"],
                "event_counts": summary["event_counts"],
                "evidence_status_counts": summary["evidence_status_counts"],
            },
            "available_runs": self.available_runs,
            "phases": phases,
            "events": events,
            "coaching_plan": coaching_plan,
            "media": media,
        }

    def _study(self) -> ExpertStudyStore:
        if self.mode != "expert-study" or self.study_store is None:
            raise RuntimeError("Study-write operations are disabled for this app")
        return self.study_store

    def _study_items(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.bootstrap()["coaching_plan"]["items"]]

    def _case_context(self) -> dict[str, Any]:
        index = next(
            (
                i for i, row in enumerate(self.available_runs)
                if row.get("run_id") == self.run_id
            ),
            0,
        )
        next_url = (
            self.available_runs[index + 1].get("url")
            if index + 1 < len(self.available_runs)
            else None
        )
        return {
            "alias": self.display_name or f"Case {index + 1:02d}",
            "index": index + 1,
            "total": len(self.available_runs),
            "next_case_url": next_url or None,
        }

    @staticmethod
    def _study_clips(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        source = list(
            item.get("review_clips") or item.get("evidence_segments") or []
        )
        if not source:
            source = [{
                "t_start_s": item.get("t_start_s"),
                "t_end_s": item.get("t_end_s"),
            }]
        padding = 3.0 if str(item.get("category") or "") == "exposure" else 5.0
        clips: list[dict[str, Any]] = []
        for row in source:
            start = max(0.0, float(row.get("t_start_s") or 0.0))
            end = max(start, float(row.get("t_end_s") or start))
            anchor = row.get("anchor_t_s")
            clips.append({
                "t_start_s": start,
                "t_end_s": end,
                "playback_start_s": max(0.0, start - padding),
                "playback_end_s": end + padding,
                "signal_time_s": (
                    float(anchor) if anchor is not None else (start + end) / 2.0
                ),
                "exact_signal_frame": anchor is not None,
                "signal_frame_idx": row.get("anchor_frame_idx_global"),
                "anatomy": row.get("anatomy") or item.get("anatomy"),
                "phase": row.get("phase") or item.get("phase"),
            })
        return clips

    @staticmethod
    def _method_details(item: Mapping[str, Any]) -> dict[str, str]:
        category = str(item.get("category") or "")
        limitation = str(item.get("claim_boundary") or "")
        if category == "safety":
            return {
                "data_used": "Time-aligned simulator anatomy and drilling activity.",
                "trigger": (
                    "The simulator indicated a possible protected-structure "
                    "boundary signal that warrants visual review."
                ),
                "selection": (
                    "Moments are grouped by anatomy and linked to recorded video."
                ),
                "limitation": limitation,
            }
        if category == "technique":
            return {
                "data_used": (
                    "Phase-level stroke pattern and procedural progress from the "
                    "recorded simulator session."
                ),
                "trigger": (
                    "The observed phase pattern is compared descriptively with "
                    "the configured reference range."
                ),
                "selection": (
                    "Only phase patterns meeting the configured comparison rule "
                    "are promoted to coaching."
                ),
                "limitation": limitation,
            }
        return {
            "data_used": (
                "Time-linked camera view, burr location, and surrounding-bone "
                "geometry from the recorded simulator session."
            ),
            "trigger": (
                "The system looked for moments when surrounding bone may have "
                "limited a direct view of the working point."
            ),
            "selection": (
                "Nearby signals are combined into short context clips."
            ),
            "limitation": limitation,
        }

    @staticmethod
    def _scale(value: Any, label: str) -> int:
        try:
            rating = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer from 1 to 5") from exc
        if rating not in {1, 2, 3, 4, 5}:
            raise ValueError(f"{label} must be an integer from 1 to 5")
        return rating

    def start_study(self, participant_id: str) -> dict[str, Any]:
        store = self._study()
        participant_id = normalize_participant_id(participant_id)
        items = self._study_items()
        previous = store.progress(participant_id, self.run_id)
        session = store.start_session(
            participant_id=participant_id,
            run_id=self.run_id,
            stimulus_ids=[str(row["feedback_id"]) for row in items],
        )
        store.add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            event_type="study_resumed" if previous else "study_started",
            details={"feedback_count": len(items)},
        )
        case = self._case_context()
        seed = f"{EXPERT_STUDY_SCHEMA}|{participant_id}"
        return {
            "schema": "expert_workflow_study_start_v1",
            "interface_version": self.interface_version,
            "participant_id": participant_id,
            "case": case,
            "media": self.media_summary(),
            "case_completed": bool(session.get("case_completed_at")),
            "workflow": {
                "mode": "recorded_session",
                "disclosure": (
                    "This experience uses a previously recorded simulator "
                    "session. Evidence retrieval and feedback assembly run locally."
                ),
                "stages": [
                    "Load recorded simulator session",
                    "Retrieve time-aligned evidence",
                    "Assemble the coaching plan",
                    "Prepare linked video review",
                ],
            },
            "completion_code": (
                "EXP-" + hashlib.sha256(seed.encode()).hexdigest()[:8].upper()
                if case["next_case_url"] is None
                else None
            ),
            "expert_proposed_moments": self.public_moments(participant_id),
        }

    def analyze_study(self, participant_id: str) -> dict[str, Any]:
        store = self._study()
        participant_id = normalize_participant_id(participant_id)
        if not store.progress(participant_id, self.run_id):
            raise ValueError("Start the expert-study experience before analysis")
        store.add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            event_type="analysis_started",
        )
        bootstrap = self.bootstrap()
        feedback = []
        for item in self._study_items():
            feedback.append({
                **{
                    key: item.get(key)
                    for key in (
                        "feedback_id", "title", "category", "priority", "message",
                        "observed_issue", "why_it_matters", "recommended_action",
                        "evidence_summary", "evidence_status", "claim_boundary",
                        "phase", "t_start_s", "t_end_s",
                    )
                },
                "clips": self._study_clips(item),
                "method_details": self._method_details(item),
            })
        store.add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            event_type="analysis_completed",
            details={"feedback_count": len(feedback)},
        )
        return {
            "schema": "expert_workflow_experience_v1",
            "interface_version": self.interface_version,
            "case": self._case_context(),
            "session": {
                "duration_s": bootstrap["run"]["duration_s"],
                "phase_count": len(bootstrap["phases"]),
                "feedback_count": len(feedback),
            },
            "phases": [
                {
                    "phase_name": row.get("phase_name"),
                    "t_start_s": row.get("t_start_s"),
                    "t_end_s": row.get("t_end_s"),
                }
                for row in bootstrap["phases"]
            ],
            "feedback": feedback,
            "media": bootstrap["media"],
            "expert_proposed_moments": self.public_moments(participant_id),
        }

    def _public_moment(
        self, participant_id: str, row: Mapping[str, Any]
    ) -> dict[str, Any]:
        public = {
            key: row.get(key)
            for key in (
                "moment_id", "t_video_s", "category", "observation",
                "coaching_message", "system_coverage", "importance",
                "interface_version", "created_at",
            )
        }
        audio = row.get("audio") or {}
        public["audio"] = (
            {
                "mime_type": audio.get("mime_type"),
                "duration_s": audio.get("duration_s"),
                "size_bytes": audio.get("size_bytes"),
                "url": (
                    "/api/expert-study/coaching-moment-audio"
                    f"?participant_id={participant_id}&moment_id={row.get('moment_id')}"
                ),
            }
            if audio
            else None
        )
        return public

    def public_moments(self, participant_id: str) -> list[dict[str, Any]]:
        return [
            self._public_moment(participant_id, row)
            for row in self._study().moments(participant_id, self.run_id)
        ]

    def add_moment(
        self, participant_id: str, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        store = self._study()
        participant_id = normalize_participant_id(participant_id)
        session = store.progress(participant_id, self.run_id)
        if not session:
            raise ValueError("Start the expert-study experience before adding a moment")
        if session.get("case_completed_at"):
            raise ValueError("This case has already been completed")
        encoded_audio = str(value.get("audio_base64") or "")
        audio_bytes = None
        if encoded_audio:
            try:
                audio_bytes = base64.b64decode(encoded_audio, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Audio recording could not be decoded") from exc
        row = store.add_moment(
            participant_id=participant_id,
            run_id=self.run_id,
            t_video_s=float(value.get("t_video_s") or 0.0),
            category=str(value.get("category") or ""),
            observation=str(value.get("observation") or ""),
            coaching_message=str(value.get("coaching_message") or ""),
            system_coverage=str(value.get("system_coverage") or ""),
            importance=str(value.get("importance") or ""),
            audio_bytes=audio_bytes,
            audio_mime_type=str(value.get("audio_mime_type") or ""),
            audio_duration_s=(
                float(value["audio_duration_s"])
                if value.get("audio_duration_s") is not None
                else None
            ),
        )
        store.add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            event_type="expert_moment_saved",
            details={
                "moment_id": row["moment_id"],
                "t_video_s": row["t_video_s"],
                "category": row["category"],
                "system_coverage": row["system_coverage"],
                "importance": row["importance"],
                "has_audio": bool(row.get("audio")),
            },
        )
        return {"ok": True, "moment": self._public_moment(participant_id, row)}

    def delete_moment(self, participant_id: str, moment_id: str) -> dict[str, Any]:
        store = self._study()
        participant_id = normalize_participant_id(participant_id)
        if store.progress(participant_id, self.run_id).get("case_completed_at"):
            raise ValueError("This case has already been completed")
        row = store.delete_moment(participant_id, self.run_id, moment_id)
        store.add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            event_type="expert_moment_deleted",
            details={"moment_id": moment_id, "t_video_s": row.get("t_video_s")},
        )
        return {"ok": True, "moment_id": moment_id}

    def complete_study(
        self, participant_id: str, answers: Mapping[str, Any]
    ) -> dict[str, Any]:
        store = self._study()
        participant_id = normalize_participant_id(participant_id)
        use_intent = str(answers.get("would_use") or "")
        if use_intent not in {"yes", "with_changes", "no", "unsure"}:
            raise ValueError("Select whether you would use this workflow")
        validated = {
            "workflow_clarity": self._scale(
                answers.get("workflow_clarity"), "Workflow clarity"
            ),
            "feedback_usefulness": self._scale(
                answers.get("feedback_usefulness"), "Feedback usefulness"
            ),
            "debrief_fit": self._scale(answers.get("debrief_fit"), "Debrief fit"),
            "would_use": use_intent,
            "most_useful": str(answers.get("most_useful") or "").strip()[:4000],
            "missing_or_concerning": str(
                answers.get("missing_or_concerning") or ""
            ).strip()[:4000],
        }
        session = store.complete(
            participant_id=participant_id,
            run_id=self.run_id,
            answers=validated,
        )
        store.add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            event_type="experience_completed",
            details={"would_use": use_intent},
        )
        case = self._case_context()
        seed = f"{EXPERT_STUDY_SCHEMA}|{participant_id}"
        return {
            "ok": True,
            "case_completed_at": session["case_completed_at"],
            "next_case_url": case["next_case_url"],
            "completion_code": (
                None
                if case["next_case_url"]
                else "EXP-" + hashlib.sha256(seed.encode()).hexdigest()[:8].upper()
            ),
        }

    def log_event(
        self,
        participant_id: str,
        stimulus_id: str | None,
        event_type: str,
        details: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        participant_id = normalize_participant_id(participant_id)
        if stimulus_id and stimulus_id not in {
            str(row["feedback_id"]) for row in self._study_items()
        }:
            raise KeyError(stimulus_id)
        row = self._study().add_event(
            participant_id=participant_id,
            run_id=self.run_id,
            stimulus_id=stimulus_id,
            event_type=event_type,
            details=details,
        )
        return {"ok": True, "recorded_at": row["recorded_at"]}


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match:
        raise ValueError("Unsupported byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length < 1:
            raise ValueError("Invalid suffix byte range")
        return max(0, size - length), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Byte range is outside the file")
    return start, min(end, size - 1)


def _handler(
    service: FeedbackService, ui_dir: Path
) -> type[BaseHTTPRequestHandler]:
    assets = {
        "/": (ui_dir / "index.html", "text/html; charset=utf-8"),
        "/index.html": (ui_dir / "index.html", "text/html; charset=utf-8"),
        "/styles.css": (ui_dir / "styles.css", "text/css; charset=utf-8"),
        "/app.js": (ui_dir / "app.js", "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "MastoidFeedback/1.0"

        def _headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")

        def _json(self, status: int, value: Mapping[str, Any]) -> None:
            body = json.dumps(dict(value), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._headers()
            self.end_headers()
            self.wfile.write(body)

        def _body(self, maximum: int = 65536) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 1 or length > maximum:
                raise ValueError(
                    f"Request body must be between 1 and {maximum} bytes"
                )
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object")
            return value

        def _file(
            self, path: Path | None, mime_type: str, *, send_body: bool = True
        ) -> None:
            if path is None or not path.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "Media not available"})
                return
            size = path.stat().st_size
            try:
                requested = parse_byte_range(self.headers.get("Range"), size)
            except (TypeError, ValueError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self._headers()
                self.end_headers()
                return
            start, end = requested or (0, size - 1)
            self.send_response(
                HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK
            )
            self.send_header("Content-Type", mime_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "private, max-age=3600")
            self._headers()
            self.end_headers()
            if not send_body:
                return
            remaining = end - start + 1
            with path.open("rb") as handle:
                handle.seek(start)
                try:
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _audio(self, parsed, *, send_body: bool = True) -> None:
            if service.mode != "expert-study":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            query = parse_qs(parsed.query)
            participant_id = (query.get("participant_id") or [""])[0]
            moment_id = (query.get("moment_id") or [""])[0]
            path, mime_type = service._study().audio(
                participant_id, service.run_id, moment_id
            )
            self._file(path, mime_type, send_body=send_body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self._json(HTTPStatus.OK, {
                        "ok": True,
                        "mode": service.mode,
                        "run_id": service.run_id,
                        "interface_version": service.interface_version,
                        "available_runs": service.available_runs,
                    })
                    return
                if (
                    parsed.path == "/api/bootstrap"
                    and service.mode == "recommendations"
                ):
                    self._json(HTTPStatus.OK, service.bootstrap())
                    return
                if parsed.path == "/api/media/video":
                    self._file(service.video_path, "video/webm")
                    return
                if parsed.path == "/api/expert-study/coaching-moment-audio":
                    self._audio(parsed)
                    return
                asset = assets.get(parsed.path)
                if asset is None or not asset[0].is_file():
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                    return
                body = asset[0].read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", asset[1])
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._headers()
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/media/video":
                    self._file(service.video_path, "video/webm", send_body=False)
                    return
                if parsed.path == "/api/expert-study/coaching-moment-audio":
                    self._audio(parsed, send_body=False)
                    return
            except (ValueError, KeyError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self._headers()
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if service.mode != "expert-study":
                self._json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "Recommendations mode is read-only"},
                )
                return
            parsed = urlparse(self.path)
            try:
                body = self._body(
                    12 * 1024 * 1024
                    if parsed.path == "/api/expert-study/coaching-moment"
                    else 65536
                )
                participant_id = str(body.get("participant_id") or "")
                if parsed.path == "/api/expert-study/start":
                    self._json(HTTPStatus.OK, service.start_study(participant_id))
                    return
                if parsed.path == "/api/expert-study/analyze":
                    self._json(HTTPStatus.OK, service.analyze_study(participant_id))
                    return
                if parsed.path == "/api/expert-study/complete-experience":
                    answers = body.get("answers")
                    if not isinstance(answers, dict):
                        raise ValueError("answers must be an object")
                    self._json(
                        HTTPStatus.OK,
                        service.complete_study(participant_id, answers),
                    )
                    return
                if parsed.path == "/api/expert-study/event":
                    details = body.get("details")
                    if details is not None and not isinstance(details, dict):
                        raise ValueError("details must be an object")
                    self._json(HTTPStatus.OK, service.log_event(
                        participant_id,
                        str(body["stimulus_id"]) if body.get("stimulus_id") else None,
                        str(body.get("event_type") or ""),
                        details,
                    ))
                    return
                if parsed.path == "/api/expert-study/coaching-moment":
                    self._json(
                        HTTPStatus.OK, service.add_moment(participant_id, body)
                    )
                    return
                if parsed.path == "/api/expert-study/delete-coaching-moment":
                    self._json(HTTPStatus.OK, service.delete_moment(
                        participant_id, str(body.get("moment_id") or "")
                    ))
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve_feedback_app(
    *,
    mode: str,
    db_path: Path,
    run_id: str,
    ui_dir: Path,
    host: str,
    port: int,
    video_path: Path | None = None,
    video_manifest_path: Path | None = None,
    ray_v3_dir: Path | None = None,
    cohort_benchmark_path: Path | None = None,
    study_state_path: Path | None = None,
    available_runs: Sequence[Mapping[str, str]] = (),
    display_name: str = "",
    interface_version: str = DEFAULT_INTERFACE_VERSION,
) -> None:
    ui_dir = _validate_ui_directory(mode, ui_dir)
    required_assets = [ui_dir / name for name in ("index.html", "styles.css", "app.js")]
    missing = [str(path) for path in required_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing UI assets: " + ", ".join(missing))
    service = FeedbackService(
        mode=mode,
        db_path=db_path,
        run_id=run_id,
        video_path=video_path,
        video_manifest_path=video_manifest_path,
        ray_v3_dir=ray_v3_dir,
        cohort_benchmark_path=cohort_benchmark_path,
        available_runs=available_runs or [{"run_id": run_id, "url": ""}],
        display_name=display_name,
        interface_version=interface_version,
        study_state_path=study_state_path,
    )
    server = ThreadingHTTPServer((host, int(port)), _handler(service, ui_dir))
    print(
        f"{mode} app ready: http://{host}:{server.server_port}/",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
