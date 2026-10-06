"""Focused tests for portable configuration and application-mode boundaries."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evidence_system.cli import (  # noqa: E402
    _direct_serve_options,
    _load_manifest,
    build_parser,
)
from evidence_system.registry import get_run, load_registry  # noqa: E402
from evidence_system.source_config import (  # noqa: E402
    load_source_roots,
    resolve_run_sources,
)
from evidence_system.web import ExpertStudyStore, FeedbackService  # noqa: E402


class ExampleConfigurationTests(unittest.TestCase):
    def test_example_registry_and_source_config_resolve_from_repository_root(self) -> None:
        records = load_registry(
            REPOSITORY_ROOT / "config" / "examples" / "runs.example.csv"
        )
        record = get_run("DEMO_CASE_01", records)
        roots = load_source_roots(
            REPOSITORY_ROOT
            / "config"
            / "examples"
            / "source_paths.example.json"
        )
        sources = resolve_run_sources(record, roots)

        self.assertEqual(record.global_run_id, "DEMO:DEMO_CASE_01")
        self.assertEqual(
            sources.raw_run_dir,
            (REPOSITORY_ROOT / "data" / "demo" / "raw" / record.run_id).resolve(),
        )
        self.assertEqual(
            sources.edt_dir,
            (
                REPOSITORY_ROOT
                / "data"
                / "demo"
                / "edt"
                / record.edt_profile
            ).resolve(),
        )
        self.assertEqual(
            sources.phase_annotation_file,
            (
                REPOSITORY_ROOT
                / "data"
                / "demo"
                / "annotations"
                / "phase_timepoints.csv"
            ).resolve(),
        )


class ApplicationBoundaryTests(unittest.TestCase):
    def test_recommendations_frontend_has_no_expert_collection_controls(self) -> None:
        recommendations = REPOSITORY_ROOT / "apps" / "recommendations"
        expert_study = REPOSITORY_ROOT / "apps" / "expert_study"
        recommendation_source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in sorted(recommendations.glob("*"))
            if path.is_file()
        )
        expert_source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in sorted(expert_study.glob("*"))
            if path.is_file()
        )

        for forbidden in (
            "/api/expert-study/",
            "expert contribution",
            "start recording",
            "study id",
            "<audio",
        ):
            self.assertNotIn(forbidden, recommendation_source)
            self.assertIn(forbidden, expert_source)

    def test_recommendations_manifest_rejects_study_state(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "config"
            / "examples"
            / "recommendations_manifest.example.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["cases"][0]["study_state"] = "results/demo/study_state.json"

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "invalid-recommendations.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not contain study_state"):
                _load_manifest(manifest)

    def test_recommendations_service_cannot_initialize_with_study_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be given a study-state path"):
            FeedbackService(
                mode="recommendations",
                db_path=Path("not-opened.sqlite3"),
                run_id="DEMO_CASE_01",
                video_path=None,
                video_manifest_path=None,
                ray_v3_dir=None,
                cohort_benchmark_path=None,
                available_runs=[],
                display_name="Synthetic demo",
                interface_version="test",
                study_state_path=Path("forbidden-study-state.json"),
            )

    def test_expert_store_persists_typed_coaching_moment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "study_state.json"
            store = ExpertStudyStore(state_path, "test-interface")
            store.start_session(
                participant_id="EXPERT_01",
                run_id="DEMO_CASE_01",
                stimulus_ids=["feedback-01"],
            )
            created = store.add_moment(
                participant_id="EXPERT_01",
                run_id="DEMO_CASE_01",
                t_video_s=4.25,
                category="safety",
                observation="Synthetic typed observation",
                coaching_message="Synthetic typed coaching message",
                system_coverage="missed",
                importance="high",
            )

            reloaded = ExpertStudyStore(state_path, "test-interface")
            moments = reloaded.moments("EXPERT_01", "DEMO_CASE_01")
            self.assertEqual(len(moments), 1)
            self.assertEqual(moments[0]["moment_id"], created["moment_id"])
            self.assertEqual(
                moments[0]["observation"], "Synthetic typed observation"
            )
            self.assertEqual(
                moments[0]["coaching_message"],
                "Synthetic typed coaching message",
            )
            self.assertIsNone(moments[0]["audio"])

    def test_direct_serve_rejects_cross_mode_ui_paths(self) -> None:
        parser = build_parser()
        recommendation_args = parser.parse_args(
            [
                "serve",
                "--mode",
                "recommendations",
                "--run-id",
                "DEMO_CASE_01",
                "--db-path",
                "results/demo/evidence.sqlite3",
            ]
        )
        recommendation_options = _direct_serve_options(recommendation_args)
        self.assertEqual(
            recommendation_options["ui_dir"],
            (REPOSITORY_ROOT / "apps" / "recommendations").resolve(),
        )
        self.assertIsNone(recommendation_options["study_state_path"])

        crossed = parser.parse_args(
            [
                "serve",
                "--mode",
                "recommendations",
                "--run-id",
                "DEMO_CASE_01",
                "--db-path",
                "results/demo/evidence.sqlite3",
                "--ui-dir",
                "apps/expert_study",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mode must use"):
            _direct_serve_options(crossed)

        expert_crossed = parser.parse_args(
            [
                "serve",
                "--mode",
                "expert-study",
                "--run-id",
                "DEMO_CASE_01",
                "--db-path",
                "results/demo/evidence.sqlite3",
                "--study-state",
                "results/demo/study_state.json",
                "--ui-dir",
                "apps/recommendations",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mode must use"):
            _direct_serve_options(expert_crossed)


if __name__ == "__main__":
    unittest.main()
