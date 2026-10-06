"""End-to-end synthetic source-to-SQLite build test."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evidence_system.pipeline import build_evidence  # noqa: E402
from evidence_system.registry import get_run, load_registry  # noqa: E402
from evidence_system.source_config import (  # noqa: E402
    SourceRoots,
    resolve_run_sources,
)
from evidence_system.web import FeedbackService  # noqa: E402
from scripts.create_demo_data import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEMO_SEED,
    FRAME_COUNT,
    create_demo_data,
)


PIPELINE_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("cv2", "h5py", "numpy", "scipy")
)


@unittest.skipUnless(
    PIPELINE_DEPENDENCIES_AVAILABLE,
    "requires the pipeline and vision optional dependencies",
)
class DemoPipelineTests(unittest.TestCase):
    def test_generation_is_byte_deterministic_and_defaults_under_data(self) -> None:
        self.assertEqual(
            DEFAULT_OUTPUT_ROOT,
            (REPOSITORY_ROOT / "data" / "demo").resolve(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            left = create_demo_data(temporary_root / "left")
            right = create_demo_data(temporary_root / "right")

            def digest(path: str | Path) -> str:
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            self.assertEqual(digest(left["hdf5_file"]), digest(right["hdf5_file"]))
            self.assertEqual(
                digest(left["phase_annotation_file"]),
                digest(right["phase_annotation_file"]),
            )
            left_edt = sorted(Path(left["edt_dir"]).glob("*.edt"))
            right_edt = sorted(Path(right["edt_dir"]).glob("*.edt"))
            self.assertEqual(
                [digest(path) for path in left_edt],
                [digest(path) for path in right_edt],
            )

    def test_generated_demo_builds_complete_evidence_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            fixture = create_demo_data(temporary_root / "demo")
            record = get_run(
                "DEMO_CASE_01",
                load_registry(
                    REPOSITORY_ROOT
                    / "config"
                    / "examples"
                    / "runs.example.csv"
                ),
            )
            sources = resolve_run_sources(
                record,
                SourceRoots(
                    data_root=Path(fixture["output_root"]) / "raw",
                    edt_root=Path(fixture["output_root"]) / "edt",
                    phase_annotation_file=Path(fixture["phase_annotation_file"]),
                ),
            )
            result_dir = temporary_root / "results" / record.run_id
            anatomy_profiles = (
                REPOSITORY_ROOT
                / "config"
                / "examples"
                / "anatomy_profiles.example.json"
            )
            with patch.dict(
                os.environ,
                {"MASTOID_ANATOMY_PROFILES": str(anatomy_profiles)},
            ):
                summary = build_evidence(
                    record=record,
                    sources=sources,
                    result_dir=result_dir,
                    project_root=REPOSITORY_ROOT,
                )

            self.assertEqual(fixture["seed"], DEMO_SEED)
            self.assertEqual(fixture["edt_file_count"], 16)
            self.assertEqual(summary["schema"], "evidence_build_summary_v1")
            self.assertEqual(summary["run_id"], record.run_id)
            self.assertEqual(summary["global_run_id"], record.global_run_id)
            self.assertEqual(summary["frame_count"], FRAME_COUNT)
            self.assertEqual(summary["removed_voxel_count"], 17)
            self.assertGreaterEqual(summary["stroke_count"], 1)
            self.assertGreater(summary["event_count"], 0)
            self.assertGreater(summary["relation_count"], 0)
            self.assertEqual(summary["store_integrity"], "ok")
            self.assertTrue(
                (result_dir / "stages" / "07_relations" / "relations.jsonl").is_file()
            )
            self.assertTrue((result_dir / "evidence_store_manifest.json").is_file())

            recommendation_service = FeedbackService(
                mode="recommendations",
                db_path=result_dir / "evidence.sqlite3",
                run_id=record.run_id,
                video_path=None,
                video_manifest_path=None,
                ray_v3_dir=None,
                cohort_benchmark_path=None,
                available_runs=[],
                display_name="Synthetic demo",
                interface_version="test",
            )
            recommendation_items = recommendation_service.bootstrap()[
                "coaching_plan"
            ]["items"]
            self.assertGreaterEqual(len(recommendation_items), 1)
            self.assertTrue(
                any(item["category"] == "safety" for item in recommendation_items)
            )

            connection = sqlite3.connect(result_dir / "evidence.sqlite3")
            try:
                self.assertEqual(
                    connection.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                )
                self.assertEqual(
                    connection.execute("PRAGMA foreign_key_check").fetchall(),
                    [],
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                    FRAME_COUNT,
                )
                for table, summary_key in (
                    ("strokes", "stroke_count"),
                    ("events", "event_count"),
                    ("relations", "relation_count"),
                ):
                    self.assertEqual(
                        connection.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0],
                        summary[summary_key],
                    )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
