"""Minimal command line interface for evidence builds and local web apps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .pipeline import build_evidence
from .project import OUTPUT_ROOT, PROJECT_ROOT, REGISTRY_PATH
from .registry import get_run, load_registry
from .source_config import (
    SourceRoots,
    load_source_roots,
    resolve_run_sources,
)
from .web import APP_MODES, serve_feedback_app


def _path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def build_command(args: argparse.Namespace) -> int:
    if args.anatomy_profiles:
        os.environ["MASTOID_ANATOMY_PROFILES"] = str(
            _path(args.anatomy_profiles)
        )
    registry_path = _path(args.registry)
    record = get_run(
        args.run_id,
        load_registry(registry_path, project_root=PROJECT_ROOT),
    )
    direct = (args.raw_run_dir, args.edt_dir, args.phase_annotation_file)
    if any(direct) and not all(direct):
        raise ValueError(
            "--raw-run-dir, --edt-dir, and --phase-annotation-file must be "
            "provided together"
        )
    if all(direct):
        roots = SourceRoots(
            data_root=_path(args.raw_run_dir),
            edt_root=_path(args.edt_dir),
            phase_annotation_file=_path(args.phase_annotation_file),
        )
        sources = resolve_run_sources(
            record,
            roots,
            raw_run_dir=roots.data_root,
            edt_dir=roots.edt_root,
            phase_annotation_file=roots.phase_annotation_file,
        )
    else:
        roots = load_source_roots(
            _path(args.source_config) if args.source_config else None,
            project_root=PROJECT_ROOT,
        )
        sources = resolve_run_sources(record, roots)
    output_root = _path(args.output_root)
    summary = build_evidence(
        record=record,
        sources=sources,
        result_dir=output_root / record.run_id,
        project_root=PROJECT_ROOT,
        build_video=args.build_video,
        video_fps=args.video_fps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = str(payload.get("app_mode") or "")
    if mode not in APP_MODES:
        raise ValueError("Manifest app_mode must be recommendations or expert-study")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain at least one case")
    access = dict(payload.get("data_access") or {})
    if mode == "recommendations":
        if access.get("expert_annotations") != "none":
            raise ValueError(
                "Recommendations manifest must disable expert annotations"
            )
        if access.get("participant_responses") != "none":
            raise ValueError(
                "Recommendations manifest must disable participant responses"
            )
        if any("study_state" in case for case in cases):
            raise ValueError(
                "Recommendations cases must not contain study_state paths"
            )
    else:
        if access.get("expert_annotations") != "read_write":
            raise ValueError(
                "Expert-study manifest must enable expert annotations"
            )
        if access.get("participant_responses") != "read_write":
            raise ValueError(
                "Expert-study manifest must enable participant responses"
            )
        if any(not str(case.get("study_state") or "").strip() for case in cases):
            raise ValueError("Every expert-study case requires study_state")
    return payload


def _manifest_serve_options(
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_path = _path(args.manifest)
    payload = _load_manifest(manifest_path)
    mode = str(payload["app_mode"])
    cases = [dict(row) for row in payload["cases"]]
    selected = next(
        (
            row for row in cases
            if args.case_id in {
                None,
                str(row.get("case_id") or ""),
                str(row.get("run_id") or ""),
            }
        ),
        None,
    )
    if selected is None:
        raise KeyError(f"Case not found in manifest: {args.case_id}")
    host = args.host or str(payload.get("host") or "127.0.0.1")
    port = args.port or int(
        selected.get("port") or payload.get("base_port") or 0
    )
    expected_ui = {
        "recommendations": PROJECT_ROOT / "apps" / "recommendations",
        "expert-study": PROJECT_ROOT / "apps" / "expert_study",
    }[mode].resolve()
    ui_dir = _path(str(payload.get("ui_directory") or ""))
    if ui_dir != expected_ui:
        raise ValueError(
            f"{mode} mode must use {expected_ui}; got {ui_dir}"
        )
    available_runs = [
        {
            "run_id": str(row["run_id"]),
            "url": f"http://{host}:{int(row.get('port') or payload.get('base_port') or 0)}/",
        }
        for row in cases
    ]
    database = _path(str(selected["evidence_database"]))
    case_root = database.parent
    ray_v3 = (
        _path(str(selected["ray_v3_directory"]))
        if selected.get("ray_v3_directory")
        else case_root / "ray_v3"
    )
    cohort = (
        _path(str(selected["cohort_benchmark"]))
        if selected.get("cohort_benchmark")
        else case_root / "cohort_benchmark.json"
    )
    return {
        "mode": mode,
        "db_path": database,
        "run_id": str(selected["run_id"]),
        "ui_dir": ui_dir,
        "host": host,
        "port": port,
        "video_path": _path(str(selected["review_video"])),
        "video_manifest_path": _path(str(selected["review_video_manifest"])),
        "ray_v3_dir": ray_v3 if ray_v3.is_dir() else None,
        "cohort_benchmark_path": cohort if cohort.is_file() else None,
        "study_state_path": (
            _path(str(selected["study_state"]))
            if mode == "expert-study"
            else None
        ),
        "available_runs": available_runs,
        "display_name": str(
            selected.get("display_name") or selected.get("case_id") or "Case"
        ),
        "interface_version": str(
            payload.get("interface_version") or "expert-feedback-v1"
        ),
    }


def _direct_serve_options(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--mode": args.mode,
        "--run-id": args.run_id,
        "--db-path": args.db_path,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Direct serve mode requires " + ", ".join(missing)
        )
    mode = str(args.mode)
    expected_ui = (
        PROJECT_ROOT
        / "apps"
        / ("recommendations" if mode == "recommendations" else "expert_study")
    ).resolve()
    ui_dir = _path(args.ui_dir) if args.ui_dir else expected_ui
    if ui_dir != expected_ui:
        raise ValueError(f"{mode} mode must use {expected_ui}; got {ui_dir}")
    state = _path(args.study_state) if args.study_state else None
    if mode == "recommendations" and state is not None:
        raise ValueError("Recommendations mode cannot use --study-state")
    if mode == "expert-study" and state is None:
        raise ValueError("Expert-study mode requires --study-state")
    return {
        "mode": mode,
        "db_path": _path(args.db_path),
        "run_id": str(args.run_id),
        "ui_dir": ui_dir,
        "host": args.host or "127.0.0.1",
        "port": args.port or (8785 if mode == "recommendations" else 8790),
        "video_path": _path(args.video) if args.video else None,
        "video_manifest_path": (
            _path(args.video_manifest) if args.video_manifest else None
        ),
        "ray_v3_dir": _path(args.ray_v3_dir) if args.ray_v3_dir else None,
        "cohort_benchmark_path": (
            _path(args.cohort_benchmark) if args.cohort_benchmark else None
        ),
        "study_state_path": state,
        "available_runs": [{"run_id": str(args.run_id), "url": ""}],
        "display_name": args.display_name or "Case 01",
        "interface_version": args.interface_version or "expert-feedback-v1",
    }


def serve_command(args: argparse.Namespace) -> int:
    options = (
        _manifest_serve_options(args)
        if args.manifest
        else _direct_serve_options(args)
    )
    serve_feedback_app(**options)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="Build evidence and SQLite from one registered HDF5 run"
    )
    build.add_argument("--run-id", required=True)
    build.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    build.add_argument("--source-config", type=Path)
    build.add_argument("--raw-run-dir", type=Path)
    build.add_argument("--edt-dir", type=Path)
    build.add_argument("--phase-annotation-file", type=Path)
    build.add_argument("--anatomy-profiles", type=Path)
    build.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    build.add_argument("--build-video", action="store_true")
    build.add_argument("--video-fps", type=float, default=10.0)
    build.set_defaults(handler=build_command)

    serve = subparsers.add_parser(
        "serve", help="Serve the read-only recommendations or expert-study app"
    )
    serve.add_argument("--manifest", type=Path)
    serve.add_argument("--case-id")
    serve.add_argument("--mode", choices=sorted(APP_MODES))
    serve.add_argument("--run-id")
    serve.add_argument("--db-path", type=Path)
    serve.add_argument("--ui-dir", type=Path)
    serve.add_argument("--video", type=Path)
    serve.add_argument("--video-manifest", type=Path)
    serve.add_argument("--ray-v3-dir", type=Path)
    serve.add_argument("--cohort-benchmark", type=Path)
    serve.add_argument("--study-state", type=Path)
    serve.add_argument("--display-name")
    serve.add_argument("--interface-version")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(handler=serve_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
