"""Generic run-registry loading and validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .project import PROJECT_ROOT, REGISTRY_PATH, resolve_project_path


@dataclass(frozen=True)
class RunRecord:
    dataset_id: str
    run_id: str
    participant_id: str
    trial_id: str
    training_year: str
    exp_id: str
    anatomy_key: str
    raw_anatomy_token: str
    anatomy_mapping_status: str
    anatomy_mapping_basis: str
    raw_run_relpath: str
    edt_profile: str
    phase_annotation_id: str
    source_status: str
    external_case_id: str
    golden_reference_path: Path

    @property
    def global_run_id(self) -> str:
        return f"{self.dataset_id}:{self.run_id}"


REGISTRY_FIELDS = (
    "dataset_id", "run_id", "participant_id", "trial_id", "training_year",
    "exp_id", "anatomy_key", "raw_anatomy_token", "anatomy_mapping_status",
    "anatomy_mapping_basis", "raw_run_relpath", "edt_profile",
    "phase_annotation_id", "source_status", "external_case_id",
    "golden_reference_path",
)


def load_registry(
    path: Path | None = None, *, project_root: Path = PROJECT_ROOT
) -> list[RunRecord]:
    registry_path = (path or REGISTRY_PATH).expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Run registry not found: {registry_path}. "
            "Pass --registry or set MASTOID_REGISTRY_PATH."
        )
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing_columns = [name for name in REGISTRY_FIELDS if name not in (rows[0] if rows else {})]
    if missing_columns:
        raise ValueError(
            "Run registry is missing columns: " + ", ".join(missing_columns)
        )
    records = [
        RunRecord(
            dataset_id=str(row["dataset_id"]).strip(),
            run_id=str(row["run_id"]).strip(),
            participant_id=str(row["participant_id"]).strip(),
            trial_id=str(row["trial_id"]).strip(),
            training_year=str(row["training_year"]).strip(),
            exp_id=str(row["exp_id"]).strip(),
            anatomy_key=str(row["anatomy_key"]).strip(),
            raw_anatomy_token=str(row["raw_anatomy_token"]).strip(),
            anatomy_mapping_status=str(row["anatomy_mapping_status"]).strip(),
            anatomy_mapping_basis=str(row["anatomy_mapping_basis"]).strip(),
            raw_run_relpath=str(row["raw_run_relpath"]).strip(),
            edt_profile=str(row["edt_profile"]).strip(),
            phase_annotation_id=str(row["phase_annotation_id"]).strip(),
            source_status=str(row["source_status"]).strip(),
            external_case_id=str(row["external_case_id"]).strip(),
            golden_reference_path=(
                resolve_project_path(row["golden_reference_path"], project_root=project_root)
                if str(row["golden_reference_path"]).strip()
                else project_root / "validation" / "not-configured.json"
            ),
        )
        for row in rows
    ]
    validate_registry(records)
    return records


def validate_registry(records: list[RunRecord]) -> None:
    if not records:
        raise ValueError("Run registry is empty")
    for field_name, values in {
        "global_run_id": [item.global_run_id for item in records],
        "external_case_id": [
            item.external_case_id for item in records if item.external_case_id
        ],
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field_name} in run registry")
    for item in records:
        if not item.dataset_id or not item.run_id:
            raise ValueError("dataset_id and run_id are required")
        if not item.anatomy_key:
            raise ValueError(f"anatomy_key is required for {item.run_id}")


def get_run(
    run_id: str,
    records: list[RunRecord] | None = None,
    *,
    registry_path: Path | None = None,
) -> RunRecord:
    candidates = records if records is not None else load_registry(registry_path)
    matches = [
        item for item in candidates
        if item.run_id == run_id or item.global_run_id == run_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one registry row for {run_id}; found {len(matches)}")
    return matches[0]

