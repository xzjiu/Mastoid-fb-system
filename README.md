# Mastoid Feedback System

Mastoid Feedback System is research software for turning recorded simulator
evidence into traceable, post-session feedback. Deterministic processing builds
frame, event, relation, and retrieval artifacts; two focused web applications
present the resulting feedback for different workflows.

The repository contains source code and synthetic examples only. Raw study
data, participant-generated content, videos, databases, credentials, and local
path maps are deliberately kept outside version control.

## Setup

Python 3.12 is the supported development runtime on Windows, macOS, and Linux.

```text
python -m venv .venv
python -m pip install --upgrade --editable ".[pipeline,vision,dev]"
python -m pytest
```

Activate the virtual environment using the command appropriate for your shell
before installing. Run `mastoid-feedback --help` to see the available commands.

## Run the synthetic demo

Create a deterministic, non-participant demo dataset and build the evidence
database plus review video:

```text
python scripts/create_demo_data.py
python run.py build --run-id DEMO_CASE_01 --build-video
```

Then choose the interface that matches the workflow:

| Interface | Command | Local address | Writes study data |
| --- | --- | --- | --- |
| Recommendations only | `python run.py serve --manifest config/examples/recommendations_manifest.example.json` | `http://127.0.0.1:8785/` | No |
| Expert study | `python run.py serve --manifest config/examples/study_manifest.example.json` | `http://127.0.0.1:8790/` | Yes, under ignored `results/` |

The recommendations interface is read-only at both the UI and service layers:
its manifest cannot contain expert annotations, participant responses, or a
study-state path. The expert-study interface uses the same automatic evidence
and recommendations, then adds the separate expert review workflow. Run the
two commands in separate terminals if both interfaces need to be open at once.

## Repository map

- `src/evidence_system/`: deterministic processing, retrieval, validation, and
  local application service.
- `apps/expert_study/`: complete expert-study workflow, including local study
  inputs collected after the feedback experience.
- `apps/recommendations/`: recommendation-only feedback experience without
  expert annotation or item-level review controls.
- `schemas/`: JSON contracts for evidence, relations, model actions, and claims.
- `config/`: shareable rules and synthetic configuration only.
- `tests/`: unit, integration, UI, and repository-hygiene checks.
- `docs/`: architecture and data-boundary documentation.

Generated files should go under `.local/` or another ignored path. See
[`docs/data-boundary.md`](docs/data-boundary.md) before connecting any external
dataset or text-model service.

Both applications consume the same validated evidence boundary. They remain
separate because the expert-study workflow records study interactions, while
the recommendations application is a read-only presentation of feedback.

## Project boundary

This is an evidence-review research system, not a medical device. Simulator
signals and automated candidates must not be presented as proof of clinical
injury, clinical ground truth, or validated coaching unless an explicit review
and validation policy supports that claim.

The project is under active collaborative development. Reproducible changes
must pass the full test suite on Windows, macOS, and Linux.
