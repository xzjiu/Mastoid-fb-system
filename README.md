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

## Generate EDT grids for real volumes

The current HDF5-to-evidence pipeline requires anatomy-matched EDT distance
grids, including all 16 structure files checked by the source audit. They are
used to calculate removed-voxel boundary penetration and drill-to-structure
distances. The simulator does not need to load EDT grids during recording for
this offline analysis, and viewing previously generated feedback does not
require regenerating them.

### Prerequisites and input layout

- An existing `EDTFromGrid` executable built for the machine where generation
  will run. It is not bundled with this repository or installed by `pip`.
  On Linux/macOS, it must have execute permission.
- Color-labelled anatomy PNG slices using the RGB labels in
  [`edt_tools/batch_generate_edt.py`](edt_tools/batch_generate_edt.py).
  The wrapper uses only the Python standard library; the external executable
  must also have its own runtime dependencies available.
- One immediate subdirectory per volume, with the PNG slices directly inside
  it. Subdirectories are not scanned recursively. Slices are sorted by the
  first number in each filename; duplicate numbers are rejected, and PNGs
  without a number are ignored. Verify that this order matches the simulator.

For example, keep inputs and outputs under the ignored `data/` directory:

```text
data/
  volumes/
    RT143_256/
      plane00000.png
      plane00001.png
      ...
    LT151_256/
      plane00000.png
      plane00001.png
      ...
  edt/                       # Created by the wrapper
```

### Run batch generation

From the repository root, activate your Python environment and run the
following command, replacing the executable path with your actual location:

```text
python edt_tools/batch_generate_edt.py --volumes data/volumes --output data/edt --exe "/absolute/path/to/EDTFromGrid"
```

The same arguments work with a compatible Windows executable; pass its quoted
path, such as `"C:/tools/EDTFromGrid.exe"`. Input and output roots must be
separate and must not be nested inside one another. The wrapper reads the PNG
folders directly; it does not read an ADF or `gui_setup.yaml`.

Each volume produces a same-named output directory, for example
`data/edt/RT143_256/`, containing these 16 files:

```text
Bone.edt                       Malleus.edt
Incus.edt                      Stapes.edt
Bony_Labyrinth.edt              IAC.edt
Superior_Vestibular_Nerve.edt    Inferior_Vestibular_Nerve.edt
Cochlear_Nerve.edt              Facial_Nerve.edt
Chorda_Tympani.edt              ICA.edt
Sinus_+_Dura.edt                 Vestibular_Aqueduct.edt
TMJ.edt                        EAC.edt
```

Each attempted structure also has a `.log` file containing the executable's
output. Existing `.edt` files are skipped without revalidation. Failed commands
or missing/empty outputs are reported, and the wrapper returns a nonzero exit
status if any generation fails or no eligible volume is found. A final `.edt`
file is published only after the command succeeds and produces a nonempty
temporary file; this is not a numerical or anatomical validity check.

Test one volume first by pointing `--volumes` to a parent directory containing
only that volume folder. Inspect the logs and verify the resulting grids before
processing the full collection. If labels or source geometry change, use a
fresh output root to avoid reusing skipped, outdated files.

### Connect generated grids to the analysis pipeline

Create machine-local configuration under ignored `config/local/`, using the
files in `config/examples/` as templates:

1. In your source-path JSON, set `edt_root` to `data/edt` (or its absolute path),
   and configure the real `data_root` and `phase_annotation_file`.
2. In your run registry CSV, set the run's `edt_profile` to its output folder
   name, such as `RT143_256`. The resolved directory is
   `edt_root / edt_profile`.
3. In your anatomy-profile JSON, add the run's `anatomy_key` with the matching
   `edt_profile` and verified physical scale, volume dimensions, and coordinate
   mapping. Do not copy the synthetic demo's numerical calibration into a real
   anatomy profile. In particular, `edt_distance_scale_mm` must correctly
   convert the generated distance values to millimetres.

With those local files prepared, build a registered run (replace `YOUR_RUN_ID`):

```text
python run.py build --run-id YOUR_RUN_ID --registry config/local/runs.csv --source-config config/local/source_paths.json --anatomy-profiles config/local/anatomy_profiles.json --build-video
```

Relative source paths are resolved from the repository root, not from the JSON
file's directory. EDT generation does not create the run registry, phase
annotations, or calibration. Reuse a volume's grids across sessions only when
the labelled anatomy and grid geometry remain unchanged. Matching folder names
alone does not establish alignment: verify resolution, axis orientation, slice
order, distance sign, and millimetre scale. Keep volumes, generated grids, and
logs under ignored `data/` or outside the repository, not in source directories.

## Repository map

- `src/evidence_system/`: deterministic processing, retrieval, validation, and
  local application service.
- `apps/expert_study/`: complete expert-study workflow, including local study
  inputs collected after the feedback experience.
- `apps/recommendations/`: recommendation-only feedback experience without
  expert annotation or item-level review controls.
- `schemas/`: JSON contracts for evidence, relations, model actions, and claims.
- `config/`: shareable rules and synthetic configuration only.
- `edt_tools/`: batch EDT generation wrapper for an external `EDTFromGrid` executable.
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
