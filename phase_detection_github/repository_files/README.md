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

## Run offline phase detection

Phase detection assigns `pre_antrum`, `antrum`, and `incus` predictions to saved
simulator recordings using a frozen V-JEPA2 encoder and a trained temporal
convolutional network (TCN). Predictions can replace manual phase annotations
in an explicitly selected model-phase build. Manual annotations remain the
default source.

### Prerequisites and input layout

- A PyTorch inference environment with `numpy`, `h5py`, and a `transformers`
  installation exposing `VJEPA2Model` and `VJEPA2VideoProcessor`. Reuse the
  model's training environment for inference; the base feedback environment
  does not require PyTorch.
- The `facebook/vjepa2-vitl-fpc64-256` encoder and processor, plus the trained
  `tcn_best.pt` checkpoint from `phase_timepoint_tcn_v1`. These model assets are
  supplied separately and are not included in the repository.
- A saved session containing HDF5 image chunks with
  `vision_data/continuous_data/time` and `vision_data/continuous_data/l_img`.
  Confirm whether the stored images use RGB or BGR channel order.

Keep recordings and checkpoints outside version control. For example:

```text
data/
  recordings/SESSION_ID/      # HDF5 session chunks
  models/tcn_best.pt          # Trained phase checkpoint
results/
  phase_predictions/         # Window probabilities and embeddings
  phase_input/               # Frame assignments and source manifest
```

### Run inference and frame assignment

From the repository root, activate the inference environment and run:

```text
python scripts/infer_phase_hdf5.py --session data/recordings/SESSION_ID --checkpoint data/models/tcn_best.pt --output results/phase_predictions --color-order bgr
python scripts/prepare_phase_input.py --predictions results/phase_predictions/phase_predictions.csv --summary results/phase_predictions/summary.json --output results/phase_input
```

Replace the session and checkpoint paths with local locations. Use
`--color-order rgb` when the stored arrays are RGB. Encoder loading is
cache-only by default; add `--allow-download` to the inference command when
initially retrieving the encoder and processor. Use a fresh inference output
directory for each run.

Inference samples 16 images per 5-second window at a 1-second stride, encodes
each window, and applies checkpoint normalization followed by the TCN.
Timestamp gaps greater than 0.5 seconds split processing into separate
segments. Frame assignment averages probabilities from covering windows;
frames without coverage receive `unknown`. Images are read directly from HDF5
without exporting individual frames.

| Output | Purpose |
| --- | --- |
| `phase_predictions/phase_predictions.csv` | Window labels and class probabilities |
| `phase_predictions/summary.json` | Model settings and source inventory |
| `phase_predictions/embeddings.pt` | Encoder embeddings |
| `phase_input/model_phase_frames.csv` | Timestamp-aligned frame labels and probabilities |
| `phase_input/phase_input_manifest.json` | Source and assignment provenance; keep beside the frame CSV |

### Connect predictions to the feedback pipeline

For a registered run that already builds with the existing recording reader
and local configuration, activate the feedback environment and add
`--model-phase-frames` to the build command:

```text
python run.py build --run-id YOUR_RUN_ID --registry config/local/runs.csv --source-config config/local/source_paths.json --anatomy-profiles config/local/anatomy_profiles.json --model-phase-frames results/phase_input/model_phase_frames.csv
```

Model mode validates source filenames, frame indices, and absolute timestamps,
then records `model_phase_prediction` provenance in the evidence outputs.
It does not edit human annotations. Existing feedback rules that reference
manual phase names require review because model labels are preserved as-is.

Phase inference supports the v3 image layout above. The existing full feedback
reader expects `data/...` HDF5 datasets; this feature does not convert v3
geometry. Inference and frame assignment can therefore run independently of a
full feedback build. The TCN uses future context and operates offline; this is
not a live simulator connection.

See [`docs/PHASE_DETECTION.md`](docs/PHASE_DETECTION.md) for checkpoint contracts,
source validation, processing options, and integration limits. Run
`python -m pytest -q` to check the phase adapter and build integration alongside
the existing tests. These checks establish software behavior, not prediction
accuracy on new recordings.

## Repository map

- `src/evidence_system/`: deterministic processing, retrieval, validation, and
  local application service.
- `apps/expert_study/`: complete expert-study workflow, including local study
  inputs collected after the feedback experience.
- `apps/recommendations/`: recommendation-only feedback experience without
  expert annotation or item-level review controls.
- `schemas/`: JSON contracts for evidence, relations, model actions, and claims.
- `config/`: shareable rules and synthetic configuration only.
- `scripts/infer_phase_hdf5.py` and `scripts/prepare_phase_input.py`: offline
  phase inference and timestamp-aligned model input.
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
