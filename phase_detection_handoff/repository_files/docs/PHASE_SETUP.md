# Phase detection setup and handoff

## Scope

This feature processes completed simulator recordings with V-JEPA2 and a
trained TCN. It saves phase predictions and frame assignments and can show them
on a local processing page. Labels are `pre_antrum`, `antrum`, `incus`, and
`unknown` for uncovered frames. See [PHASE_DETECTION.md](PHASE_DETECTION.md)
for the model and timestamp contracts.

## Current validation status

| Component | Status |
| --- | --- |
| Saved-session inference and frame assignment | Ran successfully on the user's local PC |
| Local Run Phase Detection page | Ran inference and displayed 2,000 frame assignments |
| Example output | 1,645 pre_antrum; 0 antrum; 292 incus; 63 unknown |
| Phase-only repository integration | Previously passed 21 tests on the supplied clean snapshot |
| Local runner | Six automated checks passed; real model run confirmed by the user |
| Simulator launcher button | Installed and visible in the user's local PyQt5 interface |
| Browser launch from simulator button | Unresolved: clicking the button did not open a browser on the user's WSL setup |
| Full coaching feedback from this processing page | Not implemented; outputs are saved for a separate feedback build |
| Hospital PC deployment and fresh simulator recording | Not tested |

The example counts verify execution, not prediction accuracy. The absence of
antrum predictions requires reference-label review before interpretation.
The current TCN uses future context and is for offline inference.

## Set up on a PC

Use the existing repository setup for its feedback environment. Inference runs
in a separate environment containing PyTorch, NumPy, h5py, and transformers
with `VJEPA2Model` and `VJEPA2VideoProcessor`. Reusing the tested training
inference environment is recommended; no new version lock is claimed here.

Provide separately:

- The trained `phase_timepoint_tcn_v1/tcn_best.pt` checkpoint.
- Cached `facebook/vjepa2-vitl-fpc64-256` encoder and processor files.
- A completed HDF5 recording folder with the supported v3 image datasets.

The checkpoint must contain the trained weights and normalization values
specified in PHASE_DETECTION.md. The processing page uses cached encoder files
and does not download them. For an uncached machine, provision those assets
before launching the page; the standalone inference script also supports
`--allow-download` for explicit initial model retrieval.

### Start the processing page

From the feedback repository root, use the machine's actual paths:

```bash
python phase_runner/serve.py \
  --python /path/to/inference/env/bin/python \
  --checkpoint /path/to/tcn_best.pt \
  --session /path/to/completed/session \
  --color-order bgr
```

Use `rgb` instead when the recording arrays are RGB. The server itself uses
only Python's standard library; the `--python` executable runs both processing
scripts. Keep this terminal running. Open `http://127.0.0.1:8795/` in a browser
on the same PC. Select the completed session folder and click Run Phase
Detection. Wait until recording has stopped and files have finished saving.

Each run creates a unique folder under `results/phase_jobs/` containing:

| Path | Contents |
| --- | --- |
| `predictions/` | Window prediction CSV, summary JSON, encoder embeddings |
| `phase_input/` | Frame prediction CSV and provenance manifest |
| `run.log` | Processing log |
| `job.json` | Final job status and result summary |

The page shows progress, errors, frame counts, and a frame-based timeline.
It does not rebuild the feedback database. Keep the frame CSV and its manifest
together when passing predictions into the existing `--model-phase-frames`
workflow. Full feedback still requires its supported recording layout and
already configured sources; this package does not convert geometry.

Ctrl+C stops the server. Outputs remain saved. After a restart, the page starts
with an empty display rather than reloading previous jobs.

### Add the simulator launcher button

From the feedback repository root:

```bash
python scripts/install_simulator_phase_button.py \
  --gui-dir /path/to/simulator/scripts/study_gui
```

The installer targets the inspected PyQt5 `study_gui.py`. It adds a button to
its existing grid without editing `layout.ui` or `gui_setup.yaml`, and prints
the backup path. Start the simulator study GUI in its normal environment and
working directory. The separate phase server must already be running.

This button only opens `http://127.0.0.1:8795/`. It does not start the server,
select the latest recording, or start inference. It refuses to open while the
GUI reports that a recording is active. The browser-launch action failed on
the tested WSL setup; the cause has not been established. Until resolved,
open the URL manually. Do not describe the launcher as fully tested.

The local GUI required PyQt5, pyzmq, and PyYAML. On the user's Ubuntu 24.04 WSL
installation, the Qt xcb loader also needed these system packages:
`libxcb-icccm4`, `libxcb-image0`, `libxcb-keysyms1`, `libxcb-render-util0`,
`libxcb-xinerama0`, `libxcb-xkb1`, and `libxkbcommon-x11-0`. Install missing
requirements only as needed on the destination PC; its existing simulator
runtime may already provide them.

To undo the GUI addition, restore `study_gui.py` from the exact backup path
printed by the installer. Preserve any subsequent edits before restoration.

## Checks and remaining deployment work

```bash
python -m pytest -q
python -m unittest discover -s phase_runner -p 'test_*.py'
```

The repository checks and the runner checks are separate. Runner tests simulate
model outputs and do not replace the real-model execution check.

When the simulator PC becomes available, confirm its study GUI matches the
inspected version, configure that PC's environment and model paths, test a
completed recording from that simulator, and check browser opening from the
button. Automatic session selection and a combined phase-plus-feedback run are
not included in this handoff.

## File organization

- `scripts/infer_phase_hdf5.py`, `scripts/prepare_phase_input.py`: command-line
  phase processing used by the repository integration.
- `src/evidence_system/`: phase-source validation and integration changes.
- `phase_runner/`: self-contained processing page and its two bundled inference
  scripts. These match the packaged scripts at this handoff; update both copies
  together if changing the inference implementation.
- `scripts/install_simulator_phase_button.py`: simulator launcher installer.
- `tests/` and `phase_runner/test_runner.py`: software checks.

No trained models, recordings, raw results, or simulator assets belong in this
source upload.
