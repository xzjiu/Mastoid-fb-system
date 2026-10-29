# Local phase detection runner

A local processing page for saved simulator recordings. Its Run Phase Detection
button runs V-JEPA2 + TCN inference, then frame assignment, and shows frame phase
counts, a timeline, and the process log. Both scripts run using the explicitly
configured inference Python executable. Model loading uses the existing cache.

Copy `phase_runner/` into the feedback repository root. Start from that root:

```bash
python phase_runner/serve.py --python /path/to/inference/environment/bin/python --checkpoint /path/to/tcn_best.pt --session /path/to/finished/session --color-order bgr
```

Open the printed `http://127.0.0.1:8795/` URL. Select the completed session folder
and its stored RGB/BGR convention, then press Run Phase Detection. Only one job
runs at a time. Each job gets a new directory under `results/phase_jobs/` with:

- `predictions/`: window predictions, embeddings, and inference summary.
- `phase_input/`: `model_phase_frames.csv` and `phase_input_manifest.json`.
- `run.log` and `job.json`: process output and final status.

Keep the frame CSV and manifest together. They are inputs for the existing
`--model-phase-frames` integration, subject to its source-format requirements.
This runner does not create a feedback database, launch the simulator, or add a
button inside the simulator's native interface. It is the local processing page
for testing the button-to-model workflow before that interface is connected.
The existing recommendations viewer and expert-study service remain unchanged.

Requirements: an inference Python with PyTorch, h5py, NumPy, and transformers
with VJEPA2Model/VJEPA2VideoProcessor; the cached matching encoder/processor;
and the `phase_timepoint_tcn_v1` trained checkpoint. The server itself uses only
the Python standard library. HDF5 inference expects the v3 vision layout.
Run after recording ends; active recording files are not supported. No image
export, retraining, EDT generation, or network model download is performed.

Stop with Ctrl+C. An active child process is terminated; incomplete job outputs
remain for diagnosis and are not reported as completed. Restarting the server
starts with an empty UI state; earlier files remain in their job directories.
A failed job displays its log. Each retry writes to a fresh job directory.

Validation: `python -m unittest discover -s phase_runner -p 'test_*.py'`.
Tests simulate subprocess outputs and check routing, completion, failures,
concurrent-start rejection, and result validation. They do not test GPU model
execution; that must be verified with the real checkpoint and recording.
