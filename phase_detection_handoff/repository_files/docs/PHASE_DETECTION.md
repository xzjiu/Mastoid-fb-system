# Offline phase detection

This feature adds pretrained V-JEPA2 + TCN phase predictions to the Mastoid feedback repository. The labels are pre_antrum, antrum, and incus. Existing manual-annotation builds remain the default. Model phases are selected explicitly and retain model provenance.

## Pipeline

1. Read timestamped left-camera images directly from a saved simulator HDF5 session.
2. Form 5-second windows at a 1-second stride. Split at timestamp gaps greater than 0.5 seconds; never create windows across those gaps.
3. Sample 16 images evenly in time within each window and use the V-JEPA2 processor.
4. Pool encoder tokens into a 1024-dimensional embedding.
5. Normalize with the mean and standard deviation saved in the trained TCN checkpoint.
6. Run the TCN separately on each continuous recording segment.
7. Save window probabilities. Average probabilities from windows covering each actual frame timestamp; uncovered timestamps get unknown.
8. Optionally feed those frame assignments into the existing feedback build with --model-phase-frames.

The TCN uses future context. This is saved-session inference, not a live simulator feature. It uses raw TCN probabilities; the earlier stay/advance decoder is not included. No retraining or individual-image export is required.

## Files

| File | Purpose |
| --- | --- |
| scripts/infer_phase_hdf5.py | HDF5 → V-JEPA2 embeddings → TCN window predictions |
| scripts/prepare_phase_input.py | Validate predictions and attach them to actual frame timestamps |
| src/evidence_system/model_phases.py | Validate and align model input with the feedback build's frames |
| CLI, source config, source audit, core extractor | Opt-in model source selection and frame assignment |
| Event builder, relation builder, storage | Model provenance and separate identities for repeated phase intervals |
| tests/test_phase_windows.py | Window coverage and missing-data handling |
| tests/test_phase_input.py | Probability validation, overlap aggregation, timestamps and unknown coverage |
| tests/test_model_phase_build.py | Synthetic full-build integration, provenance and mismatched recording rejection |

## Runtime setup

Inference requires a working PyTorch environment with NumPy, h5py, transformers, and the dependencies used by VJEPA2VideoProcessor. Reuse the environment that trained the model. Keep the feedback environment for the repository build. The CSV/JSON interface connects the two environments without adding PyTorch to the base feedback package.

Provide these model assets separately from GitHub:
- The facebook/vjepa2-vitl-fpc64-256 encoder and processor, normally already in the Hugging Face cache.
- tcn_best.pt from the phase_timepoint_tcn_v1 training run. It must contain model_state_dict, mean, std, and best_epoch. Normalization tensors have shape (1, 1024).

The TCN has a 1024→128 pointwise projection, four residual blocks with dilations 1/2/4/8, two kernel-3 convolutions per block, dropout 0.3, and three output classes. Checkpoint loading is strict. Match the exact encoder used during training; an arbitrary same-sized encoder is not interchangeable.

## Run phase inference

Commands below use placeholders. Replace them with local paths. Run from the repository root.

    python scripts/infer_phase_hdf5.py \
      --session /path/to/session \
      --checkpoint /path/to/tcn_best.pt \
      --output results/phase_predictions \
      --color-order bgr

The inference script currently reads the v3 layout: vision_data/continuous_data/time and vision_data/continuous_data/l_img. It combines image-bearing chunks and skips files without image timestamps. Use --max-windows 3 for a smoke test or --plan-only to inspect window coverage without loading the model. Full runs refuse an existing output directory.

By default, model loading is cache-only. Add --allow-download only if the encoder/processor is not cached and you intend to download it. --color-order describes the stored array: bgr swaps channels to RGB before preprocessing. Choose the setting from the acquisition convention. BGR was provisionally selected for the tested recording using the simulator documentation and an image comparison; it is not asserted for every recording.

Outputs: phase_predictions.csv, summary.json, and embeddings.pt. Keep these generated files local under results/.

## Attach predictions to frames

    python scripts/prepare_phase_input.py \
      --predictions results/phase_predictions/phase_predictions.csv \
      --summary results/phase_predictions/summary.json \
      --output results/phase_input

The session path defaults to summary.json; use --session /path/to/session if it moved. This script reads timestamps and image dataset shapes, not an entire image volume. It validates the source inventory, clocks, prediction probabilities, and gaps. Smoke-test outputs are rejected here.

Outputs: model_phase_frames.csv and phase_input_manifest.json. Keep them together. Their probability averaging rule is a frame-assignment rule, not an additional accuracy evaluation. Window intervals are half-open. Unknown means no usable prediction window covers that frame. Scores are uncalibrated model outputs.

## Use phases in the existing feedback build

If the run already builds successfully with the repository's supported recording format and configured anatomy assets, add the model flag:

    python run.py build \
      --run-id YOUR_REGISTERED_RUN \
      --registry config/local/runs.csv \
      --source-config config/local/source_paths.json \
      --model-phase-frames results/phase_input/model_phase_frames.csv

A valid human annotation file is not read in explicit model mode. The existing source-config parser still requires its phase_annotation_file key; keep the existing config structure. Alternatively use direct input flags:

    python run.py build \
      --run-id YOUR_REGISTERED_RUN \
      --registry config/local/runs.csv \
      --raw-run-dir /path/to/supported/session \
      --edt-dir /path/to/existing/edt \
      --anatomy-profiles config/local/anatomy_profiles.json \
      --model-phase-frames results/phase_input/model_phase_frames.csv

The supplied CSV must match every feedback frame by source filename, frame index and absolute timestamp. Model labels remain pre_antrum/antrum/incus/unknown; pre_antrum is not silently renamed early_surface or pre_drilling. Existing rules that select specific manual phase names may need separate review before interpreting their model-mode output. Repeated phases receive distinct interval IDs, and missing-data gaps break intervals. The audit, database and graph identify the source as model_phase_prediction. No human annotation file is edited.

The original feedback reader expects data/... HDF5 datasets. This focused phase package does not convert v3 geometry data or configure EDT/anatomy assets. V3 recordings can run the phase inference and frame-assignment steps independently. For a full feedback build, use the system's supported recording layout and existing configuration. The earlier local geometry adapter is not a dependency included in this feature.

## Validation and limits

Run `python -m pytest -q` from the repository root. The phase tests cover
window gaps, probability validation, overlap aggregation, source matching,
model provenance, repeated phase intervals, and preserved manual mode.
The phase-only overlay previously passed 21 tests against the supplied
repository snapshot. End-to-end GPU inference requires the separately supplied
encoder and checkpoint; the unit tests do not download or execute those assets.

The TCN checkpoint must match the architecture and class order described above.
New-recording accuracy requires independent reference labels. This feature
provides offline phase predictions and their integration contract; live
simulator streaming and hospital deployment are separate work.
