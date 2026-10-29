# Phase detection: GitHub upload package

Upload the **contents of repository_files/** to the root of the existing Mastoid-fb-system repository, preserving scripts/, src/, tests/, and docs/, and including README.md. Do not upload the enclosing repository_files folder as another level. The repository README retains its existing sections and adds “Run offline phase detection”. Detailed contracts are in docs/PHASE_DETECTION.md.

This package contains only the new phase inference and timestamp adapter scripts, their tests, pipeline documentation, and the existing repository files that need model-phase integration changes. It excludes the camera/geometry conversion work, phase-core command, EDT configuration, study recordings, output CSVs, model weights, personal paths, and environment caches.

Use a branch for review. These replacement source files are based on the supplied repository snapshot bb11406a35372ef3f62f9906516b2c1124143bf3. If GitHub has newer edits, compare and merge them instead of overwriting them blindly. FILES.json records original and packaged source hashes. The existing code still needs the rest of the original repository; this folder is an overlay, not a standalone checkout.

Local verification after copying into a clean checkout:

    conda activate mastoid-fb
    python -m pytest -q

The earlier broad local patch included phase-core and a geometry adapter. This focused package omits those additions. Its cli.py replacement removes the phase-core command; do not upload the earlier recording_adapter.py or prepare_v3_recording.py as part of this phase feature. Use this package's tests/test_model_phase_build.py in place of the earlier broader test file.

No upload, commit or deployment has been performed by this package.

Validation result: 21 tests passed. The clean test checkout included the same README example-path correction already made locally; no geometry adapter was present.
