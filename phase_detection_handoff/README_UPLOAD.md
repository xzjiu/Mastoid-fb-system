# Phase detection integration package

This folder adds offline phase detection to the existing Mastoid Feedback
System repository. It is an overlay, not a complete standalone repository.

## Upload layout

Upload the contents of `repository_files/` into the existing repository root.
Keep `scripts/`, `src/`, `tests/`, `docs/`, and `phase_runner/` at that level.
Include the updated root `README.md`. Do not add `repository_files/` as a nested
folder in the repository. Uploading the ZIP alone shares an archive; it does
not install the feature.

The source replacements are based on the supplied repository snapshot
`bb11406a35372ef3f62f9906516b2c1124143bf3`. Compare with newer repository edits
before merging replacement files, including README.md. `FILES.json` lists
packaged hashes and original snapshot hashes where applicable.

## Included

- Phase inference and frame assignment scripts.
- Opt-in model-phase integration with the existing feedback build.
- Local processing page with a Run Phase Detection button, log, and timeline.
- Installer for a Phase Detection launcher button in the inspected simulator
  study GUI; it backs up the target `study_gui.py` before editing.
- Pipeline documentation, local setup instructions, and tests.

Start with `docs/PHASE_SETUP.md` for deployment and current validation status.
The simulator installer is run against the simulator repository separately;
simply uploading it does not modify the simulator.

Recordings, participant outputs, model weights, local paths, environments,
EDT grids, and the earlier geometry adapter are not included. Nothing in this
package has been uploaded to GitHub or installed on the hospital PC by the
assistant. The user's earlier upload is separate from this new package.
