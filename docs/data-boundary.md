# Data Boundary

## Content allowed in Git

- Source code, static application assets, and JSON schemas.
- General evidence, relation, feedback, and validation policies.
- Small synthetic specifications and fixtures generated without real records.
- Documentation and expected results derived exclusively from synthetic data.

## Content kept outside Git

- Raw or transformed simulator sessions and anatomy resources.
- Real dataset manifests, case mappings, phase annotations, or review labels.
- Participant or reviewer identifiers, responses, notes, and recordings.
- Videos, images, HDF5 files, SQLite databases, model weights, and generated
  reports.
- Credentials, provider configuration, deidentification maps, and absolute
  machine paths.

Pseudonymization is not anonymization. Replacing a name with a code does not
make a real record suitable for a public repository.

The recommendation-only application does not collect expert annotation. The
expert-study application may collect local study inputs, but those records are
generated data and must remain outside Git.

## Recommended local layout

Use ignored local directories for development-only material:

```text
config/local/
data/
results/
```

Code should receive these locations through command-line options, environment
variables, or an ignored local configuration file. It must not discover data by
walking parent directories or relying on a developer's home folder.

## External-service boundary

The default pipeline is local and deterministic. If an external text-model
service is enabled, send only the minimum deidentified evidence required for
the requested operation. Do not send raw frames, media, local paths, databases,
free-form participant content, or dataset identifiers. Keep an auditable record
of the fields released and validate every returned claim locally.

## Generated outputs

Generated artifacts belong under `results/` or another ignored destination.
Backups and portable study bundles are managed separately from the
source repository. Before committing, review `git status` and the tracked-file
list for unexpected binaries, local paths, or research identifiers.
