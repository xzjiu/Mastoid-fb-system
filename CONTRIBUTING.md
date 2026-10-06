# Contributing

## Development setup

Use Python 3.12 and install the editable project with all development groups:

```text
python -m pip install --upgrade --editable ".[pipeline,vision,dev]"
python -m pytest
python -m ruff check src tests
```

CI runs the same checks on Windows, macOS, and Linux.

## Change guidelines

1. Keep deterministic evidence construction separate from presentation and
   external text-model behavior.
2. Pass dataset manifests, anatomy profiles, and source roots through explicit
   configuration. Do not add machine-specific defaults.
3. Use synthetic case identifiers and generated fixtures in tests.
4. Add or update tests for every behavior change, including validation and
   abstention paths where relevant.
5. Keep pull requests focused. Generated results and exploratory reports should
   not be committed with production changes.

## Data and privacy

Never commit raw simulator data, participant identifiers, recordings, study
responses, derived databases, videos, credentials, or absolute local paths.
Pseudonymous identifiers from a real study are still research data and do not
belong in public fixtures. Follow [`docs/data-boundary.md`](docs/data-boundary.md)
for the supported local layout.

## Pull-request checklist

- Tests and lint checks pass locally.
- The change works without access to a private dataset.
- New fixtures are synthetic and small.
- No secret, local path, participant content, or generated binary is tracked.
- User-facing claims remain supported by traceable evidence and the applicable
  validation status.

