# Contributing

Use independent synthetic data for examples, tests, issues and pull requests. Read the [data policy](docs/DATA_POLICY.md) before adding artifacts. Never attach private measurements, credentials, source-specific schemas, fitted private models or screenshots containing such information.

## Development

Create a Python 3.12 virtual environment, then run:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

The package is imported as `crfml`. Tests use artificial fixtures and cover model derivatives, causality, transport duality, numerical rejection/recovery, data contracts and publication controls. Include a focused regression check when changing mathematical behavior or a publication boundary. Documentation-only changes do not require repeating numerical experiments.

If source modules or study configurations change, regenerate the affected evidence. The current provenance format hashes the complete source-module set, so publication currently requires regenerating all four studies:

```bash
python -m crfml.cli all --output results
python scripts/export_public.py --check-tracked
python scripts/export_public.py --destination dist/review-copy
```

The export destination must be fresh. Review its manifest and contents before publication. A local working repository can contain excluded files; the publication guard deliberately refuses tracked paths outside the public profile.

## Scientific changes

State the changed assumption or question, identify the relevant baseline and keep estimation, validation and evaluation data separate. Preserve failed numerical attempts and distinguish a returned iterate from a converged fit. Report negative results and uncertainty alongside improvements. Attribute prior algorithms and distinguish new derivations from reproduced methods.

Use a separate configuration and output directory for an exploratory extension. Do not silently replace the documented benchmark after inspecting its evaluation outcomes. Explain which claims need to be revised if a guarantee's assumptions change.

## Reports and fixes

Include a minimal synthetic reproduction, the command, dependency versions and the expected versus observed behavior. Redact local usernames and paths from logs. If a concern involves actual sensitive data or a credential, do not paste it into a public issue; use private reporting when the repository provides it, or request a private contact without disclosing the material.

Original contributions are distributed under the repository's MIT license. Do not copy third-party code or datasets without preserving their applicable license and attribution.
