# Reproducibility and publication verification — 2026-11-05

The numerical artifact was reproduced in a fresh pinned Python 3.12 environment. The neutral publication copy was subsequently installed as an editable package in that isolated environment and tested without the development PYTHONPATH. The package import path resolved to the exported source.

## Observed checks

| Check | Result |
|---|---|
| Dependency consistency | pip check passed |
| Complete test suite in the neutral export | 139 passed, including seven additional tracked-publication guard tests |
| Independent automated review subset | 40 passed; see [TECHNICAL_VALIDATION.md](TECHNICAL_VALIDATION.md) |
| Full reproduction | All four CLI studies completed from the exported source |
| Machine-readable comparison | All 15 non-provenance JSON files matched after excluding recorded elapsed-time fields |
| Numerical comparison | 727,508 numeric values compared; maximum absolute difference 0 |
| Timing exclusions | 3,800 fields named learning_seconds, fit_seconds, solve_seconds or elapsed_seconds |
| Main numerical fit failures | 0 of 36 fits |
| Mechanism numerical fit failures | 0 of 36 decision fits; two rejected attempts recovered under unchanged numerical gates |
| Source/configuration/artifact validation | All four groups passed the exporter's current-byte hash checks |

The recursive JSON comparison checked keys, list lengths, nonnumeric values and numerical values. It permitted absolute tolerance 1e-8 and relative tolerance 1e-7, but observed zero numerical difference. Provenance JSON files were excluded from numerical equality because they hash artifacts containing measured timings. Their source, configuration and artifact consistency is checked separately by the exporter. This is an observed same-platform result, not a promise of bitwise numerical reproducibility across platforms.

The main study contains 540 policies and 2,160 policy/scenario evaluation records. The mechanism study contains 108 policies and 432 evaluation records. The scalar materials reference contains 24 prediction records, 36 discovery runs and 648 control trajectories. The Bayesian/adversarial reference and the separately declared six-fit convergence diagnostic are documented alongside them.

The original clean reproduction used frozen source modules throughout. Neutral publication preparation then made messages and amendment metadata in the optional local-data adapter generic. All four studies were regenerated with that revised source frozen, and comparison with the preceding clean reproduction again found zero difference across 727,508 numerical values. Mathematical algorithms and study configurations were unchanged. The local-data adapter is outside the public command import graph.

## Publication boundary

The publication export is a fresh manifest-verified copy without development Git history. Inputs consist of independent synthetic generators and public literature/software metadata. Organization-specific datasets, schemas, identifiers, images, fitted models and derived measurement results are excluded. The public command import graph has no dependency on local data intake.

The neutral export contains 112 allowlisted files plus its manifest. An independent read-only scan of its 106 text files found no known private identifiers, filenames, paths or source hashes. Manual framing review found no recipient-specific or application-oriented presentation. Only paper-level methodological references remain. No source workbooks or held-out measurements were opened during this publication review.

The tracked-file guard passed on the staged clean publication tree. Its tests include an ignored file force-added to Git; the guard rejected it before reading its contents or printing its name. The guard checks current tracked paths and working-tree artifacts, while the release process separately verifies exact Git blob and archive bytes. It does not claim to audit arbitrary historical commits.

The counts above describe the recorded numerical verification. Publication packaging checks and later CI runs should be assessed against their own manifest and commit. The [technical validation record](TECHNICAL_VALIDATION.md) reports the mathematical checks and empirical limitations; reproducibility does not imply general empirical superiority. [DATA_POLICY.md](DATA_POLICY.md) defines the public-data boundary.
