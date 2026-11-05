# Reproducibility and measurement protocol

## Four studies and their roles

The main study is decision-focused covariance learning through causal Wasserstein minimax-regret synthesis. Its configuration, sample budgets, seeds, shifts and primary comparison were fixed before its outcomes. The primary contrast is decision-trained minus likelihood-trained expected regret, using the same covariance family, data and robust-control objective.

The mechanism supplement was declared after identifying the analytic covariance-scale/radius equivalence and before observing its results. It compares fixed-trace fits and a validation-selected radius baseline. It supplements the original endpoint rather than replacing it. The earlier materials/scalar-control experiment and the Bayesian/adversarial control study remain separately reported references.

All four configurations are under configs/. The main and mechanism studies use 8, 28 or 112 estimation trajectories **plus 64 validation trajectories**, with twelve independent seeds. Training prefixes are nested across budgets within each seed. Validation draws are shared by competing estimators; evaluation draws are separate. Analytical population moments are available only to synthetic generation and subsequent scoring, except for explicitly labelled oracle comparators. Timepoints within a trajectory are not independent experimental replicates.

## Statistical and numerical interpretation

The 95% intervals resample paired seed differences with fixed bootstrap seeds. They describe finite-benchmark simulation variability, not universal performance, simultaneous coverage, or multiplicity-adjusted discoveries. Missing fits and complete-pair denominators are reported. Outer optimisation has a fixed 12-iteration cap; a returned iterate is not automatically a converged fit.

Numerical checks include primal feasibility, dual feasibility, stationarity, complementarity, independent fixed-policy transport evaluation, causal rollout reconstruction, and independently re-solved finite-difference hypergradients. Decision-layer requests first use Clarabel. On a numerical failure, a single SCS attempt may recover the same problem under unchanged acceptance thresholds; both attempts remain logged. Controller requests and conic attempts are counted separately. Solver checks do not supply the rigorous global objective-gap certificate assumed in the transfer proposition.

Known dynamics, Gaussian fitted nominal laws, perfect state measurements and a chosen radius define the main study. The radius has no asserted finite-sample coverage guarantee. Shift names refer to the oracle centre; membership in each estimated-centre ball is calculated explicitly.

## Reproduce

Install the pinned Python 3.12 environment and editable package using the README, then run:

~~~bash
python -m pytest -q
python -m crfml.cli all --output results
~~~

Use a fresh output directory for a new experiment; the CLI replaces files within its selected output. Each group contains its source/configuration hashes, artifact hashes and report. The mechanism producer also records source hashes at the start and end and rejects changed-source provenance during export. The recorded release was generated with all source modules frozen for the entire run.

Compare numerical records and policies across reruns. Wall-clock timings and solver attempt durations are measurements, not deterministic outputs. Cross-platform BLAS and solver differences may alter floating-point details or solver convergence. Byte-level source provenance does not imply cross-platform numerical identity. Git attributes preserve source line endings so a clone does not itself invalidate those hashes.

The exporter requires all four complete result groups, current source and configuration hashes, matching summary configurations, exact artifact hashes and an explicit independent-synthetic origin. It creates a fresh allowlisted tree with a manifest and refuses overwrite. Run tests and reproduction from that tree in an environment without the development PYTHONPATH.

## Data boundary

The distributed generators use invented constants and independent random draws. They have not been fitted to private observations, and the reported results are not aggregations of real measurements. Public reproduction requires no external experimental dataset or credentials.

For a new dataset, define the sampling unit, grouping and reservation before accessing evaluation outcomes. Related measurements may share an experiment, material or date; row-wise random splitting can leak information across partitions. Dataset-specific identities, paths, schemas, access records and fitted artifacts belong outside the public example tree. [DATA_POLICY.md](DATA_POLICY.md) describes what may enter a release.

## Observed release verification

The [release verification](RELEASE_VERIFICATION.md) records a pinned isolated installation, 139 passing tests in the neutral export and a complete four-study reproduction. All 727,508 compared numerical values in 15 result JSON files matched exactly, excluding measured elapsed-time fields. The recorded verification was performed locally. Remote CI results, when available, are separate observations and should be checked for the relevant commit.
