# Materials Decision Workflows

This repository is a public re-export of work done at a startup, rebuilt on independent synthetic problems and published as a single initial commit to keep the original development history and confidential data private.

This pipeline helps decide which material to test next, which candidate to recommend, and how to adjust a simulated process as new measurements arrive. It learns from simulated measurements, compares possible experiments, and checks how well its decisions hold up under uncertainty.

**All bundled observations, fitted results and figures are generated from independent synthetic problems.** They are not anonymized extracts, rescaled measurements or fitted replicas of a private dataset. The optional local-data adapter requires a caller-supplied schema; no source-specific schema or real observations are distributed. See the [data policy](docs/DATA_POLICY.md).

## Start here

Use Python 3.12 in an activated virtual environment, from the repository root:

~~~bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m crfml.cli counterexample
~~~

The last command runs a small uncertainty-selection example. To execute the complete workflow and regenerate all four studies:

~~~bash
python -m crfml.cli all --output results
~~~

The full numerical study takes longer than the small example. It uses CPU solvers and replaces artifacts in the selected output directory. Use a fresh directory to keep an existing run.

## Workflow

~~~mermaid
flowchart LR
  A[Synthetic experiments] --> B[Validate and represent observations]
  B --> C[Fit prediction and uncertainty models]
  C --> D[Select and observe the next experiment]
  D --> C
  D --> E[Recommend a material]
  E --> F[Known synthetic process model]
  F --> G[Estimate disturbance moments]
  G --> H[Synthesize a causal robust controller]
  H --> I[Evaluate cost, regret and shifted laws]
  H --> J[Differentiate validation loss]
  J --> G
~~~

The [pipeline map](docs/PIPELINE.md) covers data acquisition, processing, representation, feature engineering, model development, evaluation, discovery and feedback. Offline learning and online feedback are identified separately.

## Included studies

| Study | What it demonstrates | Command | Results |
|---|---|---|---|
| Materials learning | GP prediction, conformal diagnostics, finite-pool acquisition and bounded adaptive MPC | `python -m crfml.cli run` | [Report](results/public/REPORT.md) |
| Decision-focused learning | Covariance shrinkage learned through causal Wasserstein regret control, matched likelihood baselines and distribution shifts | `python -m crfml.cli decision` | [Report](results/decision/REPORT.md) |
| Mechanism controls | Fixed-trace covariance and validation-selected radius comparisons | `python -m crfml.cli mechanism` | [Report](results/mechanism/REPORT.md) |
| Control references | Bayesian information value, adversarial identification limits and matched adaptation | `python -m crfml.cli reference` | [Report](results/reference/REPORT.md) |

Configurations are under configs/. Each study retains its inputs, per-seed outputs, numerical diagnostics, failure records and source/configuration/artifact hashes. All examples can run without an external database, credentials or confidential files.

## Mathematical and implementation details

- **Prediction and selection:** exact GP conditioning, analytic marginal-likelihood gradients, finite-sample calibration and a selection-coverage counterexample.
- **Learning through control:** analytic gradients and Hessians, implicit solution-map differentiation, fresh-solve finite-difference checks and bounded solver recovery.
- **Robust decisions:** lifted dynamics, causal feedback, a noncausal oracle, a semidefinite formulation and an independently evaluated optimal-transport adversary.
- **Diagnostic analysis:** a covariance-error/policy-performance ordering reversal, a conditional transfer bound, scale/radius equivalence and outer-optimiser stationarity checks.

The robust controller implements the formulation of [Al Taha, Yan and Bitar (2023)](https://arxiv.org/abs/2304.06783). The [methods review](research/DECISION_FOCUSED_REVIEW.md) identifies relevant decision-focused learning, covariance estimation and differentiable optimisation work, with OpenAlex records and inspected GitHub implementations. Established methods are attributed.

## Findings and limits

The benchmark illustrates why prediction error, validation decision loss and robustness are different objectives. With 28 estimation and 64 validation trajectories, the fixed-budget decision learner improves mean IID regret relative to likelihood fitting, while worsening regret under the strongest directional shift. Fixing covariance trace weakens the IID evidence. A separate convergence diagnostic reverses one seed's IID ranking despite a lower validation loss.

These results describe the declared synthetic models and optimisation budgets. The main controller uses known dynamics, perfect state observations, unconstrained inputs and a finite horizon. Its Wasserstein radius is a design parameter with no asserted statistical coverage guarantee. The separate scalar controller has its own conditional upper-state safety result.

See [findings](docs/FINDINGS.md), [transfer theory](docs/DECISION_FOCUSED_THEORY.md), [learning implementation](docs/DECISION_LEARNING.md), [optimiser diagnostics](docs/OPTIMIZER_DIAGNOSTICS.md) and [technical validation](docs/TECHNICAL_VALIDATION.md).

## Reuse and publication

The Python import/CLI package is named `crfml`. Its model, acquisition, synthesis and data-contract modules can be inspected and adapted independently. The bundled process model is illustrative; using another process requires specifying its dynamics, costs, observation assumptions and evaluation protocol.

[CONTRIBUTING.md](CONTRIBUTING.md) describes development checks and synthetic reproductions. [Reproducibility](docs/REPRODUCIBILITY.md) and [recorded verification](docs/RELEASE_VERIFICATION.md) distinguish observed checks from cross-platform claims.

To prepare a fresh publication copy:

~~~bash
python scripts/export_public.py --destination dist/public-repo
~~~

The exporter copies only reviewed paths, validates all result provenance and writes a file manifest. It excludes local inputs, caches, logs and source Git history. It refuses to overwrite a destination. Only the resulting clean directory is intended for publication.

Original code and documentation use the [MIT license](LICENSE). Public literature metadata and referenced third-party work retain their own terms.
