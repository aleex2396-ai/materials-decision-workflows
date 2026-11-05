# End-to-end pipeline and study map

This repository connects eight stages of a materials-learning workflow, from problem definition to evaluated decisions and feedback. The main research question is whether learning a disturbance second moment for downstream causal-control regret changes decision quality relative to learning the same estimator for likelihood. The discovery loop supplies a selected synthetic material to the control study; the control problem then has known dynamics and learned exogenous uncertainty.

## Eight-stage mapping

| Workflow stage | Executed implementation | Evidence and scope |
|---|---|---|
| Problem definition | Maximize synthetic material utility subject to a defect threshold; compare scalar bounded-noise control and multi-state Wasserstein minimax-regret control; test information and covariance-scale mechanisms separately | Four public study configurations declare objectives, comparators, budgets and endpoints. Normalized synthetic quantities have no claimed physical calibration. |
| Data acquisition | Seeded material observations, independent disturbance trajectories and uniform-gain episodes; OpenAlex literature and GitHub software research; optional local workbook intake | Public generators are explicitly implemented in `simulator.py`. Private intake is separate and is never imported by public experiments. |
| Data processing | Composition/schema checks, training-only transformations, missing-target abstention, disjoint estimation/validation/test draws; positive-definite moment regularization | `data.py`, `models.py`, `decision_learning.py`; second moments use the declared zero-mean model. A singular empirical moment is not silently treated as a full-dimensional nominal law. |
| Representation | Composition fractions/process setting and fixed descriptors; lifted state/input/disturbance trajectories; causal controller mask; gain-support interval for the Bayesian reference | `simulator.py`, `dr_control.py`, `dual.py`. The lifted disturbance includes the random initial state; no future disturbance is available to the causal policy. |
| Feature engineering | Interactions and train-only scaling for material models; diagonal shrinkage and trace-scaled ridge for disturbance moments; fixed-trace normalization in the mechanism study | The covariance family is identical for likelihood and decision losses. The supplement separates correlation shaping from overall uncertainty scale. |
| Model development | Exact GP and ridge baselines; sequential acquire/observe/refit discovery; covariance tuning by likelihood or implicit control-regret gradients; analytic uniform posterior for two-stage control | `models.py`, `discovery.py`, `decision_learning.py`, `dual.py`. The GP models material responses; the moment learner models disturbance uncertainty. |
| Evaluation | Independent prediction/calibration evaluation; paired discovery/control seeds; true noncausal-oracle regret; directional shifts; actual Wasserstein-ball membership; derivative/solver checks; paired mechanism comparisons | `study.py`, `decision_study.py`, `reference_study.py`, `mechanism_study.py`. Ordinary expected regret, robust value, prediction error, safety and runtime are separate quantities. Failed solves and nonconvergence remain recorded. |
| Discovery and feedback | Material recommendation feeds a material-dependent synthetic process; scalar controller updates gain support from transitions; the multi-state policy executes causally from observed states; outer covariance learning receives validation-regret gradients; reference policy updates its Bayesian belief | The loops are executable. Covariance fitting is offline and frozen before test execution. Synthetic observations are the public verification mechanism. |

The data contracts and decision problem determine which representations and model families are meaningful. The supplied examples use composition/process descriptors and linear dynamics; alternative representations require suitable labels and validation.

## Four complementary studies

| Study | Command and configuration | Question and principal artifacts |
|---|---|---|
| Materials learning and bounded control | `python -m crfml.cli run`; `configs/study.json` | Prediction and uncertainty, sequential material selection, and scalar closed-loop control. `results/public/` contains prediction metrics, discovery traces, control trajectories, summaries, figures and provenance. |
| Decision-focused Wasserstein regret control | `python -m crfml.cli decision`; `configs/decision_study.json` | Learn a nominal second moment through the causal minimax-regret SDP, and compare with matched likelihood fitting across budgets and shifts. `results/decision/` contains estimator fits, policies, test records, failures, diagnostics, reports and provenance. |
| Information and estimator references | `python -m crfml.cli reference`; `configs/dual_study.json` | Compare informed/frozen-belief two-stage decisions; construct adversarially indistinguishable gains; compare GP, constant and measured nominal gains with identical adaptive updates. `results/reference/` retains grid/quadrature values, refinement and Monte Carlo checks, paired results and provenance. |
| Covariance-scale mechanism supplement | `python -m crfml.cli mechanism`; `configs/mechanism_study.json` plus `configs/decision_study.json` | Test whether differences persist with equal moment trace and with validation-selected radius tuning for the likelihood center. `results/mechanism/` retains fit/radius candidates, paired records, policies, failures, reports and provenance. It does not replace the primary endpoint. |

`python -m crfml.cli all --output results` executes all four with the supplied configurations. Individual commands accept output paths for separate reruns. The studies share mathematical and generator code; they regenerate declared samples rather than consuming private inputs or tuning against another study's saved test outcomes. Current measured outcomes belong in [FINDINGS.md](FINDINGS.md) and the generated reports; this map describes what executes, not which method wins.

## The learning-to-decision loop

For each public seed, constrained material discovery produces a composition/process recommendation. `material_process` maps it into an invented coupled, two-state linear process and disturbance law. The dynamics are known to every compared controller. Independent trajectory draws provide estimation, validation and held-out test data.

The estimator forms `M = (1-alpha) S + alpha diag(S) + beta scale I`, where `S` and `scale` use estimation data only. A full-dimensional zero-mean Gaussian with that moment defines the nominal law. The likelihood comparator tunes the two parameters using validation Gaussian loss. The decision learner synthesizes a causal Wasserstein minimax-regret controller, differentiates its regular interior optimum, and updates the same parameters using validation expected regret. Solver agreement, stationarity, spectral margin, Hessian conditioning and independent perturbed-solve gradient checks gate that computation.

After selection, each controller executes on the same test disturbance paths through recursive dynamics. Only measured states and reconstructed past disturbances enter its actions. The comparator knows the whole disturbance path in advance and solves the unconstrained quadratic problem; its informational advantage defines true nonnegative pathwise regret. Test reports retain cost, regret, covariance error, worst-case calculations and numerical diagnostics. A label such as `0.5r` refers to distance from the specified reference center; membership in each learned center's ball is measured separately.

The mechanism supplement addresses an exact scale relation: multiplying a nominal moment by a positive scalar changes the effective Wasserstein radius after rescaling the robust objective. It compares fixed-trace learners and a likelihood estimator with validation-tuned radius. These analyses distinguish correlation estimation from changes in conservatism; their conclusions must come from recorded outcomes.

## Distinct control guarantees

The scalar bounded-disturbance controller has an upper-state constraint and a conditional invariant backup under fixed positive gain, exact observations and a correct disturbance/gain envelope. Its known-gain `oracle` is a receding-horizon reference, not a noncausal lower bound.

The Wasserstein controller reproduces the finite-horizon linear-quadratic formulation of Al Taha, Yan and Bitar with causal disturbance feedback and a true noncausal comparator. It is distributionally robust under its declared nominal-law, radius and known-dynamics assumptions. It has no hard state/input constraints. Its Wasserstein ball is a design choice, not an asserted finite-sample confidence set. The controller families have distinct assumptions and claims.

The two-stage Bayesian reference has an exact uniform posterior and terminal action, plus a finite-grid/quadrature first-stage search. It does not claim a continuous global solution or infinite-horizon optimal dual control. [THEORY.md](THEORY.md), [DUAL_CONTROL_THEORY.md](DUAL_CONTROL_THEORY.md), [DECISION_LEARNING.md](DECISION_LEARNING.md) and [DECISION_FOCUSED_THEORY.md](DECISION_FOCUSED_THEORY.md) give the derivations.

## Using your own data

The bundled studies run entirely on independent synthetic inputs. An optional local adapter requires a caller-provided source directory and extraction schema; no organization-specific schema or example workbook is distributed. Establish sample identity, units, missing-target rules and an untouched evaluation partition before fitting. Keep raw inputs, derived outputs and fitted artifacts outside any public release unless their publication rights and disclosure risks have been reviewed. See [DATA_POLICY.md](DATA_POLICY.md) for the release boundary.

Public simulation does not establish physical chemistry, laboratory efficacy or deployed control safety. No DFT/MD engine, atomic GNN, microscopy model, toxicity model or synthesis robot is claimed. Those extensions require relevant labels, measurements and a separate validation protocol.
