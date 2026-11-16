# Inside a materials decision campaign

The repository provides two executable planning workflows. A sequential policy buys one lab operation at a time to choose a reproducible formulation under a shared budget. A batch optimiser selects a group of formulations for prediction, manufacture and multiobjective evaluation. Both use synthetic materials, instruments and costs. [Sequential campaign](../results/shared_budget/REPORT.md) · [Batch campaign](../results/campaign/REPORT.md)

## Choose the next operation under one budget

An incoming-stock assay can be useful when two plausible assay results would lead to different formulations. A re-read of an existing specimen can resolve measurement uncertainty. An independent remake can supply evidence about reproducibility. These operations compete with the cost of making a new design, and with stopping and selecting an already qualified material.

The sequential policy models a finite set of plausible material states. Each state assigns a reproducible utility to each candidate and a probability to each possible return from a lab action. Stopping earns the highest posterior expected utility among qualified materials and a fixed outside option. At history $h$ with $b$ credits, purchasing action $a$ has value

$$
Q(a\mid h,b)=-\lambda c_a+
\sum_y P(y\mid h,a)\,V(h\cup\{(a,y)\},b-c_a).
$$

Here $c_a$ is the credit price, $\lambda$ is the configured utility cost per credit, and $V$ is the best value of stopping or taking another allowable action. For each possible return $y$ the policy updates the posterior by Bayes' rule and evaluates the remaining choices. A failed preparation spends its credits and cannot qualify a new formulation. The calculation is exact for its finite model and horizon. Its interpretation depends on the supplied priors, observation likelihoods and utility. [Policy implementation](../src/materials_workflow/shared_budget.py)

The paired campaign starts from a response surface fitted to 72 synthetic preparations. The two candidate formulations use different amounts of the same incoming binder lot, 75% and 15% of the mixture. The stock viscosity therefore changes their predicted performance by different amounts. A one-credit viscometer reading distinguishes two viscosity ranges. After the reading, four credits remain to make one formulation. The report compares the resulting decisions with immediate fabrication under the same five-credit budget. The case is constructed to illustrate a decision reversal. [Case generator](../src/materials_workflow/shared_case.py) · [Paired report](../results/shared_budget/REPORT.md) · [Event trace](../results/shared_budget/trace.json)

The numbered sections below follow the batch campaign. Its three-component formulations have two measured outcomes, light transmission and fracture toughness. [Batch preset](../src/materials_workflow/preset.json)

## 1. From instrument return to an eligible observation

A submitted JSON object names the run, its independent preparation group, the source record, the intended design, the make outcome, the evidence checks, the supplier properties and the assay endpoints. Here is a complete synthetic return a reader can submit with `materials-workflow ingest --db my-campaign.sqlite3 --input return.json`.

```json
{
  "run_id": "demo-coupon-1",
  "group_id": "demo-preparation-1",
  "source_id": "example-instrument#1",
  "design": {"a": 0.30, "b": 0.45, "c": 0.25, "cure": 0.55, "route": "cast"},
  "make_status": "success",
  "evidence": {
    "instrument_check_passed": true,
    "protocol_followed": true,
    "sample_identity_confirmed": true
  },
  "descriptors": {"density": {"value": 1520, "unit": "kg/m3"}},
  "targets": {
    "transmission": {"value": 73, "unit": "%"},
    "toughness": {"value": 6600, "unit": "J/m2"}
  }
}
```

The database lands the verbatim JSON values, receipt digest and source locator. The performance row contains density 1.52 g/cm³, transmission 0.73 and toughness 6.6 kJ/m². Scattering and viscosity remain missing in the measurement record. An identical resubmission resolves to its original receipt. A changed payload claiming the same revision triggers review. A correction names the next revision and the revision it supersedes. Training and makeability queries select eligible current records. Multiple coupons from one preparation carry the same group ID. [Registry and conversion tables](../src/materials_workflow/registry.py)

| Evidence | Immediate action | Information the system retains |
|---|---|---|
| Complete evidence and a very poor material response | Use the measured endpoints | The performance surface includes the poor result. |
| Correctly attempted material fails to form | Record the failed make | The makeability model receives a negative observation. Performance has no target. |
| A standard check fails on an identified specimen | Request another measurement | The original reading stays on its receipt. A confirmed successful make can still inform makeability. |
| The manufacturing procedure deviates | Request a fresh make | A correctly measured unintended material cannot stand in for the intended design. |
| Identity is unresolved or fault evidence conflicts | Ask for review | The evidence stays available for a targeted check and an explicit revision. |

The embedded database records the submitted evidence and its digest. A physical instrument integration would also need to establish which specimen was measured and who supplied the data.

## 2. Act or Ask when the evidence disagrees

A doubtful result raises two distinct experimental costs. Admitting a compromised measurement can distort a promising region of the design space. Withholding a valid but disappointing result can make that region look better than it is. The hard-gate challenge includes both an undetected meter bias and a false instrument alarm so both mistakes appear in the evidence. [Constructed routing challenges](../src/materials_workflow/decision_replay.py)

For the softer case of discordant readings, a second policy gets three diagnostic signals measured in units of check repeatability: the standard residual, process residual and repeat disagreement. A labelled synthetic failure bank trains class-conditional densities for valid observations, measurement faults, procedure faults and mixed causes. The policy can suggest use, remeasurement or remake, and defers mixed, low-confidence and unsupported cases. Its recommendation is stored alongside the original review receipt. Changing a record’s eligibility requires new evidence through intake.

The training, calibration, same-distribution evaluation and shifted evaluation use independent generator streams and one simulated preparation per draw. For each threshold in a predeclared grid, let $n$ be the number settled and $k$ the number assigned an incorrect action. The calibration uses a one-sided exact binomial upper bound, with the error budget divided across the grid, and chooses the highest-coverage qualifying policy. An empty qualifying set sends all cases to review. Evaluation reports harmful admissions, harmful exclusions, incorrect remedies, the fraction settled and results by cause. A fixed-rule comparator sees the same three signals. The risk statement applies to independent draws from the calibration population. A drifted instrument or a referral queue with a different case mix can change that risk. [Policy and failure-bank implementation](../src/materials_workflow/adjudication.py) · [Learn then Test](https://arxiv.org/abs/2110.01052)

## 3. Complete the feature view while preserving what was measured

The three optional pre-run properties are density, scattering and viscosity. For each training fold, the model estimates group-balanced means, variances and correlations from the available measurements. A component pair with few jointly observed preparations receives stronger shrinkage. A positive-definite projection makes the covariance usable even when different pairs were observed on different subsets.

For observed entries $x_o$ and missing entries $x_m$, the Gaussian working model gives

$$
\mathbb E[x_m\mid x_o]=\mu_m+\Sigma_{mo}\Sigma_{oo}^{-1}(x_o-\mu_o),
\qquad
\mathrm{Cov}(x_m\mid x_o)=\Sigma_{mm}-\Sigma_{mo}\Sigma_{oo}^{-1}\Sigma_{om}.
$$

A complete feature view retains each field’s source, unit, mean and standard deviation. The source is observed, conditional, marginal or unavailable. A property with too few independent observations stays unavailable to the model. The materialized view is keyed to the eligible records and estimator. The original measurement remains accessible after an estimate or amendment. The training path refits the imputer inside each fold, so a held-out group contributes neither a target nor a correlation coefficient to its own prediction. Missing performance targets never become empirical training labels. Systematic missingness or a new supplier can break the conditional model. [Descriptor view](../src/materials_workflow/descriptor_view.py) · [Estimator and prediction code](../src/materials_workflow/formulation.py)

## 4. Predict a formulation before making it

Sometimes the highest-value experiment is a measurement of a missing supplier field. The `measure-or-make` command compares recipes under a declared compromise between transmission and toughness. It evaluates the possible change in the preferred recipe as information about one field becomes available. An assay is worth purchasing if that improvement outweighs the assay cost in the same utility units. The calculation assumes independent stock cards across recipes. This single-choice decision can give a different answer from the multiobjective batch planner. [Value-of-information calculation](../src/materials_workflow/value_of_information.py)

The features include the ternary mixture, pairwise mixture products, cure level and curvature, composition–cure interactions, route, route–cure interaction, supplier estimates and missingness indicators. The simplex basis works at a pure component or a zero component fraction. Inputs use only pre-manufacture information. [Scheffé’s mixture-design paper](https://doi.org/10.1111/j.2517-6161.1958.tb00299.x)

A group-weighted multivariate Bayesian response surface handles the two targets jointly in transformed coordinates. Logistic and softplus inverse links respect their ranges. The expected design cross-product accounts for uncertain training descriptors. At prediction time, the model samples uncertain supplier fields and correlated response variation and returns paired draws for the two objectives. This is a working posterior with a moment approximation for uncertain inputs. It does not fit a full latent-variable generative model of manufacture.

Five-fold grouped validation, or fewer folds for small campaigns, gives each preparation equal weight. Each endpoint has a development-mean baseline, absolute error, interval width and a count of groups whose coupons all fell inside their pointwise model intervals. The intervals describe model-based uncertainty. Marginal support checks cover mixture components, cure, route and observed supplier values. Multivariate extrapolation can still occur inside those marginal ranges. [Prediction and diagnostics](../src/materials_workflow/formulation.py)

## 5. Spend the next batch budget

With fewer than three eligible preparation groups, the planner selects affordable space-filling designs across mixture, cure and route. A pending or completed experiment occupies its coordinates. After a model can be fitted, the same interface ranks candidates using the following procedure.

1. Compute the measured nondominated transmission–toughness frontier. First average coupons within independent preparations and give those preparations equal weight within a design. Anchor and scale the objective space from measured ranges.
2. For a new design, sample joint performance and calculate the expected additional dominated area above the reference point. A design that improves one objective while sacrificing some of the other can still extend the frontier.
3. Estimate makeability from nearby adjudicated attempts with a locally weighted beta model. Weight expected area gain by makeability and a diversity factor, then divide by cost. Apply support, feasibility, pending-work and remaining-budget checks.
4. Select a batch greedily. The predicted means of eligible pending and selected designs stand in temporarily for their unknown outcomes. A proximity penalty discourages redundant slots. The packet records every score term and exclusion. This finite-pool approximation uses paired draws within each design and mean estimates for pending outcomes. Exact joint noisy batch EHVI would require a joint posterior across designs and an uncertain baseline frontier. [Planner](../src/materials_workflow/planner.py) · [Daulton, Balandat and Bakshy on parallel EHVI](https://arxiv.org/abs/2006.05078)
5. Allow an explicit independent repeat of a measured design. Its utility is the expected absolute movement of the observed frontier when the repeat updates that design’s mean. This rewards confirmation of a consequential trade-off and can also reward noisy observations. Each repeat identifies the earlier run and requires a fresh preparation. Replaying an existing receipt cannot create that evidence.

## 6. Inspect the closed loop

The runner writes its batch and pre-run predictions before generating any follow-up observations. Each selected experiment receives its own fresh preparation ID, and the simulated instrument uses the same stock that supplied the visible pre-run properties. The new returns pass through the same gate. Refit uses the updated eligible snapshot, then ranks the remaining designs with the same per-batch spend limit. Pending work stays pending across both decisions. [Campaign orchestration](../src/materials_workflow/campaign.py)

| Entry point | Inspectable output |
|---|---|
| `shared-budget` | Two executed paths from the same initial decision, the fabrication-only comparison, an action trace and a report. |
| `session` | A persistent decision ledger where the operator can reserve an action, submit its observation, inspect the budget and request the next action. |
| `demo` | Fresh campaign directory with a working SQLite ledger, a sealed pre-run decision, two model snapshots identified by their training inputs, a report, a scientific figure and an offline explorer. |
| `ingest` and `audit` | Receipt per source payload, revision lineage, reasons for exclusion, raw and canonical values, provenance for normalized measurements. |
| `features` | Current versioned supplier-property estimates or an explicitly requested historical view with a freshness indication. |
| `predict` | Two endpoint estimates, uncertainty, interval summaries, the origin of each descriptor and the support decision. |
| `measure-or-make` | A shortlist decision, expected value of each candidate assay, chance of changing the choice and its net value after assay cost. |
| `recommend` | Affordable batch, objective-space reference, makeability estimates, pending-work treatment and the alternatives considered at each slot. |

The checked-in [packet](../results/campaign/packet.json) supplies the exact inputs and decisions behind the [rendered report](../results/campaign/REPORT.md). The [publication manifest](../PUBLIC_EXPORT_MANIFEST.json) enumerates the public files. Each study has hashes for its own producer, configuration and artifacts.
