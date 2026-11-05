# Findings and interpretation

The main research question is whether a disturbance-moment estimator should be trained for density prediction or for the decisions made by a causal robust controller. Public results come entirely from independent simulation. Complete per-seed measurements, fit trajectories, numerical diagnostics and paired intervals are in the [main report](../results/decision/REPORT.md), with the [mechanism supplement](../results/mechanism/REPORT.md) kept separate.

## Prediction error does not order controller quality

The exact one-step construction gives a strict counterexample: a larger diagonal covariance error preserves the true optimal controller, whereas a smaller correlation error changes its action and increases transferred regret. This is a policy comparison, not merely a sensitivity calculation for a fixed controller.

The transfer bound identifies the relevant decision-sensitive matrix and its dependence on the transport dual's spectral margin. It is a deterministic result with an explicit global optimisation-gap premise. It neither supplies a sampling guarantee nor equates small KKT residuals with that premise.

## Ordinary decision loss can conflict with robustness

The learner optimizes ordinary validation regret after robust controller synthesis. This is a different objective from minimizing worst-case transferred regret. The benchmark exposes that difference: lower IID regret need not survive a directional shift, even though every controller was synthesized using a Wasserstein ambiguity set.

Each shifted law's actual distance to the fitted nominal centre is computed. A shift within a named distance of the oracle centre can lie outside the estimated-centre ball; the two claims must not be conflated. The main results therefore report covariance error, ordinary cost and regret, true-centre worst-case regret, and membership separately.

The [outer-optimisation diagnostic](OPTIMIZER_DIAGNOSTICS.md) shows that several capped fits remain nonstationary. In a separately declared seed-zero check, extending both objectives to a sixty-iteration cap changes the IID ordering at the middle sample budget: improved decision-training validation loss accompanies worse IID test regret. This limits any interpretation of the original finite-budget gain as an advantage of fully optimized objectives. The directional-shift disadvantage persists in all three diagnostic budgets.

## Covariance learning can also tune conservatism

The exact identity W(K;cM,r)=c W(K;M,r/sqrt(c)) shows why inflating a covariance can mimic reducing a robustness radius. The primary two-parameter family changes both scale and correlation. Its performance cannot by itself establish a benefit from better correlation estimation.

The supplementary experiment fixes the training trace for both learners and separately permits the likelihood baseline to choose its radius using the same validation data. These controls narrow the interpretation rather than supplying a replacement endpoint. The complete intervals and failed/converged fit counts remain visible, including comparisons that do not support a clear IID advantage.

## More density-model complexity need not help this objective

For the stated quadratic loss and admissible continuous nominal laws, exact Wasserstein robust evaluation depends on the second moment. Gaussian and Student nominal laws with the same second moment therefore have the same robust objective and controller. Their tail quantiles can differ.

The negative control checks both facts using a common optimal transport. This is a precise sufficiency result for the chosen problem, not a claim that higher moments never matter in control.

## The earlier workflow supplies useful reference failures

The [materials study](../results/public/REPORT.md) retains GP prediction, calibration and finite-pool discovery. The GP's IID prediction advantage did not persist under the declared covariate shift. Fixed-candidate uncertainty coverage also failed after selection in an exact independent Gaussian counterexample.

Constrained acquisition reduced some experiment violations but did not eliminate them. Queried-best quality is reported separately from the final recommendation. Violation totals include both the common initial design and subsequent queries.

Extra probing did not generally improve the original scalar tracking task. The [controlled reference study](../results/reference/REPORT.md) adds Bayesian value-of-information calculations, a matched initial-predictor comparison with the same adaptation, and an adversarial indistinguishability construction. Those tests separate information value from changes in the feedback law or noise assumptions.

## Scope

The contribution is a verified specialization of existing robust-control and implicit-differentiation methods, supported by exact diagnostic analysis and controlled experiments. Closest prior work is documented through OpenAlex retrievals and inspected GitHub repositories. No first-ever method claim is needed.

Known finite-horizon dynamics, a restricted moment family and synthetic environments limit external validity. Learning dynamics would introduce policy-dependent residuals. A statistically calibrated ambiguity radius, longer-horizon scalable differentiation, or an outer robust-transfer objective would each require a separately declared study and additional analysis. No physical chemistry or experimental safety claim follows from this benchmark.
