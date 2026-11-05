# Technical validation record

This record summarizes an independent automated review of the numerical implementation, mathematical arguments and synthetic experiments. The review used source inspection, separately reconstructed calculations and selected test reruns. It is a reproducible engineering record, not external peer review or a formal proof certificate.

The implementation combines causal robust-control synthesis, differentiated solution maps, a transferred-policy counterexample, a conditional transfer bound and experiments that test the proposed learning mechanism. Existing control and differentiable-optimisation methods are attributed. The benchmark studies a restricted covariance family and known finite-horizon dynamics.

| Component | Checks and evidence | Scope |
|---|---|---|
| Learning theory | Distinguishes moment estimation, optimized robust-value sensitivity and transferred-policy loss; derives implicit hypergradients and an exact covariance-error/policy-ranking reversal | No statistical consistency theorem, finite-sample generalization rate or calibrated ambiguity-set guarantee is established. |
| Numerical implementation | Analytic gradients and Hessians, fresh perturbed SDP solves, differentiation guards, disjoint splits, failed-attempt logs and matched ablations | Small CPU-based benchmarks; scalability and performance on unrelated datasets have not been established. |
| Optimisation and control | Lifted dynamics, causal feedback, noncausal oracle, Wasserstein duality, adversarial transport, semidefinite reformulation and conic diagnostics | Known dynamics, perfect state observations, signed unconstrained inputs and finite horizon; the separate scalar module has different safety assumptions. |

## Mathematical assessment

The lifted dynamics, causal mask, quadratic regret identity and transformed SDP agree with the stated formulation of [Al Taha, Yan and Bitar](https://arxiv.org/abs/2304.06783). The nominal model is explicitly full-dimensional Gaussian with a positive-definite second moment, including a nondegenerate initial state. The implementation therefore avoids silently substituting a singular empirical distribution into the cited theorem. The paper's controller is reproduced, not presented as new.

The [learning theory note](DECISION_FOCUSED_THEORY.md) correctly separates three objects: the optimized robust value, the robust value of a transferred controller, and ordinary validation regret. The envelope derivative of the first is not substituted for the solution-map derivative needed by the third. The analytic Hessian uses the appropriate resolvent differential, and the implicit function calculation requires an interior stationary solution and nonsingular Hessian. The code checks numerical regularity rather than assuming convexity makes every solution map differentiable.

The finite transfer proposition is valid with its stated quantifiers: fixed dynamics, policy class and radius; positive-definite moments; an attained true optimum; and an estimated pair within a genuine global objective gap eta. The trace bound follows from affine dependence on the moment matrix and comparison with the true optimal pair. Its spectral-margin factor is correct. This is a deterministic analysis, not a data-dependent deployment certificate: the true optimal sensitivity is generally unknown, and numerical stationarity does not supply the premise about eta. A large upper bound near a small spectral margin is not, by itself, proof that a particular controller actually transfers badly.

The exact one-step ordering reversal is also valid. The diagonal perturbation leaves the optimal action unchanged, while the smaller off-diagonal perturbation changes the unique minimizer and increases true transferred regret for every nonnegative radius. This is stronger than merely showing that a fixed quadratic objective weights covariance directions unequally. It disproves a universal implication; it does not prove the proposed estimator should outperform likelihood fitting on average.

The scale identity and fixed-trace quotient derivatives are correct. Fixing trace removes total-scale changes across fitted parameters within an estimation sample, while leaving estimated training scale uncertain. The radius-selection baseline supplies an additional useful comparison. Neither control implies that all remaining differences are attributable to one uniquely identified causal mechanism.

For a fixed plant and validation law, ordinary expected cost and ordinary expected regret differ by a controller-independent oracle term. The distinctive cost-versus-regret design comparison here is the **inner distributionally robust optimisation**, where the worst-case law can differ; ordinary outer regret does not become a different training objective merely by subtracting that oracle constant.

## Independent verification

The reviewer independently reran the three numerical control, decision-learning and decision-study test files in the locked version 0.2.0 environment: **40 tests passed**. These include independent Gaussian/transport calculations, causal dynamics, oracle identities, controller comparisons, derivative checks, fixed-trace behavior and rejection/recovery paths. The project-wide suite was separately reported as 132 passing tests; that larger count is not substituted for the reviewer's own rerun.

Additional checks used saved public results and separate calculations:

| Check | Independent result |
|---|---|
| Reconstruct lifted matrices and noncausal oracle directly from saved dynamics, without importing project control code; recompute every IID policy evaluation | 540 policy records checked; largest cost/regret discrepancy 2.84e-14; future-disturbance coefficients were exactly zero |
| Regenerate validation draws and recompute final decision-fit validation losses | All 36 checked; maximum discrepancy 7.11e-15 |
| Inspect final main-study controller acceptance | All 540 policies passed recorded numerical and KKT gates; maximum scaled stationarity 4.95e-8 and scaled complementarity 1.46e-6 |
| Inspect recursive/pathwise evaluation diagnostics | Minimum reported realized regret 0.04163; maximum recursive-versus-lifted discrepancy 7.99e-15 |
| Verify the transfer trace bound using an independent scalar calculation | 40 independently generated positive-definite moment pairs, radius 0.37, scalar Brent minimisation and rank-one formulas; all satisfied the bound, with minimum bound-minus-excess slack 0.01653 |
| Verify supplementary trace normalization and radius selection | Fixed-trace ratios differed from one by at most 4.44e-16; all 36 selected radii minimized their complete logged validation grids |
| Check source/artifact consistency | Main source hashes matched current source; supplementary source/artifact hashes matched, and its recorded source snapshot was unchanged during the run |

The 40-case transfer check used Gaussian random 2-by-2 matrices Z with moments ZZ-transpose + 0.2 I, independently generated for the true and estimated moments, and a fixed seed of 67019. It used the closed-form rank-one Wasserstein objective and scalar optimisation rather than the project SDP. It is a numerical corroboration of the proof, not a replacement for it.

The main learning run recorded 755 controller requests and 755 conic attempts, with no rejected attempts. The mechanism run recorded 765 requests and 767 attempts: two rejected Clarabel attempts were recovered by SCS. The inspected recovery code solves the same mathematical problem with unchanged acceptance thresholds, permits at most these two prescribed attempts, and preserves the rejected attempts. It does not change endpoints or accept a failed numerical point.

These diagnostics constitute meaningful numerical evidence. They are not interval-arithmetic certificates or rigorous global suboptimality bounds. The documentation now states that distinction correctly.

## What the experiments establish

The [main study](../results/decision/REPORT.md) contains 36 completed decision fits, 540 policies and 2,160 policy/scenario records. Each estimation budget is supplemented by 64 validation trajectories. Expectations use known benchmark moments, with independent causal Monte Carlo checks. Seed-level pairing is appropriate; time steps are not treated as independent observations.

The empirical message is mixed and informative:

| Estimation trajectories | Main IID decision-minus-likelihood regret | Fixed-trace IID difference | Main stronger directional-shift difference |
|---:|---:|---:|---:|
| 8 | -0.16390 | -0.11617 | +0.81630 |
| 28 | -0.10054 | -0.04506 | +1.03229 |
| 112 | -0.04022 | -0.02916 | +1.13014 |

Negative differences favor decision fitting. The last column is the declared top-direction shift at distance 1.5 times the primary radius from the true center. The main IID differences correspond to approximately 1.0%, 0.65% and 0.26% of likelihood-controller regret. These are modest gains. Their main-study bootstrap intervals exclude zero at the two smaller budgets, but the experiment makes no multiplicity-adjusted discovery claim.

In the [fixed-trace supplement](../results/mechanism/REPORT.md), **all three IID comparison intervals include zero**. The stronger top-direction shift still favors likelihood fitting at every budget, with all three intervals above zero. This weakens a broad “better correlation learning improves control” claim while preserving a narrower observation: training for ordinary validation decisions can trade off performance against directional robustness, even when the inner controller is robust.

The radius-tuned likelihood comparison does not establish uniform superiority either. Fixed-trace decision fitting has lower mean IID regret at the two smaller budgets, but again has higher regret under the stronger top-direction shift. The chosen radius uses the same validation observations and a complete fixed grid; it is not chosen on test outcomes. It is a supplementary mechanism comparison, not a replacement for the primary endpoint.

Ambiguity-set membership is particularly revealing. At eight estimation trajectories, the true IID law lies outside every learned estimator's chosen-radius ball in this benchmark. At 112 trajectories it lies inside all likelihood-centered balls but only one third of decision-centered balls. Thus controller robustness relative to the fitted ball must not be confused with statistical coverage of the true law. The artifacts make the distinction explicit.

The Gaussian/Student same-moment control is appropriate: identical exact quadratic robust objectives coexist with different tail quantiles. This justifies learning the relevant second moment here and shows why adding a more flexible density model without changing that moment would not help this objective.

## Optimisation-budget limitation and secondary check

The primary comparison is between **fixed-budget fitting procedures**. Only 10 of 36 decision fits and 9 of 36 likelihood fits reported outer convergence within twelve iterations. In the fixed-trace supplement, the counts are 7 and 8. These are outer fitting limits, distinct from the accepted inner controller solves.

The reviewer matched every returned primary parameter vector to its saved evaluation and computed the box-projected residual norm `||theta - clip(theta - gradient, lower, upper)||_infinity`. Decision fits had median 0.002106 and maximum 0.245918; likelihood fits had median 2.906e-5 and maximum 0.076816. Some capped fits are therefore materially nonstationary. Residual magnitudes across differently scaled objectives are not a ranking of estimator quality.

Iteration limits prevent interpreting these results as a comparison of fully optimised bilevel and likelihood objectives. Preserving the fixed budget and reporting it is appropriate. The [secondary optimisation diagnostic](OPTIMIZER_DIAGNOSTICS.md) now reports seed zero, all three estimation budgets and both objectives at a sixty-iteration cap, with unchanged data, initialization, bounds and inner numerical gates. All six fits reached first-order stopping in 6–17 iterations, with projected residuals at most 9.65e-8 and no failed conic attempts. First-order stopping does not establish global optimality.

The diagnostic demonstrates a real budget effect. With 28 estimation trajectories, the original decision fit had IID population regret 15.499338 versus likelihood's 15.517022. The longer fits reverse that ranking: decision regret becomes 15.541301 versus likelihood's 15.509260. Decision validation regret improves from 15.064065277 to 15.062874594 while its population regret worsens. This supports the distinction between optimizing validation decisions and achieving reliable transfer; it does not establish that the all-seed mean ranking would reverse under longer fitting. The stronger directional shift continues to favor likelihood fitting at all three budgets in this selected seed. Eight-trajectory fits were already converged, and the 112-trajectory decision result changes very little.

The secondary check strengthens the diagnostic interpretation while further limiting a superiority claim. It remains explicitly post-review, single-seed evidence and leaves the primary endpoints unchanged.

The release process subsequently completed the separate [reproduction and publication verification](RELEASE_VERIFICATION.md). All four studies ran in the pinned environment, dependency checks passed, and the neutral export passed 139 tests, including the additional tracked-publication guard cases. Recursive comparison found no mismatches across 15 non-provenance JSON files and 727,508 numerical values, with maximum absolute difference zero; 3,800 elapsed-time fields were excluded. These are release-process reproduction results, distinct from the independent automated review's 40-test rerun and numerical reconstructions above. The agreement is an observed same-platform result, not a cross-platform reproducibility guarantee.

## Interpretation and remaining scope

No unresolved mathematical or implementation blocker was identified for the claims currently stated. The results do not justify claiming general empirical superiority, a calibrated robustness guarantee, learned-dynamics robustness or a new control theorem. The [prior-work review](../research/DECISION_FOCUSED_REVIEW.md) appropriately identifies established decision-focused learning, differentiable optimisation and covariance-learning precedents instead of treating their combination as automatically novel.

The evidence connects an explicit estimation-and-decision question to a counterexample and conditional bound, a differentiated robust-control implementation, and experiments that test the mechanism. The resulting finding is a performance/robustness tradeoff within the stated model; the experiments do not establish a universally better estimator.
