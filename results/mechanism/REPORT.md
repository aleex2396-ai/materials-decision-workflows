# Supplementary mechanism study

This supplement investigates covariance scale and radius selection. It does not replace the frozen primary decision-study endpoint.

The fixed-trace likelihood and decision fits use identical training and validation draws, parameter bounds, initialization and iteration caps. Their nominal second moments have the same training trace. The radius baseline freezes the unnormalized likelihood fit, then selects a radius using only validation regret over the full configured grid. Known population moments are used for generating synthetic data and subsequent evaluation, never by a fitting objective.

Evaluation uses analytic expected cost and regret under the same IID and oracle-oriented shifted moments as the primary study. Shifts remain defined using the fixed primary radius, including when the fitted baseline selects another radius.

Recorded failures: 0. All missing pairs are listed in summary.json; optimizer iteration-limit returns remain reported rather than treated as converged.

| Estimator | Requested fits | Returned fits | Converged | Nonconverged |
|---|---:|---:|---:|---:|
| shape_likelihood | 36 | 36 | 8 | 28 |
| shape_decision | 36 | 36 | 7 | 29 |
| radius_tuned_likelihood | 36 | 36 | None | None |

Decision-fit numerical budget: 765 controller requests, 767 conic attempts, 2 rejected attempts and 2 recovered requests. Recovery is bounded to Clarabel followed by SCS on numerical failure, with identical acceptance thresholds. Rejected attempts remain in the trace and are not accepted as certificates.

## Primary supplementary comparison

Difference = fixed-trace decision regret minus fixed-trace likelihood regret. Negative favors decision fitting; positive favors likelihood fitting. Intervals are paired bootstrap 95% intervals over seeds and are descriptive, with no multiple-comparison correction.

| Training paths / shift | Complete pairs | Mean difference | 95% interval |
|---|---:|---:|---|
| n=8/iid | 12/12 | -0.116167 | [-0.234425, 0.00783415] |
| n=8/oracle_top_0.5r | 12/12 | 0.0919013 | [-0.0163535, 0.231331] |
| n=8/oracle_bottom_0.5r | 12/12 | -0.128077 | [-0.274618, 0.0167118] |
| n=8/oracle_top_1.5r | 12/12 | 0.633389 | [0.321406, 0.991915] |
| n=28/iid | 12/12 | -0.0450642 | [-0.149385, 0.0573298] |
| n=28/oracle_top_0.5r | 12/12 | 0.208336 | [0.0838065, 0.361596] |
| n=28/oracle_bottom_0.5r | 12/12 | -0.0352827 | [-0.186119, 0.0921404] |
| n=28/oracle_top_1.5r | 12/12 | 0.840685 | [0.546641, 1.15827] |
| n=112/iid | 12/12 | -0.0291636 | [-0.0772754, 0.0240323] |
| n=112/oracle_top_0.5r | 12/12 | 0.210667 | [0.126332, 0.30946] |
| n=112/oracle_bottom_0.5r | 12/12 | -0.0104699 | [-0.0636053, 0.0430487] |
| n=112/oracle_top_1.5r | 12/12 | 0.828128 | [0.592373, 1.07594] |

## Decision shape fitting versus likelihood plus radius selection

| Training paths / shift | Complete pairs | Mean difference | 95% interval |
|---|---:|---:|---|
| n=8/iid | 12/12 | -0.152684 | [-0.294072, -0.00128518] |
| n=8/oracle_top_0.5r | 12/12 | 0.193334 | [-0.0994535, 0.421176] |
| n=8/oracle_bottom_0.5r | 12/12 | -0.175686 | [-0.326333, -0.00662184] |
| n=8/oracle_top_1.5r | 12/12 | 1.06853 | [0.202132, 1.72735] |
| n=28/iid | 12/12 | -0.14213 | [-0.251856, -0.0432915] |
| n=28/oracle_top_0.5r | 12/12 | 0.150356 | [-0.0925894, 0.383006] |
| n=28/oracle_bottom_0.5r | 12/12 | -0.131577 | [-0.288846, 0.00712952] |
| n=28/oracle_top_1.5r | 12/12 | 0.872705 | [0.152926, 1.51548] |
| n=112/iid | 12/12 | -0.0483081 | [-0.131905, 0.0439331] |
| n=112/oracle_top_0.5r | 12/12 | 0.192425 | [-0.0265319, 0.383655] |
| n=112/oracle_bottom_0.5r | 12/12 | -0.0311017 | [-0.149127, 0.0997989] |
| n=112/oracle_top_1.5r | 12/12 | 0.807707 | [0.116899, 1.37903] |

The covariance-scale identity W(K;cM,r)=c W(K;M,r/sqrt(c)) motivates this ablation. Fixing trace removes variation in total nominal disturbance scale across hyperparameters within a fit. Training trace remains estimated, so this is not an oracle-scale experiment. Radius selection receives the same validation samples but additional controller evaluations; actual budgets are retained.

No finite-sample coverage claim is made for the Wasserstein radius. These known-dynamics synthetic results do not certify learned dynamics or physical material behavior. Read fits.json for parameter trajectories, radius candidates, numerical diagnostics and convergence flags; failures.json preserves failed partial fits.
