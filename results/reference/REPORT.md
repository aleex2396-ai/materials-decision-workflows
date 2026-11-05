# Controlled reference experiments

These supplementary references isolate information acquisition and the nominal estimator from online adaptation. They accompany the multi-state covariance-learning study; no laboratory measurements are used.

## Two-stage value of information

Both informed and frozen-belief policies observe the same next physical state. The frozen continuation keeps its old belief for action selection, but its performance is integrated under the actual conditional posterior. The first action is exhaustively searched on a finite grid; the expectation uses piecewise quadrature. Refinement is a numerical diagnostic, not a proof of continuous global optimality.

| Regime | Policy | First action | Quadrature cost | MC cost | MC mean SE | Violations |
|---|---|---:|---:|---:|---:|---:|
| low_then_high | informed | 0.50000 | 0.174100 | 0.173136 | 0.000708 | 0 |
| low_then_high | frozen | 0.49000 | 0.174280 | 0.173311 | 0.000714 | 0 |
| low_then_high | myopic | 0.12834 | 0.221594 | 0.220473 | 0.000823 | 0 |
| low_then_high | maximal_probe | 0.80000 | 0.204411 | 0.203883 | 0.000474 | 0 |
| high_both | informed | 0.80000 | 0.336711 | 0.335570 | 0.001444 | 0 |
| high_both | frozen | 0.80000 | 0.337638 | 0.336495 | 0.001436 | 0 |
| high_both | myopic | 0.80000 | 0.336711 | 0.335570 | 0.001444 | 0 |
| high_both | maximal_probe | 0.80000 | 0.336711 | 0.335570 | 0.001444 | 0 |
| low_both | informed | 0.14000 | 0.004527 | 0.004547 | 0.000025 | 0 |
| low_both | frozen | 0.14000 | 0.004565 | 0.004585 | 0.000025 | 0 |
| low_both | myopic | 0.12834 | 0.004593 | 0.004613 | 0.000026 | 0 |
| low_both | maximal_probe | 0.80000 | 0.150194 | 0.149932 | 0.000617 | 0 |
| narrow_prior | informed | 0.68000 | 0.175425 | 0.175505 | 0.000102 | 0 |
| narrow_prior | frozen | 0.68000 | 0.175425 | 0.175505 | 0.000102 | 0 |
| narrow_prior | myopic | 0.14613 | 0.248236 | 0.248365 | 0.000161 | 0 |
| narrow_prior | maximal_probe | 0.80000 | 0.179208 | 0.179279 | 0.000086 | 0 |
| large_noise | informed | 0.50000 | 0.182452 | 0.183946 | 0.000804 | 0 |
| large_noise | frozen | 0.50000 | 0.182617 | 0.184093 | 0.000804 | 0 |
| large_noise | myopic | 0.12834 | 0.230155 | 0.231594 | 0.000971 | 0 |
| large_noise | maximal_probe | 0.80000 | 0.212621 | 0.213767 | 0.000583 | 0 |

## Adversarial identification limit

Two fixed gains separated by 0.06250 generated transition histories differing by at most 5.55e-16, under matched admissible causal disturbances. The compatible interval never became narrower than 0.06250.

This is an action-adapted bounded-disturbance construction, not an IID-noise experiment. Its induction argument prevents a uniform vanishing-error claim over that adversarial class. IID uniform disturbances provide different identification information.

## Matched estimator/adaptation ablation

Every compared policy uses identical adaptive gain-set updates, robust constraints, cost weights and disturbances. Only the nominal centre changes: GP, initial physical midpoint clipped to the current set, or the directly observed noisy gain. Positive cost difference favours GP; negative favours the comparator.

| Acquisition / scenario / comparator minus GP | Seeds | Cost difference | Bootstrap 95% interval |
|---|---:|---:|---|
| random/iid/constant_adaptive-gp | 12 | 0.000431 | [-0.000162, 0.001072] |
| random/iid/measured_adaptive-gp | 12 | -0.000059 | [-0.000300, 0.000161] |
| random/bounded_shift/constant_adaptive-gp | 12 | -0.009827 | [-0.012316, -0.006807] |
| random/bounded_shift/measured_adaptive-gp | 12 | 0.000888 | [-0.000128, 0.001885] |
| greedy/iid/constant_adaptive-gp | 12 | 0.000498 | [0.000072, 0.000947] |
| greedy/iid/measured_adaptive-gp | 12 | 0.000321 | [-0.000012, 0.000769] |
| greedy/bounded_shift/constant_adaptive-gp | 12 | -0.010978 | [-0.013426, -0.008070] |
| greedy/bounded_shift/measured_adaptive-gp | 12 | 0.000368 | [-0.000763, 0.001488] |
| constrained_ei/iid/constant_adaptive-gp | 12 | 0.000551 | [-0.000064, 0.001100] |
| constrained_ei/iid/measured_adaptive-gp | 12 | 0.000046 | [-0.000298, 0.000403] |
| constrained_ei/bounded_shift/constant_adaptive-gp | 12 | -0.011429 | [-0.013959, -0.008433] |
| constrained_ei/bounded_shift/measured_adaptive-gp | 12 | -0.000726 | [-0.002070, 0.000373] |

All reference parameters and seed conventions are fixed in code/configuration; raw grid values, refinement deltas and paired seed values are retained in summary.json.
