# A measurement that changes the next formulation

Two synthetic labs start with the same preparation history, stock specifications, fitted model, action menu and credits. The incoming binder lot is different. The decision code sees the instrument return only after it reserves the assay and pays for it.

Each lab has **5 credits**. An incoming-stock viscosity assay costs one credit in the bundled preset. A fresh formulation costs four. The binder-rich formulation uses 75% of the incoming stock and the binder-lean formulation uses 15%. Both use the same casting and cure settings. The reference has an archived specimen from this lot.

| Paid viscosity result | Next make attempted | Returned specimen | Direct make without diagnostics | Credits spent |
|---|---|---|---|---:|
| 1.939 Pa s (low) | binder-rich | fresh-binder-rich | binder-lean | 5 |
| 5.939 Pa s (high) | binder-lean | fresh-binder-lean | binder-lean | 5 |

## Why the lab paid for the assay

The planner scores every affordable next action by integrating its possible returns, replanning after each one, and accounting for the remaining credits. The terminal choice is the qualified recipe with the highest posterior expected utility or the outside option. The values below include the configured charge for future spending. A successful preparation with a usable return qualifies the recipe, including when the measured performance is poor.

The synthetic preference gives equal weight to transmission (a fraction) and the ratio `toughness / (toughness + 5 kJ/m²)`. Recipe utility averages that score across successful fresh preparations. Each credit spent costs an additional 0.003 in utility.

| First action | Immediate cost (credits) | Expected net utility with optimal continuation |
|---|---:|---:|
| `assay-stock` | 1 | 0.67883 |
| `reread-reference` | 1 | 0.64858 |
| `make-binder-lean` | 4 | 0.62076 |
| `make-binder-rich` | 4 | 0.61719 |
| `stop` | 0 | 0.52671 |
| `remake-reference` | 4 | 0.51471 |

The best policy with the full menu is worth **0.67883** under the declared prior. A policy restricted to making or independently remaking formulations starts with the same budget and is worth **0.62076**. These are model expectations over all possible results.

## What changed after the read

Both executions have the same opening decision digest. The viscometer measures the incoming stock. Below 4 Pa·s its receipt is coded low, and at or above 4 Pa·s it is coded high. The policy uses this band to update the probability of a low-viscosity lot, changing the expected utility of both recipes. The archived specimen has the same prior in each branch.

| Evidence | Probability of low-viscosity lot | Binder-rich utility | Binder-lean utility |
|---|---:|---:|---:|
| Before assay | 50.0% | 0.6315 | 0.6348 |
| After low read | 99.4% | 0.7945 | 0.6719 |
| After high read | 0.6% | 0.4684 | 0.5976 |

The execution trace records the paid action, credits remaining, actual specimen and preparation receipts, qualification status and posterior before the next choice. A failed make still costs its reserved credits. Re-reading a specimen and making a fresh replicate have distinct likelihoods and identities. These two branches are a constructed demonstration of a decision reversal.

## Reproduce and inspect

```bash
python -m materials_workflow shared-budget --output .local/shared-budget
```

The output includes the public policy specification and separate SQLite sessions for each branch and comparator. An operator can also reserve an action and supply an observed return manually. A fabrication return includes the raw registry submission for the requested recipe and specimen. The session checks its disposition and observed toughness band before updating the policy.

```bash
python -m materials_workflow session init --db .local/operator.sqlite3 --spec .local/shared-budget/policy.json
python -m materials_workflow session next --db .local/operator.sqlite3
```

For the opening assay, save the following example as `receipt.json`.

```json
{"action_id":"assay-stock","measurement":{"name":"stock viscosity","unit":"Pa s","value":1.94}}
```

```bash
python -m materials_workflow session observe --db .local/operator.sqlite3 --step 1 --action assay-stock --outcome low --receipt receipt.json
python -m materials_workflow session next --db .local/operator.sqlite3
```

A value of 5.94 and the `high` band in a separate fresh session exercise the other branch. Reopening a session resumes the outstanding action. Replaying the same receipt leaves the balance and posterior intact.

[Machine-readable summary](summary.json) · [Full decisions and receipts](trace.json) · [Planner](../../src/materials_workflow/shared_budget.py) · [Synthetic case](../../src/materials_workflow/shared_case.py)
