# Outer-optimizer stationarity and iteration-budget sensitivity

The registered primary and fixed-trace studies use at most 12 outer L-BFGS-B iterations. Their results characterize that finite-budget procedure. A returned parameter vector, a solver success flag, or a small gradient residual does not establish a global optimum of the outer learning problem.

## Diagnostic definition

For theta=(alpha,log(beta)), with bounds [0,1] and [log(0.001),0], respectively, define the unit-step projected-gradient residual

\[
 \rho(\theta)=\|\theta-\Pi_{\mathrm{bounds}}(\theta-\nabla L(\theta))\|_\infty.
\]

For likelihood fitting, L is validation Gaussian negative log likelihood; for decision fitting, L is validation empirical regret. The gradient is taken from the last saved trajectory evaluation whose parameter vector exactly equals the returned theta. All 144 primary and fixed-trace fits have such an evaluation; no interpolation, additional optimizer steps or evaluation outcomes are used for this calculation.

Projection accounts for active parameter bounds. The reported magnitude uses the implemented alpha/log(beta) coordinates and a unit step, so it is not invariant to parameter or objective rescaling. In particular, its magnitude should not be read as an objective-independent comparison between likelihood and regret fitting. A zero residual is a first-order box-constrained stationarity condition; a small residual neither guarantees a global optimum nor bounds test regret.

## Existing 12-iteration results

| Study / objective | Fits | Median rho | 90th percentile | Maximum | rho < 1e-5 | Optimizer success |
|---|---:|---:|---:|---:|---:|---:|
| Primary likelihood | 36 | 2.90593e-5 | 0.0274283 | 0.0768161 | 16/36 | 9/36 |
| Primary decision | 36 | 0.00210622 | 0.0833300 | 0.245918 | 12/36 | 10/36 |
| Fixed-trace likelihood | 36 | 0.000208771 | 0.0380771 | 0.148314 | 13/36 | 8/36 |
| Fixed-trace decision | 36 | 0.00534387 | 0.0676030 | 0.145695 | 8/36 | 7/36 |

The counts below rho=1e-6 are 13, 11, 12 and 8, respectively. Counts below rho=1e-4 are 20, 14, 16 and 11. The fixed-trace decision fit at seed 9 with 8 training trajectories reports success on relative objective reduction despite rho=9.13166e-5; the other success-flagged fits are below 1e-5. Stopping flags and projected stationarity are therefore reported separately. Fits that reach the registered iteration cap remain in the original results without replacement.

## Secondary sensitivity design, declared before its outcomes

Declared at **2026-11-05 12:47:01 UTC**, after review of the saved primary and supplementary stationarity diagnostics and before running the following six sensitivity fits:

- Use seed 0 and all three original training budgets: 8, 28 and 112 trajectories. Seed 0 is selected as a focused diagnostic after the stationarity review; this is not a blind preregistered replication.
- Refit both original, unnormalized likelihood and decision covariance estimators from the same original initialization alpha=0.25, beta=0.05, using the same training draws, 64 validation draws, known dynamics, parameter bounds, positive radius and numerical acceptance gates.
- Change only the outer iteration cap from 12 to 60. Use the existing deterministic maximum-two-attempt numerical solver policy. Record every attempt and any failure; do not retune the radius, initialization or model after outcomes.
- Complete all six fits before evaluating IID and oracle-top-1.5r population regret. Report returned theta, the method-specific validation objective, projected-gradient residual, stopping flags, outer iterations, controller requests and conic attempts.
- Preserve the registered source, configuration and results files unchanged. The six sensitivity-fit logs remain under `.local/`; the public tables below are a secondary diagnostic and do not replace any primary endpoint.

The design tests local sensitivity to a larger compute budget for one selected seed. It cannot establish convergence across all seeds or global optimality, and it is not an additional method-selection step.

## Secondary sensitivity outcomes

All six fits completed without numerical failures. Each stopped on the projected-gradient convergence criterion, after 6–17 iterations; all returned residuals are below 1e-7. None required a second conic solver attempt. These are numerical first-order stopping results, not global-optimality guarantees.

The validation losses in the following table have different meanings and scales: NLL is Gaussian negative log likelihood without its constant, while decision loss is empirical validation regret. Theta is shown in the implemented (alpha,log(beta)) coordinates.

| Training paths | Objective | Returned theta | Validation loss | rho | Iterations | Success | Fit controller requests / conic attempts |
|---|---|---|---:|---:|---:|---|---:|
| 8 | NLL | (0.571601444, -0.948826725) | 7.274897064 | 1.45636e-8 | 10 | true | 0 / 0 |
| 8 | Decision | (0.987377869, 0) | 15.139624780 | 3.76144e-10 | 6 | true | 19 / 19 |
| 28 | NLL | (0.076336649, -1.346420185) | 6.129322137 | 6.39838e-9 | 16 | true | 0 / 0 |
| 28 | Decision | (0.810350464, 0) | 15.062874594 | 9.28472e-9 | 16 | true | 24 / 24 |
| 112 | NLL | (0.038798522, -2.807898366) | 5.074436838 | 5.83568e-8 | 17 | true | 0 / 0 |
| 112 | Decision | (0.453893059, 0) | 14.965708526 | 9.64220e-8 | 13 | true | 21 / 21 |

Each fit uses its stated training budget and 64 validation trajectories, with zero evaluation trajectories consumed during fitting. Objective evaluation counts are 12, 13, 18, 18, 18 and 15 in table order. The controller-request budget includes five initial gradient-verification requests and one final controller request. In addition to those fitting budgets, evaluation uses one reference-controller solve and one final controller solve for each of the six fitted covariances.

The rows below compare the unchanged registered 12-iteration outputs with the secondary 60-iteration-cap outputs on the same population moments. Costs here are **regret against the noncausal controller**, not total control cost.

| Training paths | Objective | IID regret, cap 12 | IID regret, cap 60 | Top-1.5r regret, cap 12 | Top-1.5r regret, cap 60 |
|---|---|---:|---:|---:|---:|
| 8 | NLL | 15.455778 | 15.455778 | 36.274230 | 36.274230 |
| 8 | Decision | 15.820218 | 15.820218 | 39.154713 | 39.154713 |
| 28 | NLL | 15.517022 | 15.509260 | 36.003073 | 36.007844 |
| 28 | Decision | 15.499338 | 15.541301 | 38.210971 | 38.350194 |
| 112 | NLL | 15.353097 | 15.350853 | 35.257896 | 35.262096 |
| 112 | Decision | 15.306569 | 15.306617 | 37.345014 | 37.345213 |

For 28 training trajectories, the IID ranking changes: the 12-iteration decision fit has lower regret than NLL, whereas the more nearly stationary decision fit has higher regret. Its validation regret improves from 15.064065277 to 15.062874594 while its IID population regret increases. Better validation optimization therefore does not imply better transfer, and the original results should be described as a finite-budget comparison. The top-1.5r shift still favors NLL at all three budgets in this seed. The 8-trajectory fits were already converged, and the 112-trajectory decision policy changes very little.

This single-seed diagnostic does not replace the all-seed estimates or determine their rankings after more optimization. It establishes that the finite iteration cap can matter and reports the observed direction without selecting a replacement endpoint. Complete six-fit logs, failures and evaluation diagnostics are retained locally under `.local/optimizer-sensitivity/`; registered artifacts remain unchanged.

## Reproduce the projected-gradient diagnostics

Run from the repository root with the existing project dependencies, for example by saving a code block as `optimizer_diagnostics.py` and running `python optimizer_diagnostics.py`. All paths below are repository-relative. This reads saved public-synthetic fit logs and prints every seed/budget residual as well as the aggregate distribution. It requires no new fitting or data draws.

```python
import json
from pathlib import Path
import numpy as np

def projected(fit, gradient_key):
    theta = np.asarray(fit['theta'])
    matches = [row for row in fit['trajectory']
               if np.array_equal(theta, np.asarray(row['theta']))
               and gradient_key in row]
    if not matches:
        raise ValueError('No gradient at the exact returned theta')
    gradient = np.asarray(matches[-1][gradient_key])
    projected_point = np.clip(theta-gradient, [0., np.log(.001)], [1., 0.])
    return float(np.max(np.abs(theta-projected_point)))

for study, fields in (
    ('decision', ('likelihood_fit', 'decision_fit')),
    ('mechanism', ('shape_likelihood', 'shape_decision')),
):
    rows = json.loads(Path(f'results/{study}/fits.json').read_text(encoding='utf-8'))
    for field in fields:
        key = 'hypergradient' if 'decision' in field else 'gradient'
        values = []
        for row in rows:
            fit = row[field]
            rho = projected(fit, key)
            values.append(rho)
            print(study, field, row['seed'], row['n_train'],
                  rho, fit['optimizer_success'])
        a = np.asarray(values)
        print('median/p90/max:', np.median(a), np.quantile(a, .9), a.max())
        print('below 1e-6/1e-5/1e-4:', [(a < t).sum() for t in (1e-6, 1e-5, 1e-4)])
```

## Reproduce the declared six-fit sensitivity

The following continues from the imports and `projected` function above. It reproduces the fixed diagnostic; it does not write to the registered results directories. All six fitting calls finish before population evaluation.

```python
from crfml.simulator import material_process, gaussian_trajectories
from crfml.decision_learning import (
    RegretGeometry, fit_likelihood_covariance,
    fit_decision_covariance, solve_learning_controller,
)
from crfml.dr_control import synthesize
from crfml.decision_study import shifted_moments

config = json.loads(Path('configs/decision_study.json').read_text(encoding='utf-8'))
saved = json.loads(Path('results/decision/fits.json').read_text(encoding='utf-8'))
material = next(row['material'] for row in saved if row['seed'] == 0)
problem, truth = material_process(material, config['horizon'])
geometry = RegretGeometry(problem.D, problem.Kstar, problem.causal_mask)
radius = config['radius_per_sqrt_dimension'] * np.sqrt(problem.d)
train = gaussian_trajectories(700000, truth, max(config['train_trajectories']))
validation = gaussian_trajectories(800000, truth, config['validation_trajectories'])
fits = []
for n in config['train_trajectories']:
    nll = fit_likelihood_covariance(train[:n], validation, max_iterations=60)
    dfl = fit_decision_covariance(
        train[:n], validation, geometry, radius,
        lambda M: solve_learning_controller(problem, M, radius),
        max_iterations=60,
    )
    fits.extend([(n, 'NLL', nll), (n, 'Decision', dfl)])

oracle = synthesize(problem, truth, radius, method='dr_regret')
moments = shifted_moments(problem, truth, radius, oracle['K'])
for n, method, fit in fits:
    result = synthesize(problem, fit['covariance'], radius, method='dr_regret')
    C = problem.quadratic_matrix(result['K'])
    key = 'gradient' if method == 'NLL' else 'hypergradient'
    loss = fit['validation_nll'] if method == 'NLL' else fit['validation_regret']
    print(n, method, fit['theta'], loss, projected(fit, key),
          fit['optimizer_success'], fit['optimizer_iterations'], fit['budgets'],
          np.trace(C @ moments['iid']), np.trace(C @ moments['oracle_top_1.5r']))
```
