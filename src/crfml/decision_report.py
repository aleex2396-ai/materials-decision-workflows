"""Research figures and reports generated solely from saved public experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finalize_artifacts(output):
    output = Path(output)
    p = output/'provenance.json'
    provenance = json.loads(p.read_text(encoding='utf-8'))
    import scipy,cvxpy,clarabel,scs,platform
    provenance['runtime'] = {'python':platform.python_version(),'numpy':np.__version__,
                             'scipy':scipy.__version__,'cvxpy':cvxpy.__version__,
                             'clarabel':clarabel.__version__,'scs':scs.__version__}
    provenance['artifact_sha256'] = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted(output.iterdir()) if f.is_file() and f.name != p.name}
    p.write_text(json.dumps(provenance,indent=2),encoding='utf-8')


def make_decision_report(output):
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    rows = json.loads((output/'records.json').read_text())
    fits = json.loads((output/'fits.json').read_text())
    policies = json.loads((output/'policies.json').read_text())
    config = summary['config']
    n_values = config['train_trajectories']
    plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,shift in zip(axes.flat,config['shifts']):
        entries = [summary['paired_primary'][f'n={n}/{shift}'] for n in n_values]
        good = [(i,e) for i,e in enumerate(entries) if e]
        x = [i for i,e in good]
        means = [e['mean'] for i,e in good]
        errors = [[e['mean']-e['ci95'][0] for i,e in good],[e['ci95'][1]-e['mean'] for i,e in good]]
        ax.errorbar(x,means,yerr=errors,fmt='o-',capsize=4,color='#267f96')
        ax.axhline(0,color='#697783',ls='--',lw=1)
        ax.set_xticks(range(len(n_values)),[f'{n} + 64' for n in n_values])
        ax.set_xlabel('Estimation + validation trajectories')
        ax.set_ylabel('Decision minus likelihood regret')
        ax.set_title(shift.replace('_',' '))
    fig.suptitle('Decision training: IID gain can reverse under directional shift\nPaired seed means and 95% bootstrap intervals; negative favours decision training')
    fig.savefig(output/'primary_comparison.png')
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,5),constrained_layout=True)
    oracle = {(r['seed'],r['n_train']):r['true_center_worst_regret'] for r in rows
              if r['shift']=='iid' and r['method']=='dr_regret' and r['estimator']=='oracle'}
    for estimator,color in [('empirical_ridge','#aeb3ba'),('diagonal','#b28d41'),
                             ('likelihood_shrinkage','#425e96'),('decision_shrinkage','#209b8c')]:
        selected = [r for r in rows if r['shift']=='iid' and r['method']=='dr_regret' and r['estimator']==estimator]
        ax.scatter([r['covariance_relative_frobenius'] for r in selected],
                   [r['true_center_worst_regret']-oracle[(r['seed'],r['n_train'])] for r in selected],
                   label=estimator,color=color,s=24,alpha=.65)
    ax.set_xlabel('Relative covariance Frobenius error')
    ax.set_ylabel('Excess worst-case regret at the true centre')
    ax.set_title('Prediction error and decision error are distinct endpoints')
    ax.legend(fontsize=8)
    fig.savefig(output/'covariance_tradeoff.png')
    plt.close(fig)
    completed = [f for f in fits if f['decision_fit'] is not None]
    converged = sum(f['decision_fit']['optimizer_success'] for f in completed)
    verify_error = max((max(f['decision_fit']['gradient_verification']['absolute_error']) for f in completed),default=None)
    diagnostics = [p['synthesis']['diagnostics'] for p in policies.values()]
    max_lmi = max(d['maximum_relative_lmi_violation'] for d in diagnostics)
    max_gap = max(d['objective_recompute_relative_gap'] for d in diagnostics)
    requests = sum(f['decision_fit']['budgets']['controller_requests'] for f in completed)
    attempts = sum(f['decision_fit']['budgets']['controller_attempts'] for f in completed)
    lines = ['# Decision-focused covariance learning for regret control','',
             'This is the v2 main study. It uses known coupled two-state dynamics for the selected synthetic material, a six-step causal controller, and fourteen-dimensional augmented disturbance trajectories. Each trajectory includes a nondegenerate random initial state. All public observations are independently invented simulation.', '',
             f"Fixed design: {len(config['seeds'])} independent estimation replicates; estimation sample sizes {n_values}; **64 additional validation trajectories for every estimator**; {config['test_trajectories']} independent evaluation paths per common shifted law. Estimation budgets are not total-data budgets.", '',
             f"Completed decision fits: {len(completed)}/{len(fits)}. Raised numerical fitting failures: {len(summary['failures'])}. Outer optimiser convergence flags: {converged}/{len(completed)} within the fixed {config['outer_maxiter']}-iteration budget. Returned fits that exhausted the outer budget are retained and identified in fits.json.", '',
             '## Frozen primary comparison','',
             'Decision-trained minus likelihood-trained expected regret, using the same covariance family, validation data and Wasserstein regret-control objective. Negative favours decision training. Expectations are evaluated analytically under the known benchmark law and checked by causal Monte Carlo rollouts. Complete-case means are conditional on successful numerical fits; denominators and failures are explicit. Intervals resample independent seeds, not timepoints. No multiplicity-adjusted discovery claim is made.', '',
             '| Estimation / shift | Paired seeds | Mean difference | Bootstrap 95% interval |','|---|---:|---:|---|']
    for key,result in summary['paired_primary'].items():
        if result:
            lines.append(f"| {key} | {result['n']}/{len(config['seeds'])} | {result['mean']:.5f} | [{result['ci95'][0]:.5f}, {result['ci95'][1]:.5f}] |")
        else:
            lines.append(f'| {key} | 0 | unavailable | no complete pairs |')
    lines += ['', 'The shift names refer to distance from the true/oracle centre. They do **not** assert membership in every estimated-centre ball. Each record includes its actual Gaussian Wasserstein distance and membership test.', '',
              '## IID estimator and controller comparisons','',
              '| Estimation samples | Estimator | Controller | Expected regret | Expected cost | Covariance error | True IID law inside estimated ball |',
              '|---:|---|---|---:|---:|---:|---:|']
    for n in n_values:
        for estimator in config['estimators']:
            for method in config['controller_methods']:
                key = f'n={n}/iid/{estimator}/{method}'
                if key not in summary['groups']:
                    continue
                group = summary['groups'][key]
                lines.append(f"| {n} | {estimator} | {method} | {group['expected_regret']['mean']:.4f} | {group['expected_cost']['mean']:.4f} | {group['covariance_relative_frobenius']['mean']:.4f} | {group['inside_estimated_ball']['mean']:.3f} |")
    lines += ['', '## Numerical evidence','',
              f"Largest independently re-solved hypergradient absolute error: {verify_error:.3g}. Largest relative primal LMI violation: {max_lmi:.3g}. Largest fixed-policy objective recomputation gap: {max_gap:.3g}.", '',
              f"Completed decision fits used {requests} controller requests and {attempts} conic solver attempts, including independent gradient checks. A rejected Clarabel attempt permits one SCS attempt with unchanged acceptance thresholds; both attempts remain in the logs. Outer iteration limits are not convergence certificates.", '',
              'These are numerical checks. The fixed-policy transport primal/dual equality does not itself certify the controller minimisation, and stationarity is not a rigorous global suboptimality bound. Synthesis separately records/gates conic primal feasibility, dual feasibility, stationarity and complementarity. The transfer theorem states its global objective-gap premise explicitly.', '',
              '## Analytic policy-ordering reversal','',
              'For the one-step system with A=B=terminal Q=R=1, true second moment I and radius 0.3, the larger diagonal estimation error preserves the true optimal controller. A smaller correlation error shifts the controller and creates strictly positive transferred regret. The rank-one closed form and independent scalar minimiser agree with SDP synthesis.', '',
              '| Estimator error direction | Frobenius error | True robust regret excess | Scalar/SDP action difference |','|---|---:|---:|---:|']
    for name,r in summary['policy_reversal']['rows'].items():
        lines.append(f"| {name} | {r['covariance_frobenius_error']:.5f} | {r['excess_robust_regret']:.7f} | {r['scalar_sdp_action_difference']:.3g} |")
    lines += ['', 'This is a proved counterexample for the stated reference problem, not a universal ranking of estimators.', '',
              '## Scale versus correlation: supplementary analysis','',
              'Moment scale and ambiguity radius interact: W(K;cM,r)=c W(K;M,r/sqrt(c)). The primary two-parameter estimator changes both scale and shape. Improved ordinary validation regret could therefore reflect reduced conservatism rather than better correlation estimation. Fitted alpha, beta, trace and covariance errors are retained. The separately declared fixed-trace/radius-tuning study addresses this mechanism; it does not replace the primary endpoint.', '',
              '| Estimation samples | Fit objective | Mean alpha | Mean beta | Mean fitted / training trace |',
              '|---:|---|---:|---:|---:|']
    for n in n_values:
        for field,label in [('likelihood_fit','Likelihood'),('decision_fit','Decision')]:
            selected = [f[field] for f in fits if f['n_train']==n and f[field] is not None]
            if selected:
                alpha = np.mean([f['theta'][0] for f in selected])
                beta = np.mean([np.exp(f['theta'][1]) for f in selected])
                lines.append(f'| {n} | {label} | {alpha:.4f} | {beta:.4f} | {1+beta:.4f} |')
    lines += ['',
              '## Same-moment negative control','',
              'Gaussian and Student nominal laws with the same second moment yield the same exact quadratic robust objective and controller. Their sample loss tails can differ. This verifies why density-model complexity alone cannot improve this particular objective.', '']
    if 'same_moment_negative_control' in summary:
        control = summary['same_moment_negative_control']
        lines += [f"Exact adversarial expected regret for both laws: {control['exact_adversarial_regret_both_laws']:.5f}; exact squared transport cost: {control['exact_transport_cost_both_laws']:.5f}.",'',
                  '| Law | Sample mean adversarial regret | MC standard error | 99th percentile |','|---|---:|---:|---:|']
        for law,r in control['samples'].items():
            lines.append(f"| {law} | {r['mean_adversarial_regret']:.5f} | {r['regret_mean_se']:.5f} | {r['regret_q99']:.5f} |")
    lines += ['', '## Limits','',
              'Known dynamics, perfect state observations, unconstrained signed deviation inputs, Gaussian fitted nominal laws, finite horizon, and a selected low-dimensional covariance family define the study. Radius is a design parameter, not an asserted confidence guarantee. Learned dynamics would produce policy-dependent residuals and require a separate analysis. Wasserstein expected-regret robustness is not a hard state-safety guarantee.', '',
              '![Primary paired comparison](primary_comparison.png)','', '![Covariance versus decision error](covariance_tradeoff.png)','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    finalize_artifacts(output)
