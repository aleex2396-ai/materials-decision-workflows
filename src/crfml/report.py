"""Standalone figures and an evidence report from the saved synthetic study."""
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_report(output):
    output = Path(output)
    s = json.loads((output/'summary.json').read_text())
    controls = json.loads((output/'control.json').read_text())
    plt.rcParams.update({'figure.dpi':130, 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    keys = ['nominal_gp','robust_fixed','robust_adaptive','safe_probe','constant_center','oracle']
    labels = ['Nominal GP','Fixed robust','Adaptive','Safe probe','Fixed center','Oracle']
    for ax, scenario in zip(axes, s['config']['scenarios']):
        subset = s['control']['constrained_ei/'+scenario]
        mean = [subset[k]['total_cost']['mean'] for k in keys]
        lo = [mean[i]-subset[k]['total_cost']['ci95'][0] for i,k in enumerate(keys)]
        hi = [subset[k]['total_cost']['ci95'][1]-mean[i] for i,k in enumerate(keys)]
        ax.bar(np.arange(len(keys)), mean, color=['#b9c3ce','#667c9c','#468997','#d69b35','#9895b2','#4b9a6a'])
        ax.errorbar(np.arange(len(keys)), mean, yerr=[lo,hi], fmt='none', ecolor='#28333d', capsize=2)
        ax.set_xticks(np.arange(len(keys)), labels, rotation=45, ha='right')
        ax.set_title(scenario.replace('_',' '))
        ax.set_ylabel('36-step cost (lower is better)')
    fig.suptitle('Synthetic control benchmark: paired seeds, bootstrap 95% intervals')
    fig.savefig(output/'control_cost.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True, constrained_layout=True)
    for policy in ['nominal_gp','robust_fixed','robust_adaptive','safe_probe']:
        row = next(r for r in controls if r['seed']==s['config']['seeds'][0] and
                   r['acquisition']=='constrained_ei' and r['scenario']=='bounded_shift' and r['policy']==policy)
        trajectory = row['trajectory']
        axes[0].plot([r['next_x'] for r in trajectory], label=policy)
        axes[1].plot([r['gain_width_after'] for r in trajectory], label=policy)
    axes[0].axhline(1, color='red', ls='--', label='state limit')
    axes[0].axhline(.88, color='gray', ls=':', label='reference')
    axes[0].set_ylabel('Normalised process state')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel('Admissible gain width')
    axes[1].set_xlabel('Closed-loop step')
    fig.suptitle('Illustrative first seed, bounded disturbance shift; not a laboratory trajectory')
    fig.savefig(output/'trajectory.png')
    plt.close(fig)
    lines = ['# Reproduced synthetic study', '',
             'All results below use invented public simulation only. Confidence intervals resample independent paired seeds, not time steps. Small seed counts and multiple comparisons limit inference.', '',
             '## Prediction and calibration', '', '| Distribution | Model | RMSE | Coverage |', '|---|---|---:|---:|']
    for dist, models in s['prediction'].items():
        for model, m in models.items():
            coverage = f"{m['coverage']['mean']:.3f}" if 'coverage' in m else 'n/a'
            lines.append(f"| {dist} | {model} | {m['rmse']['mean']:.4f} | {coverage} |")
    lines += ['', '## Discovery', '', '| Policy | Best queried feasible quality regret | Post-initial violations (12 queries) | Total violations (22 observations) | Recommendation feasible fraction | Utility regret among feasible recommendations |', '|---|---:|---:|---:|---:|---:|']
    for policy, m in s['discovery'].items():
        gap = m['recommended_utility_regret_if_feasible']
        gap_text = f"{gap['mean']:.4f} (n={gap['n']})" if gap else 'no feasible recommendations'
        lines.append(f"| {policy} | {m['final_regret']['mean']:.4f} | {m['sampled_constraint_violations']['mean']:.2f} | {m['total_sampled_constraint_violations']['mean']:.2f} | {m['recommendation_feasible']['mean']:.3f} | {gap_text} |")
    lines += ['', 'Feasibility-weighted expected improvement is a heuristic. Its sampled violations are reported, not hidden behind a safety claim.', '',
              '## Control on the material chosen by constrained EI', '', '| Scenario | Policy | Cost | Violations/trajectory | Final gain width |', '|---|---|---:|---:|---:|']
    for scenario in s['config']['scenarios']:
        for policy, m in s['control']['constrained_ei/'+scenario].items():
            lines.append(f"| {scenario} | {policy} | {m['total_cost']['mean']:.3f} | {m['violations']['mean']:.2f} | {m['final_width']['mean']:.4f} |")
    lines += ['', '## Predeclared primary comparison', '',
              'Safe probe minus ordinary adaptive robust control: a negative cost difference favours probing. Report all scenarios and acquisition policies; no best-seed selection.', '',
              '| Acquisition/scenario | Paired mean difference | Bootstrap 95% interval |', '|---|---:|---|']
    for k, m in s['paired_primary'].items():
        lines.append(f"| {k} | {m['mean']:.4f} | [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] |")
    c = s['selection_counterexample']
    lines += ['', '## Selection bias counterexample', '',
              f"Fixed-candidate coverage: {c['fixed_candidate_coverage']:.3f}; selected-candidate coverage: {c['selected_coverage']:.3f}; exact selected coverage: {c['selected_exact_coverage']:.3f}; Bonferroni selected coverage: {c['simultaneous_selected_coverage']:.3f}.", '',
              'The deliberately constructed Gaussian counterexample disproves a general implication from marginal coverage to post-selection coverage. It is not a claim about empirical lab residuals.', '',
              '## Limits', '',
              'Only a scalar fully observed linear plant is certified, with known decay and fixed gain. The robust guarantee is conditional on the physical gain/disturbance bounds. Out-of-envelope stress intentionally breaks those assumptions. Neither Gaussian calibration nor zero observed violations proves real-world safety. No private held-out values are used.', '',
              '![Cost comparison](control_cost.png)', '', '![Illustrative trajectory](trajectory.png)', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    provenance_path = output/'provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    provenance['artifact_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(output.iterdir()) if p.is_file() and p.name != 'provenance.json'}
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
