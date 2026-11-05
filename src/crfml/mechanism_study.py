"""Supplementary fixed-trace and radius-selection study; public synthetic only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .decision_learning import (
    RegretGeometry, fit_decision_covariance, fit_likelihood_covariance,
    second_moment, solve_learning_controller,
)
from .decision_study import gaussian_w2_squared, serializable, shifted_moments
from .discovery import discover
from .dr_control import synthesize, worst_case_quadratic
from .simulator import gaussian_trajectories, material_process
from .study import paired_summary, write_json


ESTIMATORS = ('shape_likelihood', 'shape_decision', 'radius_tuned_likelihood')


class RadiusTuningError(RuntimeError):
    def __init__(self, message, partial_fit):
        super().__init__(message)
        self.partial_fit = partial_fit


def tune_likelihood_radius(problem, moment, validation_samples, radius_multipliers):
    """Select a radius by validation regret alone, requiring the complete fixed grid.

    The covariance is already frozen. Zero radius deliberately uses the exact
    nominal controller through synthesize's documented radius-zero routing.
    A failed grid point invalidates selection rather than shrinking the grid.
    """
    validation_moment = second_moment(validation_samples)
    multipliers = np.asarray(radius_multipliers, dtype=float)
    if (multipliers.ndim != 1 or len(multipliers) < 1 or not np.isfinite(multipliers).all()
            or np.any(multipliers < 0) or len(set(multipliers.tolist())) != len(multipliers)):
        raise ValueError('Radius grid must contain distinct finite nonnegative multipliers')
    if validation_moment.shape != (problem.d, problem.d):
        raise ValueError('Validation trajectories do not match the control dimension')
    trace, solutions = [], []
    for multiplier in multipliers:
        radius = float(multiplier*np.sqrt(problem.d))
        entry = {'radius_multiplier': float(multiplier), 'radius': radius, 'status': 'attempted'}
        trace.append(entry)
        try:
            result = synthesize(problem, moment, radius, method='dr_regret')
            loss = float(np.trace(problem.quadratic_matrix(result['K'])@validation_moment))
        except Exception as error:
            entry.update(status='failed', error_type=type(error).__name__, error=str(error))
            partial = {'candidates': trace, 'controller_requests': len(trace),
                       'controller_attempts': len(trace), 'controller_solves': len(trace),
                       'validation_samples': int(len(validation_samples)), 'test_samples_accessed': 0}
            raise RadiusTuningError(str(error), partial) from error
        entry.update(status='evaluated', validation_regret=loss,
                     synthesis_diagnostics=result['diagnostics'])
        solutions.append(result)
    chosen = min(range(len(trace)), key=lambda index: (trace[index]['validation_regret'], index))
    return {'selected_index': chosen, 'selected_radius': trace[chosen]['radius'],
            'selected_multiplier': trace[chosen]['radius_multiplier'],
            'validation_regret': trace[chosen]['validation_regret'],
            'selected_controller': solutions[chosen], 'candidates': trace,
            'budgets': {'validation_samples': int(len(validation_samples)),
                        'controller_requests': len(trace), 'controller_attempts': len(trace),
                        'controller_solves': len(trace), 'test_samples_accessed': 0}}


def _source_hashes():
    source = Path(__file__).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob('*.py'))}


def _failure(seed, n_train, stage, error):
    return {'seed': seed, 'n_train': n_train, 'stage': stage,
            'error_type': type(error).__name__, 'error': str(error),
            'partial_fit': serializable(getattr(error, 'partial_fit', None))}


def _check_configs(config, decision):
    if config['origin'] != 'public_synthetic_only' or decision['origin'] != 'public_synthetic_only':
        raise ValueError('Supplementary study only accepts independent public synthetic data')
    for key in ('seeds', 'train_trajectories', 'validation_trajectories', 'outer_maxiter'):
        if config[key] != decision[key]:
            raise ValueError(f'Supplementary and decision configurations disagree on {key}')
    if (len(config['seeds']) != len(set(config['seeds'])) or not config['seeds']
            or not config['train_trajectories'] or min(config['train_trajectories']) < 1):
        raise ValueError('Use unique seeds and nonempty positive training budgets')


def summarize_mechanism(records, fits, failures, config, decision):
    """Paired seed-level analysis, preserving missing pairs and requested counts."""
    rng = np.random.default_rng(823681)
    summary = {'origin': 'public_synthetic_only', 'config': config,
               'primary_decision_config': decision, 'failures': failures,
               'paired_primary': {}, 'paired_secondary': {}, 'groups': {},
               'fit_counts': {}, 'record_count': len(records)}
    numerical_logs = []
    for fit in fits:
        if fit.get('shape_decision') is not None:
            numerical_logs.extend(fit['shape_decision'].get('controller_attempt_logs', []))
    for failure in failures:
        if failure['stage'] == 'shape_decision' and failure.get('partial_fit'):
            numerical_logs.extend(failure['partial_fit'].get('controller_attempt_logs', []))
    summary['decision_fit_numerics'] = {
        'controller_requests': len(numerical_logs),
        'conic_attempts': sum(len(log['attempts']) for log in numerical_logs),
        'rejected_conic_attempts': sum(a['status'] == 'rejected' for log in numerical_logs for a in log['attempts']),
        'recovered_controller_requests': sum(log['status'] == 'accepted' and len(log['attempts']) > 1 for log in numerical_logs),
        'failed_controller_requests': sum(log['status'] == 'failed' for log in numerical_logs)}
    for estimator in ESTIMATORS:
        available = [f[estimator] for f in fits if f.get(estimator) is not None]
        summary['fit_counts'][estimator] = {
            'requested': len(config['seeds'])*len(config['train_trajectories']),
            'returned': len(available),
            'optimizer_converged': sum(bool(f.get('optimizer_success')) for f in available)
                if estimator != 'radius_tuned_likelihood' else None,
            'optimizer_nonconverged': sum(not bool(f.get('optimizer_success')) for f in available)
                if estimator != 'radius_tuned_likelihood' else None}
    for n in config['train_trajectories']:
        for shift in decision['shifts']:
            rows = [r for r in records if r['n_train'] == n and r['shift'] == shift]
            indexed = {estimator: {r['seed']: r for r in rows if r['estimator'] == estimator}
                       for estimator in ESTIMATORS}
            for left, right, target in (
                    ('shape_decision', 'shape_likelihood', summary['paired_primary']),
                    ('shape_decision', 'radius_tuned_likelihood', summary['paired_secondary'])):
                matched = sorted(indexed[left].keys() & indexed[right].keys())
                differences = [indexed[left][s]['expected_regret']-indexed[right][s]['expected_regret']
                               for s in matched]
                key = f'n={n}/{shift}'
                target[key] = {'comparison': f'{left} minus {right}',
                               'requested_pairs': len(config['seeds']), 'complete_pairs': len(matched),
                               'included_seeds': matched,
                               'excluded_seeds': [s for s in config['seeds'] if s not in matched],
                               'difference': paired_summary(differences, rng) if matched else None}
            for estimator, rows_by_seed in indexed.items():
                if rows_by_seed:
                    group = list(rows_by_seed.values())
                    summary['groups'][f'n={n}/{shift}/{estimator}'] = {
                        key: paired_summary([r[key] for r in group], rng)
                        for key in ('expected_regret', 'expected_cost', 'covariance_relative_frobenius',
                                    'training_trace_ratio', 'true_trace_ratio', 'inside_estimated_ball',
                                    'radius_multiplier')}
    return summary


def _write_report(output, summary):
    lines = ['# Supplementary mechanism study', '',
             'This supplement investigates covariance scale and radius selection. It does not replace the frozen primary decision-study endpoint.', '',
             'The fixed-trace likelihood and decision fits use identical training and validation draws, parameter bounds, initialization and iteration caps. Their nominal second moments have the same training trace. The radius baseline freezes the unnormalized likelihood fit, then selects a radius using only validation regret over the full configured grid. Known population moments are used for generating synthetic data and subsequent evaluation, never by a fitting objective.', '',
             'Evaluation uses analytic expected cost and regret under the same IID and oracle-oriented shifted moments as the primary study. Shifts remain defined using the fixed primary radius, including when the fitted baseline selects another radius.', '',
             f"Recorded failures: {len(summary['failures'])}. All missing pairs are listed in summary.json; optimizer iteration-limit returns remain reported rather than treated as converged.", '',
             '| Estimator | Requested fits | Returned fits | Converged | Nonconverged |',
             '|---|---:|---:|---:|---:|']
    for estimator, row in summary['fit_counts'].items():
        lines.append(f"| {estimator} | {row['requested']} | {row['returned']} | {row['optimizer_converged']} | {row['optimizer_nonconverged']} |")
    numerical = summary['decision_fit_numerics']
    lines += ['', f"Decision-fit numerical budget: {numerical['controller_requests']} controller requests, {numerical['conic_attempts']} conic attempts, {numerical['rejected_conic_attempts']} rejected attempts and {numerical['recovered_controller_requests']} recovered requests. Recovery is bounded to Clarabel followed by SCS on numerical failure, with identical acceptance thresholds. Rejected attempts remain in the trace and are not accepted as certificates."]
    lines += ['', '## Primary supplementary comparison', '',
              'Difference = fixed-trace decision regret minus fixed-trace likelihood regret. Negative favors decision fitting; positive favors likelihood fitting. Intervals are paired bootstrap 95% intervals over seeds and are descriptive, with no multiple-comparison correction.', '',
              '| Training paths / shift | Complete pairs | Mean difference | 95% interval |',
              '|---|---:|---:|---|']
    for key, row in summary['paired_primary'].items():
        d = row['difference']
        cells = 'unavailable | unavailable' if d is None else f"{d['mean']:.6g} | [{d['ci95'][0]:.6g}, {d['ci95'][1]:.6g}]"
        lines.append(f"| {key} | {row['complete_pairs']}/{row['requested_pairs']} | {cells} |")
    lines += ['', '## Decision shape fitting versus likelihood plus radius selection', '',
              '| Training paths / shift | Complete pairs | Mean difference | 95% interval |',
              '|---|---:|---:|---|']
    for key, row in summary['paired_secondary'].items():
        d = row['difference']
        cells = 'unavailable | unavailable' if d is None else f"{d['mean']:.6g} | [{d['ci95'][0]:.6g}, {d['ci95'][1]:.6g}]"
        lines.append(f"| {key} | {row['complete_pairs']}/{row['requested_pairs']} | {cells} |")
    lines += ['', 'The covariance-scale identity W(K;cM,r)=c W(K;M,r/sqrt(c)) motivates this ablation. Fixing trace removes variation in total nominal disturbance scale across hyperparameters within a fit. Training trace remains estimated, so this is not an oracle-scale experiment. Radius selection receives the same validation samples but additional controller evaluations; actual budgets are retained.', '',
              'No finite-sample coverage claim is made for the Wasserstein radius. These known-dynamics synthetic results do not certify learned dynamics or physical material behavior. Read fits.json for parameter trajectories, radius candidates, numerical diagnostics and convergence flags; failures.json preserves failed partial fits.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run_mechanism_study(config_path, decision_config_path, output):
    """Run the frozen supplementary study without reading primary result artifacts."""
    config_path, decision_config_path, output = map(Path, (config_path, decision_config_path, output))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    decision = json.loads(decision_config_path.read_text(encoding='utf-8'))
    _check_configs(config, decision)
    output.mkdir(parents=True, exist_ok=True)
    source_at_start = _source_hashes()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    decision_config_hash = hashlib.sha256(decision_config_path.read_bytes()).hexdigest()
    records, fits, failures, policies = [], [], [], {}
    def checkpoint():
        for name, value in (('records', records), ('fits', fits), ('failures', failures), ('policies', policies)):
            write_json(output/f'{name}.json', serializable(value))
    for seed in config['seeds']:
        material = discover(seed, 'constrained_ei', steps=12)
        problem, true_moment = material_process(material['composition_process'], decision['horizon'])
        fixed_radius = decision['radius_per_sqrt_dimension']*np.sqrt(problem.d)
        geometry = RegretGeometry(problem.D, problem.Kstar, problem.causal_mask)
        validation = gaussian_trajectories(800000+seed, true_moment, config['validation_trajectories'])
        training_all = gaussian_trajectories(700000+seed, true_moment, max(config['train_trajectories']))
        selected = []
        # Complete every fitted selection for this seed before accessing evaluation moments.
        for n_train in config['train_trajectories']:
            print(f'mechanism study seed={seed}, training trajectories={n_train}', flush=True)
            started = perf_counter()
            train = training_all[:n_train]
            empirical_trace = float(np.trace(second_moment(train)))
            fit_record = {'seed': seed, 'n_train': n_train,
                          'material': material['composition_process'],
                          'training_trace': empirical_trace, 'fixed_radius': float(fixed_radius),
                          'shape_likelihood': None, 'shape_decision': None,
                          'radius_tuned_likelihood': None, 'unnormalized_likelihood': None,
                          'controller_solve_diagnostics': []}
            for name in ('shape_likelihood', 'shape_decision', 'radius_tuned_likelihood'):
                try:
                    if name == 'shape_likelihood':
                        fit = fit_likelihood_covariance(train, validation, trace_normalize=True,
                                                         max_iterations=config['outer_maxiter'])
                        fit_record[name] = fit
                        radius, multiplier = fixed_radius, decision['radius_per_sqrt_dimension']
                        solution = synthesize(problem, fit['covariance'], radius, method='dr_regret')
                    elif name == 'shape_decision':
                        def fresh_solver(M):
                            z, diagnostics = solve_learning_controller(problem, M, fixed_radius)
                            fit_record['controller_solve_diagnostics'].append(diagnostics)
                            return z, diagnostics
                        fit = fit_decision_covariance(train, validation, geometry, fixed_radius,
                                                      fresh_solver, trace_normalize=True,
                                                      max_iterations=config['outer_maxiter'])
                        fit_record[name] = fit
                        radius, multiplier = fixed_radius, decision['radius_per_sqrt_dimension']
                        solution = synthesize(problem, fit['covariance'], radius, method='dr_regret')
                    else:
                        density = fit_likelihood_covariance(train, validation,
                                                            max_iterations=config['outer_maxiter'])
                        fit_record['unnormalized_likelihood'] = density
                        tuning = tune_likelihood_radius(problem, density['covariance'], validation,
                                                        config['radius_multipliers'])
                        fit = {'covariance': density['covariance'], 'theta': density['theta'],
                               'radius_tuning': tuning,
                               'density_optimizer_success': density['optimizer_success'],
                               'budgets': {'training_samples': len(train),
                                           'validation_samples': len(validation),
                                           'controller_requests': tuning['budgets']['controller_requests'],
                                           'controller_attempts': tuning['budgets']['controller_attempts'],
                                           'controller_solves': tuning['budgets']['controller_solves'],
                                           'test_samples_accessed': 0}}
                        radius, multiplier = tuning['selected_radius'], tuning['selected_multiplier']
                        solution = tuning['selected_controller']
                    fit_record[name] = fit
                    policy_id = f'{seed}/{n_train}/{name}'
                    policies[policy_id] = {'synthesis': solution, 'second_moment': fit['covariance'],
                                           'radius': float(radius), 'radius_multiplier': float(multiplier)}
                    selected.append({'seed': seed, 'n_train': n_train, 'estimator': name,
                                     'fit': fit, 'solution': solution, 'radius': float(radius),
                                     'radius_multiplier': float(multiplier), 'policy_id': policy_id,
                                     'training_trace': empirical_trace})
                except Exception as error:
                    failure = _failure(seed, n_train, name, error)
                    failures.append(failure)
                    print(f"  FAILED {name}: {type(error).__name__}: {str(error)[:160]}", flush=True)
            fit_record['fit_seconds'] = perf_counter()-started
            fits.append(serializable(fit_record))
            checkpoint()
        # Known true moment is supplied only to the reference/shift construction and evaluation.
        try:
            oracle = synthesize(problem, true_moment, fixed_radius, method='dr_regret')
            moments = shifted_moments(problem, true_moment, fixed_radius, oracle['K'])
        except Exception as error:
            failures.append(_failure(seed, None, 'evaluation_reference', error))
            checkpoint()
            continue
        for item in selected:
            K, M = item['solution']['K'], item['fit']['covariance']
            regret = problem.quadratic_matrix(K)
            cost = problem.quadratic_matrix(K, loss='cost')
            robust = worst_case_quadratic(regret, true_moment, fixed_radius)
            theta = item['fit']['theta']
            for shift in decision['shifts']:
                Mtest = moments[shift]
                distance = float(np.sqrt(gaussian_w2_squared(M, Mtest)))
                records.append({'seed': seed, 'n_train': item['n_train'], 'estimator': item['estimator'],
                                'method': 'dr_regret', 'shift': shift,
                                'expected_regret': float(np.trace(regret@Mtest)),
                                'expected_cost': float(np.trace(cost@Mtest)),
                                'true_center_worst_regret_at_primary_radius': robust['value'],
                                'covariance_relative_frobenius': float(np.linalg.norm(M-true_moment, 'fro')/np.linalg.norm(true_moment, 'fro')),
                                'training_trace_ratio': float(np.trace(M)/item['training_trace']),
                                'true_trace_ratio': float(np.trace(M)/np.trace(true_moment)),
                                'alpha': float(theta[0]), 'beta': float(np.exp(theta[1])),
                                'radius': item['radius'], 'radius_multiplier': item['radius_multiplier'],
                                'estimated_center_to_test_w2': distance,
                                'inside_estimated_ball': bool(distance <= item['radius']+1e-8),
                                'oracle_center_to_test_w2': float(np.sqrt(gaussian_w2_squared(true_moment, Mtest))),
                                'policy_id': item['policy_id']})
        checkpoint()
    summary = summarize_mechanism(records, fits, failures, config, decision)
    write_json(output/'summary.json', serializable(summary))
    _write_report(output, summary)
    sources_finished = _source_hashes()
    artifact_names = ('summary.json', 'records.json', 'fits.json', 'failures.json', 'policies.json', 'REPORT.md')
    provenance = {'origin': 'public_synthetic_only', 'no_private_data_loaded': True,
                  'selection_test_samples_accessed': 0, 'evaluation': 'analytic population moments',
                  'config_sha256': config_hash, 'decision_config_sha256': decision_config_hash,
                  'source_start_sha256': source_at_start,
                  'source_sha256': sources_finished, 'sources_unchanged_during_run': source_at_start == sources_finished,
                  'artifact_sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in artifact_names},
                  'artifact_hash_excludes': ['provenance.json (self-reference)']}
    write_json(output/'provenance.json', provenance)
    return summary
