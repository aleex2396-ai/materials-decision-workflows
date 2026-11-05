"""Selection isolation, matched-trace execution and artifact integrity checks."""
import hashlib
import json

import numpy as np
import pytest

import crfml.mechanism_study as study
from crfml.decision_learning import second_moment
from crfml.simulator import gaussian_trajectories, material_process


def test_radius_selection_matches_validation_argmin_and_zero_routes_nominal():
    problem, population = material_process([.3, .45, .25, .5], horizon=2)
    validation = gaussian_trajectories(417, population, 27)
    moment = np.eye(problem.d)
    fitted = study.tune_likelihood_radius(problem, moment, validation, [0., .1, .25])
    candidates = fitted['candidates']
    assert candidates[0]['synthesis_diagnostics']['zero_radius_uses_nominal']
    assert fitted['selected_index'] == int(np.argmin([r['validation_regret'] for r in candidates]))
    K = fitted['selected_controller']['K']
    independent_loss = float(np.mean(np.einsum('bi,ij,bj->b', validation,
                                               problem.quadratic_matrix(K), validation)))
    assert abs(independent_loss-fitted['validation_regret']) < 1e-12
    assert fitted['budgets'] == {'validation_samples': 27, 'controller_solves': 3,
                                 'controller_requests': 3, 'controller_attempts': 3,
                                 'test_samples_accessed': 0}


def test_failed_radius_candidate_invalidates_selection_and_retains_trace(monkeypatch):
    problem, population = material_process([.3, .45, .25, .5], horizon=2)
    validation = gaussian_trajectories(418, population, 12)
    original, calls = study.synthesize, 0
    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError('Injected candidate synthesis failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(study, 'synthesize', failing)
    with pytest.raises(study.RadiusTuningError) as error:
        study.tune_likelihood_radius(problem, np.eye(problem.d), validation, [0., .1, .25])
    partial = error.value.partial_fit
    assert partial['controller_solves'] == calls == 2
    assert partial['candidates'][0]['status'] == 'evaluated'
    assert partial['candidates'][1]['status'] == 'failed'
    assert partial['test_samples_accessed'] == 0


def test_paired_summary_preserves_missing_seeds_and_difference_direction():
    records = []
    for seed, estimator, regret in [(0, 'shape_decision', 4.), (0, 'shape_likelihood', 5.),
                                     (1, 'shape_likelihood', 2.)]:
        records.append({'seed': seed, 'n_train': 8, 'shift': 'iid', 'estimator': estimator,
                        'expected_regret': regret, 'expected_cost': regret+2,
                        'covariance_relative_frobenius': .3, 'training_trace_ratio': 1.,
                        'true_trace_ratio': .9, 'inside_estimated_ball': False, 'radius_multiplier': .25})
    config = {'seeds': [0, 1], 'train_trajectories': [8]}
    summary = study.summarize_mechanism(records, [], [], config, {'shifts': ['iid']})
    primary = summary['paired_primary']['n=8/iid']
    assert primary['complete_pairs'] == 1
    assert primary['requested_pairs'] == 2
    assert primary['excluded_seeds'] == [1]
    assert primary['difference']['mean'] == -1.
    secondary = summary['paired_secondary']['n=8/iid']
    assert secondary['complete_pairs'] == 0
    assert secondary['difference'] is None


def test_small_study_preserves_trace_budgets_and_hashes_complete_artifacts(tmp_path, monkeypatch):
    # An independent small fixture exercises the full runner without consuming
    # the registered production seeds or reading primary outcome artifacts.
    seed, composition = 731, [.3, .45, .25, .5]
    monkeypatch.setattr(study, 'discover', lambda *args, **kwargs: {'composition_process': composition})
    config = {'origin': 'public_synthetic_only', 'seeds': [seed], 'train_trajectories': [8],
              'validation_trajectories': 16, 'outer_maxiter': 2,
              'radius_multipliers': [0., .05, .1, .25, .5]}
    decision = dict(config, horizon=2, radius_per_sqrt_dimension=.25,
                    shifts=['iid', 'oracle_top_0.5r', 'oracle_bottom_0.5r', 'oracle_top_1.5r'])
    config_path, decision_path = tmp_path/'mechanism.json', tmp_path/'decision.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    decision_path.write_text(json.dumps(decision), encoding='utf-8')
    output = tmp_path/'output'
    summary = study.run_mechanism_study(config_path, decision_path, output)
    assert summary['failures'] == []
    assert summary['record_count'] == 12
    assert all(row['complete_pairs'] == 1 for row in summary['paired_primary'].values())
    fits = json.loads((output/'fits.json').read_text())
    _, population = material_process(composition, horizon=2)
    training = gaussian_trajectories(700000+seed, population, 8)
    trace = np.trace(second_moment(training))
    for estimator in ('shape_likelihood', 'shape_decision'):
        fitted = fits[0][estimator]
        assert fitted['trace_normalize']
        assert abs(np.trace(fitted['covariance'])-trace) < 1e-12
        assert fitted['budgets']['training_samples'] == 8
        assert fitted['budgets']['validation_samples'] == 16
        assert fitted['budgets']['test_samples_accessed'] == 0
    tuning = fits[0]['radius_tuned_likelihood']['radius_tuning']
    assert len(tuning['candidates']) == 5
    assert tuning['budgets']['controller_solves'] == 5
    provenance = json.loads((output/'provenance.json').read_text())
    assert provenance['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert provenance['decision_config_sha256'] == hashlib.sha256(decision_path.read_bytes()).hexdigest()
    assert provenance['sources_unchanged_during_run']
    assert set(provenance['artifact_sha256']) == {'summary.json', 'records.json', 'fits.json',
                                                  'failures.json', 'policies.json', 'REPORT.md'}
    for name, expected in provenance['artifact_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == expected


def test_mismatched_validation_budget_is_rejected():
    base = {'origin': 'public_synthetic_only', 'seeds': [1], 'train_trajectories': [8],
            'validation_trajectories': 64, 'outer_maxiter': 12}
    with pytest.raises(ValueError, match='validation_trajectories'):
        study._check_configs(base, dict(base, validation_trajectories=65))
