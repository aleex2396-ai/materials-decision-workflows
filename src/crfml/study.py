"""Reproducible paired experiments; public synthetic sources only."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr, ndtri
from .simulator import sample, descriptors
from .models import ExactGP, Ridge, conformal_radius, metrics
from .discovery import discover
from .control import ControlConfig, rollout


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def paired_summary(values, rng, n_boot=2000):
    a = np.asarray(values, float)
    means = rng.choice(a, (n_boot, len(a)), replace=True).mean(1)
    return {'n':len(a), 'mean':float(a.mean()), 'ci95':[float(v) for v in np.quantile(means, [.025,.975])],
            'per_seed':a.tolist()}


def prediction_study(seed, config):
    train = sample(seed, config['train_size'])
    cal = sample(1000+seed, config['calibration_size'])
    x, cx = descriptors(train.x), descriptors(cal.x)
    gp = ExactGP(.025).fit(x, train.quality)
    ridge = Ridge().fit(x, train.quality)
    cm, cs = gp.predict(cx)
    radius = conformal_radius(cal.quality, cm, cs)
    outputs = []
    for shift in (False, True):
        test = sample(2000+seed+int(shift)*1000, config['test_size'], shift=shift)
        tx = descriptors(test.x)
        mean, std = gp.predict(tx)
        outputs.append({'seed':seed, 'distribution':'shift' if shift else 'iid',
                        'gp':metrics(test.quality, mean, std, radius),
                        'ridge':metrics(test.quality, ridge.predict(tx)),
                        'constant':metrics(test.quality, np.full(len(tx), train.quality.mean())),
                        'conformal_radius':radius, 'optimizer':gp.optimizer})
    return outputs


def selection_counterexample(seed=876, repetitions=20000, candidates=50, alpha=.1):
    """Equal zero truths, random prediction errors; optimizer picks highest prediction.

    Fixed-candidate Gaussian coverage is 1-alpha. After argmax, exact coverage
    is Phi(z)^M-Phi(-z)^M. This is a counterexample, not realistic material data.
    """
    rng = np.random.default_rng(seed)
    z = ndtri(1-alpha/2)
    simultaneous = ndtri(1-alpha/(2*candidates))
    errors = rng.normal(size=(repetitions, candidates))
    maximum = errors.max(1)
    return {'repetitions':repetitions, 'candidates':candidates,
            'fixed_candidate_coverage':float(np.mean(abs(errors[:,0]) <= z)),
            'selected_coverage':float(np.mean(abs(maximum) <= z)),
            'selected_exact_coverage':float(ndtr(z)**candidates-ndtr(-z)**candidates),
            'simultaneous_selected_coverage':float(np.mean(abs(maximum) <= simultaneous)),
            'pointwise_radius':float(z), 'bonferroni_radius':float(simultaneous)}


def run_study(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['origin'] != 'public_synthetic_only':
        raise ValueError('Public study accepts only synthetic origin')
    output.mkdir(parents=True, exist_ok=True)
    predictions, discovery, controllers = [], [], []
    cfg = ControlConfig()
    policies = ['nominal_gp','robust_fixed','robust_adaptive','safe_probe','constant_center','oracle']
    for seed in config['seeds']:
        print(f'seed {seed}: prediction, discovery, closed-loop control', flush=True)
        predictions.extend(prediction_study(seed, config))
        for acquisition in ('random', 'greedy', 'constrained_ei'):
            material = discover(seed, acquisition, config['discovery_steps'])
            discovery.append(material)
            rng = np.random.default_rng(30000+seed)
            base_w = rng.uniform(-1, 1, config['control_steps'])
            for scenario in config['scenarios']:
                w = cfg.wbar*base_w
                b = material['true_gain']
                if scenario == 'bounded_shift':
                    w = cfg.wbar*(base_w+1)/2  # mean shifts, support remains valid
                elif scenario == 'outside_envelope':
                    w = .08 + .02*base_w  # deliberately violates assumed bound
                    b *= 1.8
                for policy in policies:
                    result = rollout(b, material['predicted_gain'], w, policy, cfg)
                    result.update(seed=seed, acquisition=acquisition, scenario=scenario,
                                  true_b=b, learned_b=material['predicted_gain'])
                    controllers.append(result)
    rng = np.random.default_rng(451)
    summary = {'origin':'public_synthetic_only','config':config,
               'prediction':{}, 'discovery':{}, 'control':{}, 'paired_primary':{},
               'selection_counterexample':selection_counterexample()}
    for distribution in ('iid', 'shift'):
        rows = [r for r in predictions if r['distribution'] == distribution]
        summary['prediction'][distribution] = {
            model: {metric:paired_summary([r[model][metric] for r in rows], rng)
                    for metric in rows[0][model]}
            for model in ('gp','ridge','constant')}
    for acquisition in ('random','greedy','constrained_ei'):
        rows = [r for r in discovery if r['policy'] == acquisition]
        summary['discovery'][acquisition] = {
            metric:paired_summary([r[metric] for r in rows], rng)
            for metric in ('final_regret','sampled_constraint_violations','initial_constraint_violations',
                           'total_sampled_constraint_violations','recommendation_feasible')}
        gaps = [r['recommended_utility_regret_if_feasible'] for r in rows
                if r['recommended_utility_regret_if_feasible'] is not None]
        summary['discovery'][acquisition]['recommended_utility_regret_if_feasible'] = (
            paired_summary(gaps, rng) if gaps else None)
        for scenario in config['scenarios']:
            key = acquisition+'/'+scenario
            summary['control'][key] = {}
            lookup = {}
            for policy in policies:
                rows = [r for r in controllers if r['acquisition']==acquisition and
                        r['scenario']==scenario and r['policy']==policy]
                lookup[policy] = {r['seed']:r for r in rows}
                summary['control'][key][policy] = {
                    metric:paired_summary([r[metric] for r in rows], rng)
                    for metric in ('total_cost','violations','final_width','assumption_failures','solver_fallbacks')}
            summary['paired_primary'][key] = paired_summary([
                lookup['safe_probe'][s]['total_cost']-lookup['robust_adaptive'][s]['total_cost']
                for s in config['seeds']], rng, config['bootstrap_replicates'])
    write_json(output/'summary.json', summary)
    write_json(output/'predictions.json', predictions)
    write_json(output/'discovery.json', discovery)
    write_json(output/'control.json', controllers)
    provenance = {'origin':'public_synthetic_only', 'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__, 'platform':platform.platform(),
                  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
                  'no_private_data_loaded':True}
    write_json(output/'provenance.json', provenance)
    return summary
