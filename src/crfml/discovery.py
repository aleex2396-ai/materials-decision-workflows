"""Finite-candidate constrained Bayesian optimisation. Feasibility is heuristic."""
import numpy as np
from scipy.special import ndtr
from .simulator import sample, descriptors, pareto_mask
from .models import ExactGP


def expected_improvement(mean, std, incumbent):
    std = np.maximum(std, 1e-12)
    z = (mean-incumbent)/std
    return (mean-incumbent)*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)


def discover(seed, policy='constrained_ei', steps=12, pool_size=160):
    pool = sample(10000+seed, pool_size, noise=False)
    features = descriptors(pool.x)
    rng = np.random.default_rng(20000+seed)
    # Noise generated once per candidate for fair paired policy replay.
    observed_q = pool.quality + rng.normal(0, .025, pool_size)
    observed_d = pool.defect + rng.normal(0, .01, pool_size)
    observed_b = pool.gain + rng.normal(0, .006, pool_size)
    selected = [int(i) for i in rng.choice(pool_size, 10, replace=False)]
    threshold = .22
    trace = []
    for step in range(steps):
        seen = np.array(selected)
        available = np.setdiff1d(np.arange(pool_size), seen)
        qgp = ExactGP(.025).fit(features[seen], observed_q[seen])
        dgp = ExactGP(.01).fit(features[seen], observed_d[seen])
        qm, qs = qgp.predict(features[available], observation=False)
        dm, ds = dgp.predict(features[available])
        feasible_seen = observed_d[seen] <= threshold
        incumbent = np.max(observed_q[seen][feasible_seen]) if feasible_seen.any() else float(observed_q[seen].min())
        if policy == 'random':
            index = int(rng.choice(available))
        elif policy == 'greedy':
            index = int(available[np.argmax(qm)])
        elif policy == 'constrained_ei':
            score = expected_improvement(qm, qs, incumbent)*ndtr((threshold-dm)/ds)
            index = int(available[np.argmax(score)])
        else:
            raise ValueError(f'Unknown discovery policy {policy}')
        selected.append(index)
        feasible = np.array(selected)[pool.defect[selected] <= threshold]
        optimum = np.max(pool.quality[pool.defect <= threshold])
        best = np.max(pool.quality[feasible]) if len(feasible) else 0.
        trace.append({'step':step, 'candidate':index, 'simple_regret':float(optimum-best),
                      'defect_violation':bool(pool.defect[index] > threshold)})
    seen = np.array(selected)
    # Final recommendation uses only noisy observations, never hidden oracle labels.
    feasible = seen[observed_d[seen] <= threshold]
    candidates = feasible if len(feasible) else seen
    objective = observed_q[candidates]-.15*pool.cost[candidates]
    chosen = int(candidates[np.argmax(objective)])
    bgp = ExactGP(.006).fit(features[seen], observed_b[seen])
    predicted_gain = float(bgp.predict(features[[chosen]])[0][0])
    frontier = candidates[pareto_mask(np.column_stack((-observed_q[candidates], pool.cost[candidates])))]
    recommendation_feasible = bool(pool.defect[chosen] <= threshold)
    true_utility = pool.quality-.15*pool.cost
    utility_gap = (float(np.max(true_utility[pool.defect <= threshold])-true_utility[chosen])
                   if recommendation_feasible else None)
    return {'policy':policy, 'seed':seed, 'trace':trace, 'selected':selected,
            'recommendation':chosen, 'composition_process':pool.x[chosen].tolist(),
            'true_gain':float(pool.gain[chosen]), 'predicted_gain':predicted_gain,
            'observed_gain':float(observed_b[chosen]),
            'true_quality':float(pool.quality[chosen]), 'true_defect':float(pool.defect[chosen]),
            'recommendation_feasible':recommendation_feasible,
            'recommended_utility':float(true_utility[chosen]),
            'recommended_utility_regret_if_feasible':utility_gap,
            'pareto_candidates':frontier.tolist(), 'final_regret':trace[-1]['simple_regret'],
            'sampled_constraint_violations':sum(t['defect_violation'] for t in trace),
            'initial_constraint_violations':int(np.sum(pool.defect[selected[:10]] > threshold)),
            'total_sampled_constraint_violations':int(np.sum(pool.defect[selected] > threshold))}
