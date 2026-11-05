"""Known-dynamics, learned-moment control study with matched estimation budgets."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import minimize_scalar
from .dr_control import LiftedLQ, synthesize, worst_case_quadratic, symmetric_root
from .decision_learning import (RegretGeometry, second_moment, solve_learning_controller,
                                fit_decision_covariance, fit_likelihood_covariance)
from .discovery import discover
from .simulator import material_process, gaussian_trajectories, standard_normal_trajectories, same_moment_laws
from .study import paired_summary, write_json


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [serializable(v) for v in value]
    return value


def gaussian_w2_squared(A, B):
    root = symmetric_root(A)
    middle = symmetric_root(root@B@root)
    return float(max(0.,np.trace(A)+np.trace(B)-2*np.trace(middle)))


def recursive_batch(problem, K, disturbances):
    """Execute all paths from measured states and reconstructed past noise."""
    W = np.asarray(disturbances)
    n_paths = len(W)
    state = np.zeros((n_paths,problem.T+1,problem.n))
    inputs = np.zeros((n_paths,problem.T,problem.m))
    observed = np.zeros_like(W)
    state[:,0,:] = W[:,:problem.n]
    observed[:,:problem.n] = state[:,0,:]
    if np.max(np.abs(K[~problem.causal_mask])) > 1e-10:
        raise ValueError('Future-disturbance access detected')
    for t in range(problem.T):
        block = slice(t*problem.m,(t+1)*problem.m)
        inputs[:,t,:] = observed@K[block].T
        predicted = state[:,t,:]@problem.A[t].T + inputs[:,t,:]@problem.B[t].T
        state[:,t+1,:] = predicted+W[:,(t+1)*problem.n:(t+2)*problem.n]
        observed[:,(t+1)*problem.n:(t+2)*problem.n] = state[:,t+1,:]-predicted
    X,U = state.reshape(n_paths,-1),inputs.reshape(n_paths,-1)
    costs = np.einsum('bi,ij,bj->b',X,problem.Q,X)+np.einsum('bi,ij,bj->b',U,problem.R,U)
    oracle = np.einsum('bi,ij,bj->b',W,problem.Cstar,W)
    regrets = costs-oracle
    if regrets.min() < -1e-7:
        raise ArithmeticError('Negative pathwise regret against true noncausal optimum')
    lifted_x = U@problem.F.T+W@problem.G.T
    residual = float(np.max(np.abs(lifted_x-X)))
    if residual > 1e-9:
        raise ArithmeticError('Recursive and lifted dynamics disagree')
    return costs,regrets,residual


def shifted_moments(problem, true_moment, radius, oracle_K):
    values,vectors = np.linalg.eigh(problem.quadratic_matrix(oracle_K))
    moments = {'iid':true_moment}
    for name, column, factor in [('oracle_top_0.5r',-1,.5),('oracle_bottom_0.5r',0,.5),('oracle_top_1.5r',-1,1.5)]:
        v = vectors[:,column]
        amount = factor*radius/np.sqrt(v@true_moment@v)
        transport = np.eye(problem.d)+amount*np.outer(v,v)
        moment = transport@true_moment@transport.T
        measured = gaussian_w2_squared(true_moment,moment)
        if not np.isclose(measured,(factor*radius)**2,rtol=1e-7,atol=1e-8):
            raise ArithmeticError('Declared Gaussian shift distance failed verification')
        moments[name] = moment
    return moments


def policy_reversal(radius=.3, delta=.4, epsilon=.1):
    """Analytic strict covariance-error/decision-regret reversal, checked by SDP."""
    problem = LiftedLQ.time_invariant([[1.]],[[1.]],1)
    true = np.eye(2)
    estimates = {'larger_diagonal_error':np.diag([1.,1.+delta]),
                 'smaller_correlation_error':np.array([[1.,epsilon],[epsilon,1.]])}
    true_best = .5*(1+radius)**2
    rows = {}
    for name,M in estimates.items():
        fitted = synthesize(problem,M,radius)
        z = float(fitted['K'][0,0]+.5)
        rank_one = lambda a: float(2*(np.sqrt(np.array([a,.5])@M@np.array([a,.5]))+radius*np.hypot(a,.5))**2)
        direct = minimize_scalar(rank_one,bounds=(-2,2),method='bounded',options={'xatol':1e-13})
        true_risk = worst_case_quadratic(problem.quadratic_matrix(fitted['K']),true,radius)['value']
        rows[name] = {'covariance_frobenius_error':float(np.linalg.norm(M-true,'fro')),
                      'k':float(fitted['K'][0,0]),'z':z,'true_robust_regret':true_risk,
                      'excess_robust_regret':true_risk-true_best,
                      'analytic_excess':2*(1+radius)**2*z*z,
                      'independent_scalar_z':float(direct.x),
                      'scalar_sdp_action_difference':abs(z-float(direct.x)),
                      'synthesis':fitted['diagnostics']}
    return {'radius':radius,'delta':delta,'epsilon':epsilon,'true_optimal_k':-.5,
            'true_optimal_risk':true_best,'rows':rows,
            'analytic_derivative_at_zero_for_correlation_error':2*(1+radius)*epsilon}


def moment_negative_control(n=65536):
    problem,M = material_process([.3,.4,.3,.6])
    radius = .25*np.sqrt(problem.d)
    solved = synthesize(problem,M,radius)
    C = problem.quadratic_matrix(solved['K'])
    witness = worst_case_quadratic(C,M,radius)
    transport = witness['transport_map']
    rows = {}
    for law,W in same_moment_laws(51670,M,n).items():
        transformed = W@transport.T
        losses = np.einsum('bi,ij,bj->b',transformed,C,transformed)
        distances = np.sum((transformed-W)**2,axis=1)
        rows[law] = {'mean_adversarial_regret':float(losses.mean()),
                     'regret_mean_se':float(losses.std(ddof=1)/np.sqrt(n)),
                     'regret_q99':float(np.quantile(losses,.99)),
                     'mean_transport_cost':float(distances.mean()),
                     'transport_mean_se':float(distances.std(ddof=1)/np.sqrt(n))}
    return {'law_construction':'Nondegenerate Gaussian and Student(df=6), equal zero mean and population second moment',
            'samples_per_law':n,'same_controller_by_second_moment_sufficiency':True,
            'exact_adversarial_regret_both_laws':witness['value'],
            'exact_transport_cost_both_laws':radius**2,'samples':rows,
            'interpretation':'Expected quadratic robust objective agrees; tail quantiles need not agree.'}


def run_decision_study(config_path, output):
    config_path,output = Path(config_path),Path(output)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['origin'] != 'public_synthetic_only':
        raise ValueError('Only independent public simulation is accepted')
    output.mkdir(parents=True,exist_ok=True)
    records, fits, failures, policies = [],[],[],{}
    for seed in config['seeds']:
        material = discover(seed,'constrained_ei',steps=12)
        problem,true_moment = material_process(material['composition_process'],config['horizon'])
        radius = config['radius_per_sqrt_dimension']*np.sqrt(problem.d)
        geometry = RegretGeometry(problem.D,problem.Kstar,problem.causal_mask)
        oracle_policy = synthesize(problem,true_moment,radius,'dr_regret')
        moments = shifted_moments(problem,true_moment,radius,oracle_policy['K'])
        test_normals = standard_normal_trajectories(900000+seed,config['test_trajectories'],problem.d)
        test_paths = {name:test_normals@symmetric_root(M).T for name,M in moments.items()}
        # Same independent validation draw for paired estimators at every sample budget.
        validation = gaussian_trajectories(800000+seed,true_moment,config['validation_trajectories'])
        training_all = gaussian_trajectories(700000+seed,true_moment,max(config['train_trajectories']))
        for n_train in config['train_trajectories']:
            print(f'decision study seed={seed}, training trajectories={n_train}',flush=True)
            train = training_all[:n_train]
            empirical = second_moment(train)
            scale = np.trace(empirical)/problem.d
            solve_logs = []
            def solve_for_learning(M):
                point,diagnostics = solve_learning_controller(problem,M,radius)
                solve_logs.append(diagnostics)
                return point,diagnostics
            started = perf_counter()
            likelihood = fit_likelihood_covariance(train,validation,max_iterations=config['outer_maxiter'])
            learned = None
            try:
                learned = fit_decision_covariance(train,validation,geometry,radius,solve_for_learning,
                                                  max_iterations=config['outer_maxiter'])
            except (ArithmeticError,ValueError,RuntimeError) as error:
                failure = {'seed':seed,'n_train':n_train,'stage':'decision_learning','error_type':type(error).__name__,
                           'error':str(error),'partial_fit':serializable(getattr(error,'partial_fit',None))}
                failures.append(failure)
                print(f"  FAILED {failure['error_type']}: {str(error)[:200]}",flush=True)
            estimators = {'empirical_ridge':empirical+.001*scale*np.eye(problem.d),
                          'diagonal':np.diag(np.diag(empirical))+.001*scale*np.eye(problem.d),
                          'likelihood_shrinkage':likelihood['covariance'], 'oracle':true_moment}
            if learned is not None:
                estimators['decision_shrinkage'] = learned['covariance']
            fit_record = {'seed':seed,'n_train':n_train,'material':material['composition_process'],
                          'A':problem.A,'B':problem.B,'true_second_moment':true_moment,
                          'radius':radius,'likelihood_fit':likelihood,'decision_fit':learned,
                          'controller_solve_diagnostics':solve_logs,'learning_seconds':perf_counter()-started}
            fits.append(serializable(fit_record))
            for estimator,M in estimators.items():
                for method in config['controller_methods']:
                    result = synthesize(problem,M,radius,method)
                    K = result['K']
                    policy_id = f'{seed}/{n_train}/{estimator}/{method}'
                    policies[policy_id] = serializable({'synthesis':result,'second_moment':M})
                    regret_matrix = problem.quadratic_matrix(K)
                    cost_matrix = problem.quadratic_matrix(K,'cost')
                    true_robust = worst_case_quadratic(regret_matrix,true_moment,radius)
                    for shift,Mtest in moments.items():
                        costs,regrets,recursive_residual = recursive_batch(problem,K,test_paths[shift])
                        distance = np.sqrt(gaussian_w2_squared(M,Mtest))
                        record = {'seed':seed,'n_train':n_train,'estimator':estimator,'method':method,'shift':shift,
                                  'expected_regret':float(np.trace(regret_matrix@Mtest)),
                                  'expected_cost':float(np.trace(cost_matrix@Mtest)),
                                  'sample_mean_regret':float(regrets.mean()),'sample_mean_cost':float(costs.mean()),
                                  'monte_carlo_regret_se':float(regrets.std(ddof=1)/np.sqrt(len(regrets))),
                                  'minimum_pathwise_regret':float(regrets.min()),
                                  'covariance_relative_frobenius':float(np.linalg.norm(M-true_moment,'fro')/np.linalg.norm(true_moment,'fro')),
                                  'true_center_worst_regret':true_robust['value'],
                                  'estimated_center_to_test_w2':float(distance),'inside_estimated_ball':bool(distance<=radius+1e-8),
                                  'oracle_center_to_test_w2':float(np.sqrt(gaussian_w2_squared(true_moment,Mtest))),
                                  'recursive_lifted_residual':recursive_residual,
                                  'policy_id':policy_id,
                                  'true_center_witness_gap':true_robust['duality_gap']}
                        records.append(record)
            # Incremental files retain numerical failures if later work is interrupted.
            write_json(output/'records.json',records)
            write_json(output/'fits.json',fits)
            write_json(output/'failures.json',failures)
            write_json(output/'policies.json',policies)
    rng = np.random.default_rng(123123)
    summary = {'origin':'public_synthetic_only','config':config,'failures':failures,'paired_primary':{},'groups':{},
               'requested_fits':len(config['seeds'])*len(config['train_trajectories']),
               'successful_decision_fits':len(fits)-len(failures),
               'missing_pair_policy':'Complete-case paired summaries are conditional on successful fits; failures and attempted budgets are retained separately.'}
    for n in config['train_trajectories']:
        for shift in config['shifts']:
            rows = [r for r in records if r['n_train']==n and r['shift']==shift and r['method']=='dr_regret']
            decision = {r['seed']:r for r in rows if r['estimator']=='decision_shrinkage'}
            likelihood = {r['seed']:r for r in rows if r['estimator']=='likelihood_shrinkage'}
            matched = sorted(decision.keys() & likelihood.keys())
            summary['paired_primary'][f'n={n}/{shift}'] = (paired_summary([
                decision[s]['expected_regret']-likelihood[s]['expected_regret'] for s in matched],rng)
                if matched else None)
            for estimator in config['estimators']:
                for method in config['controller_methods']:
                    group = [r for r in records if r['n_train']==n and r['shift']==shift and r['method']==method and r['estimator']==estimator]
                    if group:
                        summary['groups'][f'n={n}/{shift}/{estimator}/{method}'] = {
                            key:paired_summary([r[key] for r in group],rng)
                            for key in ('expected_regret','expected_cost','covariance_relative_frobenius','inside_estimated_ball')}
    summary['policy_reversal'] = policy_reversal()
    summary['same_moment_negative_control'] = moment_negative_control()
    write_json(output/'summary.json',summary)
    provenance = {'origin':'public_synthetic_only','no_private_data_loaded':True,
                  'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(output/'provenance.json',provenance)
    return summary
