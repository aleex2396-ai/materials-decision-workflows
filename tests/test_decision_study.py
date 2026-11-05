import numpy as np
from crfml.decision_study import material_process,recursive_batch,gaussian_w2_squared,policy_reversal
from crfml.dr_control import synthesize


def test_causal_batch_matches_independent_recursive_paths():
    problem,M = material_process([.2,.3,.5,.6],3)
    K = synthesize(problem,M,.2)['K']
    W = np.random.default_rng(218).multivariate_normal(np.zeros(problem.d),M,8)
    cost,regret,residual = recursive_batch(problem,K,W)
    for i,w in enumerate(W):
        X,U = problem.causal_response(K,w)
        direct = X@problem.Q@X+U@problem.R@U
        assert np.isclose(cost[i],direct)
        assert np.isclose(regret[i],direct-w@problem.Cstar@w)
    assert residual < 1e-10


def test_gaussian_transport_for_noncommuting_covariances():
    A = np.array([[2.,.4],[.4,.7]])
    T = np.array([[1.4,.2],[.2,.8]])
    B = T@A@T
    assert np.isclose(gaussian_w2_squared(A,B),np.trace((T-np.eye(2))@A@(T-np.eye(2))))


def test_smaller_covariance_error_can_have_strictly_larger_transfer_regret():
    result = policy_reversal()
    a,b = (result['rows'][k] for k in ['larger_diagonal_error','smaller_correlation_error'])
    assert a['covariance_frobenius_error'] > b['covariance_frobenius_error']
    assert abs(a['excess_robust_regret']) < 1e-7
    assert b['excess_robust_regret'] > 1e-4
    assert np.isclose(b['excess_robust_regret'],b['analytic_excess'])
    assert b['scalar_sdp_action_difference'] < 2e-4
