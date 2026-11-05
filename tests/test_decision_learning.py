"""Independent calculus and re-solved-SDP checks for task-aware covariance learning."""
import numpy as np
import pytest

from crfml.dr_control import LiftedLQ
from crfml.decision_learning import (
    ControllerSolveError, DecisionFitError, ImplicitGradientError, RegretGeometry, fit_decision_covariance,
    fit_likelihood_covariance, gaussian_nll_gradient, implicit_hypergradient,
    reduced_hessian, reduced_value_gradient, robust_covariance_sensitivity,
    second_moment, shrinkage_covariance, solve_learning_controller,
    validation_regret, verify_hypergradient,
)


@pytest.fixture
def setup():
    # Small algebraic fixture, deliberately independent of the benchmark seeds.
    problem = LiftedLQ.time_invariant(np.array([[.65, .2], [-.1, .75]]),
                                      np.array([[.8], [.25]]), 2,
                                      state_weight=np.diag([1., .7]), input_weight=np.array([[.4]]))
    geometry = RegretGeometry(problem.D, problem.Kstar, problem.causal_mask)
    rng = np.random.default_rng(3241)
    root = np.linalg.cholesky(.55**np.abs(np.arange(problem.d)[:, None]-np.arange(problem.d)))
    training = rng.standard_normal((11, problem.d))@root.T
    validation = rng.standard_normal((41, problem.d))@root.T
    theta = np.array([.25, np.log(.05)])
    radius = .45
    return problem, geometry, training, validation, theta, radius


def test_rank_deficient_training_moment_is_smoothed_and_parameter_derivatives_match():
    moment = second_moment([[1., 2., 3.], [-1., -2., -3.]])
    theta = np.array([.3, np.log(.04)])
    M, derivatives = shrinkage_covariance(moment, theta)
    assert np.linalg.eigvalsh(M)[0] >= .04*np.trace(moment)/len(moment)-1e-12
    for j in range(2):
        direction = np.eye(2)[j]*1e-6
        finite = (shrinkage_covariance(moment, theta+direction)[0]
                  - shrinkage_covariance(moment, theta-direction)[0])/2e-6
        np.testing.assert_allclose(derivatives[j], finite, atol=2e-9, rtol=2e-8)
    with pytest.raises(ValueError, match='positive trace'):
        shrinkage_covariance(np.zeros((2, 2)), theta)
    with pytest.raises(ValueError, match='positive semidefinite'):
        shrinkage_covariance(np.diag([1., -1.]), theta)
    with pytest.raises(ValueError, match='alpha'):
        shrinkage_covariance(moment, [-.1, -3.])


def test_trace_normalization_holds_scale_and_quotient_derivatives_match():
    moment = second_moment([[1., 2., 3.], [-1., -2., -3.]])
    theta = np.array([.31, np.log(.07)])
    raw, _ = shrinkage_covariance(moment, theta)
    M, derivatives = shrinkage_covariance(moment, theta, trace_normalize=True)
    np.testing.assert_allclose(M, np.trace(moment)*raw/np.trace(raw), atol=1e-14)
    assert abs(np.trace(M)-np.trace(moment)) < 1e-12
    assert np.linalg.eigvalsh(M)[0] > 0
    np.testing.assert_allclose(np.trace(derivatives, axis1=1, axis2=2), 0., atol=1e-14)
    for j in range(2):
        direction = np.eye(2)[j]*1e-6
        finite = (shrinkage_covariance(moment, theta+direction, trace_normalize=True)[0]
                  - shrinkage_covariance(moment, theta-direction, trace_normalize=True)[0])/2e-6
        np.testing.assert_allclose(derivatives[j], finite, atol=2e-9, rtol=2e-8)
    # Default primary mapping remains exactly the unnormalized implementation.
    explicit, explicit_derivative = shrinkage_covariance(moment, theta, trace_normalize=False)
    default, default_derivative = shrinkage_covariance(moment, theta)
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(default_derivative, explicit_derivative)


def test_exact_gradient_and_hessian_match_independent_centered_differences(setup):
    _, geometry, training, _, theta, radius = setup
    M, _ = shrinkage_covariance(second_moment(training), theta)
    K = .5*geometry.Kstar*geometry.causal_mask
    largest = np.linalg.eigvalsh(geometry.regret_matrix(K))[-1]
    z = geometry.pack(K, largest+1.2)
    _, gradient = reduced_value_gradient(geometry, z, M, radius)
    hessian = reduced_hessian(geometry, z, M, radius)
    numerical_gradient, numerical_hessian = np.empty_like(z), np.empty_like(hessian)
    for j in range(len(z)):
        direction = np.eye(len(z))[j]*1e-5
        plus, pg = reduced_value_gradient(geometry, z+direction, M, radius)
        minus, mg = reduced_value_gradient(geometry, z-direction, M, radius)
        numerical_gradient[j] = (plus-minus)/2e-5
        numerical_hessian[:, j] = (pg-mg)/2e-5
    np.testing.assert_allclose(gradient, numerical_gradient, atol=1e-8, rtol=2e-7)
    np.testing.assert_allclose(hessian, numerical_hessian, atol=3e-8, rtol=3e-7)
    assert np.linalg.eigvalsh(hessian)[0] > 0


def test_covariance_sensitivity_selects_regret_directions(setup):
    problem, geometry, training, _, theta, radius = setup
    M, _ = shrinkage_covariance(second_moment(training), theta)
    z, _ = solve_learning_controller(problem, M, radius)
    sensitivity = robust_covariance_sensitivity(geometry, z, M, radius)
    K, gamma = geometry.unpack(z)
    eig, vectors = np.linalg.eigh(geometry.regret_matrix(K))
    expected = gamma*eig/(gamma-eig)
    np.testing.assert_allclose(np.diag(vectors.T@sensitivity@vectors), expected, atol=1e-10)
    # Equal-Frobenius covariance perturbations can have very different value effects.
    quiet = np.outer(vectors[:, 0], vectors[:, 0])
    sensitive = np.outer(vectors[:, -1], vectors[:, -1])
    assert abs(np.trace(sensitivity@quiet)) < 1e-9
    assert np.trace(sensitivity@sensitive) > .1
    for direction in (quiet, sensitive):
        epsilon = 2e-5
        plus = reduced_value_gradient(geometry, z, M+epsilon*direction, radius)[0]
        minus = reduced_value_gradient(geometry, z, M-epsilon*direction, radius)[0]
        assert abs((plus-minus)/(2*epsilon)-np.trace(sensitivity@direction)) < 1e-7
        # Separate envelope check: reoptimise the controller on each perturbation.
        zp, _ = solve_learning_controller(problem, M+epsilon*direction, radius)
        zm, _ = solve_learning_controller(problem, M-epsilon*direction, radius)
        vp = reduced_value_gradient(geometry, zp, M+epsilon*direction, radius)[0]
        vm = reduced_value_gradient(geometry, zm, M-epsilon*direction, radius)[0]
        assert abs((vp-vm)/(2*epsilon)-np.trace(sensitivity@direction)) < 2e-6


@pytest.mark.parametrize('trace_normalize', [False, True])
def test_hypergradient_matches_fresh_sdp_resolves(setup, trace_normalize):
    problem, geometry, training, validation, theta, radius = setup
    solves = []
    def solver(M):
        z, diagnostics = solve_learning_controller(problem, M, radius)
        solves.append(diagnostics)
        return z
    report = verify_hypergradient(geometry, theta, second_moment(training),
                                 second_moment(validation), radius, solver,
                                 step=1e-4, rtol=3e-4, atol=3e-7,
                                 trace_normalize=trace_normalize)
    assert report['passed'], report
    assert len(solves) == report['independent_controller_requests'] == 5
    assert all(d['attempt_count'] == 1 for d in solves)
    assert all(d['polishing']['stationarity_inf'] < 3e-10 for d in solves)
    assert max(d['relative_objective_gap'] for d in solves) < 1e-5


def test_covariance_scale_changes_effective_radius_exactly(setup):
    problem, geometry, training, _, theta, radius = setup
    M, _ = shrinkage_covariance(second_moment(training), theta)
    scale = 2.7
    z_scaled, _ = solve_learning_controller(problem, scale*M, radius)
    z_radius, _ = solve_learning_controller(problem, M, radius/np.sqrt(scale))
    np.testing.assert_allclose(z_scaled, z_radius, atol=2e-9, rtol=1e-9)
    value_scaled = reduced_value_gradient(geometry, z_scaled, scale*M, radius)[0]
    value_radius = reduced_value_gradient(geometry, z_radius, M, radius/np.sqrt(scale))[0]
    assert abs(value_scaled-scale*value_radius) < 2e-10


def test_bad_smoothness_stationarity_and_conditioning_are_rejected(setup):
    problem, geometry, training, validation, theta, radius = setup
    M, dM = shrinkage_covariance(second_moment(training), theta)
    val = second_moment(validation)
    z, _ = solve_learning_controller(problem, M, radius)
    result = implicit_hypergradient(geometry, z, M, dM, val, radius)
    assert result.diagnostics.linear_solve_residual < 1e-10
    assert result.diagnostics.spectral_margin > 0
    changed = z.copy()
    changed[0] += .01
    with pytest.raises(ImplicitGradientError, match='stationary'):
        implicit_hypergradient(geometry, changed, M, dM, val, radius)
    with pytest.raises(ImplicitGradientError, match='ill-conditioned'):
        implicit_hypergradient(geometry, z, M, dM, val, radius, condition_limit=1.01)
    with pytest.raises(ImplicitGradientError, match='margin'):
        implicit_hypergradient(geometry, z, M, dM, val, radius, relative_margin_floor=1.)
    outside = z.copy()
    outside[-1] = 0.
    with pytest.raises(ImplicitGradientError, match='spectral interior'):
        reduced_value_gradient(geometry, outside, M, radius)
    with pytest.raises(ValueError, match='positive radius'):
        reduced_value_gradient(geometry, z, M, 0.)
    with pytest.raises(ValueError, match='positive definite'):
        implicit_hypergradient(geometry, z, np.zeros_like(M), dM, val, radius)
    invalid_derivative = dM.copy()
    invalid_derivative[0, 0, 1] += .1
    with pytest.raises(ValueError, match='symmetric'):
        implicit_hypergradient(geometry, z, M, invalid_derivative, val, radius)


@pytest.mark.parametrize('trace_normalize', [False, True])
def test_gaussian_density_gradient_and_shared_sample_budgets(setup, trace_normalize):
    _, _, training, validation, theta, _ = setup
    train, val = second_moment(training), second_moment(validation)
    initial_loss, gradient = gaussian_nll_gradient(train, val, theta, trace_normalize=trace_normalize)
    finite = np.empty(2)
    for j in range(2):
        delta = np.eye(2)[j]*1e-5
        finite[j] = (gaussian_nll_gradient(train, val, theta+delta, trace_normalize=trace_normalize)[0]
                     - gaussian_nll_gradient(train, val, theta-delta, trace_normalize=trace_normalize)[0])/2e-5
    np.testing.assert_allclose(gradient, finite, atol=1e-8, rtol=1e-7)
    fit = fit_likelihood_covariance(training, validation, trace_normalize=trace_normalize)
    assert fit['validation_nll'] < initial_loss
    assert fit['budgets']['training_samples'] == len(training)
    assert fit['budgets']['validation_samples'] == len(validation)
    assert fit['budgets']['test_samples_accessed'] == fit['budgets']['controller_solves'] == 0
    assert fit['budgets']['optimization_evaluations'] == len(fit['trajectory'])
    assert .001-1e-12 <= np.exp(fit['theta'][1]) <= 1.
    assert fit['trace_normalize'] == trace_normalize
    if trace_normalize:
        assert abs(np.trace(fit['covariance'])-np.trace(train)) < 1e-12


@pytest.mark.parametrize('corrupt_result', [False, True])
def test_implicit_linear_solver_failure_is_not_silently_accepted(setup, monkeypatch, corrupt_result):
    problem, geometry, training, validation, theta, radius = setup
    M, dM = shrinkage_covariance(second_moment(training), theta)
    z, _ = solve_learning_controller(problem, M, radius)
    original = np.linalg.solve
    def broken_linear_solve(matrix, rhs):
        if matrix.shape == (geometry.n_variables, geometry.n_variables):
            if corrupt_result:
                return np.zeros_like(rhs)
            raise np.linalg.LinAlgError('Injected factorization failure')
        return original(matrix, rhs)
    monkeypatch.setattr(np.linalg, 'solve', broken_linear_solve)
    with pytest.raises(ImplicitGradientError, match='linear solve'):
        implicit_hypergradient(geometry, z, M, dM, second_moment(validation), radius)


@pytest.mark.parametrize('trace_normalize', [False, True])
def test_decision_fit_records_evaluations_without_test_data(setup, trace_normalize):
    problem, geometry, training, validation, theta, radius = setup
    def solver(M):
        return solve_learning_controller(problem, M, radius)
    initial_cov, _ = shrinkage_covariance(second_moment(training), theta, trace_normalize=trace_normalize)
    initial_value = validation_regret(geometry, solver(initial_cov)[0], second_moment(validation))[0]
    fit = fit_decision_covariance(training, validation, geometry, radius, solver, max_iterations=3,
                                  trace_normalize=trace_normalize)
    assert fit['validation_regret'] <= initial_value+1e-9
    assert fit['gradient_verification']['passed']
    assert fit['budgets']['controller_solves'] == len(fit['trajectory'])+6
    assert fit['budgets']['controller_attempts'] == fit['budgets']['controller_requests']
    assert fit['budgets']['test_samples_accessed'] == 0
    assert isinstance(fit['optimizer_success'], bool)
    assert fit['optimizer_message']
    assert fit['trace_normalize'] == trace_normalize
    if trace_normalize:
        assert abs(np.trace(fit['covariance'])-np.trace(second_moment(training))) < 1e-12


def test_failed_decision_fit_preserves_partial_trace_and_attempt_budget(setup):
    problem, geometry, training, validation, _, radius = setup
    solves = 0
    def solver(M):
        nonlocal solves
        solves += 1
        if solves == 2:
            raise ArithmeticError('Injected solver failure on second optimization evaluation')
        return solve_learning_controller(problem, M, radius)[0]
    with pytest.raises(DecisionFitError) as captured:
        fit_decision_covariance(training, validation, geometry, radius, solver, verify=False)
    partial = captured.value.partial_fit
    assert partial['failed_stage'] == 'optimization'
    assert partial['budgets']['controller_requests'] == 2
    assert partial['budgets']['controller_attempts'] is None  # Packed-point callback hides attempts.
    assert partial['budgets']['optimization_evaluations'] == 2
    assert partial['trajectory'][0]['status'] == 'evaluated'
    assert partial['trajectory'][1]['status'] == 'failed'
    assert partial['trajectory'][1]['error_type'] == 'ArithmeticError'
    assert partial['budgets']['test_samples_accessed'] == 0


def test_numerical_retry_preserves_gates_and_rejected_attempt(setup, monkeypatch):
    import crfml.dr_control as control
    problem, _, training, _, theta, radius = setup
    M, _ = shrinkage_covariance(second_moment(training), theta)
    original, calls = control.synthesize, []
    def reject_first(*args, **kwargs):
        calls.append((kwargs['solver'], kwargs['tolerance']))
        if len(calls) == 1:
            raise ArithmeticError("Numerical KKT validation failed: {'kkt_validation_passed': False}")
        return original(*args, **kwargs)
    monkeypatch.setattr(control, 'synthesize', reject_first)
    _, diagnostics = solve_learning_controller(problem, M, radius)
    assert calls == [('CLARABEL', 1e-8), ('SCS', 1e-8)]
    assert diagnostics['attempt_count'] == 2
    assert diagnostics['solver'] == 'SCS'
    assert diagnostics['attempts'][0]['status'] == 'rejected'
    assert diagnostics['attempts'][0]['parsed_diagnostics'] == {'kkt_validation_passed': False}
    assert diagnostics['attempts'][1]['status'] == 'accepted'
    assert diagnostics['sdp_diagnostics']['kkt_validation_passed']
    assert all(a['acceptance_tolerance'] == 1e-8 for a in diagnostics['attempts'])


def test_valid_primary_solve_does_not_trigger_fallback(setup, monkeypatch):
    import crfml.dr_control as control
    problem, _, training, _, theta, radius = setup
    M, _ = shrinkage_covariance(second_moment(training), theta)
    original, calls = control.synthesize, []
    def recording(*args, **kwargs):
        calls.append(kwargs['solver'])
        return original(*args, **kwargs)
    monkeypatch.setattr(control, 'synthesize', recording)
    _, diagnostics = solve_learning_controller(problem, M, radius)
    assert calls == ['CLARABEL']
    assert diagnostics['attempt_count'] == 1
    assert diagnostics['attempts'][0]['status'] == 'accepted'


def test_all_rejected_attempts_remain_in_failed_fit_budget(setup, monkeypatch):
    import crfml.dr_control as control
    problem, geometry, training, validation, _, radius = setup
    def reject(*args, **kwargs):
        raise ArithmeticError('Injected numerical failure')
    monkeypatch.setattr(control, 'synthesize', reject)
    with pytest.raises(DecisionFitError) as captured:
        fit_decision_covariance(training, validation, geometry, radius,
                                lambda M: solve_learning_controller(problem, M, radius))
    partial = captured.value.partial_fit
    assert partial['error_type'] == 'ControllerSolveError'
    assert partial['budgets']['controller_requests'] == 1
    assert partial['budgets']['controller_attempts'] == partial['budgets']['controller_solves'] == 2
    attempts = partial['controller_attempt_logs'][0]['attempts']
    assert [a['solver'] for a in attempts] == ['CLARABEL', 'SCS']
    assert all(a['status'] == 'rejected' for a in attempts)
