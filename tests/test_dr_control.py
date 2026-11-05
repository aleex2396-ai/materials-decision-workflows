"""Independent dynamics, algebraic, transport and optimization checks."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from crfml.dr_control import (LiftedLQ, evaluate_policy, smoothed_second_moment,
                             symmetric_root, synthesize, worst_case_quadratic,
                             _validate_kkt)


def multistate_problem(horizon=3):
    return LiftedLQ.time_invariant(
        np.array([[1.03, .2], [-.06, .92]]), np.array([[.12], [.35]]), horizon,
        state_weight=np.diag([1.4, .7]), input_weight=np.array([[.4]]))


def test_time_varying_lift_and_causal_execution_match_direct_recursion():
    rng = np.random.default_rng(140)
    A = rng.normal(size=(3, 2, 2))*.2 + np.eye(2)*.7
    B = rng.normal(size=(3, 2, 2))*.3
    problem = LiftedLQ(A, B, np.eye(8), np.eye(6))
    K = rng.normal(size=(6, 8))*.2*problem.causal_mask
    z = rng.normal(size=8)
    x, u = problem.causal_response(K, z)
    np.testing.assert_allclose(u, K@z, atol=1e-13)
    np.testing.assert_allclose(x, problem.F@u + problem.G@z, atol=1e-13)
    changed = z.copy()
    changed[-2:] += 500
    changed_x, changed_u = problem.causal_response(K, changed)
    np.testing.assert_allclose(changed_u, u, atol=1e-13)
    np.testing.assert_allclose(changed_x[:-2], x[:-2], atol=1e-13)
    assert not np.allclose(changed_x[-2:], x[-2:])


def test_noncausal_oracle_and_pointwise_regret_identity():
    problem = multistate_problem()
    rng = np.random.default_rng(141)
    K = rng.normal(size=(problem.p, problem.d))*.2*problem.causal_mask
    z = rng.normal(size=problem.d)
    u = K@z
    x = problem.F@u + problem.G@z
    ustar = problem.Kstar@z
    xstar = problem.F@ustar + problem.G@z
    cost = x@problem.Q@x + u@problem.R@u
    oracle = xstar@problem.Q@xstar + ustar@problem.R@ustar
    regret = z@problem.quadratic_matrix(K)@z
    assert cost-oracle == pytest.approx(regret, abs=1e-12)
    assert regret >= 0
    np.testing.assert_allclose(problem.D@ustar+problem.F.T@problem.Q@problem.G@z, 0, atol=1e-12)
    np.testing.assert_allclose(problem.quadratic_matrix(K, "cost"),
                               problem.quadratic_matrix(K)+problem.Cstar, atol=1e-12)
    assert np.max(np.abs(problem.Kstar[~problem.causal_mask])) > .01
    with pytest.raises(ValueError, match="Noncausal"):
        problem.causal_response(problem.Kstar, z)


@pytest.mark.parametrize("radius", [.01, .4, 2.])
def test_scalar_wasserstein_adversary_matches_closed_form(radius):
    coefficient, moment = 1.7, 2.3
    result = worst_case_quadratic(np.array([[coefficient]]), np.array([[moment]]), radius)
    expected = coefficient*(np.sqrt(moment)+radius)**2
    assert result["value"] == pytest.approx(expected, rel=1e-10)
    assert result["gamma"] == pytest.approx(coefficient*(1+np.sqrt(moment)/radius), rel=1e-10)
    assert result["transport_cost"] == pytest.approx(radius**2, rel=1e-9, abs=1e-12)
    assert result["primal_value"] == pytest.approx(expected, rel=1e-10)


def test_multivariate_worst_case_gaussian_transport_and_rotation():
    rng = np.random.default_rng(142)
    matrix = rng.normal(size=(4, 4))
    C = matrix.T@matrix*.2
    matrix = rng.normal(size=(4, 4))
    moment = matrix@matrix.T + np.eye(4)*.3
    radius = .7
    result = worst_case_quadratic(C, moment, radius)
    nominal_root = symmetric_root(moment)
    gaussian_w2_squared = np.trace(moment + result["worst_second_moment"]
        - 2*symmetric_root(nominal_root@result["worst_second_moment"]@nominal_root))
    assert gaussian_w2_squared == pytest.approx(radius**2, abs=1e-9)
    assert result["duality_gap"] < 1e-9
    assert np.linalg.eigvalsh(result["transport_map"])[0] > 0
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = worst_case_quadratic(rotation.T@C@rotation, rotation.T@moment@rotation, radius)
    assert rotated["value"] == pytest.approx(result["value"], rel=1e-11)
    values = [worst_case_quadratic(C, moment, r)["value"] for r in [0, .1, .5, 1.]]
    assert np.all(np.diff(values) > 0)


def test_nominal_solution_matches_independent_restricted_normal_equations():
    problem = multistate_problem(2)
    rng = np.random.default_rng(143)
    samples = rng.normal(size=(20, problem.d)) + .3
    moment = smoothed_second_moment(samples, .1)
    result = synthesize(problem, moment, method="nominal")
    hessian = np.kron(moment, problem.D)
    free = np.flatnonzero(problem.causal_mask.ravel(order="F"))
    rhs = hessian@problem.Kstar.ravel(order="F")
    analytic = np.zeros(problem.p*problem.d)
    analytic[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs[free])
    np.testing.assert_allclose(result["K"], analytic.reshape((problem.p, problem.d), order="F"), atol=1e-7)
    cost = evaluate_policy(problem, result["K"], moment, .2)
    assert cost["nominal_cost"]-cost["nominal_regret"] == pytest.approx(np.trace(problem.Cstar@moment), abs=1e-10)


@pytest.mark.parametrize("loss", ["cost", "regret"])
def test_sdp_matches_independent_scalar_policy_optimization(loss):
    problem = LiftedLQ(np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.diag([0., 1.]), np.eye(1))
    moment = np.array([[1., .35], [.35, 1.4]])
    radius = .4
    def direct_objective(gain):
        K = np.array([[gain, 0.]])
        return worst_case_quadratic(problem.quadratic_matrix(K, loss), moment, radius)["value"]
    direct = minimize_scalar(direct_objective, bounds=(-3., 3.), method="bounded", options={"xatol":1e-12})
    result = synthesize(problem, moment, radius, method="dr_"+loss)
    assert result["objective"] == pytest.approx(direct.fun, rel=1e-6)
    assert result["K"][0, 0] == pytest.approx(direct.x, abs=5e-5)
    assert result["K"][0, 1] == 0


def test_dr_objectives_cross_evaluate_against_feasible_baselines():
    problem = multistate_problem(3)
    rng = np.random.default_rng(144)
    samples = rng.normal(size=(25, problem.d))@np.diag(np.linspace(.5, 1.5, problem.d)) + .4
    moment = smoothed_second_moment(samples, .08)
    radius = .6
    methods = ("nominal", "energy_cost", "energy_regret", "dr_cost", "dr_regret")
    solutions = {method:synthesize(problem, moment, radius, method) for method in methods}
    values = {method:evaluate_policy(problem, solution["K"], moment, radius) for method, solution in solutions.items()}
    for metric, owner in (("wasserstein_regret", "dr_regret"), ("wasserstein_cost", "dr_cost"),
                          ("energy_regret", "energy_regret"), ("energy_cost", "energy_cost"),
                          ("nominal_cost", "nominal")):
        for candidate in methods:
            assert values[owner][metric] <= values[candidate][metric] + 3e-5
    for result in solutions.values():
        assert result["diagnostics"]["numerical_validation_passed"]
        assert result["diagnostics"]["kkt_validation_passed"]
        assert np.max(np.abs(result["K"][~problem.causal_mask])) == 0
        assert result["diagnostics"]["maximum_relative_lmi_violation"] <= 1e-7
        assert result["diagnostics"]["kkt_stationarity_max_abs"] < 1e-5
        assert result["diagnostics"]["conic_complementarity_absolute"] < 1e-5*(1+result["objective"])
        assert min(result["diagnostics"]["conic_dual_min_eigenvalues"], default=0) > -1e-7
    assert np.linalg.norm(solutions["dr_cost"]["K"]-solutions["dr_regret"]["K"]) > 1e-3


def test_zero_radius_matches_nominal_controller_and_correct_loss():
    problem = multistate_problem(2)
    moment = np.eye(problem.d)
    nominal = synthesize(problem, moment, method="nominal")
    for loss in ("cost", "regret"):
        result = synthesize(problem, moment, radius=0, method="dr_"+loss)
        np.testing.assert_allclose(result["K"], nominal["K"], atol=1e-7)
        assert result["objective"] == pytest.approx(np.trace(problem.quadratic_matrix(result["K"], loss)@moment), abs=1e-9)
        assert result["gamma"] is None


def test_nominal_law_validation_and_second_moment_not_centered_covariance():
    samples = np.array([[2., 3.], [2., 3.]])
    moment = smoothed_second_moment(samples, .1)
    np.testing.assert_allclose(moment, np.array([[4.01, 6.], [6., 9.01]]))
    assert np.linalg.eigvalsh(moment)[0] == pytest.approx(.01)
    with pytest.raises(ValueError, match="smoothing"):
        smoothed_second_moment(samples, 0.)
    with pytest.raises(ValueError, match="positive definite"):
        worst_case_quadratic(np.eye(2), np.diag([1., 0.]), .2)
    zero = worst_case_quadratic(np.zeros((2, 2)), np.eye(2), .2)
    assert zero["value"] == 0 and zero["gamma"] is None


def test_independent_conic_solvers_agree_on_small_instance():
    import cvxpy as cp
    if "SCS" not in cp.installed_solvers():
        pytest.skip("Independent SCS solver is not installed")
    problem = LiftedLQ.time_invariant(np.array([[1.]]), np.array([[1.]]), 2)
    moment = np.array([[1., .2, .1], [.2, 1.2, .3], [.1, .3, .9]])
    clarabel = synthesize(problem, moment, .4, "dr_regret", solver="CLARABEL")
    scs = synthesize(problem, moment, .4, "dr_regret", solver="SCS", tolerance=1e-6)
    assert clarabel["objective"] == pytest.approx(scs["objective"], rel=2e-5)
    np.testing.assert_allclose(clarabel["K"], scs["K"], atol=5e-4)


@pytest.mark.parametrize("failure", ["stationarity", "dual_psd", "complementarity"])
def test_kkt_gate_rejects_each_independent_failure(failure):
    # Independently control each diagnostic: PSD complementarity can be bad
    # even when stationarity and dual PSD pass, and vice versa.
    terms = [[np.array([1.]), np.array([-1.])]]
    primal, dual = [np.diag([1., 0.])], [np.diag([0., 1.])]
    if failure == "stationarity":
        terms[0][1] = np.array([-.9])
    elif failure == "dual_psd":
        dual[0] = np.diag([0., -1.])
    else:
        dual[0] = np.eye(2)
    with pytest.raises(ArithmeticError, match="KKT validation failed"):
        _validate_kkt(terms, primal, dual, 1., 1e-7)


def test_kkt_gate_uses_declared_scaling_and_rejects_nonfinite_duals():
    result = _validate_kkt([[np.array([1e8]), np.array([-1e8+1.])]],
                           [np.diag([1e7, 0.])], [np.diag([0., 1e6])], 1e7, 1e-7)
    assert result["kkt_validation_passed"]
    assert result["kkt_stationarity_max_abs"] == 1.
    assert result["kkt_stationarity_max_scaled"] < 1e-8
    assert result["kkt_stationarity_tolerance"] == pytest.approx(1e-6)
    assert result["conic_complementarity_tolerance"] == pytest.approx(1e-5)
    with pytest.raises(ArithmeticError, match="Nonfinite"):
        _validate_kkt([[0.]], [np.eye(1)], [np.array([[np.nan]])], 1., 1e-7)


def test_synthesis_rejects_corrupt_dual_even_when_solver_reports_optimal(monkeypatch):
    import cvxpy as cp
    original_solve = cp.Problem.solve

    def solve_with_corrupt_dual(program, *args, **kwargs):
        result = original_solve(program, *args, **kwargs)
        cone = next(constraint for constraint in program.constraints
                    if isinstance(constraint, cp.constraints.psd.PSD))
        cone.save_dual_value(-np.eye(cone.shape[0]))
        return result

    monkeypatch.setattr(cp.Problem, "solve", solve_with_corrupt_dual)
    problem = multistate_problem(2)
    with pytest.raises(ArithmeticError, match="KKT validation failed"):
        synthesize(problem, np.eye(problem.d), .4, "dr_regret")


def test_zero_energy_endpoint_checks_explicit_scalar_multiplier():
    problem = LiftedLQ(np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.zeros((2, 2)), np.eye(1))
    result = synthesize(problem, np.eye(2), .4, "energy_cost")
    assert result["objective"] == pytest.approx(0., abs=1e-8)
    assert not result["diagnostics"]["kkt_interior_scalar_multiplier"]
    assert result["diagnostics"]["kkt_validation_passed"]
