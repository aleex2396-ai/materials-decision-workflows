"""Covariance learning through a causal Wasserstein minimax-regret controller.

The exact controller theorem lives in ``dr_control``. This module studies the
smooth, strictly interior reduced problem and rejects unreliable derivatives.
It receives known dynamics through a regret geometry; it never learns dynamics
or accesses held-out samples. See docs/DECISION_LEARNING.md for assumptions.
"""
from dataclasses import dataclass
import ast
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import minimize


class ImplicitGradientError(RuntimeError):
    """The smooth-interior assumptions or numerical accuracy checks failed."""


class DecisionFitError(ImplicitGradientError):
    """A failed fit retaining its incomplete evaluation trace and budgets."""

    def __init__(self, message, partial_fit):
        super().__init__(message)
        self.partial_fit = partial_fit


class ControllerSolveError(ImplicitGradientError):
    """Both prescribed numerical solver attempts failed unchanged acceptance gates."""

    def __init__(self, message, attempts):
        super().__init__(message)
        self.controller_attempts = attempts
        self.attempt_count = len(attempts)


def _symmetric_matrix(value, name, *, positive_definite=False):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite square matrix')
    if not np.allclose(a, a.T, atol=1e-11, rtol=1e-11):
        raise ValueError(f'{name} must be symmetric')
    a = (a + a.T) / 2
    eig = np.linalg.eigvalsh(a)
    if (positive_definite and eig[0] <= 0) or eig[0] < -1e-10 * max(1., abs(eig[-1])):
        requirement = 'positive definite' if positive_definite else 'positive semidefinite'
        raise ValueError(f'{name} must be {requirement}')
    return a


@dataclass(frozen=True)
class RegretGeometry:
    """Known cost weight, clairvoyant controller, and admissible causal entries."""

    D: np.ndarray
    Kstar: np.ndarray
    causal_mask: np.ndarray

    def __post_init__(self):
        d = _symmetric_matrix(self.D, 'D', positive_definite=True)
        kstar = np.asarray(self.Kstar, dtype=float)
        mask = np.asarray(self.causal_mask)
        if kstar.ndim != 2 or kstar.shape[0] != len(d) or not np.isfinite(kstar).all():
            raise ValueError('Kstar has incompatible dimensions or nonfinite entries')
        if mask.shape != kstar.shape or mask.dtype.kind != 'b':
            raise ValueError('causal_mask must be boolean and match Kstar')
        object.__setattr__(self, 'D', d.copy())
        object.__setattr__(self, 'Kstar', kstar.copy())
        object.__setattr__(self, 'causal_mask', mask.copy())

    @property
    def dimension(self):
        return self.Kstar.shape[1]

    @property
    def n_variables(self):
        return int(self.causal_mask.sum()) + 1

    def pack(self, K, gamma):
        k = np.asarray(K, dtype=float)
        if k.shape != self.Kstar.shape or not np.isfinite(k).all() or not np.isfinite(gamma):
            raise ValueError('Controller has incompatible dimensions or nonfinite entries')
        if np.max(np.abs(k[~self.causal_mask]), initial=0.) > 1e-8:
            raise ValueError('Controller violates the causal mask')
        return np.r_[k[self.causal_mask], float(gamma)]

    def unpack(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.n_variables,) or not np.isfinite(z).all():
            raise ValueError('Packed controller has incompatible dimensions or nonfinite entries')
        k = np.zeros_like(self.Kstar)
        k[self.causal_mask] = z[:-1]
        return k, float(z[-1])

    def regret_matrix(self, K):
        e = np.asarray(K, dtype=float) - self.Kstar
        return e.T @ self.D @ e


def second_moment(samples):
    """Uncentered second moment for the declared zero-mean noise model, /n."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError('Samples must be a nonempty (n,d) array')
    if not np.isfinite(samples).all():
        raise ValueError('Samples must be finite')
    return samples.T @ samples / len(samples)


def shrinkage_covariance(sample_moment, theta, *, trace_normalize=False):
    """Return SPD M and derivatives for theta=(alpha, log(beta)).

    M=(1-alpha) sample_moment + alpha diag(sample_moment) + beta scale I,
    where scale=trace(sample_moment)/d is fixed from training data.
    A positive ridge makes even an undersampled empirical moment nondegenerate.
    Optional trace normalization fixes trace(M) to trace(sample_moment); the
    returned derivatives then include the exact normalization quotient rule.
    """
    moment = _symmetric_matrix(sample_moment, 'sample moment')
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError('theta must contain finite alpha and log(beta)')
    alpha, logbeta = theta
    if not 0 <= alpha <= 1 or not -700 < logbeta < 700:
        raise ValueError('alpha must be in [0,1] and log(beta) must be numerically representable')
    beta = float(np.exp(logbeta))
    eye = np.eye(len(moment))
    diagonal = np.diag(np.diag(moment))
    scale = float(np.trace(moment)/len(moment))
    if scale <= 0:
        raise ValueError('Training second moment must have strictly positive trace')
    M = (1-alpha)*moment + alpha*diagonal + beta*scale*eye
    M = _symmetric_matrix(M, 'nominal moment', positive_definite=True)
    derivatives = np.stack((diagonal-moment, beta*scale*eye))
    if trace_normalize:
        training_trace, raw_trace = float(np.trace(moment)), float(np.trace(M))
        derivative_traces = np.trace(derivatives, axis1=1, axis2=2)
        derivatives = (training_trace/raw_trace)*(derivatives
                       - derivative_traces[:, None, None]*M[None]/raw_trace)
        M = (training_trace/raw_trace)*M
    return M, derivatives


def _components(geometry, z, M, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Smooth robust differentiation requires a positive radius')
    M = _symmetric_matrix(M, 'nominal moment', positive_definite=True)
    if M.shape != (geometry.dimension, geometry.dimension):
        raise ValueError('Nominal moment dimension does not match the controller')
    K, gamma = geometry.unpack(z)
    E = K - geometry.Kstar
    C = E.T @ geometry.D @ E
    A = gamma*np.eye(geometry.dimension)-C
    margin = float(np.linalg.eigvalsh(A)[0])
    if gamma <= 0 or margin <= 0:
        raise ImplicitGradientError('Controller is outside the strict spectral interior')
    S = np.linalg.solve(A, np.eye(len(A)))
    S = (S+S.T)/2
    return K, gamma, E, C, S, M, margin


def reduced_value_gradient(geometry, z, M, radius):
    """Exact reduced robust value and analytic gradient in free K entries,gamma."""
    _, gamma, E, _, S, M, _ = _components(geometry, z, M, radius)
    tr_ms = float(np.trace(M@S))
    value = gamma*(radius**2-np.trace(M))+gamma**2*tr_ms
    grad_k = 2*gamma**2*geometry.D@E@S@M@S
    grad_gamma = radius**2-np.trace(M)+2*gamma*tr_ms-gamma**2*np.trace(M@S@S)
    return float(value), np.r_[grad_k[geometry.causal_mask], grad_gamma]


def reduced_hessian(geometry, z, M, radius):
    """Analytic Hessian, assembled from matrix differential actions.

    dS = S (dC - dgamma I) S. No autodiff or finite-difference Hessian is used.
    """
    _, gamma, E, _, S, M, _ = _components(geometry, z, M, radius)
    D = geometry.D
    hessian = np.empty((geometry.n_variables, geometry.n_variables))
    for j in range(geometry.n_variables):
        dz = np.zeros(geometry.n_variables)
        dz[j] = 1.
        dK, dg = geometry.unpack(dz)
        dC = dK.T@D@E + E.T@D@dK
        dS = S@(dC-dg*np.eye(len(S)))@S
        dk_gradient = (4*gamma*dg*D@E@S@M@S
                       + 2*gamma**2*(D@dK@S@M@S + D@E@dS@M@S + D@E@S@M@dS))
        dg_gradient = (2*dg*np.trace(M@S) + 2*gamma*np.trace(M@dS)
                       - 2*gamma*dg*np.trace(M@S@S)
                       - gamma**2*np.trace(M@(dS@S+S@dS)))
        hessian[:, j] = np.r_[dk_gradient[geometry.causal_mask], dg_gradient]
    asymmetry = np.linalg.norm(hessian-hessian.T)/max(1., np.linalg.norm(hessian))
    if asymmetry > 1e-8:
        raise ImplicitGradientError(f'Analytic Hessian lost symmetry ({asymmetry:g})')
    return (hessian+hessian.T)/2


def robust_covariance_sensitivity(geometry, z, M, radius):
    """Partial d f / d M; envelope derivative only at a regular optimum."""
    _, gamma, _, _, S, _, _ = _components(geometry, z, M, radius)
    return gamma**2*S-gamma*np.eye(geometry.dimension)


def validation_regret(geometry, z, validation_moment):
    moment = _symmetric_matrix(validation_moment, 'validation moment')
    if moment.shape != (geometry.dimension, geometry.dimension):
        raise ValueError('Validation moment dimension does not match the controller')
    K, _ = geometry.unpack(z)
    E = K-geometry.Kstar
    grad_k = 2*geometry.D@E@moment
    return float(np.trace(E.T@geometry.D@E@moment)), np.r_[grad_k[geometry.causal_mask], 0.]


@dataclass(frozen=True)
class ImplicitDiagnostics:
    stationarity_inf: float
    spectral_margin: float
    hessian_min_eigenvalue: float
    hessian_condition: float
    linear_solve_residual: float


@dataclass(frozen=True)
class ImplicitResult:
    gradient: np.ndarray
    controller_derivative: np.ndarray
    validation_regret: float
    diagnostics: ImplicitDiagnostics


def implicit_hypergradient(geometry, z, M, dM, validation_moment, radius,
                          *, stationarity_tolerance=2e-7, condition_limit=1e10,
                          relative_margin_floor=1e-8, solve_tolerance=1e-9):
    """Differentiate the interior optimum, rejecting nonregular/numerically poor inputs."""
    _, gamma, E, _, S, M, margin = _components(geometry, z, M, radius)
    _, g = reduced_value_gradient(geometry, z, M, radius)
    residual = float(np.linalg.norm(g, ord=np.inf))
    if residual > stationarity_tolerance:
        raise ImplicitGradientError(f'Controller is not stationary ({residual:g})')
    if margin/max(1., abs(gamma)) < relative_margin_floor:
        raise ImplicitGradientError('Spectral margin is too small for reliable differentiation')
    dM = np.asarray(dM, dtype=float)
    if dM.ndim != 3 or dM.shape[1:] != M.shape or not np.isfinite(dM).all():
        raise ValueError('Moment derivatives must be finite (n_parameters,d,d) matrices')
    if not np.allclose(dM, dM.transpose(0, 2, 1), atol=1e-11, rtol=1e-11):
        raise ValueError('Moment derivatives must be symmetric')
    hessian = reduced_hessian(geometry, z, M, radius)
    eig = np.linalg.eigvalsh(hessian)
    if eig[0] <= 0:
        raise ImplicitGradientError('Hessian is not positive definite; smooth uniqueness is unverified')
    condition = float(eig[-1]/eig[0])
    if condition > condition_limit:
        raise ImplicitGradientError(f'Hessian is ill-conditioned ({condition:g})')
    mixed = []
    for derivative in dM:
        grad_k = 2*gamma**2*geometry.D@E@S@derivative@S
        grad_gamma = (-np.trace(derivative)+2*gamma*np.trace(derivative@S)
                      - gamma**2*np.trace(derivative@S@S))
        mixed.append(np.r_[grad_k[geometry.causal_mask], grad_gamma])
    mixed = np.asarray(mixed).T
    try:
        dz = np.linalg.solve(hessian, -mixed)
    except np.linalg.LinAlgError as error:
        raise ImplicitGradientError('Implicit linear solve failed') from error
    solve_residual = float(np.linalg.norm(hessian@dz+mixed)/max(1., np.linalg.norm(mixed)))
    if not np.isfinite(dz).all() or solve_residual > solve_tolerance:
        raise ImplicitGradientError(f'Implicit linear solve residual is excessive ({solve_residual:g})')
    loss, loss_gradient = validation_regret(geometry, z, validation_moment)
    diagnostics = ImplicitDiagnostics(residual, margin, float(eig[0]), condition, solve_residual)
    return ImplicitResult(loss_gradient@dz, dz, loss, diagnostics)


def polish_controller(geometry, K, gamma, M, radius, *, tolerance=3e-10, max_iterations=40):
    """Newton-refine an SDP solution of the same reduced convex objective.

    Spectrally infeasible steps are backtracked. The method fails explicitly
    instead of silently differentiating a solver point with poor stationarity.
    """
    z = geometry.pack(K, gamma)
    initial_value, gradient = reduced_value_gradient(geometry, z, M, radius)
    for iteration in range(max_iterations+1):
        value, gradient = reduced_value_gradient(geometry, z, M, radius)
        if np.linalg.norm(gradient, np.inf) <= tolerance:
            return z, {'iterations': iteration, 'initial_value': initial_value,
                       'polished_value': value, 'stationarity_inf': float(np.linalg.norm(gradient, np.inf))}
        if iteration == max_iterations:
            break
        hessian = reduced_hessian(geometry, z, M, radius)
        if np.linalg.eigvalsh(hessian)[0] <= 0:
            raise ImplicitGradientError('Newton Hessian is not positive definite')
        direction = np.linalg.solve(hessian, -gradient)
        slope = float(gradient@direction)
        if slope >= 0:
            raise ImplicitGradientError('Newton direction does not decrease the robust objective')
        step = 1.
        for _ in range(60):
            candidate = z+step*direction
            try:
                new_value, new_gradient = reduced_value_gradient(geometry, candidate, M, radius)
            except ImplicitGradientError:
                step *= .5
                continue
            # Roundoff in nearly converged objectives should not block a
            # full step that demonstrably decreases the stationarity residual.
            if (new_value <= value+1e-4*step*slope
                    or (abs(new_value-value) <= 2e-13*max(1., abs(value))
                        and np.linalg.norm(new_gradient, np.inf) < np.linalg.norm(gradient, np.inf))):
                z = candidate
                break
            step *= .5
        else:
            raise ImplicitGradientError('Newton backtracking did not find a valid decreasing step')
    raise ImplicitGradientError(f'Newton refinement did not converge ({np.linalg.norm(gradient, np.inf):g})')


def solve_learning_controller(problem, M, radius, *, solver_tolerance=1e-8,
                              objective_agreement_tolerance=1e-4):
    """Fresh exact causal SDP followed by explicitly recorded stationary refinement.

    The SDP and reduced objective describe the same mathematical controller.
    Numerical objective agreement is checked before using implicit derivatives.
    At most two conic solves are attempted: Clarabel, then SCS only if numerical
    checks or the solver fail. Both use the same unchanged acceptance tolerance.
    """
    from crfml.dr_control import synthesize
    from cvxpy.error import SolverError
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Smooth robust differentiation requires a positive radius')
    geometry = RegretGeometry(problem.D, problem.Kstar, problem.causal_mask)
    attempts = []
    last_error = None
    # Deterministic numerical recovery, never a change to the nominal law,
    # objective, radius, or acceptance thresholds. A valid first solve stops.
    for solver in ('CLARABEL', 'SCS'):
        started = perf_counter()
        attempt = {'solver': solver, 'acceptance_tolerance': solver_tolerance,
                   'objective_agreement_tolerance': objective_agreement_tolerance}
        try:
            solution = synthesize(problem, M, radius, method='dr_regret', solver=solver,
                                  tolerance=solver_tolerance)
            if solution['gamma'] is None:
                raise ImplicitGradientError('Degenerate zero-regret problem has no smooth interior dual optimum')
            z, polishing = polish_controller(geometry, solution['K'], solution['gamma'], M, radius)
            gap = abs(solution['solver_objective']-polishing['polished_value'])
            scale = max(1., abs(polishing['polished_value']))
            if gap/scale > objective_agreement_tolerance:
                raise ImplicitGradientError(f'SDP and polished reduced objectives disagree ({gap/scale:g})')
        except (ArithmeticError, ImplicitGradientError, np.linalg.LinAlgError, SolverError) as error:
            parsed = None
            message = str(error)
            if '{' in message:
                try:
                    candidate = ast.literal_eval(message[message.index('{'):])
                    parsed = candidate if isinstance(candidate, dict) else None
                except (ValueError, SyntaxError):
                    pass
            attempt.update(status='rejected', elapsed_seconds=perf_counter()-started,
                           error_type=type(error).__name__, error=message,
                           parsed_diagnostics=parsed)
            attempts.append(attempt)
            last_error = error
            continue
        attempt.update(status='accepted', elapsed_seconds=perf_counter()-started,
                       sdp_diagnostics=solution['diagnostics'], polishing=polishing,
                       relative_objective_gap=float(gap/scale))
        attempts.append(attempt)
        return z, {'sdp_diagnostics': solution['diagnostics'], 'polishing': polishing,
                   'sdp_objective': float(solution['solver_objective']),
                   'relative_objective_gap': float(gap/scale), 'solver': solver,
                   'attempt_count': len(attempts), 'attempts': attempts}
    raise ControllerSolveError(f'Both prescribed controller solver attempts failed: {last_error}', attempts) from last_error


def _controller_result(result):
    """Allow callers to expose solve diagnostics without changing controller math."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, None


def verify_hypergradient(geometry, theta, sample_moment, validation_moment, radius,
                         solve_controller: Callable, *, step=2e-4, rtol=3e-3, atol=2e-6,
                         trace_normalize=False):
    """Central differences with independently re-solved controllers, never test outcomes.

    ``solve_controller(M)`` returns a packed, polished controller. At least one
    independent SDP solve per call is the responsibility of the supplied adapter.
    Verification requires alpha strictly inside its permitted interval.
    """
    theta = np.asarray(theta, dtype=float)
    if step <= 0 or not step < theta[0] < 1-step:
        raise ValueError('Central verification requires an interior alpha and positive step')
    M, dM = shrinkage_covariance(sample_moment, theta, trace_normalize=trace_normalize)
    z, _ = _controller_result(solve_controller(M))
    result = implicit_hypergradient(geometry, z, M, dM, validation_moment, radius)
    finite = np.empty(2)
    for j in range(2):
        direction = np.zeros(2)
        direction[j] = step
        plus, _ = shrinkage_covariance(sample_moment, theta+direction, trace_normalize=trace_normalize)
        minus, _ = shrinkage_covariance(sample_moment, theta-direction, trace_normalize=trace_normalize)
        plus_loss = validation_regret(geometry, _controller_result(solve_controller(plus))[0], validation_moment)[0]
        minus_loss = validation_regret(geometry, _controller_result(solve_controller(minus))[0], validation_moment)[0]
        finite[j] = (plus_loss-minus_loss)/(2*step)
    error = np.abs(finite-result.gradient)
    passed = bool(np.all(error <= atol+rtol*np.abs(finite)))
    return {'passed': passed, 'implicit': result.gradient.tolist(), 'finite_difference': finite.tolist(),
            'absolute_error': error.tolist(), 'rtol': rtol, 'atol': atol, 'step': step,
            'independent_controller_requests': 5, 'trace_normalize': bool(trace_normalize)}


def fit_decision_covariance(training_samples, validation_samples, geometry, radius,
                            solve_controller: Callable, *, initial=(.25, np.log(.05)),
                            logbeta_bounds=(np.log(.001), 0.), max_iterations=12,
                            verify=True, trace_normalize=False):
    """Tune shrinkage solely by validation regret with a full evaluation trace.

    No test inputs are accepted. ``solve_controller`` must independently solve
    the causal SDP and return a polished packed point for each supplied M, or
    ``(point, diagnostics)`` to expose actual conic attempt budgets. A packed
    point alone permits counting controller requests but hides solver attempts.
    The two sample arrays must be disjoint draws established by the caller.
    """
    train_moment = second_moment(training_samples)
    val_moment = second_moment(validation_samples)
    if train_moment.shape != val_moment.shape or len(train_moment) != geometry.dimension:
        raise ValueError('Training/validation sample dimensions must match the geometry')
    initial = np.asarray(initial, dtype=float)
    if (initial.shape != (2,) or not .001 < initial[0] < .999
            or not logbeta_bounds[0] <= initial[1] <= logbeta_bounds[1]):
        raise ValueError('Initial parameters must be interior in alpha and within ridge bounds')
    if not -700 < logbeta_bounds[0] < logbeta_bounds[1] < 700 or max_iterations < 1:
        raise ValueError('Invalid finite parameter bounds or iteration budget')
    count, attempt_count, attempts_known = 0, 0, True
    controller_attempt_logs = []
    def counted_solver(M):
        nonlocal count, attempt_count, attempts_known
        count += 1
        try:
            point, diagnostics = _controller_result(solve_controller(M))
        except Exception as error:
            if hasattr(error, 'attempt_count'):
                attempt_count += error.attempt_count
                controller_attempt_logs.append({'request': count, 'status': 'failed',
                                                'attempts': error.controller_attempts})
            else:
                attempts_known = False
            raise
        if diagnostics is not None and 'attempt_count' in diagnostics:
            attempt_count += diagnostics['attempt_count']
            controller_attempt_logs.append({'request': count, 'status': 'accepted',
                                            'attempts': diagnostics['attempts']})
        else:
            attempts_known = False
        return point
    verification = None
    trajectory = []
    def partial_record(stage):
        return {'failed_stage': stage, 'trajectory': trajectory, 'trace_normalize': bool(trace_normalize),
                'gradient_verification': verification,
                'controller_attempt_logs': controller_attempt_logs,
                'budgets': {'training_samples': int(len(training_samples)),
                            'validation_samples': int(len(validation_samples)),
                            'test_samples_accessed': 0, 'controller_requests': count,
                            'controller_attempts': attempt_count if attempts_known else None,
                            'controller_solves': attempt_count if attempts_known else None,
                            'optimization_evaluations': len(trajectory)}}
    def objective(theta):
        entry = {'theta': theta.tolist(), 'alpha': float(theta[0]),
                 'beta': float(np.exp(theta[1])), 'status': 'attempted'}
        trajectory.append(entry)
        try:
            M, dM = shrinkage_covariance(train_moment, theta, trace_normalize=trace_normalize)
            z = counted_solver(M)
            result = implicit_hypergradient(geometry, z, M, dM, val_moment, radius)
        except Exception as error:
            entry.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        entry.update(status='evaluated', validation_regret=result.validation_regret,
                     hypergradient=result.gradient.tolist(),
                     stationarity_inf=result.diagnostics.stationarity_inf,
                     hessian_condition=result.diagnostics.hessian_condition,
                     spectral_margin=result.diagnostics.spectral_margin)
        return result.validation_regret, result.gradient
    stage = 'gradient_verification'
    try:
        if verify:
            verification = verify_hypergradient(geometry, initial, train_moment, val_moment,
                                                 radius, counted_solver, trace_normalize=trace_normalize)
            if not verification['passed']:
                raise ImplicitGradientError(f'Independent hypergradient verification failed: {verification}')
        stage = 'optimization'
        result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                          bounds=((0., 1.), logbeta_bounds),
                          options={'maxiter': max_iterations, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
        stage = 'final_controller'
        M, _ = shrinkage_covariance(train_moment, result.x, trace_normalize=trace_normalize)
        z = counted_solver(M)
    except Exception as error:
        # Preserve solver/library failures as well as our own numerical guards;
        # do not swallow KeyboardInterrupt/SystemExit or continue with fallback.
        partial = partial_record(stage)
        partial.update(error_type=type(error).__name__, error=str(error))
        raise DecisionFitError(f'{stage}: {error}', partial) from error
    return {'theta': result.x, 'covariance': M, 'controller': z, 'trace_normalize': bool(trace_normalize),
            'validation_regret': validation_regret(geometry, z, val_moment)[0],
            'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
            'optimizer_iterations': int(result.nit), 'trajectory': trajectory,
            'gradient_verification': verification,
            'controller_attempt_logs': controller_attempt_logs,
            'budgets': {'training_samples': int(len(training_samples)),
                        'validation_samples': int(len(validation_samples)), 'test_samples_accessed': 0,
                        'controller_requests': count,
                        'controller_attempts': attempt_count if attempts_known else None,
                        'controller_solves': attempt_count if attempts_known else None,
                        'optimization_evaluations': len(trajectory)}}


def gaussian_nll_gradient(sample_moment, validation_moment, theta, *, trace_normalize=False):
    """Gaussian validation NLL without the constant d*log(2*pi)/2."""
    M, dM = shrinkage_covariance(sample_moment, theta, trace_normalize=trace_normalize)
    val = _symmetric_matrix(validation_moment, 'validation moment')
    if val.shape != M.shape:
        raise ValueError('Training and validation moment dimensions differ')
    inv = np.linalg.solve(M, np.eye(len(M)))
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        raise ValueError('Nominal moment lost positive definiteness')
    loss = .5*(logdet+np.trace(inv@val))
    sensitivity = .5*(inv-inv@val@inv)
    return float(loss), np.einsum('ij,kji->k', sensitivity, dM)


def fit_likelihood_covariance(training_samples, validation_samples, *,
                              initial=(.25, np.log(.05)),
                              logbeta_bounds=(np.log(.001), 0.), max_iterations=12,
                              trace_normalize=False):
    """Same covariance family and data budgets as decision fitting; density target."""
    sample_moment = second_moment(training_samples)
    val = second_moment(validation_samples)
    if sample_moment.shape != val.shape:
        raise ValueError('Training and validation dimensions must match')
    initial = np.asarray(initial, dtype=float)
    if (initial.shape != (2,) or not 0 <= initial[0] <= 1
            or not logbeta_bounds[0] <= initial[1] <= logbeta_bounds[1]):
        raise ValueError('Initial parameters must be within bounds')
    if not -700 < logbeta_bounds[0] < logbeta_bounds[1] < 700 or max_iterations < 1:
        raise ValueError('Invalid finite parameter bounds or iteration budget')
    trajectory = []
    def objective(theta):
        loss, gradient = gaussian_nll_gradient(sample_moment, val, theta, trace_normalize=trace_normalize)
        trajectory.append({'theta': theta.tolist(), 'alpha': float(theta[0]),
                           'beta': float(np.exp(theta[1])), 'validation_nll': loss,
                           'gradient': gradient.tolist()})
        return loss, gradient
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=((0., 1.), logbeta_bounds),
                      options={'maxiter': max_iterations, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
    M, _ = shrinkage_covariance(sample_moment, result.x, trace_normalize=trace_normalize)
    return {'theta': result.x, 'covariance': M, 'trace_normalize': bool(trace_normalize),
            'validation_nll': gaussian_nll_gradient(sample_moment, val, result.x,
                                                     trace_normalize=trace_normalize)[0],
            'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
            'optimizer_iterations': int(result.nit), 'trajectory': trajectory,
            'budgets': {'training_samples': int(len(training_samples)),
                        'validation_samples': int(len(validation_samples)), 'test_samples_accessed': 0,
                        'controller_requests': 0, 'controller_attempts': 0,
                        'controller_solves': 0, 'optimization_evaluations': len(trajectory)}}
