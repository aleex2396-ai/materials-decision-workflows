"""Finite-horizon W2 distributionally robust LQ cost/regret synthesis.

Implements Al Taha, Yan & Bitar (2023), Theorems 2--4, using a
congruence-transformed SDP and an independent scalar-dual evaluator.
The nominal law is declared absolutely continuous with a finite positive
definite second moment. This is unconstrained LQ feedback, not bounded MPC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import brentq


def _symmetric(matrix, dimension=None, positive=False, label="matrix"):
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite square matrix")
    if dimension is not None and value.shape != (dimension, dimension):
        raise ValueError(f"Incorrect {label} dimensions")
    if not np.allclose(value, value.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{label} must be symmetric")
    value = (value + value.T) / 2
    eigenvalues = np.linalg.eigvalsh(value)
    if positive and eigenvalues[0] <= 0:
        raise ValueError(f"{label} must be positive definite")
    if eigenvalues[0] < -1e-10 * max(1., np.linalg.norm(value, 2)):
        raise ValueError(f"{label} must be positive semidefinite")
    return value


def symmetric_root(matrix):
    value = _symmetric(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T


@dataclass
class LiftedLQ:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    F: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)
    D: np.ndarray = field(init=False)
    Droot: np.ndarray = field(init=False)
    Kstar: np.ndarray = field(init=False)
    Cstar: np.ndarray = field(init=False)
    causal_mask: np.ndarray = field(init=False)
    T: int = field(init=False)
    n: int = field(init=False)
    m: int = field(init=False)
    d: int = field(init=False)
    p: int = field(init=False)

    def __post_init__(self):
        self.A, self.B = np.asarray(self.A, float).copy(), np.asarray(self.B, float).copy()
        if self.A.ndim != 3 or self.A.shape[0] < 1 or self.A.shape[1] != self.A.shape[2]:
            raise ValueError("A must have shape (horizon, state_dimension, state_dimension)")
        self.T, self.n = self.A.shape[:2]
        if self.B.ndim != 3 or self.B.shape[:2] != (self.T, self.n) or self.B.shape[2] < 1:
            raise ValueError("B must have shape (horizon, state_dimension, input_dimension)")
        if not np.isfinite(self.A).all() or not np.isfinite(self.B).all():
            raise ValueError("Dynamics must be finite")
        self.m = self.B.shape[2]
        self.d, self.p = self.n * (self.T + 1), self.m * self.T
        self.Q = _symmetric(self.Q, self.d, label="Q")
        self.R = _symmetric(self.R, self.p, positive=True, label="R")
        self.F, self.G = np.zeros((self.d, self.p)), np.zeros((self.d, self.d))
        self.G[:self.n, :self.n] = np.eye(self.n)
        self.causal_mask = np.zeros((self.p, self.d), dtype=bool)
        for t in range(self.T):
            current, following = slice(t*self.n, (t+1)*self.n), slice((t+1)*self.n, (t+2)*self.n)
            control = slice(t*self.m, (t+1)*self.m)
            self.F[following] = self.A[t] @ self.F[current]
            self.F[following, control] += self.B[t]
            self.G[following] = self.A[t] @ self.G[current]
            self.G[following, following] += np.eye(self.n)
            self.causal_mask[control, :(t+1)*self.n] = True
        self.D = self.R + self.F.T @ self.Q @ self.F
        self.Droot = symmetric_root(self.D)
        self.Kstar = -np.linalg.solve(self.D, self.F.T @ self.Q @ self.G)
        self.Cstar = self.G.T @ self.Q @ self.G - self.Kstar.T @ self.D @ self.Kstar
        self.Cstar = (self.Cstar + self.Cstar.T) / 2

    @classmethod
    def time_invariant(cls, A, B, horizon, state_weight=None, input_weight=None):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.ndim != 2 or B.ndim != 2 or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Use matrix dynamics and a positive integer horizon")
        n, m = A.shape[0], B.shape[1]
        q = np.eye(n) if state_weight is None else np.asarray(state_weight, float)
        r = np.eye(m) if input_weight is None else np.asarray(input_weight, float)
        return cls(np.repeat(A[None], horizon, axis=0), np.repeat(B[None], horizon, axis=0),
                   np.kron(np.eye(horizon+1), q), np.kron(np.eye(horizon), r))

    def quadratic_matrix(self, K, loss="regret"):
        K = np.asarray(K, float)
        if K.shape != (self.p, self.d) or not np.isfinite(K).all():
            raise ValueError("Incorrect or nonfinite policy matrix")
        if loss == "regret":
            error = K - self.Kstar
            result = error.T @ self.D @ error
        elif loss == "cost":
            response = self.F @ K + self.G
            result = response.T @ self.Q @ response + K.T @ self.R @ K
        else:
            raise ValueError("Loss must be cost or regret")
        return (result + result.T) / 2

    def causal_response(self, K, disturbance):
        """Execute using x0 and reconstructed past disturbances, never the future."""
        K, z = np.asarray(K, float), np.asarray(disturbance, float)
        if K.shape != (self.p, self.d) or z.shape != (self.d,) or not np.isfinite(z).all() or not np.isfinite(K).all():
            raise ValueError("Incorrect response dimensions or nonfinite inputs")
        if np.max(np.abs(K[~self.causal_mask])) > 1e-10:
            raise ValueError("Noncausal feedback is not executable")
        states = [z[:self.n].copy()]
        observed = [states[0].copy()]
        inputs = []
        for t in range(self.T):
            action = K[t*self.m:(t+1)*self.m, :(t+1)*self.n] @ np.concatenate(observed)
            next_state = self.A[t] @ states[-1] + self.B[t] @ action + z[(t+1)*self.n:(t+2)*self.n]
            observed.append(next_state-self.A[t]@states[-1]-self.B[t]@action)
            states.append(next_state)
            inputs.append(action)
        return np.concatenate(states), np.concatenate(inputs)


def smoothed_second_moment(samples, smoothing):
    """Moment of the equally weighted N(sample_i, smoothing**2 I) mixture."""
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or min(samples.shape) < 1 or not np.isfinite(samples).all():
        raise ValueError("Samples must be a finite nonempty trajectory matrix")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Positive smoothing is required for an absolutely continuous nominal law")
    return samples.T @ samples / len(samples) + smoothing**2 * np.eye(samples.shape[1])


def worst_case_quadratic(C, M0, radius):
    """Theorems 2--3, independently evaluated by a bracketed scalar root solve."""
    C = _symmetric(C, label="C")
    M0 = _symmetric(M0, len(C), positive=True, label="nominal second moment")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Radius must be finite and nonnegative")
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 0.)
    weights = np.diag(eigenvectors.T @ M0 @ eigenvectors)
    largest = float(eigenvalues[-1])
    nominal = float(np.trace(C @ M0))
    if radius == 0 or largest == 0:
        return dict(value=nominal, gamma=None, transport_map=np.eye(len(C)),
                    worst_second_moment=M0.copy(), transport_cost=0., primal_value=nominal,
                    dual_value=nominal, duality_gap=0., spectral_gap=None)

    def root_function(gamma):
        with np.errstate(over="ignore", divide="ignore"):
            return float(np.sum(weights * (eigenvalues/(gamma-eigenvalues))**2) - radius**2)

    lower = np.nextafter(largest, np.inf)
    upper = largest + max(1., largest*np.sqrt(np.trace(M0))/radius)
    for _ in range(100):
        if root_function(upper) < 0:
            break
        upper = largest + 2*(upper-largest)
    else:
        raise ArithmeticError("Unable to bracket Wasserstein dual optimum")
    gamma = float(brentq(root_function, lower, upper, xtol=1e-12, rtol=1e-13))
    transport = (eigenvectors * (gamma/(gamma-eigenvalues))) @ eigenvectors.T
    moment = transport @ M0 @ transport.T
    displacement = transport - np.eye(len(C))
    budget = float(np.trace(displacement @ M0 @ displacement.T))
    primal = float(np.trace(C @ moment))
    dual = float(gamma*radius**2 + np.sum(weights*gamma*eigenvalues/(gamma-eigenvalues)))
    return dict(value=dual, gamma=gamma, transport_map=transport, worst_second_moment=moment,
                transport_cost=budget, primal_value=primal, dual_value=dual,
                duality_gap=abs(primal-dual), spectral_gap=gamma-largest)


def _validate_kkt(stationarity_terms, primal_cones, dual_cones, objective_value, tolerance):
    """Reject numerically inconsistent KKT data; this is not a global-gap bound.

    Each stationarity block is a list of additive Lagrangian-gradient terms.
    Normalization uses their infinity norms, before cancellation. Scalar
    nonnegativity cones are represented as 1-by-1 matrices.
    """
    if len(primal_cones) != len(dual_cones):
        raise ArithmeticError("Missing conic dual diagnostics")
    absolute_stationarity, scaled_stationarity = [], []
    for terms in stationarity_terms:
        arrays = [np.asarray(term, float) for term in terms]
        if not arrays or any(not np.isfinite(term).all() for term in arrays):
            raise ArithmeticError("Nonfinite or missing stationarity diagnostics")
        residual = float(np.max(np.abs(sum(arrays))))
        scale = 1. + sum(float(np.max(np.abs(term))) for term in arrays)
        absolute_stationarity.append(residual)
        scaled_stationarity.append(residual/scale)
    dual_minima, relative_dual_violations, complementarity = [], [], []
    for primal, dual in zip(primal_cones, dual_cones):
        primal, dual = np.asarray(primal, float), np.asarray(dual, float)
        if (primal.ndim != 2 or primal.shape[0] != primal.shape[1]
                or dual.shape != primal.shape or not np.isfinite(primal).all()
                or not np.isfinite(dual).all()):
            raise ArithmeticError("Nonfinite or missing conic dual diagnostics")
        dual = (dual+dual.T)/2
        minimum = float(np.linalg.eigvalsh(dual)[0])
        dual_minima.append(minimum)
        relative_dual_violations.append(max(0., -minimum)/max(1., np.linalg.norm(dual, 2)))
        complementarity.append(abs(float(np.sum(primal*dual))))
    # Local residuals use 10 times the requested tolerance. The aggregate
    # complementarity budget uses 100 times it, matching the pre-existing
    # objective-recomputation acceptance scale; no data-dependent thresholds.
    local_limit, aggregate_limit = 10*tolerance, 100*tolerance
    scaled_complementarity = sum(complementarity)/(1+abs(objective_value))
    result = dict(
        kkt_stationarity_max_abs=max(absolute_stationarity, default=0.),
        kkt_stationarity_max_scaled=max(scaled_stationarity, default=0.),
        conic_dual_min_eigenvalues=dual_minima,
        maximum_relative_conic_dual_violation=max(relative_dual_violations, default=0.),
        conic_complementarity_absolute=sum(complementarity),
        conic_complementarity_scaled=scaled_complementarity,
        kkt_stationarity_tolerance=local_limit,
        conic_dual_psd_tolerance=local_limit,
        conic_complementarity_tolerance=aggregate_limit,
    )
    result["kkt_validation_passed"] = bool(
        np.isfinite(objective_value)
        and result["kkt_stationarity_max_scaled"] <= local_limit
        and result["maximum_relative_conic_dual_violation"] <= local_limit
        and scaled_complementarity <= aggregate_limit)
    if not result["kkt_validation_passed"]:
        raise ArithmeticError(f"Numerical KKT validation failed: {result}")
    return result


def synthesize(problem: LiftedLQ, M0, radius=0., method="dr_regret", solver="CLARABEL", tolerance=1e-7, margin=1e-8):
    """Solve the causal nominal QP, energy SDP, or exact W2 cost/regret SDP.

    Numerical strictness uses the declared margin. Inadequate numerical evidence
    raises instead of returning an accepted controller. Acceptance is numerical
    evidence only, not a rigorous bound on global synthesis suboptimality.
    """
    import cvxpy as cp
    M0 = _symmetric(M0, problem.d, positive=True, label="nominal second moment")
    if method not in {"nominal", "energy_cost", "energy_regret", "dr_cost", "dr_regret"}:
        raise ValueError("Unknown synthesis method")
    if not np.isfinite([radius, tolerance, margin]).all() or radius < 0 or tolerance <= 0 or margin <= 0:
        raise ValueError("Invalid radius, numerical tolerance or margin")
    requested_method = method
    if method.startswith("dr_") and radius == 0:
        method = "nominal"
    d, p = problem.d, problem.p
    K = cp.Variable((p, d))
    H = problem.Droot @ (K-problem.Kstar)
    constraints = [K[~problem.causal_mask] == 0]
    matrices = []
    gamma, X, scalar_constraint = None, None, None
    offset = problem.Cstar if method.endswith("cost") else np.zeros((d, d))
    root = symmetric_root(M0)
    nominal_offset = float(np.trace(problem.Cstar @ M0)) if requested_method == "dr_cost" and radius == 0 else 0.
    if method == "nominal":
        objective = cp.sum_squares(H @ root) + nominal_offset
    elif method.startswith("energy_"):
        eta = cp.Variable()
        block = cp.bmat([[eta*np.eye(d)-offset, H.T], [H, np.eye(p)]])
        constraints.append(block >> 0)
        matrices.append(block)
        objective = eta
        scalar_constraint = eta >= 0
    else:
        gamma, X = cp.Variable(), cp.Variable((d, d), symmetric=True)
        small = cp.bmat([[gamma*np.eye(d)-offset, H.T], [H, np.eye(p)]])
        large = cp.bmat([[X, gamma*root, np.zeros((d, p))],
                         [gamma*root, gamma*np.eye(d)-offset, H.T],
                         [np.zeros((p, d)), H, np.eye(p)]])
        constraints.extend([X >> 0, small >> margin*np.eye(d+p), large >> 0])
        matrices.extend([X, small-margin*np.eye(d+p), large])
        objective = gamma*(radius**2-np.trace(M0)) + cp.trace(X)
        scalar_constraint = gamma >= 0
    if scalar_constraint is not None:
        constraints.append(scalar_constraint)
    program = cp.Problem(cp.Minimize(objective), constraints)
    if not program.is_dcp():
        raise ArithmeticError("Synthesis formulation is not convex under DCP rules")
    options = ({"tol_gap_abs": tolerance*.1, "tol_gap_rel": tolerance*.1,
                "tol_feas": tolerance*.1, "max_iter": 300} if solver == "CLARABEL" else
               {"eps": tolerance*.1, "max_iters": 150000} if solver == "SCS" else {})
    started = perf_counter()
    program.solve(solver=solver, **options)
    elapsed = perf_counter()-started
    if program.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or K.value is None or not np.isfinite(program.value):
        raise ArithmeticError(f"Synthesis failed: {program.status}")
    policy = np.array(K.value, float)
    causal_residual = float(np.max(np.abs(policy[~problem.causal_mask])))
    # Remove only equality-roundoff entries before independent evaluation.
    policy[~problem.causal_mask] = 0.
    loss = "cost" if requested_method.endswith("cost") else "regret"
    C = problem.quadratic_matrix(policy, loss=loss)
    witness = worst_case_quadratic(C, M0, radius)
    if method == "nominal":
        verified_objective = float(np.trace(problem.quadratic_matrix(policy, "regret") @ M0)) + nominal_offset
    elif method.startswith("energy_"):
        verified_objective = float(np.linalg.eigvalsh(C)[-1])
    else:
        verified_objective = witness["value"]
    minima, relative_violations = [], []
    for matrix in matrices:
        value = np.asarray(matrix.value, float)
        minimum = float(np.linalg.eigvalsh((value+value.T)/2)[0])
        minima.append(minimum)
        relative_violations.append(max(0., -minimum)/max(1., np.linalg.norm(value, 2)))
    gap = abs(float(program.value)-verified_objective)/(1+abs(verified_objective))
    # Independent KKT checks use conic dual matrices, not solver status alone.
    # Include explicit scalar-cone multipliers, including degenerate endpoints.
    primal_cones, dual_cones = [matrix.value for matrix in matrices], []
    if method == "nominal":
        stationarity_terms = [[(2*problem.D @ policy @ M0)[problem.causal_mask],
                               (-2*problem.D @ problem.Kstar @ M0)[problem.causal_mask]]]
        interior = True
    elif method.startswith("energy_"):
        dual = np.asarray(constraints[1].dual_value, float)
        scalar_dual = float(scalar_constraint.dual_value)
        gradient_K = -2*problem.Droot @ dual[d:, :d]
        stationarity_terms = [[gradient_K[problem.causal_mask]],
                              [1., -float(np.trace(dual[:d, :d])), -scalar_dual]]
        dual_cones = [dual, np.array([[scalar_dual]])]
        primal_cones.append(np.array([[float(eta.value)]]))
        interior = float(eta.value) > 10*tolerance
    else:
        dual_X, dual_small, dual_large = (np.asarray(item.dual_value, float) for item in constraints[1:4])
        scalar_dual = float(scalar_constraint.dual_value)
        stationarity_terms = [
            [(-2*problem.Droot @ dual_small[d:, :d])[problem.causal_mask],
             (-2*problem.Droot @ dual_large[2*d:, d:2*d])[problem.causal_mask]],
            [np.eye(d), -dual_X, -dual_large[:d, :d]],
            [radius**2, -float(np.trace(M0)), -float(np.trace(dual_small[:d, :d])),
             -float(np.trace(dual_large[d:2*d, d:2*d])),
             -2*float(np.sum(dual_large[:d, d:2*d]*root)), -scalar_dual],
        ]
        dual_cones = [dual_X, dual_small, dual_large, np.array([[scalar_dual]])]
        primal_cones.append(np.array([[float(gamma.value)]]))
        interior = float(gamma.value) > 10*tolerance
    kkt_diagnostics = _validate_kkt(stationarity_terms, primal_cones, dual_cones,
                                    float(program.value), tolerance)
    diagnostics = dict(causality_residual=causal_residual, lmi_min_eigenvalues=minima,
                       maximum_relative_lmi_violation=float(max(relative_violations, default=0.)),
                       objective_recompute_relative_gap=gap, numerical_margin=margin,
                       tolerance=tolerance, zero_radius_uses_nominal=(requested_method != method),
                       worst_case_transport_residual=abs(witness["transport_cost"]-radius**2) if witness["gamma"] is not None else 0.,
                       worst_case_duality_gap=witness["duality_gap"],
                       kkt_interior_scalar_multiplier=interior,
                       solver_spectral_gap=float(gamma.value)-float(np.linalg.eigvalsh(C)[-1]) if gamma is not None else None)
    diagnostics.update(kkt_diagnostics)
    if causal_residual > tolerance or max(relative_violations, default=0.) > tolerance or gap > max(1e-5, 100*tolerance):
        raise ArithmeticError(f"Numerical validation failed: {diagnostics}")
    if gamma is not None and diagnostics["solver_spectral_gap"] <= 0:
        raise ArithmeticError("Solver did not establish the strict quadratic-dual domain")
    diagnostics["numerical_validation_passed"] = True
    return dict(K=policy, method=requested_method, objective=verified_objective,
                solver_objective=float(program.value), gamma=witness["gamma"],
                solver_gamma=float(gamma.value) if gamma is not None else None,
                status=program.status, solver=solver, solve_seconds=elapsed, diagnostics=diagnostics)


def evaluate_policy(problem: LiftedLQ, K, M0, radius):
    M0 = _symmetric(M0, problem.d, positive=True, label="nominal second moment")
    result = {}
    for loss in ("cost", "regret"):
        C = problem.quadratic_matrix(K, loss)
        witness = worst_case_quadratic(C, M0, radius)
        result["nominal_"+loss] = float(np.trace(C @ M0))
        result["energy_"+loss] = float(np.linalg.eigvalsh(C)[-1])
        result["wasserstein_"+loss] = witness["value"]
    return result
