"""Scalar robust MPC and bounded-noise set-membership identification.

Certificate is conditional on exact state measurements, fixed b in the current
interval, known 0<=a<1, u>=0, and |w|<=wbar. GP intervals are NOT certificates.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


@dataclass(frozen=True)
class ControlConfig:
    a: float = .86
    wbar: float = .025
    xmax: float = 1.
    umax: float = .8
    reference: float = .88
    horizon: int = 6
    effort: float = .04
    slew: float = .03
    bmin: float = .12
    bmax: float = .60

    def __post_init__(self):
        if not np.isfinite(list(self.__dict__.values())).all():
            raise ValueError('Configuration must be finite')
        if not (0 <= self.a < 1 and self.wbar >= 0 and self.xmax > 0 and self.umax > 0):
            raise ValueError('Invalid dynamics/bounds')
        if self.a*self.xmax+self.wbar > self.xmax:
            raise ValueError('Zero-input backup is not invariant')
        if not (0 < self.bmin <= self.bmax) or self.horizon < 1:
            raise ValueError('Invalid gain interval/horizon')
        if self.effort <= 0 or self.slew < 0 or not isinstance(self.horizon, int):
            raise ValueError('Strictly positive effort, nonnegative slew and integer horizon required')


@dataclass
class GainSet:
    low: float
    high: float
    valid: bool = True

    def __post_init__(self):
        if not np.isfinite([self.low, self.high]).all() or not 0 < self.low <= self.high:
            raise ValueError('Gain set requires finite ordered positive bounds')

    def update(self, x, u, next_x, cfg):
        if not np.isfinite([x, u, next_x]).all():
            self.valid = False
            raise ValueError('Nonfinite transition invalidates gain certificate')
        if not self.valid:
            return False
        if u < 0:
            raise ValueError('Nonnegative input assumption violated')
        residual = next_x-cfg.a*x
        if u <= 1e-10:
            if abs(residual) > cfg.wbar+1e-9:
                self.valid = False
            return self.valid
        low = max(self.low, (residual-cfg.wbar)/u)
        high = min(self.high, (residual+cfg.wbar)/u)
        if low > high+1e-9:
            self.valid = False
            return False
        self.low, self.high = low, max(low, high)
        return True


def prediction_matrix(cfg):
    h = cfg.horizon
    return np.array([[cfg.a**(i-j) if j <= i else 0. for j in range(h)] for i in range(h)])


def solve_mpc(x, nominal_b, upper_b, cfg, previous_u=0., robust=True):
    """Convex QP via SLSQP, analytic gradient; independent feasibility checks."""
    if not np.isfinite([x, nominal_b, upper_b, previous_u]).all() or nominal_b <= 0 or upper_b <= 0:
        raise ValueError('Invalid MPC arguments')
    h, s = cfg.horizon, prediction_matrix(cfg)
    base = cfg.a**np.arange(1, h+1)*x
    response = nominal_b*s
    d = np.eye(h)-np.eye(h, k=-1)
    prev = np.zeros(h)
    prev[0] = previous_u
    q = response.T@response + cfg.effort*np.eye(h)+cfg.slew*d.T@d
    f = response.T@(base-cfg.reference)-cfg.slew*d.T@prev
    w = cfg.wbar*np.sum(s, axis=1) if robust else np.zeros(h)
    g = upper_b*s
    rhs = cfg.xmax-base-w
    result = minimize(lambda u: float(u@q@u+2*f@u), np.full(h, .15),
                      jac=lambda u: 2*q@u+2*f, bounds=Bounds(0, cfg.umax),
                      constraints=[LinearConstraint(g, -np.inf, rhs)], method='SLSQP',
                      options={'ftol': 1e-10, 'maxiter': 100})
    violation = max(0., float(np.max(g@result.x-rhs)), float(-result.x.min()),
                    float(result.x.max()-cfg.umax)) if np.isfinite(result.x).all() else float('inf')
    fallback = not result.success or violation > 1e-7
    plan = np.zeros(h) if fallback else np.maximum(result.x, 0.)
    return float(plan[0]), {'solver_success': bool(result.success), 'fallback': fallback,
                           'constraint_residual': violation,
                           'backup_certified': bool(x <= cfg.xmax+1e-9 and robust),
                           'plan': plan.tolist(), 'upper_trajectory': (base+g@plan+w).tolist()}


def safe_probe(x, gain_upper, cfg):
    if not np.isfinite([x,gain_upper]).all() or gain_upper <= 0:
        raise ValueError('Probe requires finite state and positive gain bound')
    return float(np.clip((cfg.xmax-cfg.a*x-cfg.wbar)/gain_upper, 0, cfg.umax))


def rollout(true_b, learned_b, disturbances, policy, cfg=None):
    cfg = cfg or ControlConfig()
    if not np.isfinite([true_b,learned_b]).all() or min(true_b,learned_b) <= 0 or not np.isfinite(disturbances).all():
        raise ValueError('Rollout requires finite positive gains and finite disturbances')
    permitted = {'nominal_gp', 'robust_fixed', 'robust_adaptive', 'safe_probe',
                 'constant_center', 'adaptive_constant_center', 'oracle'}
    if policy not in permitted:
        raise ValueError(f'Unknown policy {policy}')
    gain = GainSet(true_b, true_b) if policy == 'oracle' else GainSet(cfg.bmin, cfg.bmax)
    x, last_u = .2, 0.
    records = []
    for t, w in enumerate(disturbances):
        low, high = gain.low, gain.high
        valid_before = gain.valid
        if policy == 'nominal_gp':
            center, upper, robust = learned_b, learned_b, False
        elif policy == 'oracle':
            center, upper, robust = true_b, true_b, True
        elif policy == 'constant_center':
            center, upper, robust = (cfg.bmin+cfg.bmax)/2, high, True
        elif policy == 'adaptive_constant_center':
            center, upper, robust = float(np.clip((cfg.bmin+cfg.bmax)/2, low, high)), high, True
        else:
            center, upper, robust = float(np.clip(learned_b, low, high)), high, True
        u, diagnostics = solve_mpc(x, center, upper, cfg, last_u, robust)
        if not gain.valid and policy not in {'nominal_gp', 'oracle'}:
            u = 0.  # emergency stop, explicitly no longer certified
        elif policy == 'safe_probe' and t < 3:
            u = safe_probe(x, high, cfg)
        next_x = cfg.a*x + true_b*u + float(w)
        in_envelope = (low-1e-9 <= true_b <= high+1e-9 and abs(w) <= cfg.wbar+1e-9)
        if policy in {'robust_adaptive', 'adaptive_constant_center', 'safe_probe'}:
            gain.update(x, u, next_x, cfg)
        cost = (next_x-cfg.reference)**2 + cfg.effort*u*u + cfg.slew*(u-last_u)**2
        records.append({'t':t, 'x':x, 'u':u, 'next_x':next_x, 'w':float(w),
                        'cost':float(cost), 'gain_low':low, 'gain_high':high,
                        'gain_width_after':gain.high-gain.low, 'detector_valid':gain.valid,
                        'conditional_certificate':bool(robust and valid_before and x <= cfg.xmax+1e-9),
                        'actual_assumptions_hold':bool(in_envelope),
                        'violation':bool(next_x > cfg.xmax+1e-7),
                        'solver_fallback':diagnostics['fallback'],
                        'solver_residual':diagnostics['constraint_residual']})
        x, last_u = next_x, u
    return {'policy':policy, 'total_cost':sum(r['cost'] for r in records),
            'violations':sum(r['violation'] for r in records),
            'final_width':gain.high-gain.low, 'assumption_failures':sum(not r['actual_assumptions_hold'] for r in records),
            'detected_failure':not gain.valid, 'solver_fallbacks':sum(r['solver_fallback'] for r in records),
            'trajectory':records}
