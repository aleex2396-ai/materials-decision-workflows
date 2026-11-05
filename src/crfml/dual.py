"""Two-stage Bayesian control with a uniform gain belief and bounded uniform noise.

The posterior and last-stage solution are analytic. The first-stage Bellman
search is a numerical quadrature/grid reference, not an exact continuous solve.
No private data or learned GP prior is used. See docs/DUAL_CONTROL_THEORY.md.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .control import ControlConfig


@dataclass(frozen=True)
class Belief:
    low: float
    high: float

    def __post_init__(self):
        if not np.isfinite([self.low, self.high]).all() or not 0 < self.low <= self.high:
            raise ValueError('Belief requires finite ordered positive bounds')

    @property
    def mean(self):
        return (self.low+self.high)/2

    @property
    def variance(self):
        return (self.high-self.low)**2/12


def _state_check(x, previous_u, cfg):
    if not np.isfinite([x, previous_u]).all():
        raise ValueError('State and previous input must be finite')
    if x > cfg.xmax+1e-10 or not 0 <= previous_u <= cfg.umax+1e-10:
        raise ValueError('Require an upper-safe state and admissible previous input')


def safe_cap(x, belief, cfg):
    """Largest nonnegative input satisfying the one-step robust upper bound."""
    _state_check(x, 0., cfg)
    return float(np.clip((cfg.xmax-cfg.a*x-cfg.wbar)/belief.high, 0, cfg.umax))


def update_belief(x, u, next_x, belief, cfg):
    """Exact uniform posterior support, under the assumed fixed-gain model.

    A transition with zero likelihood raises; it is not repaired or truncated.
    Degenerate known-gain and zero-noise limits are supported explicitly.
    """
    if not np.isfinite([x, u, next_x]).all() or u < 0:
        raise ValueError('Transition must be finite with nonnegative input')
    z = next_x-cfg.a*x
    if u == 0:
        if abs(z) > cfg.wbar+1e-12:
            raise ValueError('Transition has zero likelihood under the noise bound')
        return belief
    low = max(belief.low, (z-cfg.wbar)/u)
    high = min(belief.high, (z+cfg.wbar)/u)
    if low > high+1e-12:
        raise ValueError('Transition has zero likelihood under the gain/noise bounds')
    # Reconcile only floating-point endpoint disagreement, never a finite gap.
    if low > high:
        low = high = (low+high)/2
    return Belief(float(low), float(high))


def _cost_array(x, u, low, high, previous_u, cfg, reference):
    mean = (low+high)/2
    variance = (high-low)**2/12
    return ((cfg.a*x+mean*u-reference)**2 + variance*u*u + cfg.wbar**2/3
            + cfg.effort*u*u + cfg.slew*(u-previous_u)**2)


def _one_step_array(x, low, high, previous_u, cfg, reference):
    mean = (low+high)/2
    second = (low*low+low*high+high*high)/3
    curvature = second+cfg.effort+cfg.slew
    unconstrained = ((reference-cfg.a*x)*mean+cfg.slew*previous_u)/curvature
    cap = np.clip((cfg.xmax-cfg.a*x-cfg.wbar)/high, 0, cfg.umax)
    return np.clip(unconstrained, 0, cap)


def expected_stage_cost(x, u, belief, previous_u=0., cfg=None, reference=None):
    """True expected next-state tracking, effort and slew cost under a belief."""
    cfg = cfg or ControlConfig()
    reference = cfg.reference if reference is None else reference
    if not np.isfinite([x, u, previous_u, reference]).all() or u < 0:
        raise ValueError('Expected cost requires finite values and nonnegative input')
    return float(_cost_array(x, u, belief.low, belief.high, previous_u, cfg, reference))


def bayes_one_step(x, belief, previous_u=0., cfg=None, reference=None):
    """Analytic Bayes-optimal last action under the robust one-step cap."""
    cfg = cfg or ControlConfig()
    reference = cfg.reference if reference is None else reference
    _state_check(x, previous_u, cfg)
    if not np.isfinite(reference):
        raise ValueError('Reference must be finite')
    u = float(_one_step_array(x, belief.low, belief.high, previous_u, cfg, reference))
    return u, expected_stage_cost(x, u, belief, previous_u, cfg, reference)


@lru_cache(maxsize=12)
def _legendre(order):
    return np.polynomial.legendre.leggauss(order)


def observation_quadrature(x, u, belief, cfg=None, order=24):
    """Return next states, weights and posterior endpoints over possible observations.

    Integrates the exact piecewise-linear convolution density using Gauss-Legendre
    nodes on each density segment. Weights are NOT silently renormalised.
    """
    cfg = cfg or ControlConfig()
    if not isinstance(order, int) or isinstance(order, bool) or order < 2:
        raise ValueError('Quadrature order must be an integer >=2')
    if not np.isfinite([x, u]).all() or u < 0:
        raise ValueError('Finite state and nonnegative input required')
    low, high, w = belief.low, belief.high, cfg.wbar
    if w == 0 and (u == 0 or low == high):
        return (np.array([cfg.a*x+low*u]), np.ones(1),
                np.array([low]), np.array([high]))
    nodes, weights = _legendre(order)
    if u == 0 or low == high:
        z = low*u+w*nodes
        return cfg.a*x+z, weights/2, np.full(order, low), np.full(order, high)
    if w == 0:
        b = (low+high)/2+(high-low)*nodes/2
        return cfg.a*x+u*b, weights/2, b, b
    knots = np.unique([low*u-w, low*u+w, high*u-w, high*u+w])
    zs, ws, ls, hs = [], [], [], []
    for left, right in zip(knots[:-1], knots[1:]):
        z = (left+right)/2+(right-left)*nodes/2
        pl = np.maximum(low, (z-w)/u)
        ph = np.minimum(high, (z+w)/u)
        density = np.maximum(ph-pl, 0)/(2*w*(high-low))
        zs.append(z)
        ws.append(weights*(right-left)/2*density)
        ls.append(pl)
        hs.append(ph)
    return cfg.a*x+np.concatenate(zs), np.concatenate(ws), np.concatenate(ls), np.concatenate(hs)


def evaluate_two_step_action(u, x, belief, previous_u=0., cfg=None,
                             reference_now=None, reference_next=None,
                             quadrature_order=24):
    """Evaluate one fixed first action with informed and frozen continuations.

    Both continuations see the next physical state. Frozen continuation chooses
    using the original belief/cap; its REAL cost uses the actual posterior.
    This common probability law is essential to an unbiased information ablation.
    """
    cfg = cfg or ControlConfig()
    now = cfg.reference if reference_now is None else reference_now
    later = cfg.reference if reference_next is None else reference_next
    _state_check(x, previous_u, cfg)
    if not np.isfinite([u, now, later]).all() or u < 0 or u > safe_cap(x, belief, cfg)+1e-10:
        raise ValueError('First action must satisfy its robust cap; references must be finite')
    states, weights, lows, highs = observation_quadrature(x, u, belief, cfg, quadrature_order)
    informed = _one_step_array(states, lows, highs, u, cfg, later)
    frozen = _one_step_array(states, belief.low, belief.high, u, cfg, later)
    adapted_cost = _cost_array(states, informed, lows, highs, u, cfg, later)
    frozen_cost = _cost_array(states, frozen, lows, highs, u, cfg, later)
    first = expected_stage_cost(x, u, belief, previous_u, cfg, now)
    return {'action':float(u), 'first_cost':first,
            'adapted_cost':float(first+weights@adapted_cost),
            'frozen_cost':float(first+weights@frozen_cost),
            'information_benefit':float(weights@(frozen_cost-adapted_cost)),
            'expected_posterior_width':float(weights@(highs-lows)),
            'weight_sum':float(weights.sum()),
            'minimum_conditional_benefit':float(np.min(frozen_cost-adapted_cost))}


def two_step_action(x, belief, previous_u=0., cfg=None, grid_size=81,
                    quadrature_order=24, frozen_belief=False,
                    reference_now=None, reference_next=None):
    """Numerical two-stage Bellman reference on a uniform safe first-action grid.

    All grid points are searched; continuation actions are analytic. Returned
    objective is a quadrature approximation, not certified continuous optimum.
    """
    cfg = cfg or ControlConfig()
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 2:
        raise ValueError('grid_size must be an integer >=2')
    _state_check(x, previous_u, cfg)
    cap = safe_cap(x, belief, cfg)
    grid = np.linspace(0, cap, grid_size) if cap > 0 else np.zeros(1)
    evaluations = [evaluate_two_step_action(float(u), x, belief, previous_u, cfg,
                    reference_now, reference_next, quadrature_order) for u in grid]
    objective = 'frozen_cost' if frozen_belief else 'adapted_cost'
    values = np.array([row[objective] for row in evaluations])
    best = int(np.argmin(values))
    return float(grid[best]), {
        'policy':'frozen_belief' if frozen_belief else 'bayesian_two_step',
        'method':'finite_grid_with_piecewise_gauss_legendre_quadrature',
        'grid_size':int(len(grid)), 'grid_step':float(cap/(grid_size-1)),
        'quadrature_order':quadrature_order, 'minimum_value':float(values[best]),
        'grid':grid.tolist(), 'values':values.tolist(), 'chosen':evaluations[best],
        'max_weight_error':float(max(abs(r['weight_sum']-1) for r in evaluations))}


def indistinguishable_disturbances(u, gain_low, gain_high, wbar):
    """Matched disturbances for two gains with identical transition histories.

    A causal adversary may apply these after seeing the chosen nonnegative input.
    Does not model IID uniform noise. Raises if the requested pair is inadmissible.
    """
    if not np.isfinite([u, gain_low, gain_high, wbar]).all() or u < 0 or wbar < 0 or not 0 < gain_low <= gain_high:
        raise ValueError('Invalid indistinguishability construction')
    offset = (gain_high-gain_low)*u/2
    if offset > wbar+1e-12:
        raise ValueError('Gain separation exceeds the disturbance masking budget')
    return float(offset), float(-offset)
