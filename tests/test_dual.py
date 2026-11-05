"""Independent numerical/analytic checks of the two-stage information study."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from crfml.control import ControlConfig
from crfml.dual import (Belief, update_belief, safe_cap, expected_stage_cost,
    bayes_one_step, observation_quadrature, evaluate_two_step_action,
    two_step_action, indistinguishable_disturbances)


def test_uniform_posterior_matches_likelihood_on_parameter_grid():
    cfg = ControlConfig(wbar=.04)
    belief = Belief(.12, .6)
    x, u, next_x = .2, .5, cfg.a*.2+.31*.5+.015
    post = update_belief(x, u, next_x, belief, cfg)
    b = np.linspace(belief.low, belief.high, 10001)
    likelihood = np.abs(next_x-cfg.a*x-b*u) <= cfg.wbar
    grid_posterior = b[likelihood]
    assert abs(grid_posterior.mean()-post.mean) < 6e-5
    assert grid_posterior.min() >= post.low-1e-12
    assert grid_posterior.max() <= post.high+1e-12
    assert update_belief(x, 0, cfg.a*x, belief, cfg) == belief
    with pytest.raises(ValueError):
        update_belief(x, 0, 100, belief, cfg)


def test_one_step_cost_and_optimizer_against_independent_integration():
    cfg = ControlConfig()
    belief, x, prev, ref = Belief(.2, .55), .35, .1, .47
    def integral_cost(u):
        # Integrate over b; E[w]=0 and Var(w)=wbar^2/3 analytically.
        value = quad(lambda b: (cfg.a*x+b*u-ref)**2, belief.low, belief.high,
                     epsabs=1e-12)[0]/(belief.high-belief.low)
        return value+cfg.wbar**2/3+cfg.effort*u*u+cfg.slew*(u-prev)**2
    u, value = bayes_one_step(x, belief, prev, cfg, ref)
    numeric = minimize_scalar(integral_cost, bounds=(0, safe_cap(x, belief, cfg)),
                              method='bounded', options={'xatol':1e-12})
    assert abs(u-numeric.x) < 1e-7
    assert abs(value-integral_cost(u)) < 1e-12


@pytest.mark.parametrize('u,width,noise', [(0,.4,.025),(.4,.4,.025),(.01,.4,.1),(.7,0,.025),(.3,.4,0),(0,.4,0)])
def test_observation_quadrature_normalizes_and_preserves_moments(u,width,noise):
    cfg = ControlConfig(wbar=noise)
    belief = Belief(.2, .2+width)
    x = .2
    states, weights, lows, highs = observation_quadrature(x,u,belief,cfg,order=16)
    mean = cfg.a*x+u*belief.mean
    assert abs(weights.sum()-1) < 1e-12
    assert abs(weights@states-mean) < 1e-12
    expected_var = u*u*belief.variance+noise**2/3
    assert abs(weights@((states-mean)**2)-expected_var) < 1e-12
    assert abs(weights@((lows+highs)/2)-belief.mean) < 1e-12
    # Law of total second moment validates posterior weights, not only density.
    assert abs(weights@((lows*lows+lows*highs+highs*highs)/3)
               -(belief.variance+belief.mean**2)) < 1e-12


def test_frozen_cost_is_evaluated_under_actual_posterior():
    cfg = ControlConfig()
    belief, x, first, ref = Belief(.12,.6), .2, .35, .55
    summary = evaluate_two_step_action(first,x,belief,cfg=cfg,reference_next=ref,quadrature_order=64)
    # Independent joint b,w Monte Carlo; draws fixed in advance.
    rng = np.random.default_rng(352)
    b = rng.uniform(belief.low,belief.high,100000)
    w = rng.uniform(-cfg.wbar,cfg.wbar,len(b))
    states = cfg.a*x+b*first+w
    second = np.array([bayes_one_step(float(s),belief,first,cfg,ref)[0] for s in states])
    conditional_real_cost = ((cfg.a*states+b*second-ref)**2+cfg.wbar**2/3
                            +cfg.effort*second**2+cfg.slew*(second-first)**2)
    expected = expected_stage_cost(x,first,belief,cfg=cfg)+conditional_real_cost.mean()
    assert abs(summary['frozen_cost']-expected) < 5e-4
    assert summary['minimum_conditional_benefit'] >= -1e-12
    assert summary['information_benefit'] > 0


def test_zero_information_cases_and_finite_grid_dominance():
    cfg = ControlConfig()
    for belief, first in [(Belief(.3,.3),.4), (Belief(.12,.6),0.)]:
        values = evaluate_two_step_action(first,.2,belief,cfg=cfg)
        assert abs(values['adapted_cost']-values['frozen_cost']) < 1e-12
    belief = Belief(.12,.6)
    _, informed = two_step_action(.2,belief,cfg=cfg,grid_size=41,reference_now=.2,reference_next=.7)
    _, frozen = two_step_action(.2,belief,cfg=cfg,grid_size=41,reference_now=.2,reference_next=.7,frozen_belief=True)
    assert informed['minimum_value'] <= frozen['minimum_value']+1e-12
    assert informed['max_weight_error'] < 1e-12


def test_strict_conditional_benefit_and_quadrature_convergence():
    cfg, belief = ControlConfig(), Belief(.12,.6)
    values = [evaluate_two_step_action(.4,.2,belief,cfg=cfg,reference_next=.5,
                                       quadrature_order=q) for q in [16,32,64]]
    assert values[-1]['information_benefit'] > 1e-5
    assert abs(values[-1]['adapted_cost']-values[-2]['adapted_cost']) < 5e-6
    # This checks a named numerical instance, not a universal convergence rate.


def test_analytic_strict_benefit_certificate():
    cfg = ControlConfig(a=0., wbar=.025, effort=.04, slew=0.)
    belief = Belief(.2, .6)
    frozen, _ = bayes_one_step(0., belief, cfg=cfg, reference=.2)
    assert abs(frozen-.375) < 1e-12
    # On the analytic event z in (.075,.085), posterior high is below .22.
    # Its probability is .005, posterior curvature >=.08, adapted u > .49.
    lower_bound = .005*.08*(.49-.375)**2
    result = evaluate_two_step_action(.5, 0., belief, cfg=cfg,
                                      reference_next=.2, quadrature_order=64)
    assert result['information_benefit'] > lower_bound


def test_stochastic_constant_excitation_width_formula():
    rng = np.random.default_rng(6129)
    noise, u, n = .025, .4, 20
    draws = rng.uniform(-noise, noise, size=(40000, n))
    widths = (2*noise-draws.max(axis=1)+draws.min(axis=1))/u
    exact = 4*noise/(u*(n+1))
    assert abs(widths.mean()-exact) < 5*widths.std(ddof=1)/np.sqrt(len(widths))


def test_indistinguishability_for_history_dependent_feedback():
    cfg = ControlConfig()
    b1, b2 = .3, .35
    x1 = x2 = .2
    prior = Belief(.12,.6)
    post1 = post2 = prior
    for t in range(30):
        # Nonconstant feedback: identical past states induce identical actions.
        u1 = min(cfg.umax, max(0., .7-.2*x1+.05*np.sin(t)))
        u2 = min(cfg.umax, max(0., .7-.2*x2+.05*np.sin(t)))
        w1, w2 = indistinguishable_disturbances(u1,b1,b2,cfg.wbar)
        y1, y2 = cfg.a*x1+b1*u1+w1, cfg.a*x2+b2*u2+w2
        assert abs(y1-y2) < 1e-12
        post1 = update_belief(x1,u1,y1,post1,cfg)
        post2 = update_belief(x2,u2,y2,post2,cfg)
        assert post1.low <= b1+1e-12 and post1.high >= b2-1e-12
        assert abs(post1.low-post2.low) < 1e-12
        x1,x2 = y1,y2
    assert post1.high-post1.low >= b2-b1-1e-12
    with pytest.raises(ValueError):
        indistinguishable_disturbances(cfg.umax,.2,.5,cfg.wbar)


def test_invalid_states_and_beliefs_fail_explicitly():
    with pytest.raises(ValueError):
        Belief(.3,.2)
    with pytest.raises(ValueError):
        two_step_action(2.,Belief(.12,.6))
    with pytest.raises(ValueError):
        evaluate_two_step_action(2.,.2,Belief(.12,.6))
    with pytest.raises(ValueError):
        observation_quadrature(.2,.3,Belief(.12,.6),order=1)
