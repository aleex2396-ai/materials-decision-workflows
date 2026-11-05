import numpy as np
import pytest
from crfml.control import ControlConfig, GainSet, prediction_matrix, solve_mpc, safe_probe, rollout


def test_analytic_upper_envelope_is_attained_at_corner():
    cfg = ControlConfig()
    u = np.linspace(.01,.4,cfg.horizon)
    x = .2
    true_states = []
    for action in u:
        x = cfg.a*x+cfg.bmax*action+cfg.wbar
        true_states.append(x)
    s = prediction_matrix(cfg)
    envelope = cfg.a**np.arange(1,cfg.horizon+1)*.2+cfg.bmax*s@u+cfg.wbar*s.sum(1)
    np.testing.assert_allclose(envelope,true_states,atol=1e-12)


def test_safe_probe_and_backup_at_extreme_parameters():
    cfg = ControlConfig()
    for x in np.linspace(-.3,1,21):
        u = safe_probe(x,cfg.bmax,cfg)
        assert cfg.a*x+cfg.bmax*u+cfg.wbar <= cfg.xmax+1e-12
        assert cfg.a*x+cfg.wbar <= cfg.xmax+1e-12
    with pytest.raises(ValueError):
        ControlConfig(wbar=.2)


def test_set_membership_contains_truth_and_contraction_bound():
    cfg = ControlConfig()
    gain = GainSet(.12,.6)
    b, x, u = .31,.4,.7
    assert gain.update(x,u,cfg.a*x+b*u+.01,cfg)
    assert gain.low <= b <= gain.high
    assert gain.high-gain.low <= 2*cfg.wbar/u+1e-12
    old = (gain.low,gain.high)
    assert gain.update(x,0,cfg.a*x,cfg)
    assert (gain.low,gain.high) == old
    assert not gain.update(x,u,100.,cfg)
    assert not gain.valid


def test_robust_rollouts_safe_at_adversarial_bound():
    cfg = ControlConfig()
    for b in [cfg.bmin,cfg.bmax,.33]:
        for policy in ['robust_fixed','robust_adaptive','safe_probe','constant_center']:
            result = rollout(b,.2,np.full(20,cfg.wbar),policy,cfg)
            assert result['violations'] == 0
            assert result['assumption_failures'] == 0
            assert not result['detected_failure']


def test_qp_feasible_and_better_than_feasible_zero_control():
    cfg = ControlConfig()
    u, info = solve_mpc(.2,.3,.6,cfg)
    assert info['solver_success'] and not info['fallback']
    assert 0 < u <= cfg.umax
    assert max(info['upper_trajectory']) <= cfg.xmax+1e-7
    s = prediction_matrix(cfg)
    plan = np.array(info['plan'])
    base = cfg.a**np.arange(1,cfg.horizon+1)*.2
    objective = np.sum((base+.3*s@plan-cfg.reference)**2)+cfg.effort*np.sum(plan**2)+cfg.slew*np.sum(np.diff(np.r_[0.,plan])**2)
    assert objective < np.sum((base-cfg.reference)**2)


def test_broken_bounds_are_visible_not_claimed_safe():
    result = rollout(.9,.2,np.full(20,.2),'safe_probe')
    assert result['assumption_failures'] > 0
    assert result['detected_failure']
    assert any(not t['conditional_certificate'] for t in result['trajectory'])


def test_nonfinite_measurement_invalidates_certificate():
    gain = GainSet(.12,.6)
    with pytest.raises(ValueError):
        gain.update(.2,.4,float('nan'),ControlConfig())
    assert not gain.valid
    for interval in [(float('nan'),.6),(.6,.1),(-.1,.6)]:
        with pytest.raises(ValueError):
            GainSet(*interval)


def test_one_step_qp_matches_independent_closed_form():
    cfg = ControlConfig(horizon=1)
    rng = np.random.default_rng(721)
    for _ in range(25):
        x, b, upper, last_u = rng.uniform(.1,.95), .3, .6, rng.uniform(0,.8)
        unconstrained = (b*(cfg.reference-cfg.a*x)+cfg.slew*last_u)/(b*b+cfg.effort+cfg.slew)
        cap = min(cfg.umax,(cfg.xmax-cfg.a*x-cfg.wbar)/upper)
        expected = np.clip(unconstrained,0,cap)
        actual, info = solve_mpc(x,b,upper,cfg,last_u)
        assert not info['fallback']
        assert actual == pytest.approx(expected,abs=1e-8)


def test_known_gain_reference_has_no_parameter_uncertainty():
    result = rollout(.31,.25,np.zeros(5),'oracle')
    assert result['final_width'] == 0
    assert result['assumption_failures'] == 0
