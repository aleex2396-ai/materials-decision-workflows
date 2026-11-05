"""Factorial controller ablations and complete discovery-budget accounting."""
import numpy as np
import pytest

from crfml.control import ControlConfig, rollout
from crfml.discovery import discover
from crfml.simulator import sample


@pytest.mark.parametrize("true_gain", [.18, .55])
def test_adaptive_constant_center_matches_midpoint_adaptive_control(true_gain):
    cfg = ControlConfig(horizon=3)
    disturbances = np.random.default_rng(513).uniform(-cfg.wbar, cfg.wbar, 16)
    midpoint = (cfg.bmin + cfg.bmax) / 2
    expected = rollout(true_gain, midpoint, disturbances, "robust_adaptive", cfg)
    actual = rollout(true_gain, .27, disturbances, "adaptive_constant_center", cfg)
    assert actual["policy"] == "adaptive_constant_center"
    # Equal trajectories include input, cost, set updates and certificate state.
    assert actual["trajectory"] == expected["trajectory"]
    for field in ("total_cost", "final_width", "violations", "assumption_failures",
                  "detected_failure", "solver_fallbacks"):
        assert actual[field] == expected[field]
    assert actual["final_width"] < cfg.bmax - cfg.bmin
    assert actual["violations"] == 0
    # The ablation removes the learned center without changing adaptation.
    different_learned_gain = rollout(true_gain, .58, disturbances, "adaptive_constant_center", cfg)
    assert different_learned_gain["trajectory"] == actual["trajectory"]


def test_adaptive_constant_center_has_same_failure_response():
    cfg = ControlConfig(horizon=3)
    midpoint = (cfg.bmin + cfg.bmax) / 2
    disturbances = np.full(10, .15)
    expected = rollout(.9, midpoint, disturbances, "robust_adaptive", cfg)
    actual = rollout(.9, .2, disturbances, "adaptive_constant_center", cfg)
    assert actual["trajectory"] == expected["trajectory"]
    assert actual["detected_failure"]
    assert actual["assumption_failures"] > 0


def test_measured_gain_is_the_queried_observation_and_both_centers_run():
    seed, pool_size = 17, 36
    result = discover(seed, "random", steps=2, pool_size=pool_size)
    chosen = result["recommendation"]
    assert chosen in result["selected"]
    pool = sample(10000 + seed, pool_size, noise=False)
    noise = np.random.default_rng(20000 + seed)
    noise.normal(0, .025, pool_size)
    noise.normal(0, .01, pool_size)
    measured = pool.gain + noise.normal(0, .006, pool_size)
    assert result["observed_gain"] == pytest.approx(measured[chosen], abs=1e-14)
    assert result["observed_gain"] != result["true_gain"]
    cfg = ControlConfig(horizon=3)
    for field in ("predicted_gain", "observed_gain"):
        assert np.isfinite(result[field]) and result[field] > 0
        diagnostic = rollout(result["true_gain"], result[field], np.zeros(5), "robust_adaptive", cfg)
        assert np.isfinite(diagnostic["total_cost"])
        assert diagnostic["violations"] == 0


def test_discovery_violations_include_initial_designs_without_changing_old_endpoint():
    seed, pool_size, steps = 19, 40, 3
    result = discover(seed, "random", steps=steps, pool_size=pool_size)
    pool = sample(10000 + seed, pool_size, noise=False)
    chosen = result["selected"]
    assert len(chosen) == 10 + steps
    assert result["initial_constraint_violations"] == int(np.sum(pool.defect[chosen[:10]] > .22))
    assert result["sampled_constraint_violations"] == sum(row["defect_violation"] for row in result["trace"])
    assert result["total_sampled_constraint_violations"] == int(np.sum(pool.defect[chosen] > .22))
    assert result["total_sampled_constraint_violations"] == (
        result["initial_constraint_violations"] + result["sampled_constraint_violations"])
