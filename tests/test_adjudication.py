"""Selective risk, synthetic held-out scoring and review-case interface."""
import json

import numpy as np
import pytest
from scipy.stats import binom

from materials_workflow.adjudication import (
    ACTIONS, ALPHA, GRID, _calibrate, _metrics, _policy, _upper_bound, run_shadow_study,
)


def test_one_sided_bounds_match_inverted_binomial_cdf_and_empty_cases():
    tail = ALPHA / len(GRID)
    for errors, n in [(0, 54), (4, 40), (49, 50)]:
        assert binom.cdf(errors, n, _upper_bound(errors, n, tail)) == pytest.approx(tail)
    assert _upper_bound(0, 0, tail) == _upper_bound(5, 5, tail) == 1


def test_none_pass_defer_all_instead_of_relaxing_risk_limit():
    count = 70
    scores = (np.ones(count, dtype=int), np.ones(count), np.ones(count, dtype=bool))
    selected = _calibrate(scores, np.zeros(count, dtype=int))
    assert selected["selected_threshold"] is None
    assert all(_policy(scores, None) == "ask")


def test_shared_loss_counts_and_deferrals():
    actions = np.array(["use", "remeasure", "ask", "remake", "use", "remeasure", "ask"])
    truth = np.array([0, 0, 0, 1, 2, 1, 3])
    result = _metrics(actions, truth, np.ones(len(truth), dtype=bool))
    assert result["groups"] == 7 and result["settled"] == 5
    assert result["selective_risk"] == pytest.approx(3 / 5)
    assert result["harmful_admissions"] == result["harmful_exclusions"] == result["incorrect_remedies"] == 1
    assert result["valid_poor_groups"] == 3 and result["valid_poor_admitted"] == 1
    assert sum(result["by_cause"]["valid"]["actions"].values()) == 3


def test_known_diagnostics_fit_selective_policy_and_evaluation_is_paired():
    cases = {
        "standard": [3.2, 0.6, 2.2], "process": [0.5, 3.2, 1.8], "valid": [0.5, 0.4, 0.6],
        "mixed": [2.8, 2.7, 2.3], "novel": [12, 17, 9],
        "missing": [None, 1.0, 2.0], "text": [1.2, "1.5", 1.0], "nan": [float("nan"), 0, 1],
    }
    report = run_shadow_study(39, cases)
    decisions = report["decisions"]
    assert {key: decisions[key]["action"] for key in cases} == {
        "standard": "remeasure", "process": "remake", "valid": "use", "mixed": "ask",
        "novel": "ask", "missing": "ask", "text": "ask", "nan": "ask",
    }
    assert decisions["novel"]["confidence"] is None
    calibration = report["calibration"]
    passing = [row for row in calibration["thresholds"] if row["passes"]]
    winner = max(passing, key=lambda row: (row["coverage"], row["threshold"]))
    assert calibration["selected_threshold"] == winner["threshold"]
    assert winner["risk_upper"] <= calibration["risk_limit"]
    for distribution in ("iid", "shifted"):
        comparison = report["evaluation"][distribution]
        assert comparison["model"]["groups"] == comparison["baseline"]["groups"] == 1500
        for name in comparison["model"]["by_cause"]:
            assert comparison["model"]["by_cause"][name]["groups"] == comparison["baseline"]["by_cause"][name]["groups"]
        for metrics in comparison.values():
            assert metrics["settled"] + metrics["asks"] == metrics["groups"]
            assert sum(metrics[key] for key in ("harmful_admissions", "harmful_exclusions", "incorrect_remedies")) == metrics["settled_errors"]
    assert report["evaluation"]["shifted"]["model"]["selective_risk"] > calibration["risk_limit"]
    assert json.dumps(report, allow_nan=False) == json.dumps(run_shadow_study(39, cases), allow_nan=False)


def test_mixed_predictions_and_out_of_support_always_defer():
    scores = (np.array([0, 1, 2, 3, 0]), np.full(5, 0.99),
              np.array([True, True, True, True, False]))
    assert _policy(scores, 0.95).tolist() == [*ACTIONS[:3], "ask", "ask"]
