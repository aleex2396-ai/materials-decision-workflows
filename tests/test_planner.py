import json

import numpy as np
import pytest

from materials_workflow.planner import _front, _improvement, recommend_batch


def design(cure, route="cast"):
    return {"a": 0.2, "b": 0.3, "c": 0.5, "cure": cure, "route": route}


def observation(run, group, cure, transmission, toughness):
    return {"run_id": run, "group_id": group, "design": design(cure),
            "targets": {"transmission": transmission, "toughness": toughness}}


def make(run, group, cure, success):
    return {"run_id": run, "group_id": group, "design": design(cure), "success": success}


def candidate(ident, cure, cost=1., **kwargs):
    return {"id": ident, "cost": cost, "design": design(cure), **kwargs}


class GivenDraws:
    """Deliberately return known paired outcomes for hand-computable decision tests."""

    def __init__(self, predictions=None):
        self.predictions = predictions or {}

    def predict(self, design, descriptors=None):
        pair = self.predictions.get(design["cure"], [(.8, 12.), (.85, 13.)])
        status = "extrapolation" if design["route"] == "press" else "in_support"
        return {"draws": dict(zip(("transmission", "toughness"), zip(*pair))),
                "applicability": {"status": status, "reasons": []}}


def plan(model, candidates, observed, made=(), pending=(), budget=3., batch_size=2):
    return recommend_batch(model, candidates, observed, made, pending,
                           budget=budget, batch_size=batch_size, seed=4)


def test_monte_carlo_area_and_negative_coordinates_match_rectangles():
    frontier = _front([(1., 2.), (2., 1.), (1., 0.5), (1., 2.)])
    draws = np.array([[1.5, 1.5], [3., 3.], [2., 1.], [-1., 15.], [10., -1.]])
    assert _improvement(draws, frontier) == pytest.approx([.25, 6., 0., 0., 0.])


def test_frontier_is_group_balanced_and_dominated_observations_have_zero_gain():
    observed = [observation("x1", "g1", 0., .7, 6.), observation("x2", "g1", 0., .7, 6.),
                observation("x3", "g2", 0., .9, 8.), observation("y", "g3", 1., .75, 8.)]
    result = plan(GivenDraws({.4: [(.75, 6.)]}), [candidate("new", .4)], observed)
    assert result["frontier"][0]["targets"] == pytest.approx({"transmission": .8, "toughness": 7.})
    assert result["frontier"][0]["independent_groups"] == 2
    assert len(result["frontier"]) == 2
    assert not result["actions"] and result["stop_reason"] == "no_positive_marginal_utility"
    json.dumps(result, allow_nan=False)


def test_failures_and_cost_change_ranking_without_corrupting_performance_front():
    observed = [observation("base", "g0", .5, .48, 6.)]
    evidence = [make("fail" + str(i), str(i), .05, False) for i in range(7)]
    evidence += [make("ok", "g8", .96, True)]
    candidates = [candidate("uncertain", .07), candidate("likely", .94)]
    res = plan(GivenDraws(), candidates, observed, evidence, batch_size=1)
    assert res["actions"][0]["id"] == "likely"
    assert len(res["frontier"]) == 1
    cheaper = [candidate("uncertain", .07, .01), candidate("likely", .94)]
    assert plan(GivenDraws(), cheaper, observed, evidence, batch_size=1)["actions"][0]["id"] == "uncertain"
    clustered = [make("rep" + str(i), "single_group", .05, False) for i in range(8)]
    result = plan(GivenDraws(), [candidate("near", .06)], observed, clustered)
    terms = result["actions"][0]["score_terms"]
    assert terms["effective_feasibility_groups"] <= 1.
    assert terms["makeability_probability"] >= 1 / 3


def test_batch_accounts_for_pending_tradeoff_and_nearby_designs_and_budget():
    observed = [observation("base", "g1", .1, .5, 6.)]
    candidates = [candidate("pending-copy", .2), candidate("unneeded", .25), candidate("better", .7, .2),
                  candidate("alias", .7, .3), candidate("unaffordable", .49, 3.)]
    pred = GivenDraws({.2: [(.8, 11.)], .25: [(.8, 11.)], .7: [(.75, 14.)]})
    res = plan(pred, candidates, observed, pending=[candidate("busy", .2)], budget=.4)
    assert [x["id"] for x in res["actions"]] == ["better"]
    assert res["spent"] == .2
    reasons = {(t["id"], t.get("reason")) for t in res["trace"]}
    assert {("pending-copy", "pending_design_or_id"), ("alias", "batch_duplicate"),
            ("unaffordable", "over_budget")} <= reasons
    terms = res["actions"][0]["score_terms"]
    assert 0 < terms["diversity"] < 1
    assert np.isclose(terms["nearest_pending_or_selected"], .5)


def test_explicit_repeats_update_existing_frontier_mean():
    measured = [observation("original", "g1", .3, .7, 10.)]
    candidates = [candidate("implicit", .3), candidate("confirm", .3, repeat_of="original"),
                  candidate("mismatch", .4, repeat_of="original")]
    res = plan(GivenDraws({.3: [(.3, 4.), (.8, 11.)]}), candidates, measured)
    assert [x["action"] for x in res["actions"]] == ["independent_repeat"]
    assert res["actions"][0]["fresh_preparation_required"]
    assert res["actions"][0]["score_terms"]["expected_area_utility"] > 0
    reasons = {t["id"]: t["reason"] for t in res["trace"] if "reason" in t}
    assert reasons["implicit"] == "already_attempted_without_repeat_request"
    assert reasons["mismatch"] == "repeat_requires_matched_measured_run"
    # A repeat matching the current mean exactly cannot resolve uncertainty in this test.
    result = plan(GivenDraws({.3: [(.7, 10.)]}), candidates, measured)
    assert result["actions"] == []


def test_invalid_pending_duplicates_out_of_support_no_reference_and_wrong_feasibility_labels():
    obs = [observation("base", "g0", .2, .55, 7.)]
    cases = [candidate("overflow", 1.2), candidate("negative", .9, -1.),
             {"id": "unknown-route", "design": design(.5, "press"), "cost": 1.},
             candidate("base", .4)]
    res = plan(GivenDraws(), cases, obs)
    assert res["actions"] == []
    assert {t["reason"] for t in res["trace"]} >= {"outside_model_support", "used_run_id"}
    assert plan(GivenDraws(), cases, [])["stop_reason"] == "no_measured_reference"
    with pytest.raises(ValueError, match="distinct"):
        plan(GivenDraws(), [candidate("same", .5), candidate("same", .6)], obs)
    with pytest.raises(ValueError, match="adjudicated"):
        plan(GivenDraws(), [], obs, [make("wrong", "g1", .1, "failed")])
