from copy import deepcopy
from dataclasses import replace
import json

import pytest

from materials_workflow.shared_budget import SharedBudgetPlanner


def make(ident, recipe, cost=2, likelihoods=None):
    return {"id": ident, "kind": "make", "recipe": recipe, "cost": cost,
            "specimen": "coupon-" + ident, "success_outcomes": ["made"],
            "likelihoods": likelihoods or {"made": [1, 1]}}


def assay(ident="assay", cost=1):
    return {"id": ident, "kind": "assay", "cost": cost,
            "likelihoods": {"low": [0.9, 0.1], "high": [0.1, 0.9]}}


def short_campaign():
    return {"prior": [0.6, 0.4], "budget": 3, "utility_per_credit": 0.02,
            "utilities": {"A": [1, 0], "B": [0, 1]},
            "actions": [assay(), make("make-A", "A"), make("make-B", "B")]}


def test_assay_competes_with_manufacture_and_drives_the_observed_choice():
    planner = SharedBudgetPlanner(short_campaign())
    start = planner.initial_state()
    choice = planner.select(start)
    # With a one-credit assay and a two-credit make: P(correct)=.6*.9+.4*.9.
    assert choice["action_id"] == "assay"
    assert choice["expected_value"] == pytest.approx(.9 - .02*3)
    assert choice["outcomes"] == pytest.approx({"low": .58, "high": .42})
    assert choice["terminal_recipe"] is None
    values = {r["action_id"]: r["expected_value"] for r in choice["alternatives"]}
    assert values["make-A"] == pytest.approx(.6 - .02*2)
    assert values["make-B"] == pytest.approx(.4 - .02*2)
    for label, make_id, posterior in (("low", "make-A", [.54/.58, .04/.58]),
                                      ("high", "make-B", [.06/.42, .36/.42])):
        after = planner.observe(start, "assay", label)
        assert planner.belief(after) == pytest.approx(posterior)
        assert after.remaining == 2
        assert planner.select(after)["action_id"] == make_id
        done = planner.observe(after, make_id, "made")
        assert done.remaining == 0
        assert done.qualified == {make_id[-1]}
        assert done.specimens == {"coupon-" + make_id}
        assert planner.select(done)["action_id"] == "stop"
        assert planner.select(done)["terminal_recipe"] == make_id[-1]
    assert start.remaining == 3 and start.history == ()
    json.dumps(choice, allow_nan=False)


def test_unaffordable_assay_followed_by_manufacture_is_never_bought():
    spec = short_campaign()
    spec["budget"] = 2
    planner = SharedBudgetPlanner(spec)
    first = planner.select(planner.initial_state())
    assert first["action_id"] == "make-A"
    expensive = short_campaign()
    expensive["utility_per_credit"] = .4
    planner = SharedBudgetPlanner(expensive)
    assert planner.select(planner.initial_state())["kind"] == "stop"


@pytest.mark.parametrize("kind", ["reread", "remake"])
def test_confirmation_can_win_if_it_changes_the_best_reproducible_recipe(kind):
    spec = short_campaign()
    probe = dict(assay("verify"), kind=kind, recipe="A", specimen="old-A")
    if kind == "remake":
        probe.update(specimen="fresh-A", requires_specimens=["old-A"], success_outcomes=["low", "high"])
    spec.update(budget=1, actions=[probe], initial_qualified=["A", "B"], initial_specimens=["old-A"])
    planner = SharedBudgetPlanner(spec)
    start = planner.initial_state()
    assert planner.select(start)["terminal_recipe"] == "A"
    assert planner.select(start)["kind"] == kind
    after = planner.observe(start, "verify", "high")
    assert planner.select(after)["terminal_recipe"] == "B"
    assert after.qualified == {"A", "B"}
    assert after.specimens == ({"old-A", "fresh-A"} if kind == "remake" else {"old-A"})


def test_failed_make_spends_budget_without_creating_reread_or_qualification():
    first = make("trial", "A", 1, {"made": [.8, .2], "failed": [.2, .8]})
    retry = dict(make("repeat", "A", 1), kind="remake")
    read = dict(assay("read"), kind="reread", recipe="A", specimen="coupon-trial")
    planner = SharedBudgetPlanner(dict(short_campaign(), actions=[first, retry, read]))
    start = planner.initial_state()
    assert planner.select(start)["ineligible"] == {"read": "missing_specimen", "repeat": "no_qualified_preparation"}
    failed = planner.observe(start, "trial", "failed")
    assert failed.remaining == 2 and len(failed.history) == 1
    assert not failed.qualified and not failed.specimens
    assert planner.belief(failed) == pytest.approx([.12/.44, .32/.44])
    assert planner.select(failed)["action_id"] == "stop"
    with pytest.raises(ValueError, match="already_executed"):
        planner.observe(failed, "trial", "made")
    passed = planner.observe(start, "trial", "made")
    assert {r["action_id"] for r in planner.select(passed)["alternatives"]} == {"read", "repeat", "stop"}


def test_shared_latent_bias_prevents_rereading_identical_evidence_for_extra_information():
    first = dict(assay("first"), likelihoods={"low": [1, 0], "high": [0, 1]})
    second = dict(first, id="second", kind="reread", specimen="old", recipe="A")
    spec = dict(short_campaign(), actions=[first, second], initial_specimens=["old"],
                initial_qualified=["A", "B"], utility_per_credit=0)
    planner = SharedBudgetPlanner(spec)
    after = planner.observe(planner.initial_state(), "first", "low")
    assert planner.select(after)["action_id"] == "stop"
    with pytest.raises(ValueError, match="zero probability"):
        planner.observe(after, "second", "high")
    assert planner.belief(after) == (1., 0.)


def test_ties_and_public_state_integrity():
    spec = short_campaign()
    spec.update(actions=[assay("z",2), assay("y"), assay("a")], initial_qualified=["A", "B"],
                budget=2, utility_per_credit=0)
    # Each probe would reveal the lot perfectly. Equal reward favors the
    # cheaper probe, with ID as a final tie-break.
    spec["actions"] = [dict(a, likelihoods={"low": [1, 0], "high": [0, 1]}) for a in spec["actions"]]
    planner = SharedBudgetPlanner(spec)
    assert planner.select(planner.initial_state())["action_id"] == "a"
    different_order = dict(spec, actions=list(reversed(spec["actions"])))
    other = SharedBudgetPlanner(different_order)
    assert planner.select(planner.initial_state()) == other.select(other.initial_state())
    with pytest.raises(ValueError, match="resources"):
        planner.select(replace(planner.initial_state(), remaining=100))
    with pytest.raises(ValueError, match="Unknown outcome"):
        planner.observe(planner.initial_state(), "a", "undeclared")


def test_invalid_cost_model_and_sample_reuse_are_rejected():
    spec = short_campaign()
    invalid = deepcopy(spec)
    invalid["actions"][0]["likelihoods"]["high"][0] = .6
    with pytest.raises(ValueError, match="sum to one"):
        SharedBudgetPlanner(invalid)
    invalid = deepcopy(spec)
    invalid["actions"][0]["cost"] = 0
    with pytest.raises(ValueError, match="integer"):
        SharedBudgetPlanner(invalid)
    invalid = deepcopy(spec)
    invalid["actions"][2]["specimen"] = invalid["actions"][1]["specimen"]
    with pytest.raises(ValueError, match="unique specimen"):
        SharedBudgetPlanner(invalid)
