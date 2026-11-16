"""Physical fork, specimen lineage and failure semantics for the public synthetic lab."""

import hashlib
import json

import pytest

from materials_workflow.registry import verify_preparation_return
from materials_workflow.shared_budget import SharedBudgetPlanner
from materials_workflow.shared_case import CaseLab, build_case


@pytest.fixture(scope="module")
def case():
    return build_case({"seed": 5837, "budget": 5, "utility_per_credit": 0.003})


def test_only_purchased_stock_result_forks_the_next_physical_make(case):
    spec, context = case
    policy = SharedBudgetPlanner(spec)
    menu = {a["id"]: a for a in spec["actions"]}
    first = policy.select(policy.initial_state())
    assert first["action_id"] == "assay-stock"
    assert first["ineligible"] == {}
    assert {row["kind"] for row in first["alternatives"]} == {"assay", "make", "remake", "reread", "stop"}
    direct = max((row for row in first["alternatives"] if row["kind"] == "make"),
                 key=lambda row: row["expected_value"])
    actual, meter_noise = {}, []
    for world, expected in zip(context["demonstration_worlds"], ("binder-rich", "binder-lean"), strict=True):
        lab = CaseLab(context, world, seed=17)
        try:
            state = policy.initial_state()
            band, stock = lab.execute(menu[first["action_id"]])
            assert stock["recipe"] is None
            assert band == ("low" if stock["measurement"]["value"] < 4 else "high")
            meter_noise.append(stock["measurement"]["value"] - (2 if band == "low" else 6))
            state = policy.observe(state, first["action_id"], band)
            assert state.remaining == 4
            next_action = policy.select(state)
            assert next_action["kind"] == "make"
            observation, made = lab.execute(menu[next_action["action_id"]])
            assert made["recipe"] == expected
            assert made["registry_action"] == "use"
            verify_preparation_return(menu[next_action["action_id"]], observation, made)
            raw = json.dumps(made["registry_record"], sort_keys=True, separators=(",", ":"))
            assert made["registry_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
            state = policy.observe(state, next_action["action_id"], observation)
            assert state.remaining == 0
            assert policy.select(state)["terminal_recipe"] == expected
            actual[band] = expected
        finally:
            lab.close()
    assert meter_noise[0] == pytest.approx(meter_noise[1])
    assert actual == {"low": "binder-rich", "high": "binder-lean"}
    assert actual["low"] != direct["recipe"]


def test_old_specimen_and_fresh_remake_have_distinct_preparation_ancestry(case):
    spec, context = case
    menu = {a["id"]: a for a in spec["actions"]}
    lab = CaseLab(context, "low/weak", seed=17)
    try:
        reread_result, original = lab.execute(menu["reread-reference"])
        remake_result, fresh = lab.execute(menu["remake-reference"])
        verify_preparation_return(menu["reread-reference"], reread_result, original)
        verify_preparation_return(menu["remake-reference"], remake_result, fresh)
        assert original["existing_specimen"] and fresh["repeats_preparation"] == original["group_id"]
        assert fresh["specimen_id"] != original["specimen_id"]
        assert fresh["group_id"] != original["group_id"]
        assert fresh["registry_record"]["repeats_preparation"] == original["group_id"]
        with pytest.raises(ValueError, match="already executed"):
            lab.execute(menu["reread-reference"])
    finally:
        lab.close()


def test_failed_new_make_spends_credits_without_qualifying_the_recipe(case):
    spec, context = case
    policy = SharedBudgetPlanner(spec)
    lab = CaseLab(context, "high/weak", seed=41)
    try:
        make = next(a for a in spec["actions"] if a["id"] == "make-binder-rich")
        label, receipt = lab.execute(make)
        assert label == "failed" and receipt["registry_action"] == "feasibility_only"
        verify_preparation_return(make, label, receipt)
        assert receipt["specimen_id"] is None and receipt["targets"] == {}
        state = policy.observe(policy.initial_state(), make["id"], label)
        assert state.remaining == 1 and "binder-rich" not in state.qualified
    finally:
        lab.close()


def test_preparation_requires_trustworthy_evidence_and_matches_the_reservation(case):
    spec, context = case
    make = next(a for a in spec["actions"] if a["id"] == "make-binder-rich")
    lab = CaseLab(context, "low/weak", seed=17)
    try:
        label, receipt = lab.execute(make)
    finally:
        lab.close()
    with pytest.raises(ValueError, match="raw registry_record"):
        verify_preparation_return(make, label, {"registry_action": "use"})
    raw = receipt["registry_record"]
    for changes, error in [
        ({"evidence": dict(raw["evidence"], instrument_check_passed=False)}, "Registry returned remeasure"),
        ({"stock_lot": "another-stock"}, "stock_lot"),
        ({"specimen_id": "another-specimen"}, "specimen_id"),
        ({"design": dict(raw["design"], cure=0.1)}, "design"),
    ]:
        with pytest.raises(ValueError, match=error):
            verify_preparation_return(make, label, {"registry_record": dict(raw, **changes)})
    # The authoritative endpoint can arrive in J/m2. The threshold is in canonical kJ/m2.
    raw = dict(raw, targets=dict(raw["targets"], toughness={"value": 6900, "unit": "J/m2"}))
    verify_preparation_return(make, "below", {"registry_record": raw})
    with pytest.raises(ValueError, match="normalized preparation endpoint"):
        verify_preparation_return(make, "above", {"registry_record": raw})
