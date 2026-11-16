import json

import pytest

from materials_workflow.cold_start import cold_start_batch


def design(a, b, cure=0.5, route="cast"):
    return dict(a=a, b=b, c=1-a-b, cure=cure, route=route)


def candidate(ident, d, cost=1.0, **extras):
    return dict(id=ident, design=d, cost=cost, **extras)


def plan(candidates, training=(), feasibility=(), pending=(), budget=5., size=2):
    return cold_start_batch(candidates, training, feasibility, pending,
                            budget=budget, batch_size=size, seed=97)


def test_empty_history_starts_near_centre_then_covers_distant_route_and_composition():
    centre = design(1/3, 1/3)
    library = [candidate("centre", centre), candidate("near", design(.35, .33)),
               candidate("far", design(1., 0., cure=1., route="press"))]
    result = plan(library)
    assert [a["id"] for a in result["actions"]] == ["centre", "far"]
    assert [a["score_terms"]["anchor_designs"] for a in result["actions"]] == [0, 1]
    assert result["method"] == "cold_start_space_filling"
    assert result["actions"] == plan(reversed(library))["actions"]
    assert result["spent"] == 2.0 and result["remaining_budget"] == 3.0
    json.dumps(result, allow_nan=False)


def test_pending_and_failed_history_cannot_be_reissued_and_pending_is_already_paid():
    d0, d1, d2 = design(1., 0.), design(0., 1.), design(0., 0.)
    pending = [candidate("busy", d0, cost=200)]
    failed = [dict(run_id="attempt", group_id="old", design=d1, success=False)]
    library = [candidate("busy-alias", d0), candidate("failed-alias", d1),
               candidate("expensive", design(.1, .2), cost=.6), candidate("cheap", d2, cost=.4),
               candidate("duplicate", d2, cost=.5), candidate("usedid", d1, repeat_of="attempt")]
    result = plan(library, feasibility=failed, pending=pending, budget=.5)
    assert [a["id"] for a in result["actions"]] == ["cheap"]
    assert result["remaining_budget"] == .1
    rejected = {t["id"]: t["reason"] for t in result["trace"] if "reason" in t}
    assert rejected["busy-alias"] == "pending_design_or_id"
    assert rejected["failed-alias"] == "already_attempted_without_repeat_request"
    assert rejected["duplicate"] == "batch_duplicate"
    assert rejected["expensive"] == "over_budget"
    assert rejected["usedid"] == "repeat_requires_matched_measured_run"


def test_measured_and_pending_positions_are_anchors_independent_of_outcome_magnitude():
    old = design(.15, .30)
    observed = [dict(run_id="r", group_id="g", design=old, targets=dict(transmission=.2, toughness=3.))]
    library = [candidate("repeat", old, repeat_of="r"), candidate("adjacent", design(.16, .31)),
               candidate("far", design(.4, .50, route="press")), candidate("r", design(.25, .4))]
    result = plan(library, training=observed, size=1)
    assert result["actions"][0]["id"] == "far"
    rejected = {t["id"]: t["reason"] for t in result["trace"] if "reason" in t}
    assert rejected == {"repeat": "confirmation_is_not_a_new_design", "r": "used_run_id"}
    observed[0]["targets"] = dict(transmission=.95, toughness=77.)
    assert plan(library, training=observed, size=1) == result


def test_bad_library_and_empty_affordable_set():
    malformed = candidate("bad-mix", dict(a=.8, b=.6, c=0., cure=.5, route="cast"))
    result = plan([malformed, candidate("too-costly", design(.5, .5), 9.)])
    assert not result["actions"] and result["stop_reason"] == "no_eligible_affordable_design"
    leaked = candidate("leaky", design(.4, .4), descriptors={"toughness": 1})
    unusable = candidate("bad-datum", design(.3, .3), descriptors={"density": -3.})
    result = plan([leaked, unusable, candidate("valid", design(.4, .3))])
    assert [row["id"] for row in result["actions"]] == ["valid"]
    with pytest.raises(ValueError, match="distinct"):
        plan([malformed, malformed])
    with pytest.raises(ValueError, match="nonnegative"):
        plan([], budget=-1.)
