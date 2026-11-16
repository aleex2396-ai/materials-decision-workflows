import json

import pytest

from materials_workflow.value_of_information import measure_or_make


class ThresholdStock:
    """Fictional fixture with an analytical two-regime optical utility."""

    def descriptor_estimates(self, supplied):
        def item(key):
            if key in supplied:
                return dict(mean=supplied[key], sd=0, used_in_model=key == "density")
            if key == "density":
                return dict(mean=3, sd=.15, used_in_model=True)
            return dict(mean=None, sd=None, used_in_model=False)
        return {key: item(key) for key in ("density", "scattering", "viscosity")}

    def predict(self, design, supplied):
        if design == "stock" and "density" in supplied:
            t = .8 if supplied["density"] >= 3 else .2
        elif design == "dominates":
            t = .95
        else:
            t = .5
        return {"applicability": {"status": "extrapolation" if design == "outside" else "in_support"},
                "draws": {"transmission": [t, t], "toughness": [5., 5.]}}


def shortlist(other="constant"):
    return [{"id": "A", "design": "stock", "descriptors": {}},
            {"id": "B", "design": other, "descriptors": {"density": 3.0}}]


def test_known_switching_value_and_cost_same_utility_units():
    # Stock utility is 0.35 or 0.65 with equal probability; other utility
    # is 0.50. An ideal assay with a boundary at the median is worth 0.075.
    result = measure_or_make(ThresholdStock(), shortlist(), seed=7, credit_cost=.02)
    assay = result["candidates"][0]["assays"][0]
    assert result["make_now_expected_utility"] == pytest.approx(.5)
    assert assay["expected_gain"] == pytest.approx(.075)
    assert assay["expected_net_value"] == pytest.approx(.055)
    assert assay["switch_probability"] == pytest.approx(.5)
    assert sum(b["probability"] for b in assay["bins"]) == pytest.approx(1)
    assert result["recommended_action"]["action"] == "measure"
    assert result["recommended_action"]["descriptor"] == "density"
    assert result == measure_or_make(ThresholdStock(), list(reversed(shortlist())), seed=7, credit_cost=.02)
    expensive = measure_or_make(ThresholdStock(), shortlist(), seed=7, credit_cost=.08)
    assert expensive["recommended_action"]["action"] == "make"
    assert expensive["candidates"][0]["assays"][0]["expected_gain"] == pytest.approx(.075)
    json.dumps(result, allow_nan=False)


def test_no_value_if_no_switch_possible_even_with_uncertain_property():
    result = measure_or_make(ThresholdStock(), shortlist("dominates"), seed=31)
    assay = result["candidates"][0]["assays"][0]
    assert result["recommended_action"]["candidate_id"] == "B"
    assert result["recommended_action"]["action"] == "make"
    assert assay["expected_gain"] == pytest.approx(0)
    assert assay["switch_probability"] == 0
    assert measure_or_make(ThresholdStock(), shortlist()[:1], seed=31)["candidates"][0]["assays"][0]["expected_gain"] == 0


def test_only_supported_shortlisted_designs_enter_the_decision():
    result = measure_or_make(ThresholdStock(), shortlist("outside"), seed=3)
    assert result["recommended_action"]["candidate_id"] == "A"
    assert result["candidates"][1]["status"] == "extrapolation"
    assert result["candidates"][0]["assays"][0]["fixed_alternative"] is None
    assert measure_or_make(ThresholdStock(), [], seed=3)["recommended_action"]["action"] == "abstain"
    with pytest.raises(ValueError, match="nonnegative"):
        measure_or_make(ThresholdStock(), shortlist(), seed=4, credit_cost=-.1)


def test_assays_share_one_card_prior_and_resolve_correlated_unreported_fields():
    class CorrelatedStock(ThresholdStock):
        def descriptor_estimates(self, supplied):
            result = super().descriptor_estimates(supplied)
            density = result["density"]
            result["scattering"] = dict(mean=2*density["mean"], sd=2*density["sd"], used_in_model=True)
            return result

    result = measure_or_make(CorrelatedStock(), shortlist(), seed=44, credit_cost=0)
    assays = {row["descriptor"]: row for row in result["candidates"][0]["assays"]}
    # A perfect supplier relationship makes measuring either quantity equally valuable.
    for key in ("density", "scattering"):
        assay = assays[key]
        implied_prior = sum(row["conditional_utility"] * row["probability"] for row in assay["bins"])
        assert implied_prior == pytest.approx(result["candidates"][0]["expected_utility"])
        assert assay["expected_gain"] == pytest.approx(.075)
        assert assay["switch_probability"] == pytest.approx(.5)
