import json

import numpy as np
import pytest
from scipy.special import expit

from materials_workflow.formulation import PerformanceModel


def membrane_rows():
    rng = np.random.default_rng(40)
    rows = []
    for i in range(35):
        a, b, c = rng.dirichlet([2, 2, 2])
        cure = rng.uniform(0.15, 0.8)
        route = ("press", "cast")[i % 2]
        stock = 1 + rng.uniform(-0.6, 0.6)
        for j in range(2):
            descriptors = {"density": stock, "scattering": 2*stock, "viscosity": 0.5 + 3*stock}
            rows.append({"run_id": f"coupon-{i}-{j}", "group_id": f"prep-{i}",
                         "design": dict(a=a, b=b, c=c, cure=cure, route=route),
                         "descriptors": descriptors,
                         "targets": {"transmission": expit(6*a - b + 0.5*stock),
                                     "toughness": 1 + 7*b + 3*cure + stock}})
    return rows


def test_whole_group_validation_preprocessing_and_predictive_draws():
    rows = membrane_rows()
    model = PerformanceModel().fit(rows)
    valid = model.diagnostics["validation"]
    held_groups = []
    for fold in valid["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])
        held_groups.extend(fold["validation_groups"])
    assert sorted(held_groups) == sorted({r["group_id"] for r in rows})
    for metrics in valid["targets"].values():
        assert metrics["model_mae"] < metrics["mean_baseline_mae"]
    design = dict(a=1.0, b=0.0, c=0.0, cure=0.99, route="press")
    result = model.predict(design)
    assert result["applicability"]["status"] == "extrapolation"
    transmission = np.array(result["draws"]["transmission"])
    toughness = np.array(result["draws"]["toughness"])
    assert np.all((0 <= transmission) & (transmission <= 1)) and np.all(toughness > 0)
    assert transmission.shape == toughness.shape
    assert result["targets"]["transmission"]["sd"] == pytest.approx(transmission.std(ddof=1))
    assert result == model.predict(design)   # common random numbers for reproducible planning
    assert model.descriptor_estimates() == result["descriptors"]
    previous_hash = model.training_sha256
    model.fit(reversed(rows))
    assert previous_hash == model.training_sha256 and result == model.predict(design)
    json.dumps(result, allow_nan=False)
    json.dumps(model.diagnostics, allow_nan=False)


def test_conditional_posterior_uses_visible_supplier_signal_and_propagates_missingness():
    rows = membrane_rows()
    model = PerformanceModel(seed=10).fit(rows)
    design = rows[0]["design"]
    low = model.predict(design, {"density": 0.65})
    high = model.predict(design, {"density": 1.45})
    unknown = model.predict(design)
    assert low["descriptors"]["density"]["source"] == "observed"
    assert high["descriptors"]["scattering"]["source"] == "conditional"
    assert high["descriptors"]["scattering"]["mean"] > low["descriptors"]["scattering"]["mean"]
    assert 0 < high["descriptors"]["scattering"]["sd"] < unknown["descriptors"]["scattering"]["sd"]
    assert unknown["descriptors"]["scattering"]["source"] == "marginal"
    full = {key: info["mean"] for key, info in unknown["descriptors"].items()}
    point = model.predict(design, full)
    assert not np.allclose(point["draws"]["toughness"], unknown["draws"]["toughness"])


def test_unobserved_training_channel_is_disabled_and_repeated_coupons_do_not_reweight_groups():
    rows = membrane_rows()
    for row in rows:
        row["descriptors"].pop("viscosity")
    first = PerformanceModel().fit(rows).predict(rows[0]["design"], {"viscosity": 3.0})
    assert first["descriptors"]["viscosity"] == dict(mean=3.0, sd=0.0, used_in_model=False, source="observed")
    many = rows + [dict(rows[0], run_id=f"extra-{i}") for i in range(8)]
    second = PerformanceModel().fit(many).predict(rows[0]["design"])
    assert second["descriptors"]["viscosity"] == dict(mean=None, sd=None, used_in_model=False, source="unavailable")
    for key in first["draws"]:
        np.testing.assert_allclose(first["draws"][key], second["draws"][key], atol=1e-10)


def test_reject_bad_design_leaked_descriptor_identity_and_insufficient_groups():
    rows = membrane_rows()
    with pytest.raises(ValueError, match="three independent"):
        PerformanceModel().fit(rows[:4])
    with pytest.raises(ValueError, match="Duplicate training run"):
        PerformanceModel().fit(rows + [rows[0]])
    model = PerformanceModel().fit(rows[:6])
    with pytest.raises(ValueError, match="pre-run"):
        model.predict(rows[0]["design"], {"transmission": 0.99})
    with pytest.raises(ValueError, match="sum to one"):
        model.predict(dict(rows[0]["design"], c=1))
    with pytest.raises(ValueError, match="finite"):
        model.predict(rows[0]["design"], {"density": np.inf})
