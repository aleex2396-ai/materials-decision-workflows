"""Synthetic intake, provenance and eligibility checks for the optical membrane demo."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from materials_workflow.registry import Registry


def observation(run="coupon-1"):
    return {
        "run_id": run, "group_id": "preparation-a", "source_id": "fictional-return#" + run,
        "make_status": "success", "design": {"a": 0.2, "b": 0.5, "c": 0.3, "cure": 0.6, "route": "cast"},
        "evidence": {"protocol_followed": True, "sample_identity_confirmed": True, "instrument_check_passed": True},
        "targets": {"transmission": {"value": 44.0, "unit": "%"}, "toughness": {"value": 2500, "unit": "J/m2"}},
        "descriptors": {"density": {"value": 1200, "unit": "kg/m3"}},
    }


def test_landing_processing_and_traceable_units(tmp_path):
    record = observation()
    with Registry(tmp_path / "lab.sqlite") as registry:
        receipt = registry.ingest(record)
        assert receipt["action"] == "use"
        assert registry.ingest(record) == receipt
        assert len(registry.audit()) == 1
        stored = registry.submission(receipt["submission_id"])
        assert stored["raw"] == record
        assert stored["provenance"]["density"] == dict(
            source_id=record["source_id"], field="descriptors.density", raw_value=1200,
            raw_unit="kg/m3", value=1.2, unit="g/cm3",
        )
        [ready] = registry.training_rows()
        assert set(ready) == {"run_id", "group_id", "design", "descriptors", "targets"}
        assert ready["targets"] == {"transmission": 0.44, "toughness": 2.5}
        assert ready["descriptors"] == {"density": 1.2}
        assert registry.feasibility_rows()[0]["success"] is True
    with Registry(tmp_path / "lab.sqlite") as reopened:
        assert reopened.training_rows() == [ready]
        json.dumps(reopened.audit(), allow_nan=False)


@pytest.mark.parametrize("evidence,action,feasible", [
    ({"instrument_check_passed": False}, "remeasure", True),
    ({"protocol_followed": False}, "remake", False),
    ({"sample_identity_confirmed": False}, "ask", False),
    ({"instrument_check_passed": False, "protocol_followed": False}, "ask", False),
    ({"instrument_check_passed": "false"}, "ask", False),
    ({"unknown_check": True}, "ask", False),
])
def test_route_from_independent_checks(evidence, action, feasible):
    with Registry(":memory:") as registry:
        record = observation()
        record["evidence"].update(evidence)
        receipt = registry.ingest(record)
        assert receipt["action"] == action
        assert bool(registry.feasibility_rows()) is feasible
        assert not registry.training_rows()


def test_observed_poor_membrane_and_confirmed_make_failure_are_different():
    poor = observation()
    poor["targets"]["transmission"]["value"] = 0
    with Registry(":memory:") as registry:
        assert registry.ingest(poor)["action"] == "use"
        failure = dict(observation("failed"), make_status="failed", targets={})
        assert registry.ingest(failure)["action"] == "feasibility_only"
        result = {row["run_id"]: row["success"] for row in registry.feasibility_rows()}
        assert result == {"coupon-1": True, "failed": False}
        assert [row["run_id"] for row in registry.training_rows()] == ["coupon-1"]


def test_missing_assay_check_preserves_only_confirmed_makeability():
    with Registry(":memory:") as registry:
        missing = observation("unknown")
        del missing["evidence"]
        assert registry.ingest(missing)["code"] == "missing_check_protocol_followed"
        confirmed = observation("made")
        del confirmed["evidence"]["instrument_check_passed"]
        assert registry.ingest(confirmed)["code"] == "missing_check_instrument_check_passed"
        failed = dict(observation("failed"), make_status="failed", targets={})
        del failed["evidence"]["instrument_check_passed"]
        assert registry.ingest(failed)["action"] == "feasibility_only"
        soft = dict(observation("soft"), issue="discordant_replicates")
        del soft["evidence"]["sample_identity_confirmed"]
        assert registry.ingest(soft)["code"] == "missing_check_sample_identity_confirmed"
        assert not registry.training_rows()
        assert {r["run_id"]: r["success"] for r in registry.feasibility_rows()} == {"made": True, "failed": False}


@pytest.mark.parametrize("unit,value,action", [
    ("fraction", 1.01, "remeasure"), ("fraction", -1, "remeasure"),
    ("fraction", True, "remeasure"), ("fraction", float("nan"), "remeasure"),
    ("?", 0.2, "ask"),
])
def test_target_defects_have_no_numeric_label(unit, value, action):
    record = observation()
    record["targets"]["transmission"] = {"value": value, "unit": unit}
    with Registry(":memory:") as registry:
        assert registry.ingest(record)["action"] == action
        assert not registry.training_rows()


def test_missing_target_nonclosing_recipe_and_unidentified_record():
    missing = observation("unread")
    del missing["targets"]["toughness"]
    bad = observation("bad-recipe")
    bad["design"]["c"] = 0.1
    unknown = observation("unknown-source")
    del unknown["source_id"]
    with Registry(":memory:") as registry:
        assert registry.ingest(missing)["action"] == "remeasure"
        assert registry.ingest(bad)["action"] == "ask"
        assert registry.ingest(unknown)["action"] == "ask"
        assert len(registry.audit()) == 3 and not registry.training_rows()
        assert [r["run_id"] for r in registry.feasibility_rows()] == ["unread"]


def test_corrections_conflicts_and_duplicate_sources():
    record = observation()
    with Registry(":memory:") as registry:
        first = registry.ingest(record)
        alias = dict(record, run_id="second-id")
        assert registry.ingest(alias)["code"] == "source_claimed_by_another_run"
        assert len(registry.training_rows()) == 1
        conflicting = deepcopy(record)
        conflicting["targets"]["transmission"]["value"] = 52
        assert registry.ingest(conflicting)["code"] == "conflicting_revision"
        assert not registry.training_rows() and not registry.feasibility_rows()
        assert registry.submission(first["submission_id"])["review_required"]
        assert registry.ingest(dict(conflicting, revision=3, supersedes=1))["code"] == "invalid_revision_lineage"
        second = registry.ingest(dict(conflicting, revision=2, supersedes=1))
        assert second["action"] == "use"
        assert registry.training_rows()[0]["targets"]["transmission"] == 0.52
        archived = registry.submission(first["submission_id"])
        assert archived["raw"] == record and not archived["is_current"]
        assert len(registry.audit()) == 5


def test_concurrent_replay_creates_one_landing(tmp_path):
    path = tmp_path / "race.sqlite"
    with Registry(path):
        pass

    def replay(_):
        with Registry(path) as registry:
            return registry.ingest(observation())["submission_id"]

    with ThreadPoolExecutor(max_workers=4) as workers:
        assert len(set(workers.map(replay, range(12)))) == 1
    with Registry(path) as registry:
        assert len(registry.audit()) == len(registry.training_rows()) == 1
