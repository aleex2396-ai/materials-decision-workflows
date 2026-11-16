"""Cross-process reservations and observed branches of the shared-budget lab controller."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

from materials_workflow.shared_experiment import run_shared_demo
from materials_workflow.shared_session import BudgetSession, fingerprint


@pytest.fixture
def spec():
    make = {"kind": "make", "cost": 2, "success_outcomes": ["made"],
            "likelihoods": {"made": [1, 1], "failed": [0, 0]}}
    return {"prior": [.6, .4], "budget": 3, "utility_per_credit": .02,
            "utilities": {"A": [1, 0], "B": [0, 1]},
            "actions": [{"id": "check-stock", "kind": "assay", "cost": 1,
                         "likelihoods": {"low": [.9, .1], "high": [.1, .9]}},
                        dict(make, id="make-A", recipe="A", specimen="coupon-A"),
                        dict(make, id="make-B", recipe="B", specimen="coupon-B")]}


def receipt(reservation, label):
    action = reservation["decision"]
    return {"action_id": reservation["action_id"], "kind": action["kind"],
            "observation": label, "cost": action["cost"], "run_id": "receipt-" + reservation["action_id"]}


def test_restarts_and_retries_preserve_sealed_requests_and_exactly_once_debits(tmp_path, spec):
    db = tmp_path / "run.sqlite3"
    with BudgetSession(db, spec=spec) as session:
        initial = session.status()
        assert initial["budget"] == initial["remaining"] == 3
        assert initial["events"] == []
        first = session.next()
        assert first["action_id"] == "check-stock"
        assert "outcome" not in first and "receipt" not in first
        assert first["decision_sha256"] == fingerprint({k: v for k, v in first.items() if k != "decision_sha256"})
    with BudgetSession(db) as session:
        assert session.next() == first
        assert session.status()["spent"] == 1
        evidence = receipt(first, "high")
        accepted = session.observe(1, first["action_id"], "high", evidence)
        assert accepted["after"]["posterior"] == pytest.approx([.06/.42, .36/.42])
        second = session.next()
        assert second["action_id"] == "make-B" and second["credits_after"] == 0
    with BudgetSession(db, spec=spec) as session:
        # Replaying an already completed prior request after the next reservation
        # must return its original receipt and leave the current pending make intact.
        assert session.observe(1, first["action_id"], "high", evidence) == accepted
        assert session.next() == second
        with pytest.raises(ValueError):
            session.observe(1, first["action_id"], "high", dict(evidence, run_id="different-record"))
        returned = session.observe(2, "make-B", "made", receipt(second, "made"))
        assert returned["after"]["specimens"] == ["coupon-B"]
        stop = session.next()
        assert stop["decision"]["terminal_recipe"] == "B"
        assert session.next() == stop
        status = session.status()
        assert status["spent"] == 3 and status["remaining"] == 0 and status["finished"]
        assert [(e["step"], e["action_id"]) for e in status["events"]] == [(1, "check-stock"), (2, "make-B"), (3, "stop")]
    with pytest.raises(ValueError, match="different policy"):
        BudgetSession(db, spec=dict(spec, budget=4))


def test_wrong_envelope_impossible_or_out_of_order_result_cannot_advance_state(tmp_path, spec):
    with BudgetSession(tmp_path / "run.sqlite3", spec=spec) as session:
        first = session.next()
        evidence = receipt(first, "low")
        before = session.status()
        attempts = [
            (2, "check-stock", "low", evidence),
            (1, "make-A", "low", evidence),
            (1, "check-stock", "undeclared", dict(evidence, observation="undeclared")),
            (1, "check-stock", "low", dict(evidence, action_id="make-A")),
            (1, "check-stock", "low", dict(evidence, observation="high")),
            (1, "check-stock", "low", dict(evidence, cost=2)),
        ]
        for args in attempts:
            with pytest.raises(ValueError):
                session.observe(*args)
            assert session.status() == before
        session.observe(1, "check-stock", "low", evidence)
        make = session.next()
        assert make["action_id"] == "make-A"
        pending = session.status()
        with pytest.raises(ValueError, match="zero probability"):
            session.observe(2, "make-A", "failed", receipt(make, "failed"))
        assert session.status() == pending
        assert session.next() == make


def test_thresholded_assay_receipt_requires_matching_reading_and_units(tmp_path, spec):
    spec["actions"][0]["observable"] = {"name": "stock viscosity", "unit": "Pa s", "threshold": 2.0,
                                        "below": "low", "at_or_above": "high"}
    with BudgetSession(tmp_path / "assay.sqlite3", spec=spec) as session:
        reserved = session.next()
        pending = session.status()
        evidence = receipt(reserved, "high")
        reading = {"name": "stock viscosity", "unit": "Pa s", "value": 2.0}
        for invalid in (dict(reading, value=1.94), dict(reading, unit="mPa s")):
            with pytest.raises(ValueError):
                session.observe(1, "check-stock", "high", dict(evidence, measurement=invalid))
            assert session.status() == pending
        accepted = session.observe(1, "check-stock", "high", dict(evidence, measurement=reading))
        assert accepted["outcome"] == "high"
        assert session.next()["action_id"] == "make-B"


def test_independent_concurrent_connections_reserve_and_return_only_one_step(tmp_path, spec):
    db = tmp_path / "run.sqlite3"
    with BudgetSession(db, spec=spec):
        pass
    parties = 4
    barrier = Barrier(parties, timeout=10)

    def reserve_and_return(_):
        with BudgetSession(db) as session:
            barrier.wait()
            reserved = session.next()
            barrier.wait()
            result = session.observe(reserved["step"], reserved["action_id"], "low", receipt(reserved, "low"))
            barrier.wait()
            return reserved, result, session.next()

    with ThreadPoolExecutor(max_workers=parties) as executor:
        transactions = list(executor.map(reserve_and_return, range(parties)))
    assert all(t == transactions[0] for t in transactions)
    with BudgetSession(db) as session:
        status = session.status()
        assert len(status["events"]) == 2
        assert status["spent"] == 3 and status["pending"] == "make-A"
        assert status["events"][0]["outcome"] == "low"


def test_a_specimen_reread_requires_a_new_instrument_return(tmp_path):
    design = {"a": 0.5, "b": 0.25, "c": 0.25, "cure": 0.5, "route": "cast"}
    contract = {"recipe": "A", "specimen": "coupon-A", "group_id": "prep-A", "stock_lot": "lot",
                "design": design, "cost": 1, "registry_return": {"target": "toughness", "threshold": 7,
                "below": "below", "at_or_above": "above", "failure": "failed"}}
    make = dict(contract, id="make-A", kind="make", success_outcomes=["above", "below"],
                likelihoods={"above": [.5, .5], "below": [.5, .5], "failed": [0, 0]})
    reread = dict(contract, id="read-A", kind="reread", likelihoods={"above": [.95, .05], "below": [.05, .95]})
    spec = {"budget": 2, "prior": [.5, .5], "initial_qualified": ["B"],
            "utilities": {"A": [1, 0], "B": [.5, .5]}, "actions": [make, reread]}
    raw = {"run_id": "run-1", "group_id": "prep-A", "source_id": "meter-1", "stock_lot": "lot",
           "design": design, "specimen_id": "coupon-A", "make_status": "success",
           "targets": {"toughness": {"value": 8, "unit": "kJ/m2"},
                       "transmission": {"value": 0.7, "unit": "fraction"}},
           "evidence": {"instrument_check_passed": True, "protocol_followed": True, "sample_identity_confirmed": True}}
    with BudgetSession(tmp_path / "reread.sqlite3", spec=spec) as session:
        assert session.next()["action_id"] == "make-A"
        session.observe(1, "make-A", "above", {"registry_record": raw})
        assert session.next()["action_id"] == "read-A"
        with pytest.raises(ValueError, match="already used"):
            session.observe(2, "read-A", "above", {"registry_record": raw})
        assert session.status()["pending"] == "read-A"
        new_read = dict(raw, run_id="run-2", source_id="meter-2")
        session.observe(2, "read-A", "above", {"registry_record": new_read})
        assert session.next()["decision"]["terminal_recipe"] == "A"


def test_cli_observation_persists_across_independent_processes(tmp_path, spec):
    # The operator may run each command from a different process, outside the
    # repository; use the checkout currently under test for package discovery.
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), env.get("PYTHONPATH", "")])
    db = tmp_path / "operator.sqlite3"
    spec_path = tmp_path / "policy.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def operate(command, *args):
        result = subprocess.run([sys.executable, "-m", "materials_workflow", "session", command,
                                 "--db", str(db), *map(str, args)], cwd=tmp_path, env=env,
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    assert operate("init", "--spec", spec_path)["policy_sha256"] == fingerprint(spec)
    request = operate("next")
    assert request["action_id"] == "check-stock" and request["credits_after"] == 2
    proof = tmp_path / "evidence.json"
    proof.write_text(json.dumps(receipt(request, "high")), encoding="utf-8")
    accepted = operate("observe", "--action", "check-stock", "--step", 1, "--outcome", "high",
                       "--receipt", proof)
    assert accepted["decision_sha256"] == request["decision_sha256"]
    changed = operate("next")
    assert changed["before"] == accepted["after"]
    assert changed["action_id"] == "make-B" and changed["credits_after"] == 0


def test_public_demo_proves_policy_identity_executed_fork_and_equal_budget_control(tmp_path):
    output = tmp_path / "paired"
    summary = run_shared_demo(output)
    trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
    spec = json.loads((output / "policy.json").read_text(encoding="utf-8"))
    assert summary["opening_action"] == "assay-stock"
    assert summary["opening_net_value"] > summary["make_only_net_value"]
    paths = trace["episodes"]
    starts = [branch["shared"]["events"][0] for branch in paths]
    assert all(first["decision_sha256"] == summary["shared_opening_decision_sha256"] for first in starts)
    assert all(first["policy_sha256"] == fingerprint(spec) for first in starts)
    assert starts[0]["before"] == starts[1]["before"] and starts[0]["decision"] == starts[1]["decision"]
    outcomes, makes = [], []
    for branch in paths:
        shared, control = branch["shared"], branch["make_only"]
        measurement, make, stop = shared["events"]
        assert measurement["credits_after"] == make["credits_before"]
        assert make["before"] == measurement["after"]
        assert make["receipt"]["recipe"] == make["request"]["recipe"]
        assert make["receipt"]["registry_action"] == "use"
        assert stop["action_id"] == "stop" and shared["remaining"] == 0
        assert shared["budget"] == control["budget"] == spec["budget"]
        outcomes.append(measurement["receipt"]["observation"])
        makes.append(make["receipt"]["recipe"])
    assert dict(zip(outcomes, makes)) == {"low": "binder-rich", "high": "binder-lean"}
    assert sum(row["switch_vs_direct"] for row in summary["branches"]) == 1
