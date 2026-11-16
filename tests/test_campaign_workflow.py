"""Exercise the public entry points with only the packaged synthetic campaign."""

from hashlib import sha256
import json

import pytest

from materials_workflow.__main__ import main
from materials_workflow.campaign import run_demo, rows_hash
from materials_workflow.registry import Registry


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_decisions_and_versioned_features_from_the_registered_pre_run_batch(tmp_path):
    out = tmp_path / "demo"
    summary = run_demo(out)
    first = read(out / "pre_run_decision.json")
    evidence = read(out / "packet.json")
    assert evidence["replay_receipt_matches"]
    assert set(summary["gate_counts"]) == {"use", "remeasure", "remake", "ask", "feasibility_only"}
    assert [a["id"] for a in first["proposal"]["actions"]] == summary["selected_ids"]
    assert first["proposal"] == evidence["initial_plan"]
    assert summary["spent"] <= summary["budget"]
    assert len(summary["selected_ids"]) == summary["feedback_rows"] == summary["config"]["batch_size"]
    assert set(summary["next_ids"]).isdisjoint(summary["selected_ids"] + evidence["pending_ids"])
    assert evidence["training_snapshot"]["pre_run_file_sha256"] == sha256((out / "pre_run_decision.json").read_bytes()).hexdigest()
    for outcome in evidence["execution"]:
        assert outcome["pre_run_prediction"] == first["predictions"][outcome["candidate_id"]]
        assert outcome["group_id"] not in {r["group_id"] for r in evidence["initial_rows"]}
    assert first["training_sha256"] == rows_hash(evidence["initial_rows"])
    with Registry(out / "campaign.sqlite3") as ledger:
        assert evidence["next_plan"]["training_sha256"] == rows_hash(ledger.training_rows())
        assert len(ledger.audit()) == summary["initial_attempts"] + summary["feedback_rows"]
    db = str(out / "campaign.sqlite3")
    current = tmp_path / "features.json"
    main(["features", "--db", db, "--out", str(current)])
    assert read(current)["is_current"]
    assert read(current)["counts"]["conditional"] > 0
    older = evidence["descriptor_views"]["before"]["snapshot_id"]
    main(["features", "--db", db, "--snapshot", older, "--out", str(current)])
    assert not read(current)["is_current"]
    decisions = tmp_path / "recommendations.json"
    main(["recommend", "--db", db, "--candidates", str(out / "candidates.json"), "--pending", str(out / "pending.json"),
          "--budget", str(summary["budget"]), "--batch-size", str(summary["config"]["batch_size"]),
          "--seed", str(summary["config"]["seed"]), "--out", str(decisions)])
    assert read(decisions) == evidence["next_plan"]
    assert evidence["descriptor_assay_decision"]["recommended_action"]["action"] in {"make", "measure"}
    with pytest.raises(FileExistsError):
        run_demo(out)
