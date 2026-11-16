"""Projection of model-based estimates is separate from original measurements and training."""
import pytest

from materials_workflow.descriptor_view import materialize_descriptors, read_descriptors
from materials_workflow.formulation import PerformanceModel
from materials_workflow.registry import Registry


def entered(index):
    observed = {}
    if index != 5:
        observed["density"] = {"value": 1 + 0.1 * index, "unit": "g/cm3"}
    if index < 4:
        observed["scattering"] = {"value": 1 + 0.2 * index, "unit": "1/mm"}
    return {
        "run_id": "run-" + str(index), "group_id": "mix-" + str(index), "source_id": "fixture#" + str(index),
        "design": {"a": 0.45, "b": 0.25, "c": 0.30, "cure": 0.1 + 0.1 * index, "route": "cast"},
        "evidence": {"instrument_check_passed": True, "protocol_followed": True, "sample_identity_confirmed": True},
        "make_status": "success", "descriptors": observed,
        "targets": {"transmission": {"value": 0.6 + 0.01 * index, "unit": "fraction"},
                    "toughness": {"value": 1.1 + 0.1 * index, "unit": "kJ/m2"}},
    }


def test_materialization_preserves_raw_and_tracks_all_descriptor_sources(tmp_path):
    with Registry(tmp_path / "lab.sqlite") as db:
        for index in range(6):
            db.ingest(entered(index))
        rows = db.training_rows()
        model = PerformanceModel().fit(rows)
        summary = materialize_descriptors(db, model, rows)
        sid = summary["snapshot_id"]
        assert materialize_descriptors(db, model, rows[::-1]) == summary
        assert summary["row_count"] == 6 and summary["descriptor_count"] == 18
        assert summary["counts"] == {"observed": 9, "unavailable": 6, "conditional": 1, "marginal": 2}
        view = read_descriptors(db, sid)
        assert view["is_current"]
        conditional = next(r for r in view["records"] if r["source"] == "conditional")
        assert conditional["run_id"] == "run-4" and conditional["descriptor"] == "scattering"
        assert conditional["mean"] is not None and conditional["sd"] > 0 and conditional["used_in_model"]
        unavailable = next(r for r in view["records"] if r["source"] == "unavailable")
        assert unavailable["mean"] is None and unavailable["sd"] is None and not unavailable["used_in_model"]
        assert db.training_rows() == rows and len(db.audit()) == 6
    with Registry(tmp_path / "lab.sqlite") as reopened:
        assert read_descriptors(reopened, sid) == view
        with pytest.raises(KeyError):
            read_descriptors(reopened, "missing-snapshot")


def test_same_numeric_amendment_changes_automatic_identity_and_preserves_history():
    with Registry(":memory:") as db:
        for index in range(6):
            db.ingest(entered(index))
        rows = db.training_rows()
        model = PerformanceModel().fit(rows)
        before = materialize_descriptors(db, model, rows)
        amended = dict(entered(1), revision=2, supersedes=1)
        db.ingest(amended)
        assert db.training_rows() == rows
        assert not read_descriptors(db, before["snapshot_id"])["is_current"]
        with pytest.raises(ValueError, match="already identifies"):
            materialize_descriptors(db, model, rows, before["snapshot_id"])
        after = materialize_descriptors(db, model, rows)
        assert before["snapshot_id"] != after["snapshot_id"]
        assert before["training_sha256"] == after["training_sha256"]
        assert read_descriptors(db, after["snapshot_id"])["is_current"]
        db.ingest(entered(6))
        with pytest.raises(ValueError, match="currently eligible"):
            materialize_descriptors(db, model, rows)
        with pytest.raises(ValueError, match="different training snapshot"):
            materialize_descriptors(db, model, db.training_rows())
        assert not read_descriptors(db, after["snapshot_id"])["is_current"]


def test_unknown_snapshot_and_tampering():
    with Registry(":memory:") as db:
        with pytest.raises(KeyError):
            read_descriptors(db, "no-snapshots")
        for index in range(4):
            db.ingest(entered(index))
        rows = db.training_rows()
        model = PerformanceModel().fit(rows)
        materialize_descriptors(db, model, rows, "named")
        with db.db:
            db.db.execute("UPDATE descriptor_values SET sd=5 WHERE descriptor='density'")
        with pytest.raises(ValueError, match="integrity"):
            read_descriptors(db, "named")
        with pytest.raises(ValueError, match="integrity"):
            materialize_descriptors(db, model, rows, "named")
