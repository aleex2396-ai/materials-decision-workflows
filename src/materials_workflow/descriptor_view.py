"""Immutable, snapshot-specific descriptor estimates for inspection and downstream reporting.

Fitting and cross-validation read the registry's measured rows directly. They do
not read these full-snapshot estimates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .registry import DESCRIPTORS, UNITS


SOURCES = ("observed", "conditional", "marginal", "unavailable")
FORMULA = {
    "version": "descriptor-view-v1",
    "observed": "Original unit-normalized input; sd=0 denotes no imputation uncertainty.",
    "conditional": "Gaussian mean and SD conditional on the available, correlated descriptors.",
    "marginal": "Group-weighted fitted mean and SD when observed descriptors are uninformative.",
    "unavailable": "No estimate when the descriptor is missing and fewer than two training groups observed it.",
    "meaning_of_sd": "Working model uncertainty for missing inputs. Instrument error is outside this estimate.",
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _eligibility(registry):
    rows = registry.training_rows()  # Registry orders by run_id.
    current = {r["run_id"]: r["sha256"] for r in registry.audit()
               if r["is_current"] and not r["review_required"] and r["action"] == "use"}
    return rows, current


def _tables(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS descriptor_snapshots (
            snapshot_id TEXT PRIMARY KEY, summary_json TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS descriptor_values (
            snapshot_id TEXT NOT NULL REFERENCES descriptor_snapshots(snapshot_id),
            run_id TEXT NOT NULL, group_id TEXT NOT NULL, receipt_sha256 TEXT NOT NULL,
            descriptor TEXT NOT NULL, unit TEXT NOT NULL, source TEXT NOT NULL,
            mean REAL, sd REAL, used_in_model INTEGER NOT NULL,
            PRIMARY KEY(snapshot_id, run_id, descriptor)
        )
    """)


def _stored_values(db, snapshot_id):
    records = [dict(row) for row in db.execute("""
        SELECT run_id,group_id,receipt_sha256,descriptor,unit,source,mean,sd,used_in_model
        FROM descriptor_values WHERE snapshot_id=? ORDER BY run_id,descriptor
    """, (snapshot_id,))]
    for row in records:
        row["used_in_model"] = bool(row["used_in_model"])
    return records


def materialize_descriptors(registry, model, training_rows, snapshot_id: str | None = None) -> dict:
    """Persist one estimated/observed value per descriptor and eligible run, atomically."""
    if snapshot_id is not None and (not isinstance(snapshot_id, str) or not snapshot_id.strip() or snapshot_id != snapshot_id.strip()):
        raise ValueError("An explicit snapshot ID must be a nonempty string")
    rows = sorted(training_rows, key=lambda r: r["run_id"])
    train_hash = _sha(rows)
    if model.training_sha256 != train_hash:
        raise ValueError("Descriptor model was fitted on a different training snapshot")
    with registry.db as db:
        db.execute("BEGIN IMMEDIATE")
        _tables(db)
        eligible, receipts = _eligibility(registry)
        if _sha(eligible) != train_hash:
            raise ValueError("Provided training rows differ from currently eligible revisions")
        records = []
        counts = dict.fromkeys(SOURCES, 0)
        for run in rows:
            estimates = model.descriptor_estimates(run["descriptors"])
            for name in DESCRIPTORS:
                estimate = dict(estimates[name])
                # SQLite stores either sign of zero as +0.0; hash that same representation.
                if estimate["mean"] == 0.0:
                    estimate["mean"] = 0.0
                counts[estimate["source"]] += 1
                records.append(dict(run_id=run["run_id"], group_id=run["group_id"],
                                    receipt_sha256=receipts[run["run_id"]], descriptor=name,
                                    unit=UNITS[name][0], **estimate))
        code_hash = hashlib.sha256(Path(__file__).with_name("formulation.py").read_bytes()).hexdigest()
        receipt_hash = _sha(receipts)
        if snapshot_id is None:
            snapshot_id = _sha([FORMULA, train_hash, receipt_hash, code_hash, model.seed])
        metadata = dict(snapshot_id=snapshot_id, formula=FORMULA, estimator_sha256=code_hash,
                        training_sha256=train_hash, receipt_sha256=receipt_hash,
                        model_seed=model.seed, row_count=len(rows), descriptor_count=len(records), counts=counts)
        metadata["view_sha256"] = _sha([metadata, records])
        existing = db.execute("SELECT summary_json FROM descriptor_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if existing:
            if existing[0] != _json(metadata):
                raise ValueError("Snapshot ID already identifies different revisions, model or estimates")
            if _stored_values(db, snapshot_id) != records:
                raise ValueError("Stored descriptor snapshot failed integrity check")
            return metadata
        db.execute("INSERT INTO descriptor_snapshots VALUES (?,?)", (snapshot_id, _json(metadata)))
        db.executemany("""
            INSERT INTO descriptor_values VALUES
                (:snapshot_id,:run_id,:group_id,:receipt_sha256,:descriptor,:unit,:source,:mean,:sd,:used_in_model)
        """, [dict(snapshot_id=snapshot_id, **row) for row in records])
    return metadata


def read_descriptors(registry, snapshot_id: str) -> dict:
    """Return the named historical view, with an explicit current-eligibility check."""
    with registry.db as db:
        db.execute("BEGIN")
        table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='descriptor_snapshots'").fetchone()
        if not table:
            raise KeyError(snapshot_id)
        summary = db.execute("SELECT summary_json FROM descriptor_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if summary is None:
            raise KeyError(snapshot_id)
        metadata = json.loads(summary[0])
        records = _stored_values(db, snapshot_id)
        check = metadata.copy()
        old_hash = check.pop("view_sha256")
        if _sha([check, records]) != old_hash:
            raise ValueError("Stored descriptor snapshot failed integrity check")
        eligible, receipts = _eligibility(registry)
        current = _sha(eligible) == metadata["training_sha256"] and _sha(receipts) == metadata["receipt_sha256"]
        return dict(metadata, is_current=current, records=records)
