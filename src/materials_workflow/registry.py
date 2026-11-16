"""SQLite intake for synthetic optical-membrane records.

The source locator identifies one record within an artifact. The digest identifies
the received JSON. Measurements retain both their submitted and canonical units.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3


UNITS = {
    "density": ("g/cm3", {"g/cm3": 1.0, "kg/m3": 0.001}),
    "scattering": ("1/mm", {"1/mm": 1.0, "1/m": 0.001}),
    "viscosity": ("Pa s", {"Pa s": 1.0, "mPa s": 0.001}),
    "transmission": ("fraction", {"fraction": 1.0, "%": 0.01}),
    "toughness": ("kJ/m2", {"kJ/m2": 1.0, "J/m2": 0.001}),
}
DESCRIPTORS = ("density", "scattering", "viscosity")
TARGETS = ("transmission", "toughness")
STATUS = {
    "use": "accepted", "remeasure": "quarantined", "remake": "quarantined",
    "ask": "pending_review", "feasibility_only": "feasibility_only",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _identifier(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _design(record):
    design = record.get("design")
    if not isinstance(design, dict) or set(design) != {"a", "b", "c", "cure", "route"}:
        return None
    if design["route"] not in ("cast", "press"):
        return None
    values = {key: _number(design[key]) for key in ("a", "b", "c", "cure")}
    if any(value is None or not 0 <= value <= 1 for value in values.values()):
        return None
    if abs(sum(values[key] for key in ("a", "b", "c")) - 1) > 1e-8:
        return None
    return dict(values, route=design["route"])


def _measurements(record, section, names):
    """Read a section without guessing units or silently substituting zero."""
    values, provenance, issues = {}, {}, []
    supplied = record.get(section, {})
    if not isinstance(supplied, dict) or set(supplied).difference(names):
        return values, provenance, [("ask", "unrecognized_" + section)]
    for name in names:
        observation = supplied.get(name)
        if observation is None:
            if section == "targets":
                issues.append(("remeasure", "missing_" + name))
            continue
        if not isinstance(observation, dict):
            issues.append(("ask", "unlabeled_" + name))
            continue
        canonical_unit, scales = UNITS[name]
        unit = observation.get("unit")
        if not isinstance(unit, str) or unit not in scales:
            issues.append(("ask", "unrecognized_unit_" + name))
            continue
        raw = _number(observation.get("value"))
        if raw is None:
            issues.append(("remeasure", "invalid_value_" + name))
            continue
        value = raw * scales[unit]
        if value < 0 or (name == "transmission" and value > 1) or (name in ("density", "toughness") and value == 0):
            issues.append(("remeasure", "outside_physical_range_" + name))
            continue
        values[name] = value
        provenance[name] = {
            "source_id": record["source_id"], "field": section + "." + name,
            "raw_value": observation["value"], "raw_unit": unit,
            "value": value, "unit": canonical_unit,
        }
    return values, provenance, issues


def _diagnosis(record):
    """Resolve observed checks and optional operator reports without consulting a simulator.

    An absent check supplies no diagnosis; _assess checks completeness for each use.
    A failed check identifies a possible
    cause. A concordant report adds no second cause; a contrary report, multiple
    causes, or an unfamiliar check requires clarification.
    """
    checks = {
        "instrument_check_passed": "instrument",
        "protocol_followed": "procedure",
        "sample_identity_confirmed": "ambiguous",
    }
    evidence = record.get("evidence", {})
    issue = record.get("issue", "none")
    if not isinstance(evidence, dict) or any(k not in checks or type(v) is not bool for k, v in evidence.items()):
        return "ask", "ambiguous_check_evidence"
    if issue == "discordant_replicates":
        return "ask", "discordant_replicates" if all(evidence.values()) else "multiple_possible_causes"
    if issue not in ("none", "instrument", "procedure", "ambiguous"):
        return "ask", "unresolved_evidence"
    findings = {checks[key] for key, passed in evidence.items() if not passed}
    if issue != "none":
        if any(passed and checks[key] == issue for key, passed in evidence.items()):
            return "ask", "conflicting_evidence"
        findings.add(issue)
    if len(findings) > 1:
        return "ask", "multiple_possible_causes"
    if not findings:
        return None, None
    cause = findings.pop()
    return {"ambiguous": ("ask", "unresolved_identity"),
            "procedure": ("remake", "procedure_deviation"),
            "instrument": ("remeasure", "instrument_issue")}[cause]


def _assess(record):
    """Return disposition, ordered reasons, normalized row, conversions and known make result."""
    for key in ("run_id", "group_id", "source_id"):
        if not _identifier(record.get(key)):
            return "ask", ["invalid_" + key], None, {}, None
    design = _design(record)
    if design is None:
        return "ask", ["invalid_design"], None, {}, None
    base = {"run_id": record["run_id"], "group_id": record["group_id"], "design": design}
    action, reason = _diagnosis(record)
    if reason == "ambiguous_check_evidence":
        return action, [reason], base, {}, None
    made = record.get("make_status")
    if made not in ("success", "failed"):
        return "ask", ["unknown_make_outcome"], base, {}, None
    # Identity and procedure establish a makeability outcome. Successful assay
    # data also require an instrument check. Missing evidence never implies pass.
    evidence = record.get("evidence", {})
    required_make = ("protocol_followed", "sample_identity_confirmed")
    missing = ["missing_check_" + key for key in required_make if key not in evidence]
    if missing:
        return "ask", missing, base, {}, None
    if action in ("ask", "remake") and reason != "discordant_replicates":
        return action, [reason], base, {}, None
    if made == "failed":
        supplied_targets = record.get("targets", {})
        if not isinstance(supplied_targets, dict) or any(v is not None for v in supplied_targets.values()):
            return "ask", ["measurements_on_failed_make"], base, {}, None
        if reason == "discordant_replicates":
            return "ask", ["discordant_readings_on_failed_make"], base, {}, None
        return "feasibility_only", ["confirmed_failed_make"], base, {}, False
    if "instrument_check_passed" not in evidence:
        return "ask", ["missing_check_instrument_check_passed"], base, {}, True
    if action:
        return action, [reason], base, {}, True
    descriptors, conversions, issues = _measurements(record, "descriptors", DESCRIPTORS)
    targets, target_conversions, target_issues = _measurements(record, "targets", TARGETS)
    conversions.update(target_conversions)
    issues.extend(target_issues)
    if issues:
        # Resolve ambiguous meaning before choosing a physical intervention.
        issues.sort(key=lambda row: row[0] != "ask")
        return issues[0][0], [reason for _, reason in issues], base, conversions, True
    return "use", ["validated"], dict(base, descriptors=descriptors, targets=targets), conversions, True


class Registry:
    """Store submissions atomically. Only unambiguous current revisions feed the models."""

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY,
                sha256 TEXT NOT NULL UNIQUE,
                raw_json TEXT NOT NULL,
                run_id TEXT, revision INTEGER, source_id TEXT,
                action TEXT NOT NULL, reasons_json TEXT NOT NULL,
                canonical_json TEXT, provenance_json TEXT NOT NULL, made INTEGER,
                received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS heads (
                run_id TEXT PRIMARY KEY,
                submission_id INTEGER NOT NULL REFERENCES submissions(id),
                review_required INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS source_owners (
                source_id TEXT PRIMARY KEY, run_id TEXT NOT NULL
            );
        """)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.db.close()

    def ingest(self, record: dict) -> dict:
        """Stage a JSON-compatible input and append exactly one disposition per unique payload.

        Corrections provide revision N+1 and supersedes N. Two different payloads
        claiming the current revision suspend its eligibility until a correction.
        """
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        fields = record if isinstance(record, dict) else {}
        run = fields.get("run_id") if _identifier(fields.get("run_id")) else None
        source = fields.get("source_id") if _identifier(fields.get("source_id")) else None
        revision = fields.get("revision", 1)
        if type(revision) is not int or not 1 <= revision <= 2**63 - 1:
            revision = None
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            prior = self.db.execute("SELECT id FROM submissions WHERE sha256=?", (digest,)).fetchone()
            if prior:
                return self._disposition(prior["id"])
            head = self.db.execute("""
                SELECT s.revision, s.id FROM heads h JOIN submissions s ON s.id=h.submission_id
                WHERE h.run_id=?
            """, (run,)).fetchone()
            previous = fields.get("supersedes")
            if head:
                next_revision = (revision == head["revision"] + 1 and
                                 type(previous) is int and previous == head["revision"])
            else:
                next_revision = revision == 1 and previous is None
            advance = bool(run and next_revision)
            if not next_revision:
                code = "invalid_revision_lineage"
                if head and head["revision"] == revision:
                    code = "conflicting_revision"
                    self.db.execute("UPDATE heads SET review_required=1 WHERE run_id=?", (run,))
                action, reasons, row, provenance, made = "ask", [code], None, {}, None
            else:
                action, reasons, row, provenance, made = _assess(fields)
                owner = self.db.execute("SELECT run_id FROM source_owners WHERE source_id=?", (source,)).fetchone()
                if source and run and owner and owner["run_id"] != run:
                    action, reasons, row, provenance, made = "ask", ["source_claimed_by_another_run"], None, {}, None
                elif source and run:
                    self.db.execute("INSERT OR IGNORE INTO source_owners VALUES (?, ?)", (source, run))
            receipt = self.db.execute("""
                INSERT INTO submissions
                    (sha256,raw_json,run_id,revision,source_id,action,reasons_json,canonical_json,provenance_json,made)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (digest, raw, run, revision, source, action, json.dumps(reasons),
                  json.dumps(row) if row else None, json.dumps(provenance), made))
            if advance:
                self.db.execute("""
                    INSERT INTO heads (run_id,submission_id) VALUES (?,?)
                    ON CONFLICT(run_id) DO UPDATE SET submission_id=excluded.submission_id, review_required=0
                """, (run, receipt.lastrowid))
            return self._disposition(receipt.lastrowid)

    _AUDIT_QUERY = """
        SELECT s.id AS submission_id,s.sha256,s.run_id,s.revision,s.source_id,s.action,s.reasons_json,s.received_at,
               h.submission_id IS NOT NULL AS is_current, COALESCE(h.review_required,0) AS review_required
        FROM submissions s LEFT JOIN heads h ON h.submission_id=s.id
    """

    @staticmethod
    def _audit_dict(row):
        result = dict(row)
        result["reasons"] = json.loads(result.pop("reasons_json"))
        result["code"] = result["reasons"][0]
        result["status"] = STATUS[result["action"]]
        for flag in ("is_current", "review_required"):
            result[flag] = bool(result[flag])
        return result

    def _disposition(self, submission_id):
        row = self.db.execute(self._AUDIT_QUERY + " WHERE s.id=?", (submission_id,)).fetchone()
        if row is None:
            raise KeyError(submission_id)
        return self._audit_dict(row)

    def audit(self) -> list[dict]:
        """Return all disposition receipts in landing order, including superseded attempts."""
        return [self._audit_dict(row) for row in self.db.execute(self._AUDIT_QUERY + " ORDER BY s.id")]

    def submission(self, submission_id: int) -> dict:
        """Inspect original submitted values, normalized values and field-level provenance."""
        result = self._disposition(submission_id)
        row = self.db.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
        return dict(result, raw=json.loads(row["raw_json"]),
                    canonical=json.loads(row["canonical_json"]) if row["canonical_json"] else None,
                    provenance=json.loads(row["provenance_json"]))

    def training_rows(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT s.canonical_json FROM heads h JOIN submissions s ON h.submission_id=s.id
            WHERE h.review_required=0 AND s.action='use' ORDER BY s.run_id
        """)
        return [json.loads(row[0]) for row in rows]

    def feasibility_rows(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT s.canonical_json,s.made FROM heads h JOIN submissions s ON h.submission_id=s.id
            WHERE h.review_required=0 AND s.made IS NOT NULL ORDER BY s.run_id
        """)
        result = []
        for row in rows:
            canonical = json.loads(row["canonical_json"])
            result.append({key: canonical[key] for key in ("run_id", "group_id", "design")}
                          | {"success": bool(row["made"])})
        return result


def verify_preparation_return(request, outcome, receipt):
    """Check one reserved preparation or specimen reread against its raw registry record."""
    record = receipt.get("registry_record")
    if not isinstance(record, dict):
        raise ValueError("A preparation return requires a raw registry_record.")
    rule = request["registry_return"]
    failed = outcome == rule["failure"]
    specimen = None if failed else request["specimen"]
    if failed and request["kind"] == "reread":
        raise ValueError("A specimen reread cannot report a failed preparation.")
    expected = {"design": request["design"], "stock_lot": request["stock_lot"],
                "group_id": request["group_id"], "specimen_id": specimen,
                "repeats_preparation": request.get("repeats_preparation")}
    for field, reserved in expected.items():
        if record.get(field) != reserved:
            raise ValueError(f"Raw preparation {field} differs from the reservation.")
    for field in ("run_id", "group_id", "stock_lot", "specimen_id", "repeats_preparation"):
        if field in receipt and receipt[field] != record.get(field):
            raise ValueError(f"Receipt {field} differs from the raw preparation record.")
    # Replay the actual intake gate; a claimed disposition or digest alone is insufficient.
    with Registry(":memory:") as registry:
        disposition = registry.ingest(record)
        expected_action = "feasibility_only" if failed else "use"
        if disposition["action"] != expected_action:
            raise ValueError(f"Registry returned {disposition['action']} ({disposition['code']}); "
                             f"the submitted outcome requires {expected_action}.")
        for field, value in (("registry_sha256", disposition["sha256"]), ("registry_action", expected_action)):
            if field in receipt and receipt[field] != value:
                raise ValueError(f"Receipt {field} differs from the registry intake.")
        if failed:
            if receipt.get("measurement") is not None or receipt.get("targets"):
                raise ValueError("A failed preparation cannot supply a measured endpoint.")
            return
        canonical = registry.submission(disposition["submission_id"])["canonical"]
    target = rule["target"]
    value = canonical["targets"][target]
    band = rule["below"] if value < rule["threshold"] else rule["at_or_above"]
    if outcome != band:
        raise ValueError("Reported band disagrees with the normalized preparation endpoint.")
    if "targets" in receipt and receipt["targets"] != canonical["targets"]:
        raise ValueError("Receipt targets differ from the normalized preparation endpoints.")
    if "measurement" in receipt:
        reading = receipt["measurement"]
        if (not isinstance(reading, dict) or reading.get("name") != target
                or reading.get("unit") != UNITS[target][0] or reading.get("value") != value
                or reading.get("band", band) != band or reading.get("threshold", rule["threshold"]) != rule["threshold"]):
            raise ValueError("Receipt measurement differs from the normalized preparation endpoint.")
