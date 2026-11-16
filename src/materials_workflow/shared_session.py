"""Reserve actions and apply observations against one persistent campaign budget."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path
import sqlite3

from .registry import verify_preparation_return
from .shared_budget import SharedBudgetPlanner


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def _verify_reading(rule, outcome, reading):
    if (not isinstance(reading, dict) or reading.get("name") != rule["name"]
            or reading.get("unit") != rule["unit"]):
        raise ValueError("The receipt needs the reserved measurement in its specified units.")
    value = reading.get("value")
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError("The measurement must be a finite number.")
    observed_band = rule["below"] if value < rule["threshold"] else rule["at_or_above"]
    if observed_band != outcome or reading.get("band", outcome) != outcome:
        raise ValueError("The reported outcome disagrees with the measured value.")


class BudgetSession:
    """Single in-flight action. Reserving consumes its credits, including failed attempts."""

    def __init__(self, path, *, spec=None):
        path = Path(path).resolve()
        mode = "rwc" if spec is not None else "rw"
        self.db = sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE" if spec is not None else "BEGIN")
                tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables and spec is not None:
                    self.db.execute("CREATE TABLE policy (spec TEXT NOT NULL, digest TEXT NOT NULL)")
                    self.db.execute("""CREATE TABLE steps (
                        step INTEGER PRIMARY KEY, action_id TEXT NOT NULL UNIQUE, sealed TEXT NOT NULL,
                        outcome TEXT, receipt TEXT, posterior TEXT)""")
                    self.db.execute("INSERT INTO policy VALUES (?,?)", (encode(spec), fingerprint(spec)))
                elif tables != {"policy", "steps"}:
                    raise ValueError("This database is not a shared-budget session.")
                entry = self.db.execute("SELECT spec, digest FROM policy").fetchone()
                self.spec = json.loads(entry["spec"])
                self.spec_hash = entry["digest"]
                if fingerprint(self.spec) != self.spec_hash:
                    raise ValueError("Stored policy specification failed its digest check.")
                if spec is not None and fingerprint(spec) != self.spec_hash:
                    raise ValueError("The session already has a different policy specification.")
                self.policy = SharedBudgetPlanner(self.spec)
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def _state(self):
        state = self.policy.initial_state()
        rows = self.db.execute("SELECT action_id,outcome FROM steps WHERE outcome IS NOT NULL ORDER BY step")
        for row in rows:
            state = self.policy.observe(state, row["action_id"], row["outcome"])
        return state

    def _describe(self, state):
        posterior = list(self.policy.belief(state))
        return {"posterior": posterior, "qualified": sorted(state.qualified),
                "specimens": sorted(state.specimens),
                "expected_recipe_utility": {
                    recipe: sum(p * value for p, value in zip(posterior, values, strict=True))
                    for recipe, values in self.spec["utilities"].items()}}

    @staticmethod
    def _entry(row):
        event = json.loads(row["sealed"])
        if row["outcome"] is not None:
            event.update(outcome=row["outcome"], receipt=json.loads(row["receipt"]),
                         after=json.loads(row["posterior"]))
        return event

    def next(self):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            last = self.db.execute("SELECT * FROM steps ORDER BY step DESC LIMIT 1").fetchone()
            if last is not None and last["outcome"] is None:
                return self._entry(last)
            state = self._state()
            decision = self.policy.select(state)
            action = next((a for a in self.spec["actions"] if a["id"] == decision["action_id"]), {})
            request = {k: v for k, v in action.items() if k != "likelihoods"}
            event = {"step": 1 if last is None else last["step"] + 1,
                     "policy_sha256": self.spec_hash, "action_id": decision["action_id"],
                     "credits_before": state.remaining, "credits_after": state.remaining - decision["cost"],
                     "before": self._describe(state), "decision": decision, "request": request}
            event["decision_sha256"] = fingerprint(event)
            self.db.execute("INSERT INTO steps (step,action_id,sealed) VALUES (?,?,?)",
                            (event["step"], event["action_id"], encode(event)))
            return event

    def observe(self, step, action_id, outcome, receipt):
        if type(step) is not int or step < 1 or not isinstance(receipt, dict) or not receipt:
            raise ValueError("Observation requires a positive integer step and a nonempty evidence receipt.")
        encoded = encode(receipt)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT * FROM steps WHERE step=?", (step,)).fetchone()
            if row is None or row["action_id"] != action_id or action_id == "stop":
                raise ValueError("Observation does not match a reserved experimental action.")
            request = json.loads(row["sealed"])["request"]
            for field in ("kind", "cost", "recipe", "stock_lot"):
                if field in receipt and receipt[field] != request.get(field):
                    raise ValueError(f"Receipt {field} differs from the reservation.")
            if receipt.get("action_id", action_id) != action_id or receipt.get("observation", outcome) != outcome:
                raise ValueError("Receipt action or observation differs from the submitted result.")
            if row["outcome"] is not None:
                if row["outcome"] != outcome or row["receipt"] != encoded:
                    raise ValueError("Conflicting observation for an already completed action.")
                return self._entry(row)
            if "registry_return" in request:
                verify_preparation_return(request, outcome, receipt)
                raw = receipt["registry_record"]
                previously_used = self.db.execute("""
                    SELECT step FROM steps
                    WHERE json_extract(receipt, '$.registry_record.run_id') = ?
                       OR json_extract(receipt, '$.registry_record.source_id') = ?
                    """, (raw["run_id"], raw["source_id"])).fetchone()
                if previously_used:
                    raise ValueError("That instrument return was already used for another action.")
            if "observable" in request:
                _verify_reading(request["observable"], outcome, receipt.get("measurement"))
            state = self.policy.observe(self._state(), action_id, outcome)
            self.db.execute("UPDATE steps SET outcome=?,receipt=?,posterior=? WHERE step=?",
                            (outcome, encoded, encode(self._describe(state)), step))
            return self._entry(self.db.execute("SELECT * FROM steps WHERE step=?", (step,)).fetchone())

    def status(self):
        with self.db:
            self.db.execute("BEGIN")
            events = [self._entry(row) for row in self.db.execute("SELECT * FROM steps ORDER BY step")]
            last = events[-1] if events else {}
            budget = self.spec["budget"]
            remaining = last.get("credits_after", budget)
            done = last.get("action_id") == "stop"
            pending = last.get("action_id") if not done and "outcome" not in last else None
            return {"policy_sha256": self.spec_hash, "budget": budget, "remaining": remaining,
                    "spent": budget - remaining, "pending": pending, "finished": done, "events": events}
