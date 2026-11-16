"""Budgeted space-filling design when no reliable performance surface exists."""
from __future__ import annotations

from decimal import Decimal
from numbers import Integral

import numpy as np

from .planner import (_as_design, _candidate_spec, _design_key, _distance,
                      _feasibility_evidence, _number, _observed)


def cold_start_batch(candidates, training_rows, feasibility_rows, pending, *, budget, batch_size, seed):
    """Cover a finite, permitted design library without consulting predicted outcomes.

    Start near the simplex/process centre when there is no history. Subsequently
    maximize the minimum standardized distance to all attempted, in-flight and
    selected designs. Among equally spaced options prefer lower cost, then use a
    seeded tie-break independent of input ordering. Budget denotes new spend.
    """
    budget = _number(budget, "budget")
    if budget < 0 or isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 0:
        raise ValueError("budget and integer batch_size must be nonnegative")
    remaining = Decimal(str(budget))
    measured, _, observed_runs = _observed(training_rows)
    _, made_runs = _feasibility_evidence(feasibility_rows)
    if any(made_runs[run] != key for run, key in observed_runs.items() if run in made_runs):
        raise ValueError("performance and makeability evidence disagree on design identity")
    pending = list(pending)
    pending_keys = {_design_key(record["design"]) for record in pending}
    pending_ids = {record["id"] for record in pending}
    tried_keys = set(measured) | set(made_runs.values())
    anchors = tried_keys | pending_keys
    candidates = list(candidates)
    ids = [record.get("id") for record in candidates]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be distinct nonempty strings")
    tie = dict(zip(sorted(ids), np.random.default_rng(seed).random(len(ids))))
    actions, trace, eligible = [], [], []
    for record in candidates:
        ident = record["id"]
        try:
            key, cost, repeat = _candidate_spec(record, pending_ids=pending_ids, pending_keys=pending_keys,
                                                observed_runs=observed_runs, made_runs=made_runs,
                                                tried_keys=tried_keys, remaining=remaining)
            if repeat is not None:
                raise ValueError("confirmation_is_not_a_new_design")
        except ValueError as error:
            trace.append({"id": ident, "stage": "admissibility", "reason": str(error)})
            continue
        eligible.append((ident, key, cost))

    while eligible and len(actions) < batch_size:
        slot = len(actions) + 1
        ranked = []
        for ident, key, cost in eligible:
            if key in anchors or Decimal(str(cost)) > remaining:
                trace.append({"id": ident, "stage": "batch", "slot": slot,
                              "reason": "batch_duplicate" if key in anchors else "over_remaining_budget"})
                continue
            if anchors:
                score = min(_distance(key, other) for other in anchors)
                basis = "nearest_attempted_pending_or_selected"
            else:
                centre = (1/3, 1/3, 1/3, 0.5, key[-1])
                score = 1 / (1 + _distance(key, centre))
                basis = "proximity_to_design_centre"
            terms = {"score": score, "basis": basis, "cost": cost, "anchor_designs": len(anchors)}
            trace.append({"id": ident, "stage": "ranking", "slot": slot, "score_terms": terms})
            ranked.append((score, -cost, tie[ident], ident, key, terms))
        if not ranked:
            break
        _, neg_cost, _, ident, key, terms = max(ranked)
        cost = -neg_cost
        actions.append({"id": ident, "design": _as_design(key), "action": "run", "repeat_of": None,
                        "fresh_preparation_required": True, "cost": cost, "score_terms": terms,
                        "rationale": ("Begin with the affordable design closest to the design-space centre."
                                      if not anchors else "Cover a design farthest from attempted and planned preparations.")})
        anchors.add(key)
        remaining -= Decimal(str(cost))
        eligible = [entry for entry in eligible if entry[0] != ident]

    return {"method": "cold_start_space_filling", "seed": int(seed), "actions": actions, "trace": trace,
            "frontier": [], "reference_point": None, "budget": budget,
            "spent": float(Decimal(str(budget)) - remaining), "remaining_budget": float(remaining),
            "stop_reason": "batch_full" if len(actions) == batch_size else "no_eligible_affordable_design"}
