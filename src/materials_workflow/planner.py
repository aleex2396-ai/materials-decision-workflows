"""Finite-library, two-objective batch planning for the invented membrane study."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from numbers import Integral, Real

import numpy as np
from .formulation import _descriptor_vector


OBJECTIVES = ("transmission", "toughness")
FACTORS = ("a", "b", "c", "cure")
# Planning conventions on the invented scales. Prevent division by zero for a flat history.
MINIMUM_SPANS = np.array([0.05, 0.5])
KERNEL_WIDTH = 0.35
DIVERSITY_WIDTH = 0.20


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _design_key(design):
    if not isinstance(design, dict):
        raise ValueError("design must be an object")
    if set(design) != {*FACTORS, "route"}:
        raise ValueError("unexpected_or_missing_design_field")
    x = tuple(_number(design.get(k), k) for k in FACTORS)
    if any(v < 0 or v > 1 for v in x) or abs(sum(x[:3]) - 1) > 1e-8:
        raise ValueError("invalid fractions or cure setting")
    route = design.get("route")
    if route not in ("cast", "press"):
        raise ValueError("unknown route")
    return (*[round(v, 8) for v in x], route)


def _as_design(key):
    return dict(zip((*FACTORS, "route"), key))


def _distance(left, right):
    # Pure-A to pure-B, or opposite ends of the cure axis, are both unit distance.
    mixture = sum((a - b) ** 2 for a, b in zip(left[:3], right[:3])) / 2
    return float(np.sqrt(mixture + (left[3] - right[3]) ** 2 + (0.65 * (left[4] != right[4])) ** 2))


def _observed(rows):
    """Average within each preparation first, then weight independent preparations equally."""
    by_design = defaultdict(lambda: defaultdict(list))
    observed_runs = {}
    for row in rows:
        run, group = row["run_id"], row["group_id"]
        if not run or not group or run in observed_runs:
            raise ValueError("training rows require distinct run IDs and a preparation group")
        key = _design_key(row["design"])
        y = [_number(row["targets"][name], name) for name in OBJECTIVES]
        if not 0 <= y[0] <= 1 or y[1] <= 0:
            raise ValueError("measured performance outside the declared physical scale")
        by_design[key][group].append(y)
        observed_runs[run] = key
    means = {key: np.mean([np.mean(y, axis=0) for y in groups.values()], axis=0)
             for key, groups in by_design.items()}
    counts = {key: len(groups) for key, groups in by_design.items()}
    return means, counts, observed_runs


def _feasibility_evidence(rows):
    evidence = defaultdict(list)
    seen = {}
    for row in rows:
        run, group = row["run_id"], row["group_id"]
        if not run or not group or run in seen or type(row.get("success")) is not bool:
            raise ValueError("makeability requires one adjudicated boolean per identified attempt")
        key = _design_key(row["design"])
        evidence[group].append((key, row["success"]))
        seen[run] = key
    return evidence, seen


def _makeability(key, evidence):
    """Local beta pseudo-counts. Any one preparation contributes at most one unit."""
    total = successes = 0.0
    for attempts in evidence.values():
        weights = np.array([np.exp(-0.5 * (_distance(key, k) / KERNEL_WIDTH) ** 2)
                            for k, _ in attempts])
        weight = float(max(weights))
        rate = float(np.average([int(ok) for _, ok in attempts], weights=weights))
        successes += weight * rate
        total += weight
    return {"probability": (1 + successes) / (2 + total), "effective_groups": total}


def _front(points):
    """Non-dominated unique points, increasing first objective and decreasing second."""
    best, frontier = -np.inf, []
    for x, y in sorted({tuple(point) for point in points}, reverse=True):
        if y > best:
            frontier.append((float(x), float(y)))
            best = y
    return frontier[::-1]


def _improvement(draws, front):
    """Area gained over a 2D maximisation frontier, with reference at the origin."""
    x, y = draws.T
    area = np.zeros_like(x)
    left = 0.
    for right, height in front:
        width = np.maximum(np.minimum(x, right) - left, 0)
        area += width * np.maximum(y - height, 0)
        left = right
    return area + np.maximum(x - left, 0) * np.maximum(y, 0)


def _utility_draws(prediction):
    draws = np.column_stack([prediction["draws"][name] for name in OBJECTIVES]).astype(float)
    if draws.ndim != 2 or draws.shape[1] != 2 or not draws.size or not np.isfinite(draws).all():
        raise ValueError("paired predictive draws must be finite and nonempty")
    # Saturate utility at physical limits. Never modify an observation or model prediction.
    draws[:, 0] = np.clip(draws[:, 0], 0, 1)
    draws[:, 1] = np.maximum(draws[:, 1], 0)
    return draws


def _objective_dict(values):
    return {name: float(value) for name, value in zip(OBJECTIVES, values)}


def _candidate_spec(candidate, *, pending_ids, pending_keys, observed_runs, made_runs, tried_keys, remaining):
    """Shared admission checks for outcome-guided and space-filling planning."""
    ident, repeat = candidate["id"], candidate.get("repeat_of")
    key = _design_key(candidate.get("design"))
    _descriptor_vector(candidate.get("descriptors"))
    cost = _number(candidate.get("cost"), "cost")
    if cost <= 0:
        raise ValueError("cost_must_be_positive")
    if ident in pending_ids or key in pending_keys:
        raise ValueError("pending_design_or_id")
    if ident in observed_runs or ident in made_runs:
        raise ValueError("used_run_id")
    if repeat is not None and (not isinstance(repeat, str) or observed_runs.get(repeat) != key):
        raise ValueError("repeat_requires_matched_measured_run")
    if key in tried_keys and repeat is None:
        raise ValueError("already_attempted_without_repeat_request")
    if Decimal(str(cost)) > remaining:
        raise ValueError("over_budget")
    return key, cost, repeat


def recommend_batch(model, candidates, training_rows, feasibility_rows, pending, *, budget, batch_size, seed):
    """Plan only new spend. Pending experiments have already been budgeted by the caller.

    Candidates are objects with id, design, cost, optional descriptors and optional repeat_of.
    repeat_of names a valid measured run of exactly that design. A repeat must be prepared
    independently. Pending entries carry id and design; known recipes are treated as repeats.
    Model.predict(design, descriptors) supplies paired within-candidate outcome draws and
    applicability. Training inputs are the caller's eligible development rows only.
    """
    budget = _number(budget, "budget")
    if budget < 0 or isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 0:
        raise ValueError("budget and integer batch_size must be nonnegative")
    remaining = Decimal(str(budget))
    measured, group_counts, observed_runs = _observed(training_rows)
    evidence, made_runs = _feasibility_evidence(feasibility_rows)
    if any(made_runs[run] != key for run, key in observed_runs.items() if run in made_runs):
        raise ValueError("performance and makeability evidence disagree on design identity")
    pending = list(pending)
    pending_keys = {_design_key(record["design"]) for record in pending}
    pending_ids = {record["id"] for record in pending}
    occupied = pending_keys.copy()
    tried_keys = set(made_runs.values()) | set(measured)
    candidates = list(candidates)
    ids = [c.get("id") for c in candidates]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be distinct nonempty strings")

    actions, trace = [], []
    report = {
        "actions": actions, "trace": trace, "frontier": [], "reference_point": None,
        "objectives": {"transmission": "maximize fraction", "toughness": "maximize kJ/m2"},
        "budget": budget, "spent": 0., "remaining_budget": budget, "seed": int(seed),
        "method": "greedy cost- and makeability-weighted Monte Carlo hypervolume improvement",
        "utility": {
            "new_design": "expected normalized Pareto area gained by a measured outcome",
            "independent_repeat": "expected absolute area change when an independent group updates this design's mean",
            "batch": "include in-support new pending/selected predictive means as temporary frontier points; downweight nearby designs",
            "score": "expected area utility * local makeability probability * diversity factor / cost",
            "reference": "measured minima minus 10% of spans; spans are measured ranges floored at 0.05 fraction and 0.5 kJ/m2",
            "makeability": "Beta(1,1) local kernel estimate, Gaussian width 0.35 in design space, one effective vote per preparation",
            "diversity": "1 - exp(-distance^2 / (2 * 0.20^2)); minimum distance to a pending or selected design",
            "prediction": "paired within-design observation draws, with transmission utility saturated to [0,1] and toughness floored at zero",
        },
        "limits": ["Model-based, uncalibrated outcome and makeability uncertainty.",
                   "Greedy cost efficiency is approximate; pending mean fantasies assume successful makes and do not model joint batch covariance.",
                   "Confirmation utility measures movement in the observed frontier; it can value measurement scatter. It is not expected reduction in posterior loss."],
    }
    if not measured:
        report["stop_reason"] = "no_measured_reference"
        return report

    measured_array = np.asarray(list(measured.values()))
    scales = np.maximum(np.ptp(measured_array, axis=0), MINIMUM_SPANS)
    reference = measured_array.min(axis=0) - 0.1 * scales
    norm = {key: (value - reference) / scales for key, value in measured.items()}
    report["reference_point"] = _objective_dict(reference)
    report["normalization_spans"] = _objective_dict(scales)
    boundary = set(_front(measured.values()))
    report["frontier"] = [{"design": _as_design(key), "targets": _objective_dict(measured[key]),
                           "independent_groups": group_counts[key]}
                          for key in sorted(measured) if tuple(measured[key]) in boundary]

    fantasies = []
    for record in pending:
        key = _design_key(record["design"])
        if key not in measured:
            prediction = model.predict(record["design"], record.get("descriptors"))
            if prediction["applicability"]["status"] == "in_support":
                fantasies.append((_utility_draws(prediction).mean(axis=0) - reference) / scales)

    eligible = []
    tie_break = dict(zip(sorted(ids), np.random.default_rng(seed).random(len(ids))))
    for candidate in candidates:
        ident = candidate["id"]
        try:
            key, cost, repeat = _candidate_spec(candidate, pending_ids=pending_ids, pending_keys=pending_keys,
                                                observed_runs=observed_runs, made_runs=made_runs,
                                                tried_keys=tried_keys, remaining=remaining)
        except ValueError as err:
            trace.append({"id": ident, "stage": "admissibility", "reason": str(err)})
            continue
        prediction = model.predict(candidate["design"], candidate.get("descriptors"))
        if prediction["applicability"]["status"] != "in_support":
            trace.append({"id": ident, "stage": "admissibility", "reason": "outside_model_support",
                          "details": prediction["applicability"]["reasons"]})
            continue
        draws = (_utility_draws(prediction) - reference) / scales
        eligible.append((ident, key, cost, repeat, draws, _makeability(key, evidence)))

    while eligible and len(actions) < batch_size:
        scores = []
        for ident, key, cost, repeat, draws, feasibility in eligible:
            if key in occupied or Decimal(str(cost)) > remaining:
                reason = "batch_duplicate" if key in occupied else "over_remaining_budget"
                trace.append({"id": ident, "stage": "batch", "slot": len(actions) + 1, "reason": reason})
                continue
            observed_front = [value for other, value in norm.items() if repeat is None or other != key]
            # Negative fantasy coordinates cannot contribute a rectangle above the reference.
            front = _front([p for p in observed_front + fantasies if min(p) > 0])
            if repeat is None:
                improvement = _improvement(draws, front)
            else:
                n = group_counts[key]
                updated = (n * norm[key] + draws) / (n + 1)
                old = _improvement(norm[key][None, :], front)[0]
                improvement = abs(_improvement(updated, front) - old)
            gain = float(np.mean(improvement))
            nearest = min((_distance(key, other) for other in occupied), default=None)
            diversity = 1. if nearest is None else float(-np.expm1(-0.5 * (nearest / DIVERSITY_WIDTH) ** 2))
            score = gain * feasibility["probability"] * diversity / cost
            terms = {"expected_area_utility": gain, "makeability_probability": feasibility["probability"],
                     "effective_feasibility_groups": feasibility["effective_groups"], "diversity": diversity,
                     "nearest_pending_or_selected": nearest, "cost": cost, "score": float(score),
                     "predictive_draws": len(draws)}
            trace.append({"id": ident, "stage": "ranking", "slot": len(actions) + 1, "score_terms": terms})
            scores.append((score, tie_break[ident], ident, key, cost, repeat, draws, terms))
        if not scores:
            report["stop_reason"] = "no_eligible_affordable_design"
            break
        score, _, ident, key, cost, repeat, draws, terms = max(scores, key=lambda entry: entry[:2])
        if score <= 0:
            report["stop_reason"] = "no_positive_marginal_utility"
            break
        action = "independent_repeat" if repeat is not None else "run"
        actions.append({"id": ident, "design": _as_design(key), "action": action, "cost": cost,
                        "repeat_of": repeat, "fresh_preparation_required": True, "score_terms": terms,
                        "rationale": ("Confirm this measured trade-off on a fresh preparation."
                                      if repeat is not None else "Test this design for a predicted improvement of the observed trade-off.")})
        if repeat is None:
            fantasies.append(draws.mean(axis=0))
        occupied.add(key)
        remaining -= Decimal(str(cost))
        eligible = [entry for entry in eligible if entry[0] != ident]
    report["spent"] = float(Decimal(str(budget)) - remaining)
    report["remaining_budget"] = float(remaining)
    report.setdefault("stop_reason", "batch_full" if len(actions) == batch_size else "no_eligible_affordable_design")
    return report
