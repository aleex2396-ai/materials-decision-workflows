"""Shadow adjudication of three soft diagnostic signals on fictional failure banks.

Each bank samples one diagnostic vector per independent preparation. Its simulator
labels are visible to fitting/calibration/scoring only. Nothing here writes to the
registry or changes the eligibility of an observation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from scipy.stats import beta


CAUSES = ("valid", "measurement", "procedure", "mixed")
ACTIONS = np.array(["use", "remeasure", "remake", "ask"])
GRID = (0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98)
ALPHA = 0.05
RISK_LIMIT = 0.05
MAX_DISTANCE_SQUARED = 16.3
FEATURES = ("standard_residual", "process_residual", "repeat_disagreement")


def _bank(rng, count, shifted=False):
    # One row per independently drawn group. Values are residual magnitudes in
    # units of check repeatability, with correlated signal and mixture overlap.
    priors = [0.48, 0.23, 0.20, 0.09] if not shifted else [0.30, 0.24, 0.20, 0.26]
    cause = rng.choice(4, count, p=priors)
    centers = np.array([[0.45, 0.5, 0.7], [2.6, 0.6, 1.8],
                        [0.55, 2.5, 1.45], [2.25, 2.1, 2.1]])
    if shifted:
        # A drifted standard loses sensitivity to measurement errors; some
        # mixed failures likewise resemble ordinary measurement failures.
        centers[1] = [1.65, 0.75, 1.45]
        centers[3] = [2.15, 1.25, 1.85]
    scatter = rng.normal(size=(count, 3)) @ np.array(
        [[0.67, 0.07, 0.18], [0.0, 0.69, 0.18], [0.0, 0.0, 0.68]])
    x = np.abs(centers[cause] + (1.25 if shifted else 1.0) * scatter)
    # Simulated true performance is independent of the diagnostics. Its lowest
    # quintile is reported separately; neither model nor baseline sees it.
    poor = rng.uniform(size=count) < 0.20
    return x, cause, poor


def _fit(x, cause):
    means = np.stack([x[cause == k].mean(axis=0) for k in range(4)])
    pooled = np.cov((x - means[cause]).T)
    fits = []
    for k, mean in enumerate(means):
        group = x[cause == k]
        covariance = 0.8 * np.cov(group.T) + 0.2 * pooled + np.eye(3) * 1e-6
        log_weight = np.log(len(group) / len(x)) - 0.5 * np.linalg.slogdet(covariance)[1]
        fits.append((mean, np.linalg.inv(covariance), log_weight))
    return fits


def _scores(fits, x):
    squared, scores = [], []
    for mean, precision, log_weight in fits:
        residual = x - mean
        distance = (residual @ precision * residual).sum(axis=1)
        squared.append(distance)
        scores.append(log_weight - 0.5 * distance)
    probs = softmax(np.stack(scores, axis=1), axis=1)
    return probs.argmax(axis=1), probs.max(axis=1), np.min(squared, axis=0) <= MAX_DISTANCE_SQUARED


def _policy(scores, threshold):
    cause, confidence, supported = scores
    # Mixed causes always defer. Rejected support is part of the calibrated policy.
    if threshold is None:
        return np.full(len(cause), "ask")
    return np.where((confidence >= threshold) & supported, ACTIONS[cause], "ask")


def _baseline(x):
    standard, process, disagreement = x.T
    valid = (standard < 1.15) & (process < 1.15) & (disagreement < 1.45)
    measurement = (standard >= 1.7) & (process < 1.35) & (disagreement >= 0.6)
    procedure = (standard < 1.35) & (process >= 1.7) & (disagreement >= 0.6)
    return np.select([valid, measurement, procedure], ACTIONS[:3], default="ask")


def _upper_bound(errors, settled, tail):
    if errors == settled:
        return 1.0
    return float(beta.isf(tail, errors + 1, settled - errors))


def _calibrate(scores, cause):
    sweep = []
    for threshold in GRID:
        action = _policy(scores, threshold)
        settled = int((action != "ask").sum())
        errors = int(((action != "ask") & (action != ACTIONS[cause])).sum())
        upper = _upper_bound(errors, settled, ALPHA / len(GRID))
        sweep.append(dict(threshold=threshold, settled=settled, errors=errors,
                          coverage=settled / len(cause), risk_upper=upper, passes=upper <= RISK_LIMIT))
    passing = [row for row in sweep if row["passes"]]
    selected = max(passing, key=lambda row: (row["coverage"], row["threshold"])) if passing else None
    return dict(groups=len(cause), alpha=ALPHA, risk_limit=RISK_LIMIT, thresholds=sweep,
                selected_threshold=selected["threshold"] if selected else None,
                selected_risk_upper=selected["risk_upper"] if selected else None)


def _metrics(action, cause, poor):
    settled = action != "ask"
    expected = ACTIONS[cause]
    wrong = settled & (action != expected)
    harmful_admission = (action == "use") & (cause != 0)
    harmful_exclusion = settled & (action != "use") & (cause == 0)
    poor_valid = poor & (cause == 0)
    per_cause = {}
    for k, name in enumerate(CAUSES):
        subset = cause == k
        per_cause[name] = dict(groups=int(subset.sum()),
                               actions={a: int(((action == a) & subset).sum()) for a in ACTIONS.tolist()})
    return {
        "groups": len(cause), "settled": int(settled.sum()), "asks": int((~settled).sum()),
        "coverage": float(settled.mean()), "settled_errors": int(wrong.sum()),
        "selective_risk": float(wrong.sum() / settled.sum()) if settled.any() else None,
        "harmful_admissions": int(harmful_admission.sum()), "harmful_exclusions": int(harmful_exclusion.sum()),
        "incorrect_remedies": int((wrong & ~harmful_admission & ~harmful_exclusion).sum()),
        "valid_poor_groups": int(poor_valid.sum()), "valid_poor_admitted": int(((action == "use") & poor_valid).sum()),
        "by_cause": per_cause,
    }


def _decision(fits, threshold, signals):
    if not isinstance(signals, (tuple, list)) or len(signals) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (float, int, np.integer, np.floating)) for value in signals
    ):
        return dict(action="ask", reason="invalid_signals", predicted_cause=None, confidence=None)
    # Bound conversion before squaring extremely large, out-of-domain inputs.
    if any(not 0 <= value <= 100 for value in signals):
        return dict(action="ask", reason="signals_out_of_support", predicted_cause=None, confidence=None)
    scores = _scores(fits, np.array([signals], dtype=float))
    cause, confidence, supported = scores
    action = str(_policy(scores, threshold)[0])
    if not supported[0]:
        return dict(action="ask", reason="signals_out_of_support", predicted_cause=None, confidence=None)
    elif cause[0] == 3:
        reason = "mixed_evidence"
    elif threshold is None or confidence[0] < threshold:
        reason = "insufficient_calibrated_confidence"
    else:
        reason = "supported_shadow_suggestion"
    return dict(action=action, reason=reason, predicted_cause=CAUSES[cause[0]], confidence=float(confidence[0]))


def run_shadow_study(seed: int, cases: dict[str, list[float]]) -> dict:
    """Evaluate once per fresh bank and suggest actions for caller-supplied review cases."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    train, labels, _ = _bank(streams[0], 1100)
    model = _fit(train, labels)
    calibrators, truth, _ = _bank(streams[1], 1200)
    calibration = _calibrate(_scores(model, calibrators), truth)
    threshold = calibration["selected_threshold"]
    evaluation = {}
    for index, name in enumerate(("iid", "shifted"), start=2):
        evidence, actual, poor = _bank(streams[index], 1500, shifted=name == "shifted")
        evaluation[name] = {"model": _metrics(_policy(_scores(model, evidence), threshold), actual, poor),
                            "baseline": _metrics(_baseline(evidence), actual, poor)}
    return {
        "seed": int(seed), "mode": "shadow_only", "training_groups": len(labels),
        "decisions": {run: _decision(model, threshold, signals) for run, signals in cases.items()},
        "calibration": calibration, "evaluation": evaluation,
        "method": {
            "features": list(FEATURES), "training_unit": "one independent synthetic preparation per row",
            "classifier": "Class-conditional Gaussians, 20% pooled covariance shrinkage and empirical class priors.",
            "selection": "Highest calibration coverage passing the simultaneous upper bound; highest threshold breaks ties. Defer all if none pass.",
            "bound": "One-sided exact binomial upper bounds on error conditional on settling, Bonferroni across the fixed grid.",
            "scope": "The risk bound concerns fresh independent draws from the calibration generator. Cases routed to review can have a different mix of causes. Calibration for that subset or a physical lab requires new labelled groups from the intended referral process. Shifted scores are descriptive. No registry mutation.",
            "loss": "Every incorrect settled action incurs one error. Ask defers. Using a compromised result is harmful admission; intervening on a valid result is harmful exclusion. Other settled errors are incorrect remedies.",
            "baseline": "Fixed rules on the same three signals. A single large check residual and corroborating disagreement triggers intervention; three small residuals permit use; other cases defer.",
            "references": ["https://arxiv.org/abs/2110.01052", "https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm"],
        },
    }
