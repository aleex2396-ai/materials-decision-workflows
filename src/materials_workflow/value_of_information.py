"""Price one supplier measurement before choosing one fictional membrane to make."""
from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .registry import DESCRIPTORS


def _physical(name, value):
    return np.isfinite(value) and (value > 0 if name == "density" else value >= 0)


def _expected_utility(prediction):
    """A bounded synthetic preference on a new run, before its outcome is observed."""
    t = np.asarray(prediction["draws"]["transmission"], float)
    k = np.asarray(prediction["draws"]["toughness"], float)
    if (t.ndim != 1 or not len(t) or t.shape != k.shape or not np.isfinite(t).all()
            or not np.isfinite(k).all() or np.any((t < 0) | (t > 1)) or np.any(k < 0)):
        raise ValueError("Expected aligned physical draws for both membrane endpoints")
    return float((0.5*t + 0.5*k/(k + 5.0)).mean())


def _card_bank(model, candidate, normals):
    """Sample complete stock cards once, sharing the bank across all possible assays."""
    ident, design = candidate["id"], candidate["design"]
    supplied = candidate.get("descriptors") or {}
    prediction = model.predict(design, supplied)
    entry = {"id": ident, "status": prediction["applicability"]["status"], "assays": []}
    if entry["status"] != "in_support":
        return entry, None
    known = {key: value for key, value in supplied.items() if value is not None}
    summary = model.descriptor_estimates(known)
    missing = [key for key in DESCRIPTORS if key not in known and summary[key]["used_in_model"]]
    entry["unavailable_descriptors"] = [key for key in DESCRIPTORS if key not in known and key not in missing]
    if not missing:
        entry.update(expected_utility=_expected_utility(prediction), retained_cards=1, proposed_cards=1)
        return entry, None

    cards, utilities = [], []
    # Sequential conditional normals sample a joint Gaussian. Reject the entire
    # card if any entry is unphysical or beyond training support. Truncating
    # each marginal independently would destroy the conditional correlations.
    proposed = 0
    for normal in normals:
        proposed += 1
        card = dict(known)
        for index, key in enumerate(missing):
            conditional = summary[key] if index == 0 else model.descriptor_estimates(card)[key]
            value = conditional["mean"] + conditional["sd"] * normal[DESCRIPTORS.index(key)]
            if not _physical(key, value):
                break
            card[key] = float(value)
        else:
            result = model.predict(design, card)
            if result["applicability"]["status"] == "in_support":
                cards.append([card[key] for key in missing])
                utilities.append(_expected_utility(result))
                if len(cards) == 128:
                    break
    entry.update(retained_cards=len(cards), proposed_cards=proposed)
    if len(cards) < 16:
        entry["status"] = "insufficient_supported_stock_cards"
        return entry, None
    entry["expected_utility"] = float(np.mean(utilities))
    return entry, (missing, np.array(cards), np.array(utilities))


def _price(ident, key, values, utility, current, fallback, base, cost):
    """The same empirical prior and fixed alternative are used on both sides of EVSI."""
    # Approximate a scalar assay by reporting an interval. No equal observed
    # value is split between bins; a constant measurement therefore has zero value.
    cuts = np.unique(np.quantile(values, np.arange(1, 8)/8))
    cuts = cuts[(cuts > values.min()) & (cuts < values.max())]
    labels = np.searchsorted(cuts, values, side="right")
    bins = []
    after = switches = 0.0
    for i in sorted(set(labels)):
        mask = labels == i
        probability = float(mask.mean())
        conditional = float(utility[mask].mean())
        chosen = ident
        if fallback and (fallback["expected_utility"] > conditional or
                         (fallback["expected_utility"] == conditional and current != ident)):
            chosen = fallback["id"]
        best = max(conditional, fallback["expected_utility"]) if fallback else conditional
        switches += probability * (chosen != current)
        after += probability * best
        bins.append({"lower_inclusive": None if i == 0 else float(cuts[i - 1]),
                     "upper_exclusive": None if i == len(cuts) else float(cuts[i]),
                     "probability": probability, "conditional_utility": conditional,
                     "make_candidate_id": chosen})
    gain = max(after - base, 0.0) if fallback else 0.0
    return {"candidate_id": ident, "descriptor": key, "expected_gain": gain, "credit_cost": cost,
            "expected_net_value": gain-cost, "switch_probability": switches,
            "fixed_alternative": fallback["id"] if fallback else None, "bins": bins}


def measure_or_make(model, shortlist, *, seed: int, credit_cost: float = 0.01):
    """Evaluate a single binned assay, then choose just one supported recipe.

    The shortlisted recipes have independent latent stock cards. A result can
    update the measured recipe's utility, while all other expected utilities
    remain fixed. A joint bank retains within-card descriptor correlations.
    """
    if (isinstance(credit_cost, bool) or not isinstance(credit_cost, Real)
            or not np.isfinite(credit_cost) or credit_cost < 0):
        raise ValueError("credit_cost must be a finite nonnegative utility amount")
    cost = float(credit_cost)
    shortlist = list(shortlist)
    ids = [r["id"] for r in shortlist]
    if any(not isinstance(ident, str) or not ident.strip() for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError("Shortlisted recipes require distinct IDs")
    # Scrambled low-discrepancy draws, reused across recipes. Each predictive
    # call also uses the fitted model's fixed within-candidate random numbers.
    samples = qmc.Sobol(len(DESCRIPTORS), scramble=True, seed=seed).random_base2(10)
    normals = ndtri(np.clip(samples, 1e-12, 1-1e-12))
    entries, banks = [], {}
    for candidate in sorted(shortlist, key=lambda r: r["id"]):
        entry, bank = _card_bank(model, candidate, normals)
        entries.append(entry)
        if bank is not None:
            banks[entry["id"]] = bank
    ranked = sorted((e for e in entries if "expected_utility" in e),
                    key=lambda e: (-e["expected_utility"], e["id"]))
    choice = {"action": "abstain", "candidate_id": None, "descriptor": None,
              "credit_cost": 0.0, "expected_net_value": 0.0}
    if ranked:
        current, base = ranked[0]["id"], ranked[0]["expected_utility"]
        choice.update(action="make", candidate_id=current)
        for entry in ranked:
            ident = entry["id"]
            if ident not in banks:
                continue
            keys, stock, utilities = banks[ident]
            fallback = next((r for r in ranked if r["id"] != ident), None)
            for col, key in enumerate(keys):
                assay = _price(ident, key, stock[:, col], utilities, current, fallback, base, cost)
                entry["assays"].append(assay)
                if assay["expected_net_value"] > choice["expected_net_value"] + 1e-12:
                    choice.update(action="measure", candidate_id=ident, descriptor=key,
                                  credit_cost=cost, expected_net_value=assay["expected_net_value"])
    return {
        "recommended_action": choice, "candidates": entries,
        "make_now_candidate_id": ranked[0]["id"] if ranked else None,
        "make_now_expected_utility": ranked[0]["expected_utility"] if ranked else None,
        "assumptions": [
            "Choose one shortlisted recipe. Fixed fictional utility = 0.5*transmission + 0.5*toughness/(toughness+5 kJ/m²).",
            "Assay credit_cost is in the same utility units; it is separate from the batch-planning budget.",
            "One assay reports one of up to eight supplier-property intervals accurately. No later assay is priced.",
            "Stock cards are independent across recipes. Within-card conditional Gaussian draws retain supplier correlations.",
            "Utility averages the complete-card predictor before and after revealing an interval. Acquisition of a value does not change the material.",
            "Missing fields are treated as unreported supplier properties; informative selection or assay error would require another observation model.",
            "Finite quadrature retains up to 128 positive, in-support cards from 1024 proposals and requires 16 if sampling is needed.",
            "The working posterior and assay values are uncalibrated. The calculation concerns the supplied shortlist and this synthetic utility.",
        ],
    }
