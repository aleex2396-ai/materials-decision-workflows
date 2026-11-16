"""Invented optical-composite campaign. No file input, fitted constants or external data."""

from __future__ import annotations

from hashlib import sha256
import numpy as np

DESCRIPTORS = ("density", "scattering", "viscosity")
UNITS = ("g/cm3", "1/mm", "Pa s")


def design_from(rng, index):
    fractions = rng.dirichlet([1.5, 1.5, 1.5])
    a, b = (round(float(x), 5) for x in fractions[:2])
    return {"a": a, "b": b, "c": round(1 - a - b, 5),
            "cure": round(float(rng.uniform(0.08, 0.92)), 4),
            "route": ("cast", "press")[index % 2]}


def material_card(design, rng):
    """Synthetic values available from a stock certificate before manufacture."""
    a, b, c = (design[name] for name in ("a", "b", "c"))
    stock = rng.normal(0, 0.08)
    density = 1.35 + 0.3 * b + 0.18 * c + stock
    scattering = 0.35 + 0.85 * b + 0.2 * c + 0.8 * stock + rng.normal(0, 0.03)
    viscosity = 1.2 + 1.1 * density + 0.7 * scattering + rng.normal(0, 0.1)
    return dict(zip(DESCRIPTORS, map(float, (density, scattering, viscosity))))


def instrument_return(run_id, group_id, design, properties, visible, rng, *, case="normal",
                      alternate_units=False, prep_offset=0.0):
    a, b, c = (design[name] for name in ("a", "b", "c"))
    cure = design["cure"]
    press = design["route"] == "press"
    # Two conflicting objectives, a nonlinear mixture effect and a shared preparation effect.
    logit = (0.5 + 3.3 * a - 1.8 * b + 1.2 * a * c + 0.7 * cure - 0.3 * press
             - 0.8 * properties["scattering"] + prep_offset + rng.normal(0, 0.065))
    transmission = float(1 / (1 + np.exp(-logit)))
    toughness = float(1.6 + 7 * b + 1.9 * c + 4.1 * a * b + 1.7 * cure + 1.4 * press
                      + 0.12 * properties["viscosity"] + 2 * prep_offset + rng.normal(0, 0.16))
    evidence = {"instrument_check_passed": case != "meter",
                "protocol_followed": case != "procedure",
                "sample_identity_confirmed": case != "ambiguous"}
    if case == "mixed":
        evidence.update(instrument_check_passed=False, protocol_followed=False)
    if case == "poor":
        transmission, toughness = 0.18, 1.4
    targets = {"transmission": {"value": transmission * (100 if alternate_units else 1),
                                "unit": "%" if alternate_units else "fraction"},
               "toughness": {"value": toughness * (1000 if alternate_units else 1),
                             "unit": "J/m2" if alternate_units else "kJ/m2"}}
    if case == "missing_endpoint":
        targets.pop("toughness")
    if case == "failed":
        targets = {}
    return {"run_id": run_id, "group_id": group_id, "source_id": "simulated-instrument#" + run_id,
            "design": dict(design), "make_status": "failed" if case == "failed" else "success",
            "evidence": evidence, "descriptors": {
                k: {"value": properties[k], "unit": unit}
                for k, unit in zip(DESCRIPTORS, UNITS) if k in visible
            }, "targets": targets}


def initial_returns(config):
    """Two coupons per fresh preparation. Withhold selected supplier properties by design."""
    rng = np.random.default_rng(config["seed"])
    cases = {2: "meter", 4: "failed", 9: "procedure", 12: "ambiguous",
             15: "mixed", 20: "poor", 25: "missing_endpoint"}
    records = []
    for i in range(config["preparation_groups"]):
        design = design_from(rng, i)
        card = material_card(design, rng)
        offset = float(rng.normal(0, 0.09))
        visible = [name for j, name in enumerate(DESCRIPTORS) if (i + 2 * j) % 6 != 2]
        for coupon in range(2):
            case = cases.get(i, "normal")
            if coupon and case in {"meter", "ambiguous", "missing_endpoint", "poor"}:
                case = "normal"
            record = instrument_return(f"G{i:03d}-{coupon}", f"G{i:03d}", design, card, visible,
                                       rng, case=case, alternate_units=bool(coupon), prep_offset=offset)
            if i in (7, 17) and coupon == 0:
                record["issue"] = "discordant_replicates"
                record["diagnostic_signals"] = ([2.4, 0.3, 1.8] if i == 7 else [1.1, 1.3, 1.1])
            records.append(record)
    return records


def _candidate_stock(design, campaign_seed, candidate_id):
    # Regenerate the same entire certificate at execution. A planner only receives
    # the measured subset. Independent streams keep costs and selection out of stock generation.
    key = sha256(f"{campaign_seed}/stock/{candidate_id}".encode()).digest()
    return material_card(design, np.random.default_rng(int.from_bytes(key[:16], "big")))


def candidate_pool(config, known_run):
    rng = np.random.default_rng(config["seed"] + 1)
    candidates = []
    for i in range(config["candidate_count"]):
        design = design_from(rng, i)
        ident = f"M{i:03d}"
        card = _candidate_stock(design, config["seed"], ident)
        # These partial stock specifications are known before the assay.
        visible = {k: card[k] for j, k in enumerate(DESCRIPTORS) if (i + j) % 3 == 0}
        cost = round(1 + design["cure"] + 0.7 * (design["route"] == "press") + 0.3 * design["c"], 2)
        candidates.append({"id": ident, "design": design, "cost": cost,
                           "descriptors": visible})
    repeat_stock = _candidate_stock(known_run["design"], config["seed"], "confirm-reference")
    candidates.append({"id": "confirm-reference", "design": known_run["design"], "cost": 1.4,
                       "descriptors": {"density": repeat_stock["density"]},
                       "repeat_of": known_run["run_id"]})
    return candidates


def execute_synthetic(actions, candidates, campaign_seed, *, round_id):
    """Return independent preparation groups, carrying forward the proposed design and stock evidence."""
    lookup = {c["id"]: c for c in candidates}
    records = []
    for action in actions:
        candidate = lookup[action["id"]]
        key = sha256(f"{campaign_seed}/{round_id}/{action['id']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:16], "big"))
        card = _candidate_stock(candidate["design"], campaign_seed, action["id"])
        if any(card[k] != v for k, v in candidate.get("descriptors", {}).items()):
            raise ValueError("The candidate disagrees with the simulated stock certificate.")
        group = f"{round_id}-{action['id']}"
        record = instrument_return("return-" + group, group, candidate["design"], card,
                                   candidate.get("descriptors", {}), rng, prep_offset=rng.normal(0, 0.09))
        records.append(record)
    return records
