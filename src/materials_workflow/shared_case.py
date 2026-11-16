"""A stock-lot decision experiment with a fitted prior and a separate synthetic lab.

The lab equations create a historical calibration panel and future physical returns.
The policy receives a finite prior compiled from the public fitted model and a
declared instrument model. It never receives the chosen world's index or noise.
"""

from __future__ import annotations

from hashlib import sha256
import math

import numpy as np
from scipy.special import expit, logit, ndtr

from .formulation import PerformanceModel
from .registry import Registry


DESIGNS = {
    "binder-rich": {"a": 0.75, "b": 0.15, "c": 0.10, "route": "cast", "cure": 0.65},
    "binder-lean": {"a": 0.15, "b": 0.55, "c": 0.30, "route": "cast", "cure": 0.65},
    "reference": {"a": 0.48, "b": 0.32, "c": 0.20, "route": "cast", "cure": 0.12},
}
LOT_VISCOSITY = {"low": 2.0, "high": 6.0}  # Pa s
ARCHIVE_OFFSETS = {"weak": -0.35, "strong": 0.35}  # fixed log-strength offsets in one old coupon
WORLD_IDS = [f"{lot}/{condition}" for lot in LOT_VISCOSITY for condition in ARCHIVE_OFFSETS]
LOT_ID = "incoming-stock-01"
ARCHIVE = "reference-archive"
ARCHIVE_GROUP = "archived-reference-prep"
CARRIER_VISCOSITY = 4.0
VISCOMETER_SD = 0.8
PREP_SD = 0.11
STRENGTH_METER_SD = 0.10
STRENGTH_CUT = 7.0  # kJ/m²


def _blend_viscosity(design, lot):
    # A stipulated rule of this simulator, not a physical mixing law.
    return design["a"] * lot + (1 - design["a"]) * CARRIER_VISCOSITY


def _true_centres(design, lot):
    """Latent logit transmission and log toughness before prep or instrument noise."""
    a = design["a"]
    delta = _blend_viscosity(design, lot) - CARRIER_VISCOSITY
    cure = design["cure"] - 0.5
    return 0.5 + 0.45*a - 0.90*delta + 1.25*cure, 1.90 + 0.15*a - 0.27*delta + 0.50*cure


def _endpoints(centres, prep_offset, rng):
    t, k = centres
    return {"transmission": float(expit(t + prep_offset + rng.normal(0, 0.035))),
            "toughness": float(np.exp(k + prep_offset + rng.normal(0, STRENGTH_METER_SD)))}


def _fresh_make_utility(centres):
    # Integrate only true preparation variation. Meter scatter does not make a
    # formulation more valuable. The same preference is used throughout.
    z, w = np.polynomial.hermite.hermgauss(16)
    delta = math.sqrt(2) * PREP_SD * z
    transmission = expit(centres[0] + delta)
    strength_term = expit(centres[1] + delta - math.log(5))
    return float(w @ (0.5 * transmission + 0.5 * strength_term) / math.sqrt(math.pi))


def _failure_probability(design, lot):
    # More viscous blends are harder to make. A failure spends the full make cost.
    return float(expit(-4.2 + 0.85 * (_blend_viscosity(design, lot) - CARRIER_VISCOSITY)))


def _historical_model(seed):
    """Balanced calibration panel with past stocks; neither challenge lot is included."""
    rng = np.random.default_rng(seed)
    rows = []
    for lot in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
        for recipe, design in DESIGNS.items():
            for prep in range(4):
                group = f"historical-{lot:g}-{recipe}-{prep}"
                responses = _endpoints(_true_centres(design, lot), rng.normal(0, PREP_SD), rng)
                rows.append({"run_id": group, "group_id": group, "design": design,
                             "descriptors": {"viscosity": _blend_viscosity(design, lot)},
                             "targets": responses})
    return PerformanceModel(seed=seed).fit(rows)


def _action_likelihoods(kind, recipe, predicted):
    """Instrument/make observation likelihoods under each possible world."""
    odds = {}
    for world in WORLD_IDS:
        lot_band, condition = world.split("/")
        lot = LOT_VISCOSITY[lot_band]
        if kind == "assay":
            high = float(ndtr((lot - CARRIER_VISCOSITY) / VISCOMETER_SD))
            labels = {"low": 1 - high, "high": high}
        else:
            strength = predicted[recipe][lot_band][1]
            if kind == "reread":
                # The archived coupon already has its preparation offset. A new
                # read adds meter noise alone, even if future tests share the lot.
                above = float(ndtr((strength + ARCHIVE_OFFSETS[condition] - math.log(STRENGTH_CUT))
                                   / STRENGTH_METER_SD))
                labels = {"above": above, "below": 1 - above}
            else:
                above = float(ndtr((strength - math.log(STRENGTH_CUT)) / math.hypot(PREP_SD, STRENGTH_METER_SD)))
                fail = _failure_probability(DESIGNS[recipe], lot)
                labels = {"above": (1 - fail)*above, "below": (1 - fail)*(1 - above), "failed": fail}
        for label, probability in labels.items():
            odds.setdefault(label, []).append(probability)
    return odds


def build_case(config):
    """Build one public prior for every branch and a separate lab execution context."""
    seed = config["seed"]
    model = _historical_model(seed)
    predicted = {}
    for recipe, design in DESIGNS.items():
        predicted[recipe] = {}
        for band, lot in LOT_VISCOSITY.items():
            forecast = model.predict(design, {"viscosity": _blend_viscosity(design, lot)})
            if forecast["applicability"]["status"] != "in_support":
                raise ValueError(f"Unsupported scenario {recipe}/{band}")
            # Predictive medians anchor the preparation's centres. The public
            # process/measurement calibration supplies the two variance components.
            t = float(np.median(forecast["draws"]["transmission"]))
            k = float(np.median(forecast["draws"]["toughness"]))
            predicted[recipe][band] = (float(logit(t)), math.log(k))
    utilities = {
        recipe: [_fresh_make_utility(centres[world.split("/")[0]]) for world in WORLD_IDS]
        for recipe, centres in predicted.items()
    }
    actions = [{"id": "assay-stock", "kind": "assay", "cost": 1, "stock_lot": LOT_ID,
                "observable": {"name": "stock viscosity", "unit": "Pa s", "threshold": CARRIER_VISCOSITY,
                               "below": "low", "at_or_above": "high"}},
               {"id": "reread-reference", "kind": "reread", "recipe": "reference", "cost": 1,
                "specimen": ARCHIVE},
               {"id": "remake-reference", "kind": "remake", "recipe": "reference", "cost": 4,
                "specimen": "fresh-reference", "success_outcomes": ["above", "below"]}]
    for recipe in ("binder-rich", "binder-lean"):
        actions.append({"id": "make-" + recipe, "kind": "make", "recipe": recipe, "cost": 4,
                        "specimen": "fresh-" + recipe, "success_outcomes": ["above", "below"]})
    for action in actions:
        if "recipe" in action:
            action["stock_lot"] = LOT_ID
            action["design"] = DESIGNS[action["recipe"]].copy()
            action["group_id"] = ARCHIVE_GROUP if action["kind"] == "reread" else "independent-prep-" + action["recipe"]
            if action["kind"] == "remake":
                action["repeats_preparation"] = ARCHIVE_GROUP
            action["registry_return"] = {"target": "toughness", "threshold": STRENGTH_CUT,
                                         "below": "below", "at_or_above": "above", "failure": "failed"}
        action["likelihoods"] = _action_likelihoods(action["kind"], action.get("recipe"), predicted)
    spec = {"budget": config.get("budget", 5), "utility_per_credit": config.get("utility_per_credit", 0.003),
            "prior": [0.25] * 4, "utilities": utilities, "initial_qualified": ["reference"],
            "initial_specimens": [ARCHIVE], "actions": actions}
    context = {
        "world_ids": list(WORLD_IDS), "demonstration_worlds": ["low/weak", "high/weak"],
        "description": "Two formulations share an incoming stock lot. The binder-rich recipe uses "
                       "75% of it; the binder-lean recipe uses 15%. The carrier is 4 Pa s. A paid "
                       "viscometer assay reports whether the stock reads below or above 4 Pa s.",
        "training": {"estimator": "PerformanceModel, group-weighted Bayesian mixture response surface",
                     "groups": model.diagnostics["training_groups"], "snapshot_sha256": model.training_sha256,
                     "seed": seed, "historical_stock_levels_pa_s": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                     "target_units": {"transmission": "fraction", "toughness": "kJ/m2"},
                     "population_utility": "E[0.5*transmission + 0.5*toughness/(toughness+5)] "
                                           "on a successful fresh preparation, excluding meter error"},
        "initial_evidence": {"stock_lot": LOT_ID, "carrier_viscosity_pa_s": CARRIER_VISCOSITY,
                             "reference_specimen": ARCHIVE, "reference_group": ARCHIVE_GROUP,
                             "stock_assay_sd_pa_s": VISCOMETER_SD, "strength_cut_kj_m2": STRENGTH_CUT},
        "assumptions": [
            "The balanced prior describes the population after enrollment with an accepted reference specimen.",
            "A stock assay observes one noisy reading; only its band enters the policy.",
            "The archived specimen retains its preparation offset on rereading. A fresh make draws a new offset.",
            "The synthetic lab uses its own population response equations. The policy's scenario responses come "
            "from the fitted historical surrogate, with declared make-failure and instrument noise models.",
        ],
        "actions": actions,
    }
    return spec, context


class CaseLab:
    """Execute purchases. One action-specific noise stream supports paired counterfactuals."""

    def __init__(self, context, world_id, *, seed):
        if world_id not in context["world_ids"]:
            raise ValueError("Unknown synthetic execution world")
        band, condition = world_id.split("/")
        self._lot = LOT_VISCOSITY[band]
        self._archive_offset = ARCHIVE_OFFSETS[condition]
        self._seed = seed
        self._actions = {a["id"]: a for a in context["actions"]}
        self._performed = set()
        self._registry = Registry(":memory:")

    def close(self):
        self._registry.close()

    def realized_utility(self, recipe):
        """Evaluator only: expected true material utility across independent preparations."""
        return _fresh_make_utility(_true_centres(DESIGNS[recipe], self._lot))

    def execute(self, action):
        action_id = action["id"]
        if action_id in self._performed:
            raise ValueError("This action purchase was already executed")
        action = self._actions[action_id]  # Execute the declared physical operation and cost.
        kind, recipe = action["kind"], action.get("recipe")
        key = sha256(f"lab/{self._seed}/{action_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:16], "big"))
        receipt = {"action_id": action_id, "kind": kind, "cost": action["cost"], "recipe": recipe,
                   "run_id": "return-" + action_id, "stock_lot": LOT_ID}
        if kind == "assay":
            value = float(max(0, self._lot + rng.normal(0, VISCOMETER_SD)))
            label = "high" if value >= CARRIER_VISCOSITY else "low"
            receipt.update(group_id=None, specimen_id=None,
                           measurement={"name": "stock viscosity", "value": value, "unit": "Pa s",
                                        "threshold": CARRIER_VISCOSITY, "band": label})
        else:
            reread = kind == "reread"
            specimen = action["specimen"]
            group = action["group_id"]
            failed = False if reread else rng.random() < _failure_probability(DESIGNS[recipe], self._lot)
            evidence = {"protocol_followed": True, "sample_identity_confirmed": True}
            targets = {}
            label = "failed"
            measurement = None
            if not failed:
                prep_offset = self._archive_offset if reread else rng.normal(0, PREP_SD)
                targets = _endpoints(_true_centres(DESIGNS[recipe], self._lot), prep_offset, rng)
                label = "above" if targets["toughness"] >= STRENGTH_CUT else "below"
                evidence["instrument_check_passed"] = True
                measurement = {"name": "toughness", "value": targets["toughness"], "unit": "kJ/m2",
                               "threshold": STRENGTH_CUT, "band": label}
            record = {"run_id": receipt["run_id"], "group_id": group, "source_id": "synthetic-fixture#" + action_id,
                      "stock_lot": LOT_ID, "specimen_id": None if failed else specimen,
                      "design": DESIGNS[recipe], "evidence": evidence, "make_status": "failed" if failed else "success",
                      "targets": {k: {"value": value, "unit": "fraction" if k == "transmission" else "kJ/m2"}
                                  for k, value in targets.items()}}
            if kind == "remake":
                record["repeats_preparation"] = ARCHIVE_GROUP
            disposition = self._registry.ingest(record)
            receipt.update(group_id=group, specimen_id=None if failed else specimen, measurement=measurement,
                           targets=targets, registry_record=record,
                           registry_action=disposition["action"], registry_sha256=disposition["sha256"])
            if kind == "remake":
                receipt["repeats_preparation"] = ARCHIVE_GROUP
            elif reread:
                receipt["existing_specimen"] = True
        self._performed.add(action_id)
        receipt["observation"] = label
        return label, receipt
