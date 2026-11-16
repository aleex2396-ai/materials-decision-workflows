"""Exact Bayesian planning for a finite materials campaign with one spend limit.

Each public hypothesis assigns a reproducible terminal utility to every recipe and
a likelihood to each observable result of a one-shot action. Observations are
conditionally independent given the hypothesis. A shared lot or persistent
specimen bias can be included in that hypothesis to model correlated readings.
The policy sees the prior, model and receipts. The executor supplies the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isclose, isfinite, log
from numbers import Integral, Real


def _name(value, field):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a nonempty, trimmed string")
    return value


def _names(values, field):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a sequence of names")
    names = tuple(_name(value, field) for value in values)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate name in {field}")
    return frozenset(names)


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _credits(value, field, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum} credits")
    return int(value)


def _vector(values, size, field):
    if not isinstance(values, (tuple, list)) or not values or len(values) != size:
        raise ValueError(f"{field} needs one entry per world")
    return tuple(_number(x, field) for x in values)


@dataclass(frozen=True)
class BudgetState:
    """Public observations and resources after completed actions; no latent world ID."""

    remaining: int
    history: tuple[tuple[str, str], ...]
    qualified: frozenset[str]
    specimens: frozenset[str]


@dataclass(frozen=True)
class _Action:
    id: str
    kind: str
    cost: int
    recipe: str | None
    specimen: str | None
    requires: frozenset[str]
    success: frozenset[str]
    likelihoods: dict[str, tuple[float, ...]]


class SharedBudgetPlanner:
    """Finite-horizon Bellman policy over stop, assay, make, reread and remake.

    Costs are positive integer credits. The hard budget applies to every action.
    utility_per_credit optionally values saved credit in terminal-utility units.
    At stop only qualified recipes and the outside option may be selected. A
    successful make or independent remake supplies a fresh specimen and qualifies
    its recipe. A reread uses a specimen already present. The caller may list
    additional specimen prerequisites for any action.
    """

    def __init__(self, spec):
        prior = spec["prior"]
        if not isinstance(prior, (list, tuple)):
            raise ValueError("prior must be a sequence of world probabilities")
        weights = _vector(prior, len(prior), "prior")
        if min(weights) < 0 or not isclose(fsum(weights), 1.0, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Prior probabilities must be nonnegative and sum to one")
        self._prior = tuple(p / fsum(weights) for p in weights)
        self._utilities = {_name(r, "recipe"): _vector(u, len(weights), "utilities")
                           for r, u in spec["utilities"].items()}
        self._budget = _credits(spec["budget"], "budget", 0)
        self._price = _number(spec.get("utility_per_credit", 0), "utility_per_credit")
        if self._price < 0:
            raise ValueError("utility_per_credit must be nonnegative")
        self._outside = _number(spec.get("outside_utility", 0), "outside_utility")
        qualified = _names(spec.get("initial_qualified", []), "initial_qualified")
        specimens = _names(spec.get("initial_specimens", []), "initial_specimens")
        if qualified - self._utilities.keys():
            raise ValueError("Unknown initially qualified recipe")
        self._initial = BudgetState(self._budget, (), qualified, specimens)

        actions = [self._parse_action(row) for row in spec["actions"]]
        self._actions = {a.id: a for a in actions}
        if "stop" in self._actions or len(self._actions) != len(actions):
            raise ValueError("Action IDs must be unique; stop is reserved")
        # This order implements the tie-break after preferring stop.
        self._ordered = sorted(actions, key=lambda a: (a.cost, a.id))
        creates = [a.specimen for a in actions if a.kind in ("make", "remake")]
        if len(set(creates)) != len(creates) or specimens.intersection(creates):
            raise ValueError("Each manufacturing action must create a fresh, unique specimen ID")
        possible_specimens = specimens.union(creates)
        owners = {a.specimen: a.recipe for a in actions if a.kind in ("make", "remake")}
        for action in actions:
            if action.requires - possible_specimens:
                raise ValueError("Unknown prerequisite specimen")
            if action.kind == "reread" and owners.setdefault(action.specimen, action.recipe) != action.recipe:
                raise ValueError("A specimen cannot be reread as two different formulations")
        self._beliefs = {(): self._prior}
        self._values = {}

    def _parse_action(self, row):
        ident = _name(row["id"], "action ID")
        kind = row["kind"]
        if kind not in ("make", "remake", "reread", "assay"):
            raise ValueError(f"Unknown action kind for {ident}")
        recipe = row.get("recipe")
        if (kind != "assay" or recipe is not None) and recipe not in self._utilities:
            raise ValueError(f"Missing or unknown recipe for {ident}")
        cost = _credits(row["cost"], "action cost", 1)
        likelihoods = {_name(k, "outcome"): _vector(v, len(self._prior), "likelihoods")
                       for k, v in row["likelihoods"].items()}
        if not likelihoods or any(p < 0 or p > 1 for v in likelihoods.values() for p in v):
            raise ValueError(f"Invalid observation probabilities for {ident}")
        if any(not isclose(fsum(column), 1., rel_tol=0, abs_tol=1e-10)
               for column in zip(*likelihoods.values())):
            raise ValueError(f"Outcome probabilities must sum to one in each world for {ident}")
        if "observable" in row:
            observable = row["observable"]
            _name(observable["name"], "observable name")
            _name(observable["unit"], "observable unit")
            _number(observable["threshold"], "observation threshold")
            bands = {observable["below"], observable["at_or_above"]}
            if len(bands) != 2 or bands != likelihoods.keys():
                raise ValueError(f"The measurement bands must match the two outcomes for {ident}")
        success = _names(row.get("success_outcomes", []), "success_outcomes")
        requires = _names(row.get("requires_specimens", []), "requires_specimens")
        specimen = row.get("specimen")
        if kind in ("make", "remake", "reread"):
            _name(specimen, "specimen")
        if kind in ("make", "remake"):
            if not success or success - likelihoods.keys():
                raise ValueError(f"Specify valid successful manufacture outcomes for {ident}")
        elif success:
            raise ValueError("Only manufacture can qualify a new specimen")
        if kind == "reread":
            requires = requires.union([specimen])
        return _Action(ident, kind, cost, recipe, specimen, requires, success, likelihoods)

    def initial_state(self):
        return self._initial

    def _posterior(self, evidence):
        if evidence not in self._beliefs:
            # Work in log space so a long sequence of rare readings retains
            # relative evidence. Canonical receipt order shares equivalent subtrees.
            scores = []
            for i, prior in enumerate(self._prior):
                weights = [prior] + [self._actions[a].likelihoods[o][i] for a, o in evidence]
                scores.append(fsum(log(p) for p in weights) if all(weights) else float("-inf"))
            peak = max(scores)
            if peak == float("-inf"):
                raise ValueError("Observed history has zero probability under the world model")
            weights = [exp(s - peak) for s in scores]
            total = fsum(weights)
            self._beliefs[evidence] = tuple(p / total for p in weights)
        return self._beliefs[evidence]

    def belief(self, state):
        self._check_state(state)
        return self._posterior(tuple(sorted(state.history)))

    @staticmethod
    def _ineligible(state, action, used):
        if action.id in used:
            return "already_executed"
        if action.cost > state.remaining:
            return "insufficient_credits"
        if not action.requires <= state.specimens:
            return "missing_specimen"
        if action.kind == "make" and action.recipe in state.qualified:
            return "recipe_already_qualified"
        if action.kind == "remake" and action.recipe not in state.qualified:
            return "no_qualified_preparation"
        return None

    def _advance(self, state, action, label):
        qualified, specimens = state.qualified, state.specimens
        if label in action.success:
            qualified = qualified.union([action.recipe])
            specimens = specimens.union([action.specimen])
        return BudgetState(state.remaining - action.cost, state.history + ((action.id, label),),
                           qualified, specimens)

    def _check_state(self, state):
        # Public callers can deserialize or construct a state. Verify that its
        # resources and eligibility follow from the receipts before trusting it.
        if not isinstance(state, BudgetState):
            raise ValueError("A BudgetState is required")
        rebuilt = self._initial
        used = set()
        for action_id, label in state.history:
            action = self._actions.get(action_id)
            if action is None or label not in action.likelihoods or self._ineligible(rebuilt, action, used):
                raise ValueError("Invalid action history for this campaign")
            used.add(action_id)
            rebuilt = self._advance(rebuilt, action, label)
        if rebuilt != state:
            raise ValueError("State resources do not match its observed history")
        self._posterior(tuple(sorted(state.history)))

    def observe(self, state, action_id, result):
        """Debit one legal action and condition on its returned label, including failed makes."""
        self._check_state(state)
        action = self._actions.get(action_id)
        if action is None:
            raise ValueError(f"Unknown executable action: {action_id}")
        reason = self._ineligible(state, action, {k for k, _ in state.history})
        if reason:
            raise ValueError(f"Cannot execute {action_id}: {reason}")
        if result not in action.likelihoods:
            raise ValueError(f"Unknown outcome for {action_id}")
        next_state = self._advance(state, action, result)
        self._posterior(tuple(sorted(next_state.history)))
        return next_state

    def _terminal(self, state, belief):
        best, value = None, self._outside
        for recipe in sorted(state.qualified):
            utility = fsum(p * u for p, u in zip(belief, self._utilities[recipe]))
            if utility > value:
                best, value = recipe, utility
        return best, value

    def _action_value(self, state, action, belief):
        probabilities = {label: fsum(p * q for p, q in zip(belief, likelihood))
                         for label, likelihood in action.likelihoods.items()}
        rewards = []
        for label, probability in probabilities.items():
            if probability > 0:
                following = self._advance(state, action, label)
                rewards.append(probability * self._value(following))
        return fsum(rewards) - self._price * action.cost, probabilities

    def _value(self, state):
        key = tuple(sorted(state.history))
        if key not in self._values:
            belief = self._posterior(key)
            _, best = self._terminal(state, belief)
            used = {k for k, _ in key}
            for action in self._ordered:
                if self._ineligible(state, action, used) is None:
                    value, _ = self._action_value(state, action, belief)
                    best = max(best, value)
            self._values[key] = best
        return self._values[key]

    def select(self, state):
        """Rank all legal actions by remaining expected utility net of future spending."""
        belief = self.belief(state)
        terminal, value = self._terminal(state, belief)
        best = dict(action_id="stop", kind="stop", recipe=terminal, cost=0,
                    expected_value=value, outcomes={})
        alternatives = [best]
        excluded = {}
        used = {k for k, _ in state.history}
        for action in self._ordered:
            reason = self._ineligible(state, action, used)
            if reason:
                excluded[action.id] = reason
                continue
            action_value, probabilities = self._action_value(state, action, belief)
            alternative = dict(action_id=action.id, kind=action.kind, recipe=action.recipe,
                               cost=action.cost, expected_value=action_value, outcomes=probabilities)
            alternatives.append(alternative)
            if action_value > best["expected_value"] + 1e-12:
                best = alternative
        return dict(best, terminal_recipe=terminal, terminal_value=value,
                    alternatives=alternatives, ineligible=excluded)
