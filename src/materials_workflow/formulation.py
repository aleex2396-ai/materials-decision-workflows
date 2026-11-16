"""Group-aware inference for a fictional three-component optical membrane.

Descriptors are optional supplier characterisations available before a make. Only
the caller's eligible training snapshot belongs in fit(). Run outcomes never
enter the descriptor distribution or a candidate's feature vector.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
import json
from numbers import Real

import numpy as np
from scipy.special import expit, logit
from .registry import DESCRIPTORS, TARGETS


_MIXTURE = ("a", "b", "c")
_ROUTES = ("cast", "press")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _design_vector(design):
    if not isinstance(design, Mapping) or set(design) != {*_MIXTURE, "cure", "route"}:
        raise ValueError("Design requires exactly a, b, c, cure and route")
    a, b, c = [_number(design[key], key) for key in _MIXTURE]
    cure = _number(design["cure"], "cure")
    if min(a, b, c) < 0 or max(a, b, c) > 1 or not np.isclose(a + b + c, 1, rtol=0, atol=1e-8):
        raise ValueError("Mixture fractions must be nonnegative and sum to one")
    if not 0 <= cure <= 1 or design["route"] not in _ROUTES:
        raise ValueError("Cure must lie in [0, 1] and route must be cast or press")
    route = float(design["route"] == "press")
    # With an intercept, (a,b) spans the linear simplex basis: c=1-a-b.
    # Scheffé pair terms admit structural zeroes and pure constituents exactly.
    return np.array([a, b, a*b, a*c, b*c, cure, cure*(1-cure), a*cure, b*cure, route, route*cure])


def _descriptor_vector(values):
    if values is None:
        values = {}
    if not isinstance(values, Mapping) or set(values) - set(DESCRIPTORS):
        raise ValueError("Expected only pre-run density, scattering or viscosity")
    result = np.array([np.nan if values.get(key) is None else _number(values[key], key)
                       for key in DESCRIPTORS])
    if np.any(result < 0) or result[0] == 0:
        raise ValueError("Density must be positive and other supplier characterisations nonnegative")
    return result


def _group_weights(groups):
    counts = Counter(groups)
    return np.array([1.0 / counts[group] for group in groups])


class _DescriptorDistribution:
    """Training-only Gaussian with pairwise correlations on co-observations."""

    def __init__(self, values, groups):
        self.mean = np.zeros(len(DESCRIPTORS))
        self.scale = np.zeros(len(DESCRIPTORS))
        self.counts = np.array([len(set(groups[np.isfinite(col)])) for col in values.T])
        self.active = self.counts >= 2
        correlation = np.eye(len(DESCRIPTORS))
        self.limits = {}
        for i, key in enumerate(DESCRIPTORS):
            known = np.isfinite(values[:, i])
            if self.active[i]:
                weights = _group_weights(groups[known])
                column = values[known, i]
                self.mean[i] = np.average(column, weights=weights)
                self.scale[i] = np.sqrt(np.average((column - self.mean[i])**2, weights=weights))
                self.limits[key] = [float(column.min()), float(column.max())]
            for j in range(i):
                pair = known & np.isfinite(values[:, j])
                pair_groups = groups[pair]
                n = len(set(pair_groups))
                if n < 3:
                    continue
                weights = _group_weights(pair_groups)
                data = values[pair][:, [i, j]]
                centered = data - np.average(data, axis=0, weights=weights)
                scatter = (centered.T * weights) @ centered
                denominator = np.sqrt(scatter[0, 0] * scatter[1, 1])
                if denominator > 0:
                    correlation[i, j] = correlation[j, i] = n / (n + 3) * scatter[0, 1] / denominator
        # Different pairs can have different observed subsets. Project the
        # pairwise correlation estimate to a positive-definite correlation matrix.
        eigenvalues, axes = np.linalg.eigh(correlation)
        correlation = (axes * np.maximum(eigenvalues, 1e-6)) @ axes.T
        normalizer = np.sqrt(np.diag(correlation))
        self.covariance = correlation / np.outer(normalizer, normalizer) * np.outer(self.scale, self.scale)

    def condition(self, observed):
        known = np.isfinite(observed) & self.active
        missing = ~np.isfinite(observed) & self.active
        mean = np.where(known, observed, self.mean)
        covariance = np.zeros_like(self.covariance)
        if np.any(missing):
            conditional = self.covariance[np.ix_(missing, missing)]
            informative = known & (self.scale > 0)
            if np.any(informative):
                cross = self.covariance[np.ix_(missing, informative)]
                known_cov = self.covariance[np.ix_(informative, informative)]
                gain = np.linalg.solve(known_cov, cross.T).T
                mean[missing] += gain @ (observed[informative] - self.mean[informative])
                conditional = conditional - gain @ cross.T
            covariance[np.ix_(missing, missing)] = (conditional + conditional.T) / 2
        return mean, covariance

    def summary(self, observed, mean, covariance):
        summary = {}
        for i, key in enumerate(DESCRIPTORS):
            if np.isfinite(observed[i]):
                source, value, sd = "observed", float(observed[i]), 0.0
            elif not self.active[i]:
                source, value, sd = "unavailable", None, None
            else:
                informative = np.isfinite(observed) & (np.abs(self.covariance[i]) > 0)
                source = "conditional" if np.any(informative) else "marginal"
                value, sd = float(mean[i]), float(np.sqrt(max(covariance[i, i], 0)))
            summary[key] = dict(source=source, mean=value, sd=sd, used_in_model=bool(self.active[i]))
        return summary


def _targets(rows):
    y = np.array([[_number(row["targets"][key], key) for key in TARGETS] for row in rows])
    if np.any((y[:, 0] < 0) | (y[:, 0] > 1)) or np.any(y[:, 1] <= 0):
        raise ValueError("Transmission must lie in [0, 1] and toughness must be positive")
    return y


class _ResponseSurface:
    """Group-weighted Bayesian multivariate ridge with missing-covariate moments."""

    def __init__(self, rows):
        groups = np.array([row["group_id"] for row in rows])
        weights = _group_weights(groups)
        designs = np.stack([_design_vector(row["design"]) for row in rows])
        descriptors = np.stack([_descriptor_vector(row.get("descriptors")) for row in rows])
        self.dist = _DescriptorDistribution(descriptors, groups)
        self.design_size = designs.shape[1]
        moments = [self.dist.condition(row) for row in descriptors]
        means = np.stack([pair[0] for pair in moments])
        uncertainties = np.stack([pair[1] for pair in moments])
        masks = ~np.isfinite(descriptors) & self.dist.active
        matrix = np.column_stack([designs, means, masks])
        self.center = np.average(matrix, axis=0, weights=weights)
        variances = np.average((matrix - self.center)**2, axis=0, weights=weights)
        desc_slice = slice(self.design_size, self.design_size + len(DESCRIPTORS))
        imputed_scatter = np.einsum("n,nij->ij", weights, uncertainties)
        variances[desc_slice] += np.diag(imputed_scatter) / weights.sum()
        self.scale = np.where(variances > 1e-12, np.sqrt(variances), 1)
        x = self._standardize(matrix)
        y = _targets(rows)
        # Finite link values for observations reported at the optical boundaries;
        # inverse softplus keeps positive-response predictive moments finite.
        linked = np.column_stack([logit(np.clip(y[:, 0], 1e-5, 1-1e-5)),
                                  y[:, 1] + np.log(-np.expm1(-y[:, 1]))])
        self.y_center = np.average(linked, axis=0, weights=weights)
        self.y_scale = np.maximum(np.sqrt(np.average((linked - self.y_center)**2,
                                                      axis=0, weights=weights)), 0.02)
        z = (linked - self.y_center) / self.y_scale
        # Fractional likelihood weights assign total mass one per preparation.
        # E[X'X] integrates the conditional descriptor covariance in the expected
        # quadratic loss. This is a working posterior, not exact latent-X inference.
        precision = (x.T * weights) @ x + np.eye(x.shape[1])
        block = slice(desc_slice.start + 1, desc_slice.stop + 1)
        precision[block, block] += imputed_scatter / np.outer(self.scale[desc_slice], self.scale[desc_slice])
        self.precision_inverse = np.linalg.solve(precision, np.eye(x.shape[1]))
        cross = (x.T * weights) @ z
        self.beta = np.linalg.solve(precision, cross)
        # Identity inverse-Wishart scale with d+2 prior degrees of freedom.
        scale = np.eye(len(TARGETS)) + (z.T * weights) @ z - cross.T @ self.beta
        self.df = 3 + len(set(groups))
        self.noise_factor = np.linalg.cholesky((scale + scale.T) / (2*self.df))
        self.routes = {row["design"]["route"] for row in rows}
        self.ranges = {key: (min(row["design"][key] for row in rows), max(row["design"][key] for row in rows))
                       for key in (*_MIXTURE, "cure")}

    def _standardize(self, matrix):
        return np.column_stack([np.ones(len(matrix)), (matrix - self.center) / self.scale])

    def predict(self, design, descriptors, seed):
        d = _design_vector(design)
        observed = _descriptor_vector(descriptors)
        mu, cov = self.dist.condition(observed)
        rng = np.random.default_rng(seed)
        n = 256
        # Paired antithetic draws reduce MC noise and provide common random
        # numbers across candidate evaluations. Only within-candidate draws are joint.
        normal = rng.standard_normal((n // 2, len(DESCRIPTORS) + len(TARGETS)))
        normal = np.vstack([normal, -normal])
        eigenvalues, axes = np.linalg.eigh(cov)
        sqrt_cov = (axes * np.sqrt(np.maximum(eigenvalues, 0))) @ axes.T
        desc_draws = mu + normal[:, :len(DESCRIPTORS)] @ sqrt_cov
        missing = ~np.isfinite(observed) & self.dist.active
        matrix = np.column_stack([np.tile(d, (n, 1)), desc_draws, np.tile(missing, (n, 1))])
        x = self._standardize(matrix)
        variance_multiplier = 1 + np.einsum("ij,ij->i", x @ self.precision_inverse, x)
        chi = np.tile(rng.chisquare(self.df, n // 2), 2)
        noise = normal[:, len(DESCRIPTORS):] @ self.noise_factor.T
        latent = x @ self.beta + np.sqrt(variance_multiplier * self.df / chi)[:, None] * noise
        transformed = self.y_center + self.y_scale * latent
        draws = np.column_stack([expit(transformed[:, 0]), np.logaddexp(0, transformed[:, 1])])
        reasons = [f"{key} outside observed range" for key, (low, high) in self.ranges.items()
                   if not low - 1e-12 <= design[key] <= high + 1e-12]
        if design["route"] not in self.routes:
            reasons.append("route absent from training")
        for i, key in enumerate(DESCRIPTORS):
            if key in self.dist.limits and np.isfinite(observed[i]):
                low, high = self.dist.limits[key]
                if not low <= observed[i] <= high:
                    reasons.append(f"{key} outside observed range")
        return draws, self.dist.summary(observed, mu, cov), reasons


class PerformanceModel:
    """Fit only eligible rows and predict two endpoints of a fictional membrane."""

    def __init__(self, seed=17):
        self.seed = int(seed)
        self.diagnostics = {}
        self.training_sha256 = None
        self._surface = None

    def fit(self, rows):
        rows = list(rows)
        if any(not isinstance(r.get(k), str) or not r[k].strip() for r in rows for k in ("run_id", "group_id")):
            raise ValueError("Every row requires run and preparation-group identities")
        if len({row["run_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate training run identity")
        rows.sort(key=lambda row: row["run_id"])
        groups = sorted({row["group_id"] for row in rows})
        if len(groups) < 3:
            raise ValueError("Need at least three independent training groups")
        # Split by group before estimating the imputer, feature scaling, response
        # priors or baseline. Each group contributes equally to the reported MAE.
        fold_count = min(len(groups), 5)
        group_errors, baseline_errors = [], []
        whole_group_coverage, group_widths = [], []
        folds = []
        for fold in range(fold_count):
            held = set(groups[fold::fold_count])
            train = [row for row in rows if row["group_id"] not in held]
            surface = _ResponseSurface(train)
            train_groups = [row["group_id"] for row in train]
            baseline = np.average(_targets(train), axis=0, weights=_group_weights(train_groups))
            for group in sorted(held):
                valid = [row for row in rows if row["group_id"] == group]
                draws = np.stack([surface.predict(r["design"], r.get("descriptors"), self.seed)[0]
                                  for r in valid])
                predicted = draws.mean(axis=1)
                actual = _targets(valid)
                group_errors.append(np.abs(actual - predicted).mean(0))
                baseline_errors.append(np.abs(actual - baseline).mean(0))
                lower, upper = np.quantile(draws, [0.05, 0.95], axis=1)
                whole_group_coverage.append(((actual >= lower) & (actual <= upper)).all(0))
                group_widths.append((upper - lower).mean(0))
            folds.append(dict(train_groups=sorted(set(train_groups)), validation_groups=sorted(held)))
        surface = _ResponseSurface(rows)
        model_mae, baseline_mae = np.mean(group_errors, axis=0), np.mean(baseline_errors, axis=0)
        snapshot = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._surface = surface
        self.training_sha256 = sha256(snapshot.encode("utf-8")).hexdigest()
        self.diagnostics = {
            "training_rows": len(rows), "training_groups": len(groups),
            "descriptor_observed_groups": dict(zip(DESCRIPTORS, surface.dist.counts.tolist())),
            "validation": {"method": "whole_group_cross_validation", "n_groups": len(groups), "folds": folds,
                           "targets": {key: {"model_mae": float(model_mae[i]), "mean_baseline_mae": float(baseline_mae[i]),
                                             "groups_fully_inside_pointwise_90_interval": int(np.sum(whole_group_coverage, 0)[i]),
                                             "group_mean_90_interval_width": float(np.mean(group_widths, 0)[i])}
                                       for i, key in enumerate(TARGETS)}},
            "uncertainty": "uncalibrated_model_based",
            "assumptions": ["Pairwise Gaussian supplier correlations estimated within eligible training folds.",
                            "Each group has unit weight. Group dependence is approximated, not fitted as a random effect.",
                            "Working posterior integrates missing-input second moments with empirical response scaling.",
                            "Reported uncertainty includes new-run variation; predictive coverage requires fresh groups.",
                            "Support screening checks marginal ranges and route membership only."],
        }
        return self

    def descriptor_estimates(self, descriptors=None):
        """Summarize the fitted supplier distribution for supplied partial evidence."""
        if self._surface is None:
            raise ValueError("Fit a training snapshot before prediction")
        observed = _descriptor_vector(descriptors)
        mean, covariance = self._surface.dist.condition(observed)
        return self._surface.dist.summary(observed, mean, covariance)

    def predict(self, design, descriptors=None):
        if self._surface is None:
            raise ValueError("Fit a training snapshot before prediction")
        draws, summaries, reasons = self._surface.predict(design, descriptors, self.seed)
        return {
            "targets": {key: {"mean": float(draws[:, i].mean()), "sd": float(draws[:, i].std(ddof=1))}
                        for i, key in enumerate(TARGETS)},
            "draws": {key: draws[:, i].tolist() for i, key in enumerate(TARGETS)},
            "descriptors": summaries,
            "applicability": {"status": "extrapolation" if reasons else "in_support", "reasons": reasons},
            "uncertainty": "uncalibrated_model_based",
        }
