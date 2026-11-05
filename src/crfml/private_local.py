"""Schema-driven local development smoke study; never a public results exporter."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .data import PrivateStore, extract_numeric_table
from .models import ExactGP, metrics


def run_development_study(private_dir: str | Path, schema_path: str | Path) -> dict:
    """Fit only declared development trajectories and report chronological forecasts.

    This diagnostic checks an actual source-to-model path. It cannot establish
    material-design generalization, closed-loop control validity, or calibrated
    coverage under dependence. Schemas and all outputs stay in the private tree.
    """
    destination = Path(private_dir).resolve()
    schemas = Path(schema_path).resolve()
    if not schemas.is_relative_to(destination):
        raise ValueError("Confidential schema must live inside the private directory")
    spec = json.loads(schemas.read_text(encoding="utf-8"))
    if spec.get("study_scope") != "development_trajectory_smoke_only":
        raise ValueError("This runner only supports exploratory development diagnostics")
    if not 0 < float(spec["train_fraction"]) < 1:
        raise ValueError("Train fraction must lie strictly between zero and one")
    store = PrivateStore(destination)
    records, rejected = [], []
    for table in spec["tables"]:
        book = store.read_development_workbook(table["source"])
        result = extract_numeric_table(book, **table["schema"])
        records.extend(result["records"])
        rejected.extend(dict(source=table["source"], **row) for row in result["rejected"])
    ordered_feature = spec["ordered_feature"]
    groups = defaultdict(list)
    for row in records:
        groups[(row["provenance"]["source"], row["experiment_id"])].append(row)
    trajectories = []
    normalized = []
    for (source, identity), rows in groups.items():
        valid, previous = [], -np.inf
        for row in rows:
            value = row["features"][ordered_feature]
            if value <= previous:
                rejected.append(dict(source=source, row=row["provenance"]["row"],
                                     reason="nonincreasing_elapsed_time_ambiguous_do_not_repair"))
                continue
            previous = value
            valid.append(row)
        normalized.extend(valid)
        if len(valid) < 8:
            trajectories.append(dict(experiment_id=identity, status="abstain_insufficient_trajectory"))
            continue
        n_train = max(3, int(len(valid) * float(spec["train_fraction"])))
        if n_train >= len(valid):
            raise ValueError("Need subsequent development observations for forecasting")
        x = np.array([[row["features"][ordered_feature]] for row in valid])
        y = np.array([row["target"] for row in valid])
        # All preprocessing and hyperparameter fitting use the early prefix only.
        y_mean, y_scale = y[:n_train].mean(), max(y[:n_train].std(), 1e-8)
        gp = ExactGP(noise=float(spec["standardized_noise_assumption"])).fit(
            x[:n_train], (y[:n_train] - y_mean) / y_scale)
        mean, std = gp.predict(x[n_train:])
        mean, std = mean * y_scale + y_mean, std * y_scale
        persistence = np.full(len(y) - n_train, y[n_train - 1])
        linear_coefficients = np.polyfit(x[:n_train, 0], y[:n_train], 1)
        linear = np.polyval(linear_coefficients, x[n_train:, 0])
        trajectories.append(dict(experiment_id=identity, source=source,
                                 status="descriptive_development_forecast", observations=len(valid),
                                 train_observations=n_train, forecast_observations=len(y) - n_train,
                                 target_unit=valid[0]["target_unit"],
                                 gp=metrics(y[n_train:], mean, std),
                                 persistence=metrics(y[n_train:], persistence),
                                 linear=metrics(y[n_train:], linear),
                                 optimizer=gp.optimizer,
                                 forecast=[dict(time=float(t), observed=float(a), gp_mean=float(b), gp_std=float(c))
                                           for t, a, b, c in zip(x[n_train:, 0], y[n_train:], mean, std)]))
    report = dict(status="development_only", normalized_observations=len(normalized),
                  source_date_blocks=len(set(row["group_id"] for row in normalized)),
                  trajectories=trajectories, exclusions=rejected,
                  material_generalization_status="abstain_insufficient_independent_experiments",
                  private_test_status="sealed_unread",
                  caveats=["Timepoints within a run are correlated, not independent experiments.",
                           "This is an exploratory chronological forecast on development records, not the sealed test.",
                           "A measured proxy is not automatically the target property; no physical calibration is inferred.",
                           "Photo evidence and execution fidelity are not automatically converted to labels.",
                           "Metadata-based selection or prior knowledge can limit confirmatory interpretation.",
                           "Gaussian intervals are model assumptions, not a validated coverage guarantee.",
                           "A fixed standardized noise is a diagnostic assumption, not an estimated sensor accuracy."])
    (destination / "normalized_development_records.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    (destination / "development_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
