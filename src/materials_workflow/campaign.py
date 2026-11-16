"""Assemble a full, isolated campaign and preserve the evidence for each decision."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import scipy

from .adjudication import run_shadow_study
from .cold_start import cold_start_batch
from .decision_replay import replay_gate
from .descriptor_view import materialize_descriptors
from .formulation import PerformanceModel
from .planner import recommend_batch
from .registry import Registry
from .synthetic import candidate_pool, execute_synthetic, initial_returns
from .value_of_information import measure_or_make

PACKAGE = Path(__file__).parent
PRESET = PACKAGE / "preset.json"
SNAPSHOT_FILES = ("summary.json", "packet.json", "REPORT.md", "decision_map.png", "explorer.html")


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def rows_hash(rows):
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.name: file_hash(p) for p in sorted(PACKAGE.glob("*.py"))}


def concise_prediction(model, candidate):
    response = model.predict(candidate["design"], candidate.get("descriptors"))
    response["training_sha256"] = model.training_sha256
    draws = response.pop("draws")
    for name, samples in draws.items():
        response["targets"][name]["model_interval_90"] = np.quantile(samples, [0.05, 0.95]).tolist()
    return response


def design_batch(registry, candidates, pending, *, seed, budget, batch_size, model=None):
    with registry.db as db:
        db.execute("BEGIN")
        rows = registry.training_rows()
        feasibility = registry.feasibility_rows()
    settings = {"seed": seed, "budget": budget, "batch_size": batch_size}
    training_hash = rows_hash(rows)
    if len({r["group_id"] for r in rows}) < 3:
        plan = cold_start_batch(candidates, rows, feasibility, pending, **settings)
    else:
        model = model if model is not None else PerformanceModel(seed=seed).fit(rows)
        if model.training_sha256 != training_hash:
            raise ValueError("The fitted model belongs to a different eligible observation snapshot.")
        plan = recommend_batch(model, candidates, rows, feasibility, pending, **settings)
    plan["training_sha256"] = training_hash
    plan["makeability_sha256"] = rows_hash(feasibility)
    return plan


def _execute(config_path, output):
    from .reporting import write_report

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("origin") != "public_synthetic_only" or config["preparation_groups"] < 8:
        raise ValueError("The showcase needs a synthetic preset and at least eight preparation groups.")
    if config["candidate_count"] < 4 or config["batch_size"] < 1 or config["budget"] <= 0:
        raise ValueError("The showcase needs four candidates and a positive batch size and budget.")
    before_sources = source_hashes()
    seed = config["seed"]
    settings = {"seed": seed, "budget": config["budget"], "batch_size": config["batch_size"]}
    returns = initial_returns(config)
    with Registry(output / "campaign.sqlite3") as registry:
        receipts = [registry.ingest(rec) for rec in returns]
        # Demonstrate exactly-once landing from a replay of the first instrument payload.
        replay = registry.ingest(returns[0])
        rows = registry.training_rows()
        groups = {row["group_id"] for row in rows}
        model = PerformanceModel(seed=seed).fit(rows)
        before_view = materialize_descriptors(registry, model, rows)
        known = max(rows, key=lambda row: row["targets"]["transmission"])
        pool = candidate_pool(config, known)
        pending = pool[:2]
        with Registry(":memory:") as empty:
            cold = design_batch(empty, pool, pending, **settings)
        plan = design_batch(registry, pool, pending, model=model, **settings)
        by_id = {candidate["id"]: candidate for candidate in pool}
        # Score and seal first. The feedback generator cannot contribute to this fit or proposal.
        selected = [action["id"] for action in plan["actions"]]
        frozen = {key: concise_prediction(model, by_id[key]) for key in selected}
        shortlist = [by_id[key] for key in selected[:4]] or pool[2:6]
        assay_decision = measure_or_make(model, shortlist, seed=seed,
                                         credit_cost=config["descriptor_assay_cost_in_utility_units"])
        snapshot = {"before": rows_hash(rows), "pre_run_decision_sha256": rows_hash([plan, frozen]),
                    "before_model_seed": seed}
        write_json(output / "pre_run_decision.json", {"training_sha256": snapshot["before"],
                   "proposal": plan, "predictions": frozen})
        snapshot["pre_run_file_sha256"] = file_hash(output / "pre_run_decision.json")
        execution = []
        observed = execute_synthetic(plan["actions"], pool, seed, round_id="round-1")
        for ident, record in zip(selected, observed, strict=True):
            receipt = registry.ingest(record)
            entry = registry.submission(receipt["submission_id"])
            canonical = entry["canonical"] or {}
            execution.append({"candidate_id": ident, "run_id": record["run_id"],
                              "group_id": record["group_id"], "action": receipt["action"],
                              "receipt_sha256": receipt["sha256"], "pre_run_prediction": frozen[ident],
                              "observed": canonical.get("targets")})
        final_rows = registry.training_rows()
        next_model = PerformanceModel(seed=seed).fit(final_rows)
        after_view = materialize_descriptors(registry, next_model, final_rows)
        # Each batch has its own equal budget. Carry unresolved pending work across the boundary.
        remaining = [candidate for candidate in pool if candidate["id"] not in selected]
        next_plan = design_batch(registry, remaining, pending, model=next_model, **settings)
        snapshot["after"] = rows_hash(final_rows)
        snapshot["snapshot_scope"] = "eligible performance rows, including actual preparation group and design"
        counts = dict(sorted(Counter(row["action"] for row in receipts).items()))
        focus = next((by_id[k] for k in selected if len(by_id[k]["descriptors"]) < 3), pool[2])
        example = {"candidate": focus, "prediction": concise_prediction(model, focus)}
        soft_cases = {rec["run_id"]: rec["diagnostic_signals"] for rec, receipt in zip(returns, receipts, strict=True)
                      if receipt["action"] == "ask" and receipt["code"] == "discordant_replicates"}
        audit = [{k: v for k, v in entry.items() if k != "received_at"} for entry in registry.audit()]
        packet = {"origin": config["origin"], "initial_rows": rows, "audit": audit, "candidates": pool,
                  "pending_ids": [c["id"] for c in pending], "cold_start": cold, "initial_plan": plan,
                  "descriptor_assay_decision": assay_decision,
                  "next_plan": next_plan, "prediction_example": example, "execution": execution,
                  "descriptor_views": {"before": before_view, "after": after_view},
                  "diagnostics": model.diagnostics, "gate_replay": replay_gate(seed + 3),
                  "shadow_study": run_shadow_study(seed + 4, soft_cases), "training_snapshot": snapshot,
                  "replay_receipt_matches": replay["submission_id"] == receipts[0]["submission_id"]}
        summary = {"origin": config["origin"], "name": config["name"], "config": config,
                   "initial_rows": len(rows), "initial_groups": len(groups), "initial_attempts": len(returns),
                   "gate_counts": counts, "feedback_rows": len(final_rows) - len(rows),
                   "final_groups": len({row["group_id"] for row in final_rows}),
                   "spent": plan["spent"], "budget": plan["budget"], "selected_ids": selected,
                   "next_ids": [action["id"] for action in next_plan["actions"]]}
    write_json(output / "summary.json", summary)
    write_json(output / "packet.json", packet)
    write_json(output / "candidate.json", focus)
    write_json(output / "candidates.json", remaining)
    write_json(output / "pending.json", pending)
    write_json(output / "shortlist.json", shortlist)
    write_json(output / "sample_return.json", returns[0])
    write_report(output, summary, packet)
    end_sources = source_hashes()
    if before_sources != end_sources:
        raise RuntimeError("Producer source changed during the campaign. Run again from the final version.")
    write_json(output / "provenance.json", {
        "origin": config["origin"], "no_private_data_loaded": True,
        "config_sha256": sha256(config_bytes).hexdigest(), "source_start_sha256": before_sources,
        "source_sha256": end_sources, "sources_unchanged_during_run": True,
        "artifact_sha256": {name: file_hash(output / name) for name in SNAPSHOT_FILES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    })
    return summary


def run_demo(output, config=PRESET):
    """Publish a complete run into a fresh directory only after all stages succeed."""
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("Choose a fresh output directory; a campaign already exists there.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as staging:
        work = Path(staging) / "campaign"
        work.mkdir()
        summary = _execute(Path(config), work)
        work.rename(target)
    return summary
