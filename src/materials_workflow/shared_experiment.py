"""Run an observation-driven policy and a fabrication-only comparator in paired synthetic labs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import platform

import numpy as np
import scipy

from .campaign import file_hash, source_hashes, write_json
from .shared_case import CaseLab, build_case
from .shared_session import BudgetSession, fingerprint


PRESET = Path(__file__).with_name("shared_budget_preset.json")


def run_episode(db_path, spec, context, world_id, seed):
    """The policy receives one paid observation at a time. Only the lab knows the world."""
    lab = CaseLab(context, world_id, seed=seed)
    action_by_id = {action["id"]: action for action in spec["actions"]}
    try:
        with BudgetSession(db_path, spec=spec) as session:
            while True:
                reservation = session.next()
                action_id = reservation["action_id"]
                if action_id == "stop":
                    break
                outcome, receipt = lab.execute(action_by_id[action_id])
                session.observe(reservation["step"], action_id, outcome, receipt)
            history = session.status()
        terminal = history["events"][-1]["decision"]["terminal_recipe"]
        # The evaluator accesses the simulator only after the policy stops.
        utility = spec.get("outside_utility", 0.0) if terminal is None else lab.realized_utility(terminal)
    finally:
        lab.close()
    return dict(history, terminal_recipe=terminal, simulated_terminal_utility=utility,
                simulated_net_utility=utility - spec.get("utility_per_credit", 0.0) * history["spent"])


def _first(episode, *kinds):
    return next((e for e in episode["events"] if e["decision"]["kind"] in kinds), None)


def _branch_summary(world, shared, baseline):
    assay = _first(shared, "assay")
    manufacture = _first(shared, "make", "remake")
    direct = _first(baseline, "make", "remake")
    made_recipe = manufacture["decision"]["recipe"] if manufacture else None
    direct_recipe = direct["decision"]["recipe"] if direct else None
    make_receipt = manufacture["receipt"] if manufacture else {}
    return {"world": world, "assay_outcome": assay["outcome"] if assay else None,
            "measurement": assay["receipt"].get("measurement") if assay else None,
            "next_make": made_recipe, "make_outcome": manufacture["outcome"] if manufacture else None,
            "specimen_id": make_receipt.get("specimen_id"), "registry_action": make_receipt.get("registry_action"),
            "terminal_recipe": shared["terminal_recipe"], "spent": shared["spent"],
            "remaining": shared["remaining"], "direct_make": direct_recipe,
            "switch_vs_direct": made_recipe != direct_recipe,
            "shared_net_utility": shared["simulated_net_utility"], "direct_net_utility": baseline["simulated_net_utility"]}


def _report(summary, trace):
    starts = trace["episodes"][0]
    opening = starts["shared"]["events"][0]
    menu = opening["decision"]["alternatives"]
    lines = ["# A measurement that changes the next formulation", "",
             "Two synthetic labs start with the same preparation history, stock specifications, fitted model, "
             "action menu and credits. The incoming binder lot is different. The decision code sees the "
             "instrument return only after it reserves the assay and pays for it.", "",
             f"Each lab has **{summary['budget']} credits**. An incoming-stock viscosity assay costs one credit "
             "in the bundled preset. A fresh formulation costs four. The binder-rich formulation uses 75% of the "
             "incoming stock and the binder-lean formulation uses 15%. Both use the same casting and cure settings. "
             "The reference has an archived specimen from this lot.", "",
             "| Paid viscosity result | Next make attempted | Returned specimen | Direct make without diagnostics | Credits spent |",
             "|---|---|---|---|---:|"]
    for row in summary["branches"]:
        m = row["measurement"]
        reading = f"{m['value']:.3f} {m['unit']} ({row['assay_outcome']})" if m else "No assay"
        specimen = row['specimen_id'] or ('failed preparation' if row['make_outcome'] == 'failed' else 'none')
        lines.append(f"| {reading} | {row['next_make'] or 'none'} | {specimen} | {row['direct_make'] or 'none'} | {row['spent']} |")
    lines += ["", "## Why the lab paid for the assay", "",
              "The planner scores every affordable next action by integrating its possible returns, replanning after "
              "each one, and accounting for the remaining credits. The terminal choice is the qualified recipe with "
              "the highest posterior expected utility or the outside option. The values below include the configured "
              "charge for future spending. A successful preparation with a usable return qualifies the recipe, "
              "including when the measured performance is poor.", "",
              "The synthetic preference gives equal weight to transmission (a fraction) and the ratio "
              "`toughness / (toughness + 5 kJ/m²)`. Recipe utility averages that score across successful "
              "fresh preparations. Each credit spent costs an additional "
              f"{trace['public_spec'].get('utility_per_credit', 0):g} in utility.", "",
              "| First action | Immediate cost (credits) | Expected net utility with optimal continuation |",
              "|---|---:|---:|"]
    for option in sorted(menu, key=lambda row: -row["expected_value"]):
        lines.append(f"| `{option['action_id']}` | {option['cost']} | {option['expected_value']:.5f} |")
    lines += ["", f"The best policy with the full menu is worth **{summary['opening_net_value']:.5f}** "
              f"under the declared prior. A policy restricted to making or independently remaking formulations "
              f"starts with the same budget and is worth **{summary['make_only_net_value']:.5f}**. "
              "These are model expectations over all possible results.", "",
              "## What changed after the read", "",
              "Both executions have the same opening decision digest. The viscometer measures the incoming stock. "
              "Below 4 Pa·s its receipt is coded low, and at or above 4 Pa·s it is coded high. The policy uses this "
              "band to update the probability of a low-viscosity lot, changing the expected utility of both recipes. "
              "The archived specimen has the same prior in each branch.", "",
              "| Evidence | Probability of low-viscosity lot | Binder-rich utility | Binder-lean utility |",
              "|---|---:|---:|---:|"]
    def ranking(label, evidence):
        p_low = sum(p for world, p in zip(trace["world_order"], evidence["posterior"], strict=True)
                    if world.startswith("low/"))
        ranks = evidence["expected_recipe_utility"]
        return f"| {label} | {p_low:.1%} | {ranks['binder-rich']:.4f} | {ranks['binder-lean']:.4f} |"

    lines.append(ranking("Before assay", opening["before"]))
    for row, branch in zip(summary["branches"], trace["episodes"], strict=True):
        assay = _first(branch["shared"], "assay")
        if assay:
            lines.append(ranking(f"After {row['assay_outcome']} read", assay["after"]))
    lines += ["", "The execution trace records the paid action, credits remaining, actual specimen and preparation "
              "receipts, qualification status and posterior before the next choice. A failed make still costs its "
              "reserved credits. Re-reading a specimen and making a fresh replicate have distinct likelihoods and "
              "identities. These two branches are a constructed demonstration of a decision reversal.", "",
              "## Reproduce and inspect", "",
              "```bash", "python -m materials_workflow shared-budget --output .local/shared-budget", "```", "",
              "The output includes the public policy specification and separate SQLite sessions for each branch "
              "and comparator. An operator can also reserve an action and supply an observed return manually. "
              "A fabrication return includes the raw registry submission for the requested recipe and specimen. "
              "The session checks its disposition and observed toughness band before updating the policy.", "",
              "```bash",
              "python -m materials_workflow session init --db .local/operator.sqlite3 --spec .local/shared-budget/policy.json",
              "python -m materials_workflow session next --db .local/operator.sqlite3",
              "```", "",
              "For the opening assay, save the following example as `receipt.json`.", "",
              "```json", '{"action_id":"assay-stock","measurement":{"name":"stock viscosity","unit":"Pa s","value":1.94}}', "```", "",
              "```bash",
              "python -m materials_workflow session observe --db .local/operator.sqlite3 --step 1 --action assay-stock --outcome low --receipt receipt.json",
              "python -m materials_workflow session next --db .local/operator.sqlite3",
              "```", "",
              "A value of 5.94 and the `high` band in a separate fresh session exercise the other branch. "
              "Reopening a session resumes the outstanding "
              "action. Replaying the same receipt leaves the balance and posterior intact.", "",
              "[Machine-readable summary](summary.json) · [Full decisions and receipts](trace.json) · "
              "[Planner](../../src/materials_workflow/shared_budget.py) · [Synthetic case](../../src/materials_workflow/shared_case.py)", ""]
    return "\n".join(lines)


def _run(output, config_path):
    before = source_hashes()
    config = json.loads(config_path.read_bytes())
    if config.get("origin") != "public_synthetic_only":
        raise ValueError("The demonstration requires a public synthetic preset.")
    spec, context = build_case(config)
    direct_spec = dict(spec, actions=[a for a in spec["actions"] if a["kind"] in {"make", "remake"}])
    write_json(output / "policy.json", spec)
    episodes, branches = [], []
    for branch, world in enumerate(context["demonstration_worlds"], 1):
        shared = run_episode(output / f"branch-{branch}.sqlite3", spec, context, world, config["execution_seed"])
        direct = run_episode(output / f"direct-{branch}.sqlite3", direct_spec, context, world, config["execution_seed"])
        episodes.append({"world": world, "shared": shared, "make_only": direct})
        branches.append(_branch_summary(world, shared, direct))
    start = episodes[0]["shared"]["events"][0]
    if any(e["shared"]["events"][0]["decision_sha256"] != start["decision_sha256"] for e in episodes):
        raise RuntimeError("Paired branches did not start from the same decision.")
    direct_start = episodes[0]["make_only"]["events"][0]
    trace = {"origin": config["origin"], "public_spec": spec, "policy_sha256": fingerprint(spec),
             "world_order": context["world_ids"], "training": context["training"],
             "initial_evidence": context["initial_evidence"], "assumptions": context["assumptions"],
             "description": context["description"], "episodes": episodes}
    summary = {"origin": config["origin"], "config": config, "budget": spec["budget"],
               "opening_action": start["action_id"], "direct_make": direct_start["decision"]["recipe"],
               "opening_net_value": start["decision"]["expected_value"],
               "make_only_net_value": direct_start["decision"]["expected_value"],
               "shared_opening_decision_sha256": start["decision_sha256"], "branches": branches}
    write_json(output / "trace.json", trace)
    write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary, trace), encoding="utf-8", newline="\n")
    if source_hashes() != before:
        raise RuntimeError("Producer source changed during the campaign.")
    write_json(output / "provenance.json", {
        "origin": config["origin"], "no_private_data_loaded": True, "config_sha256": file_hash(config_path),
        "source_start_sha256": before, "source_sha256": before, "sources_unchanged_during_run": True,
        "artifact_sha256": {name: file_hash(output / name) for name in ("REPORT.md", "summary.json", "trace.json")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}})
    return summary


def run_shared_demo(output, config=PRESET):
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("Choose a fresh directory for the paired campaign.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as staging:
        work = Path(staging) / "campaign"
        work.mkdir()
        result = _run(work, Path(config))
        work.rename(target)
    return result
