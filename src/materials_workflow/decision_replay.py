"""Inspect routing mistakes with labelled, independent synthetic challenges."""

import numpy as np

from .registry import Registry
from .synthetic import DESCRIPTORS, design_from, instrument_return, material_card


def replay_gate(seed):
    # The gate sees only the return. The external challenge key supplies the intended truth.
    scenarios = [
        ("ordinary", "normal", True), ("valid low response", "poor", True),
        ("meter check failed", "meter", False), ("execution deviation", "procedure", False),
        ("conflicting clues", "mixed", False), ("wrong identity", "ambiguous", False),
        ("confirmed failed make", "failed", False), ("absent assay value", "missing_endpoint", False),
        ("undetected meter fault", "normal", False), ("false instrument alarm", "meter", True),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    with Registry(":memory:") as ledger:
        for i, (name, case, usable) in enumerate(scenarios):
            design = design_from(rng, i)
            run_id = f"challenge-{i:02d}"
            record = instrument_return(run_id, run_id, design, material_card(design, rng),
                                       DESCRIPTORS, rng, case=case)
            if name == "undetected meter fault":
                record["targets"]["transmission"]["value"] *= 0.72
            outcome = ledger.ingest(record)
            action = outcome["action"]
            disposition = ("deferred" if action == "ask" else
                           "harmful_admission" if action == "use" and not usable else
                           "harmful_exclusion" if action != "use" and usable else "consistent")
            rows.append({"scenario": name, "usable_by_construction": usable, "action": action,
                         "initial_routing": disposition})
    settled = [row for row in rows if row["initial_routing"] != "deferred"]
    return {"basis": "one constructed case per scenario; counts describe this challenge set",
            "cases": rows, "settled": len(settled), "total": len(rows),
            "harmful_admissions": sum(r["initial_routing"] == "harmful_admission" for r in rows),
            "harmful_exclusions": sum(r["initial_routing"] == "harmful_exclusion" for r in rows)}
