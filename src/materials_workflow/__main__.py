"""Command line for materials experiment intake, prediction and planning."""

import argparse
import json
from pathlib import Path
import sqlite3

from .campaign import PRESET, concise_prediction, design_batch, run_demo, write_json
from .descriptor_view import materialize_descriptors, read_descriptors
from .formulation import PerformanceModel
from .registry import Registry
from .shared_experiment import PRESET as SHARED_PRESET, run_shared_demo
from .shared_session import BudgetSession
from .value_of_information import measure_or_make


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="materials-workflow", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a full synthetic campaign in a fresh directory")
    demo.add_argument("--output", type=Path, default=Path(".local/campaign"))
    demo.add_argument("--config", type=Path, default=PRESET)
    paired = commands.add_parser("shared-budget", help="execute two evidence-dependent campaigns with one budget each")
    paired.add_argument("--output", type=Path, default=Path(".local/shared-budget"))
    paired.add_argument("--config", type=Path, default=SHARED_PRESET)
    live = commands.add_parser("session", help="run the shared-budget policy one observation at a time")
    operations = live.add_subparsers(dest="operation", required=True)
    init = operations.add_parser("init", help="register a public hypothesis and action specification")
    init.add_argument("--spec", type=Path, required=True)
    reserve = operations.add_parser("next", help="reserve the chosen action or resume the outstanding one")
    record = operations.add_parser("observe", help="complete an already reserved action with its observed result")
    record.add_argument("--step", type=int, required=True)
    record.add_argument("--action", required=True)
    record.add_argument("--outcome", required=True, help="one of the reserved action's observable labels")
    record.add_argument("--receipt", type=Path, required=True, help="nonempty JSON object containing the evidence")
    history = operations.add_parser("status", help="inspect decisions, observations and remaining credits")
    for operation in (init, reserve, record, history):
        operation.add_argument("--db", type=Path, required=True)
        operation.add_argument("--out", type=Path, help="write JSON output to this path")
    ingest = commands.add_parser("ingest", help="land a JSON record or list and return the dispositions")
    ingest.add_argument("--input", required=True, type=Path)
    audit = commands.add_parser("audit", help="inspect receipts, raw submissions and normalized values")
    audit.add_argument("--submission", type=int)
    features = commands.add_parser("features", help="materialize a current descriptor view or read a named snapshot")
    features.add_argument("--snapshot", help="read the historical view with this identity")
    predict = commands.add_parser("predict", help="predict performance for a specified design")
    predict.add_argument("--candidate", type=Path, required=True)
    measure = commands.add_parser("measure-or-make", help="price a supplier assay against choosing from a shortlist")
    measure.add_argument("--shortlist", type=Path, required=True)
    measure.add_argument("--cost", type=float, default=0.01, help="cost in normalized decision utility")
    plan = commands.add_parser("recommend", help="select affordable designs or independent repeats")
    plan.add_argument("--candidates", type=Path, required=True)
    plan.add_argument("--pending", type=Path, help="JSON list of designs already scheduled")
    plan.add_argument("--budget", type=float, required=True)
    plan.add_argument("--batch-size", type=int, default=3)
    for command in (ingest, audit, predict, plan, features, measure):
        command.add_argument("--db", required=True, type=Path, help="SQLite campaign ledger")
        command.add_argument("--out", type=Path, help="write JSON output to this path")
    for command in (predict, plan, features, measure):
        command.add_argument("--seed", type=int, default=5837)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            payload = run_demo(args.output, args.config)
        elif args.command == "shared-budget":
            payload = run_shared_demo(args.output, args.config)
        elif args.command == "session":
            with BudgetSession(args.db, spec=load(args.spec) if args.operation == "init" else None) as session:
                if args.operation == "next":
                    payload = session.next()
                elif args.operation == "observe":
                    payload = session.observe(args.step, args.action, args.outcome, load(args.receipt))
                else:
                    payload = session.status()
        else:
            if args.command != "ingest" and not args.db.is_file():
                raise ValueError("The specified database does not exist.")
            with Registry(args.db) as registry:
                if args.command == "ingest":
                    records = load(args.input)
                    payload = [registry.ingest(r) for r in (records if isinstance(records, list) else [records])]
                elif args.command == "audit":
                    payload = registry.audit() if args.submission is None else registry.submission(args.submission)
                elif args.command == "features":
                    if args.snapshot:
                        payload = read_descriptors(registry, args.snapshot)
                    else:
                        rows = registry.training_rows()
                        model = PerformanceModel(seed=args.seed).fit(rows)
                        view = materialize_descriptors(registry, model, rows)
                        payload = read_descriptors(registry, view["snapshot_id"])
                elif args.command == "predict":
                    model = PerformanceModel(seed=args.seed).fit(registry.training_rows())
                    payload = concise_prediction(model, load(args.candidate))
                elif args.command == "measure-or-make":
                    model = PerformanceModel(seed=args.seed).fit(registry.training_rows())
                    payload = measure_or_make(model, load(args.shortlist), seed=args.seed, credit_cost=args.cost)
                else:
                    payload = design_batch(registry, load(args.candidates), load(args.pending) if args.pending else [],
                                           seed=args.seed, budget=args.budget, batch_size=args.batch_size)
        if getattr(args, "out", None):
            write_json(args.out, payload)
        else:
            print(json.dumps(payload, indent=2, allow_nan=False))
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
