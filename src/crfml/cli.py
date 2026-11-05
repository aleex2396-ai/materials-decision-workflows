import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Synthetic materials learning and control research')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', default='configs/study.json')
    run.add_argument('--output', default='results/public')
    decision = sub.add_parser('decision')
    decision.add_argument('--config', default='configs/decision_study.json')
    decision.add_argument('--output', default='results/decision')
    reference = sub.add_parser('reference')
    reference.add_argument('--config', default='configs/dual_study.json')
    reference.add_argument('--output', default='results/reference')
    mechanism = sub.add_parser('mechanism')
    mechanism.add_argument('--config', default='configs/mechanism_study.json')
    mechanism.add_argument('--decision-config', default='configs/decision_study.json')
    mechanism.add_argument('--output', default='results/mechanism')
    all_studies = sub.add_parser('all')
    all_studies.add_argument('--output',default='results')
    counter = sub.add_parser('counterexample')
    args = parser.parse_args()
    if args.command == 'run':
        from .study import run_study
        from .report import make_report
        run_study(args.config, args.output)
        make_report(Path(args.output))
    elif args.command == 'decision':
        from .decision_study import run_decision_study
        from .decision_report import make_decision_report
        run_decision_study(args.config,args.output)
        make_decision_report(args.output)
    elif args.command == 'reference':
        from .reference_study import run_reference_study
        run_reference_study(args.config,args.output)
    elif args.command == 'mechanism':
        from .mechanism_study import run_mechanism_study
        run_mechanism_study(args.config,args.decision_config,args.output)
    elif args.command == 'all':
        from .study import run_study
        from .report import make_report
        from .decision_study import run_decision_study
        from .decision_report import make_decision_report
        from .reference_study import run_reference_study
        from .mechanism_study import run_mechanism_study
        destination = Path(args.output)
        run_study('configs/study.json',destination/'public')
        make_report(destination/'public')
        run_decision_study('configs/decision_study.json',destination/'decision')
        make_decision_report(destination/'decision')
        run_reference_study('configs/dual_study.json',destination/'reference')
        run_mechanism_study('configs/mechanism_study.json','configs/decision_study.json',destination/'mechanism')
    else:
        import json
        from .study import selection_counterexample
        print(json.dumps(selection_counterexample(), indent=2))


if __name__ == '__main__':
    main()
