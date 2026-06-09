"""
ExpLens CLI — generate an experiment report from the command line.

Usage:
    python cli.py --run entity/project/run_id --output ./reports/my_run
    python cli.py --csv path/to/log.csv --name "baseline run"
    python cli.py --json path/to/log.json
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="ExpLens — Automated ML Experiment Narratives"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=str,
                        help="WandB run path: entity/project/run_id")
    source.add_argument("--csv", type=str,
                        help="Path to a CSV training log")
    source.add_argument("--json", type=str,
                        help="Path to a JSON training log")

    parser.add_argument("--name", type=str, default="",
                        help="Run name (for CSV/JSON inputs)")
    parser.add_argument("--output", type=str, default="./reports",
                        help="Output directory for report and charts")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip consistency checking")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress to stdout")

    args = parser.parse_args()

    # Load run data
    if args.verbose:
        print("Loading run data...")

    if args.run:
        from src.connectors.wandb_connector import load_run
        run = load_run(args.run)
    elif args.csv:
        from src.connectors.local_connector import load_from_csv
        run = load_from_csv(args.csv, run_name=args.name or "local_run")
    elif args.json:
        from src.connectors.local_connector import load_from_json
        run = load_from_json(args.json)

    # Extract facts
    if args.verbose:
        print(f"Extracted facts for: {run.run_name}")

    from src.analysis.extractor import extract_facts
    facts = extract_facts(run)

    if args.verbose:
        print(f"Detected {len(facts.facts)} measurable facts")
        print(f"Patterns: {[p.title for p in __import__('src.analysis.patterns', fromlist=['detect_patterns']).detect_patterns(facts)]}")

    # Generate narrative
    if args.verbose:
        print("Generating narrative...")

    from src.narrator.generator import ExperimentNarrator
    narrator = ExperimentNarrator()
    result = narrator.generate(facts)

    # Report consistency
    if not args.no_check:
        failed = result.failed_checks()
        if failed:
            print(f"\n⚠️  {len(failed)} consistency issue(s) detected:")
            for section, check in failed:
                print(f"  [{section}] Claimed: {check.claim_text}")
                print(f"  Actual: {check.actual_value}")
        else:
            if args.verbose:
                print("✓ All claims consistent with data")

    # Build report
    output_dir = os.path.join(args.output, run.run_name.replace("/", "_"))
    from src.report.builder import build_report
    report_path = build_report(result, output_dir)

    print(f"\nReport saved to: {report_path}")
    if run.url:
        print(f"WandB run: {run.url}")


if __name__ == "__main__":
    main()
