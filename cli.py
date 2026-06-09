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

