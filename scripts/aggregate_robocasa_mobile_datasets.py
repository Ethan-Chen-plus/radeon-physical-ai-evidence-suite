#!/usr/bin/env python3
"""Aggregate audited RoboCasa365 mobile task datasets into one LeRobot root."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="datawhale-eai/robocasa365-mobile-3task")
    parser.add_argument("--task", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [args.source_root / task for task in args.task]
    missing = [str(root) for root in roots if not (root / "meta" / "info.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset metadata: {missing}")
    if args.output_root.exists():
        if any(args.output_root.iterdir()):
            raise FileExistsError(f"aggregation output must be absent or empty: {args.output_root}")
        args.output_root.rmdir()
    aggregate_datasets(
        repo_ids=args.task,
        aggr_repo_id=args.repo_id,
        roots=roots,
        aggr_root=args.output_root,
    )
    print(f"aggregated {len(roots)} datasets into {args.output_root}")


if __name__ == "__main__":
    main()
