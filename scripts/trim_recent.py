#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from forex_helper.data import get_hourly_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim merged hourly history to recent years.")
    parser.add_argument("--pair", required=True, help="Forex pair symbol, e.g. EURUSD=X.")
    parser.add_argument(
        "--years",
        type=int,
        default=4,
        help="Number of years to retain from the most recent data (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/EURUSD_X_recent.csv"),
        help="Destination CSV for the trimmed series.",
    )
    parser.add_argument(
        "--preload-dir",
        type=Path,
        default=Path("data/preload"),
        help="Directory containing preprocessed preload data (default: data/preload).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw downloads (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hourly, _ = get_hourly_history(
        args.pair,
        preload_dir=args.preload_dir,
        raw_dir=args.raw_dir,
    )
    cutoff = hourly.index.max() - pd.Timedelta(days=365 * args.years)
    recent = hourly.loc[hourly.index >= cutoff]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recent.to_csv(args.output)
    print(f"Saved {len(recent):,} rows to {args.output}")


if __name__ == "__main__":
    main()
