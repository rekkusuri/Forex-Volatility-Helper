#!/usr/bin/env python
"""
CLI utility to download hourly forex data via Yahoo Finance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forex_helper.data import DownloadConfig, download_hourly_data, persist_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        required=True,
        help="Yahoo Finance symbol for the forex pair, e.g. EURUSD=X.",
    )
    parser.add_argument("--start", help="Inclusive start date, e.g. 2020-01-01.")
    parser.add_argument("--end", help="Exclusive end date, e.g. 2024-01-01.")
    parser.add_argument(
        "--interval",
        default="1h",
        help="Data interval (default: 1h). Hourly data recommended.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store the downloaded CSV (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DownloadConfig(
        pair=args.pair,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    df = download_hourly_data(config)
    output_file = args.output / f"{args.pair.replace('=', '_')}_{args.interval}.csv"
    persist_dataframe(df, output_file)
    print(f"Saved {len(df):,} rows to {output_file}")


if __name__ == "__main__":
    main()
