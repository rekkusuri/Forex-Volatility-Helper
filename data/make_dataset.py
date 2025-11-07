#!/usr/bin/env python
"""
Transform raw hourly forex data into weekly modelling features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forex_helper.data import get_hourly_history
from forex_helper.preprocessing import (
    add_hourly_pip_features,
    build_weekly_dataset,
    infer_pip_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        required=True,
        help="Forex pair symbol (e.g. EURUSD=X) used to infer pip size.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in weeks (default: 1).",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=96,
        help="Minimum hourly candles required per week to keep it (default: 96).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory to store the processed dataset (default: data/processed).",
    )
    parser.add_argument(
        "--weekly-lags",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Lagged weeks to include as features (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--preload-dir",
        type=Path,
        default=Path("data/preload"),
        help="Directory containing preload GMT+0 data to merge before processing.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw downloads to merge (default: data/raw).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional extra raw CSV to include (in addition to --raw-dir glob).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_sources = [args.input] if args.input else None
    hourly_df, meta = get_hourly_history(
        args.pair,
        preload_dir=args.preload_dir,
        raw_dir=args.raw_dir,
        extra_sources=extra_sources,
    )
    if meta.get("preload_rows") is not None:
        print(
            f"Loaded {meta['preload_rows']:,} rows from preload "
            f"{meta.get('preload_path') or ''}".strip()
        )
    if meta.get("raw_files"):
        print(
            f"Added {meta['raw_rows']:,} rows from {meta['raw_files']} raw download(s)."
        )
    print(f"Total unique hourly candles: {meta['final_rows']:,}.")

    pip_size = infer_pip_size(args.pair)
    hourly = add_hourly_pip_features(hourly_df, pip_size)
    dataset = build_weekly_dataset(
        hourly,
        horizon_weeks=args.horizon,
        weekly_lags=tuple(sorted(set(args.weekly_lags))),
        min_hours_per_week=args.min_hours,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.pair.replace('=', '_')}_h{args.horizon}.csv"
    processed = dataset.features.copy()
    processed["target_future_range_pips"] = dataset.target
    processed.to_csv(out_path, index=True)

    retained_weeks = len(processed)
    print(f"Saved dataset with {retained_weeks} rows to {out_path}")


if __name__ == "__main__":
    main()
