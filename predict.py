#!/usr/bin/env python
"""
Generate volatility forecasts and exceedance probabilities for a forex pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import joblib
import pandas as pd

from forex_helper.data import get_hourly_history
from forex_helper.model import VolatilityModel
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
        help="Forex pair symbol (e.g. EURUSD=X). Used to infer pip size.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model artifact .joblib file.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in weeks (must match the trained model).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[300.0, 400.0, 500.0],
        help="List of volatility thresholds (in pips) for exceedance probabilities.",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=96,
        help="Minimum hourly candles per week required for features (default: 96).",
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
        help="Directory with downloaded hourly CSVs (default: data/raw).",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        help="Optional extra raw CSV to include (in addition to --raw-dir glob).",
    )
    parser.add_argument(
        "--weekly-lags",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Lagged weeks to include when building features (default: 1 2 3 4). "
        "Must match the configuration used during training.",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> VolatilityModel:
    payload = joblib.load(model_path)
    artifact = payload["model"] if isinstance(payload, dict) else payload
    if not isinstance(artifact, VolatilityModel):
        raise TypeError(f"Model artifact at {model_path} is not a VolatilityModel.")
    return artifact


def main() -> None:
    args = parse_args()
    extra_sources = [args.raw_csv] if args.raw_csv else None
    df, _ = get_hourly_history(
        args.pair,
        preload_dir=args.preload_dir,
        raw_dir=args.raw_dir,
        extra_sources=extra_sources,
    )

    pip_size = infer_pip_size(args.pair)
    hourly = add_hourly_pip_features(df, pip_size)
    dataset = build_weekly_dataset(
        hourly,
        horizon_weeks=args.horizon,
        weekly_lags=tuple(sorted(set(args.weekly_lags))),
        min_hours_per_week=args.min_hours,
    )

    model = load_model(args.model)
    print("=== Volatility Forecast ===")
    print(f"Pair: {args.pair}")
    print(f"Horizon: next {args.horizon} week(s)")
    print(f"Model type: {model.model_type}")

    if model.model_type in {"tree", "quantile"}:
        if dataset.features.empty:
            raise ValueError(
                "No complete weekly feature rows available. "
                "Consider relaxing --min-hours or extending the data window."
            )
        feature_vector = dataset.features.tail(1)

        missing_cols = set(model.feature_columns or []) - set(feature_vector.columns)
        if missing_cols:
            raise ValueError(
                f"Missing feature columns required by the model: {missing_cols}"
            )

        prediction = model.predict(feature_vector)[-1]
        print(f"Predicted pip range (mean/median): {prediction:.2f}")

        if model.model_type == "quantile":
            quantile_preds = model.predict_quantiles(feature_vector)
            print("\n--- Quantile Forecasts ---")
            for level in sorted(quantile_preds):
                value = quantile_preds[level][-1]
                print(f"Q{int(level * 100):02d}: {value:.2f} pips")

        probabilities = model.probability_exceed(
            feature_vector, thresholds=args.thresholds
        )
        print("\n=== Exceedance Probabilities ===")
        for threshold, values in probabilities.items():
            prob = values[-1] * 100
            print(f"P(range > {threshold:.0f} pips) = {prob:.1f}%")

    elif model.model_type == "garch":
        close_series = dataset.full_features.get("close_last")
        if close_series is None or close_series.dropna().empty:
            raise ValueError(
                "GARCH inference requires the 'close_last' column. "
                "Rebuild the dataset or ensure close prices are present."
            )
        latest_close = float(close_series.dropna().iloc[-1])
        expected_pips = model.predict_from_close(latest_close)
        std_return = model._forecast_return_std()
        std_pips = std_return * latest_close / model.pip_size

        print(f"Latest weekly close: {latest_close:.5f}")
        print(f"Expected absolute move: {expected_pips:.2f} pips")
        print(f"Std. deviation (pips): {std_pips:.2f}")

        probabilities = model.probability_exceed_from_close(
            latest_close, thresholds=args.thresholds
        )
        print("\n=== Exceedance Probabilities ===")
        for threshold, prob in probabilities.items():
            print(f"P(|return| > {threshold:.0f} pips) = {prob * 100:.1f}%")
    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")


if __name__ == "__main__":
    main()
