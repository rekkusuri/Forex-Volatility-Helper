#!/usr/bin/env python
"""
Train a forex volatility forecasting model from a processed dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd

from forex_helper.model import (
    VolatilityModel,
    evaluate_quantile_models,
    evaluate_tree_model,
    persist_model,
    train_quantile_models,
    train_tree_model,
)
from forex_helper.preprocessing import infer_pip_size, train_test_split_time_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the processed dataset CSV created by data/make_dataset.py.",
    )
    parser.add_argument(
        "--pair",
        required=True,
        help="Forex pair symbol (e.g. EURUSD=X) to record metadata.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in weeks (default: 1).",
    )
    parser.add_argument(
        "--model-type",
        choices=("tree", "quantile"),
        default="tree",
        help="Estimator family to train (default: tree).",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Quantile levels to fit when --model-type=quantile.",
    )
    parser.add_argument(
        "--tree-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for the tree regressor (default: 0.05).",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=6,
        help="Maximum depth for each tree in the regressor (default: 6).",
    )
    parser.add_argument(
        "--tree-max-iter",
        type=int,
        default=400,
        help="Number of boosting iterations for the tree regressor (default: 400).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models"),
        help="Directory to store the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/metrics"),
        help="Directory to store model evaluation metrics (JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset, index_col=0, parse_dates=True)
    target_col = "target_future_range_pips"

    if target_col not in df.columns:
        raise ValueError(f"Dataset missing '{target_col}' column.")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split_time_series(features, target)

    pair = args.pair
    pip_size = infer_pip_size(pair)
    feature_columns = list(features.columns)

    if args.model_type == "tree":
        estimator = train_tree_model(
            X_train,
            y_train,
            learning_rate=args.tree_learning_rate,
            max_depth=args.tree_max_depth,
            max_iter=args.tree_max_iter,
        )
        metrics = evaluate_tree_model(estimator, X_train, y_train, X_test, y_test)
        artifact = VolatilityModel(
            model_type="tree",
            horizon_weeks=args.horizon,
            pip_size=pip_size,
            feature_columns=feature_columns,
            estimator=estimator,
            residual_std=metrics.rmse_train,
        )
    else:
        quantile_levels = sorted({float(q) for q in args.quantiles})
        if not quantile_levels:
            raise ValueError("At least one quantile level must be provided.")
        quantile_models = train_quantile_models(
            X_train,
            y_train,
            quantiles=quantile_levels,
        )
        metrics = evaluate_quantile_models(
            quantile_models, X_train, y_train, X_test, y_test
        )

        median_level = min(quantile_levels, key=lambda q: abs(q - 0.5))
        residuals = y_train - quantile_models[median_level].predict(X_train)
        residual_std = float(np.std(residuals, ddof=1))

        artifact = VolatilityModel(
            model_type="quantile",
            horizon_weeks=args.horizon,
            pip_size=pip_size,
            feature_columns=feature_columns,
            estimator=quantile_models[median_level],
            residual_std=residual_std,
            quantile_estimators=dict(quantile_models),
            quantile_levels=quantile_levels,
        )

    model_dir = args.model_output
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.pair.replace('=', '_')}_h{args.horizon}.joblib"

    persist_model(artifact, metrics=metrics, model_path=model_path)

    metrics_dir = args.metrics_output
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.pair.replace('=', '_')}_h{args.horizon}.json"
    metrics_payload = asdict(metrics)
    metrics_payload["extra"] = {
        key: float(value) for key, value in metrics_payload.get("extra", {}).items()
    }
    with metrics_path.open("w") as fp:
        json.dump(
            {
                "pair": args.pair,
                "horizon_weeks": args.horizon,
                "model_type": args.model_type,
                "metrics": metrics_payload,
            },
            fp,
            indent=2,
        )

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
