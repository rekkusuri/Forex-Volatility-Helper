#!/usr/bin/env python
"""
Train a GARCH volatility model on weekly forex closes.
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

import pandas as pd

from forex_helper.garch import train_garch_model
from forex_helper.model import persist_model
from forex_helper.preprocessing import infer_pip_size


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
        "--order",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("P", "Q"),
        help="GARCH order (p, q). Default: 1 1.",
    )
    parser.add_argument(
        "--dist",
        choices=("normal", "t", "skewt", "ged"),
        default="normal",
        help="Distribution assumption for the innovations (default: normal).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of observations reserved for evaluation (default: 0.2).",
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

    if "close_last" not in df.columns:
        raise ValueError(
            "Processed dataset must include the 'close_last' column derived from weekly aggregation."
        )

    weekly_close = df["close_last"].astype(float)
    pip_size = infer_pip_size(args.pair)
    p, q = args.order

    result = train_garch_model(
        weekly_close,
        pip_size=pip_size,
        horizon_weeks=args.horizon,
        p=p,
        q=q,
        dist=args.dist,
        test_size=args.test_size,
    )

    model_dir = args.model_output
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.pair.replace('=', '_')}_h{args.horizon}_garch.joblib"

    persist_model(result.artifact, metrics=result.metrics, model_path=model_path)

    metrics_dir = args.metrics_output
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.pair.replace('=', '_')}_h{args.horizon}_garch.json"
    metrics_payload = asdict(result.metrics)
    metrics_payload["extra"] = {
        key: float(value) for key, value in metrics_payload.get("extra", {}).items()
    }
    with metrics_path.open("w") as fp:
        json.dump(
            {
                "pair": args.pair,
                "horizon_weeks": args.horizon,
                "model_type": "garch",
                "order": {"p": p, "q": q},
                "dist": args.dist,
                "metrics": metrics_payload,
            },
            fp,
            indent=2,
        )

    print(f"GARCH model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
