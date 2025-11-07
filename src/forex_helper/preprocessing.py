"""
Preprocessing utilities to engineer features for weekly forex volatility forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def infer_pip_size(symbol: str) -> float:
    """
    Infer pip size from the forex symbol.

    EURUSD -> 0.0001, USDJPY -> 0.01, etc.
    """

    base_symbol = symbol.replace("=X", "").upper()
    if base_symbol.endswith("JPY"):
        return 0.01
    return 0.0001


def add_hourly_pip_features(df: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    """
    Compute hourly pip-based metrics required for downstream aggregation.
    """

    hourly = df.copy()
    hourly.attrs["pip_size"] = pip_size
    hourly["pip_close"] = hourly["close"] / pip_size
    hourly["pip_change"] = hourly["close"].diff() / pip_size
    hourly["abs_pip_change"] = hourly["pip_change"].abs()
    hourly["pip_high_low"] = (hourly["high"] - hourly["low"]) / pip_size

    prev_close = hourly["close"].shift()
    true_range = np.maximum(
        hourly["high"] - hourly["low"],
        np.maximum((hourly["high"] - prev_close).abs(), (hourly["low"] - prev_close).abs()),
    )
    hourly["true_range_pips"] = true_range / pip_size
    return hourly


@dataclass
class WeeklyDataset:
    features: pd.DataFrame
    target: pd.Series
    feature_columns: Iterable[str]
    full_features: pd.DataFrame


def _future_range(series: pd.Series, horizon: int) -> pd.Series:
    """
    Compute aggregated future range across the specified horizon using
    forward-looking weekly ranges.
    """

    forward = series.shift(-1)
    rolled = forward.rolling(window=horizon, min_periods=horizon).sum()
    if horizon > 1:
        rolled = rolled.shift(-(horizon - 1))
    return rolled


def build_weekly_dataset(
    hourly: pd.DataFrame,
    horizon_weeks: int = 1,
    weekly_lags: Iterable[int] = (1, 2, 3, 4),
    min_hours_per_week: int = 96,
) -> WeeklyDataset:
    """
    Aggregate hourly data into a weekly feature matrix and future volatility targets.

    Parameters
    ----------
    hourly:
        Hourly OHLCV DataFrame with pip features (see `add_hourly_pip_features`).
    horizon_weeks:
        Forecast horizon in weeks (1 = next week, 2 = next two weeks, ...).
    weekly_lags:
        Lag periods (in weeks) to include as autoregressive features.
    min_hours_per_week:
        Minimum number of hourly observations required for a week to stay in the sample.
    """

    if "pip_change" not in hourly.columns or "true_range_pips" not in hourly.columns:
        raise ValueError(
            "Hourly dataframe must include pip features. "
            "Call `add_hourly_pip_features` first."
        )

    hourly = hourly.copy()
    if isinstance(hourly.index, pd.DatetimeIndex) and hourly.index.tz is not None:
        hourly.index = hourly.index.tz_convert("UTC").tz_localize(None)

    agg_map = {
        "close": "last",
        "pip_change": ["mean", "std"],
        "abs_pip_change": ["mean", "max"],
        "pip_high_low": ["mean", "max"],
        "true_range_pips": ["mean", "sum"],
    }
    if "volume" in hourly.columns:
        agg_map["volume"] = "sum"

    weekly = hourly.resample("W-MON", label="left", closed="left").agg(agg_map)
    weekly.columns = ["_".join(filter(None, col)).strip("_") for col in weekly.columns]
    pip_size = hourly.attrs.get("pip_size", 1.0)

    high_max = hourly["high"].resample("W-MON", label="left", closed="left").max()
    low_min = hourly["low"].resample("W-MON", label="left", closed="left").min()
    weekly["week_range_pips"] = (high_max - low_min) / pip_size

    hours_in_week = hourly["close"].resample("W-MON", label="left", closed="left").count()
    weekly["hours_in_week"] = hours_in_week
    weekly = weekly[weekly["hours_in_week"] >= min_hours_per_week]

    weekly["future_range_pips"] = _future_range(weekly["week_range_pips"], horizon_weeks)

    for lag in weekly_lags:
        weekly[f"range_pips_lag_{lag}"] = weekly["week_range_pips"].shift(lag)
        weekly[f"true_range_sum_lag_{lag}"] = weekly["true_range_pips_sum"].shift(lag)
        weekly[f"abs_pip_mean_lag_{lag}"] = weekly["abs_pip_change_mean"].shift(lag)
        weekly[f"pip_std_lag_{lag}"] = weekly["pip_change_std"].shift(lag)

    feature_columns = [
        col
        for col in weekly.columns
        if col
        not in {
            "future_range_pips",
            "hours_in_week",
            "volume_sum",
        }
    ]

    features = weekly[feature_columns]
    target = weekly["future_range_pips"]

    dataset = pd.concat([features, target], axis=1)
    cleaned = dataset.dropna()
    target_series = cleaned.pop("future_range_pips")
    return WeeklyDataset(
        features=cleaned,
        target=target_series,
        feature_columns=features.columns,
        full_features=features,
    )


def train_test_split_time_series(
    features: pd.DataFrame, target: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Deterministic chronological split for time-series modelling.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_idx = int(len(features) * (1 - test_size))
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]
    return X_train, X_test, y_train, y_test
