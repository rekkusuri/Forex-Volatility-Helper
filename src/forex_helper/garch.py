"""
Classical volatility modelling utilities (e.g., GARCH) for forex returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

from .model import ModelMetrics, VolatilityModel


@dataclass
class GarchTrainingResult:
    artifact: VolatilityModel
    metrics: ModelMetrics


def _prepare_returns(close_prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
    close = close_prices.dropna().astype(float)
    returns = np.log(close).diff().dropna()
    aligned_close = close.loc[returns.index]
    return returns, aligned_close


def _safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    num = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(1 - num / denom)


def train_garch_model(
    weekly_close: pd.Series,
    *,
    pip_size: float,
    horizon_weeks: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    test_size: float = 0.2,
    return_scale: float = 100.0,
) -> GarchTrainingResult:
    """
    Fit a univariate GARCH(p, q) model on weekly log returns.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be within (0, 1).")

    returns, aligned_close = _prepare_returns(weekly_close)
    if len(returns) < max(p, q) + 5:
        raise ValueError("Insufficient data to fit the requested GARCH order.")

    scaled_returns = returns * return_scale
    model = arch_model(
        scaled_returns,
        vol="Garch",
        p=p,
        q=q,
        dist=dist,
        mean="Constant",
        rescale=False,
    )

    result = model.fit(update_freq=0, disp="off")
    cond_vol_scaled = result.conditional_volatility
    cond_vol = cond_vol_scaled / return_scale

    split_idx = int(len(returns) * (1 - test_size))
    train_idx = returns.index[:split_idx]
    test_idx = returns.index[split_idx:]

    scale_factor = aligned_close / pip_size
    actual_abs_pips = (returns.abs() * scale_factor).astype(float)
    predicted_abs_pips = (cond_vol * scale_factor).astype(float)

    mae_train = float(
        np.mean(
            np.abs(actual_abs_pips.loc[train_idx] - predicted_abs_pips.loc[train_idx])
        )
    )
    mae_test = float(
        np.mean(np.abs(actual_abs_pips.loc[test_idx] - predicted_abs_pips.loc[test_idx]))
    )
    rmse_train = float(
        np.sqrt(
            np.mean(
                (actual_abs_pips.loc[train_idx] - predicted_abs_pips.loc[train_idx]) ** 2
            )
        )
    )
    rmse_test = float(
        np.sqrt(
            np.mean(
                (actual_abs_pips.loc[test_idx] - predicted_abs_pips.loc[test_idx]) ** 2
            )
        )
    )

    r2_train = _safe_r2(
        actual_abs_pips.loc[train_idx], predicted_abs_pips.loc[train_idx]
    )
    r2_test = _safe_r2(actual_abs_pips.loc[test_idx], predicted_abs_pips.loc[test_idx])

    extra: Dict[str, float] = {
        "mean_abs_actual_test": float(actual_abs_pips.loc[test_idx].mean()),
        "mean_abs_predicted_test": float(predicted_abs_pips.loc[test_idx].mean()),
    }

    metrics = ModelMetrics(
        mae_train=mae_train,
        mae_test=mae_test,
        rmse_train=rmse_train,
        rmse_test=rmse_test,
        r2_train=r2_train,
        r2_test=r2_test,
        extra=extra,
    )

    artifact = VolatilityModel(
        model_type="garch",
        horizon_weeks=horizon_weeks,
        pip_size=pip_size,
        return_scale=return_scale,
        garch_result=result,
        metadata={
            "order": {"p": p, "q": q},
            "dist": dist,
            "training_index": returns.index.tolist(),
        },
    )

    return GarchTrainingResult(artifact=artifact, metrics=metrics)
