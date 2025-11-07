"""
Model training and inference utilities for forex volatility forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _pinball_loss(y_true: pd.Series, y_pred: np.ndarray, alpha: float) -> float:
    """
    Compute the pinball (quantile) loss for a given alpha.
    """

    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


@dataclass
class ModelMetrics:
    mae_train: float
    mae_test: float
    rmse_train: float
    rmse_test: float
    r2_train: float
    r2_test: float
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class VolatilityModel:
    """
    Serializable model artifact supporting multiple estimator families.
    """

    model_type: str
    horizon_weeks: int
    pip_size: float
    feature_columns: Optional[List[str]] = None
    estimator: Optional[Any] = None
    residual_std: Optional[float] = None
    quantile_estimators: Optional[Dict[float, Any]] = None
    quantile_levels: Optional[List[float]] = None
    return_scale: Optional[float] = None
    garch_result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Common helpers -------------------------------------------------

    def _require_features(self) -> List[str]:
        if not self.feature_columns:
            raise ValueError("This model artifact does not define feature columns.")
        return self.feature_columns

    # --- Point forecasts -------------------------------------------------

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict the expected pip range for the configured horizon.
        """

        if self.model_type in {"tree", "quantile"}:
            X = features[self._require_features()]
            if self.model_type == "tree":
                return self.estimator.predict(X)
            median_level = self._median_quantile_level()
            estimator = self.quantile_estimators[median_level]
            return estimator.predict(X)
        if self.model_type == "garch":
            raise ValueError(
                "GARCH artifacts do not accept tabular features. "
                "Use `predict_from_close` instead."
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict_from_close(self, latest_close: float) -> float:
        """
        Predict the expected pip range using the latest closing price.
        Only applicable to GARCH-based artifacts.
        """

        if self.model_type != "garch":
            raise ValueError("predict_from_close is only available for GARCH artifacts.")
        std_return = self._forecast_return_std()
        pip_scale = latest_close / self.pip_size
        expected_abs_move = std_return * pip_scale * np.sqrt(2 / np.pi)
        return float(expected_abs_move)

    # --- Quantile utilities ---------------------------------------------

    def _median_quantile_level(self) -> float:
        if not self.quantile_levels:
            raise ValueError("Quantile levels not populated for this artifact.")
        return min(self.quantile_levels, key=lambda q: abs(q - 0.5))

    def predict_quantiles(self, features: pd.DataFrame) -> Dict[float, np.ndarray]:
        if self.model_type != "quantile":
            raise ValueError("Quantile predictions only available for quantile models.")
        X = features[self._require_features()]
        return {q: est.predict(X) for q, est in self.quantile_estimators.items()}

    def _quantile_based_std(
        self, quantile_preds: Mapping[float, np.ndarray]
    ) -> np.ndarray:
        """
        Approximate residual standard deviation from quantile forecasts.
        """

        if len(quantile_preds) < 2:
            if self.residual_std is None:
                raise ValueError("Insufficient information to approximate variance.")
            return np.full_like(next(iter(quantile_preds.values())), self.residual_std)

        sorted_levels = sorted(quantile_preds.keys())
        low_level = sorted_levels[0]
        high_level = sorted_levels[-1]
        low_pred = quantile_preds[low_level]
        high_pred = quantile_preds[high_level]
        z_diff = norm.ppf(high_level) - norm.ppf(low_level)
        z_diff = z_diff if z_diff != 0 else 1.0
        std_est = (high_pred - low_pred) / z_diff
        std_est = np.where(std_est <= 0, self.residual_std or 1.0, std_est)
        return std_est

    # --- GARCH utilities ------------------------------------------------

    def _forecast_return_std(self) -> float:
        if self.model_type != "garch":
            raise ValueError("Return volatility is only defined for GARCH artifacts.")
        if self.garch_result is None:
            raise ValueError("GARCH result not present in artifact.")
        forecast = self.garch_result.forecast(horizon=self.horizon_weeks, reindex=False)
        variance_matrix = forecast.variance.values
        variance_scaled = variance_matrix[-1, self.horizon_weeks - 1]
        scale = self.return_scale or 1.0
        std_return = float(np.sqrt(max(variance_scaled, 0.0))) / scale
        return std_return

    # --- Probability estimates -----------------------------------------

    def probability_exceed(
        self, features: pd.DataFrame, thresholds: Iterable[float]
    ) -> Dict[float, np.ndarray]:
        """
        Estimate exceedance probabilities for pip thresholds.
        """

        if self.model_type == "tree":
            preds = self.predict(features)
            std = self.residual_std or np.std(preds)
            std = max(float(std), 1e-6)
            return {
                threshold: 1 - norm.cdf((threshold - preds) / std) for threshold in thresholds
            }

        if self.model_type == "quantile":
            quantile_preds = self.predict_quantiles(features)
            median_level = self._median_quantile_level()
            center_pred = quantile_preds[median_level]
            std_est = self._quantile_based_std(quantile_preds)
            std_est = np.where(std_est <= 0, self.residual_std or 1.0, std_est)
            results: Dict[float, np.ndarray] = {}
            for threshold in thresholds:
                z = (threshold - center_pred) / std_est
                results[threshold] = 1 - norm.cdf(z)
            return results

        raise ValueError(
            "Tabular exceedance probabilities not supported for model_type "
            f"{self.model_type}. Use probability_exceed_from_close."
        )

    def probability_exceed_from_close(
        self, latest_close: float, thresholds: Iterable[float]
    ) -> Dict[float, float]:
        """
        Estimate exceedance probabilities for GARCH models using the latest close.
        """

        if self.model_type != "garch":
            raise ValueError(
                "probability_exceed_from_close available only for GARCH artifacts."
            )
        std_return = self._forecast_return_std()
        std_pips = std_return * latest_close / self.pip_size
        std_pips = max(std_pips, 1e-6)
        results: Dict[float, float] = {}
        for threshold in thresholds:
            z = threshold / std_pips
            results[threshold] = float(2 * (1 - norm.cdf(z)))
        return results


# ---------------------------------------------------------------------------
# Gradient boosted tree regression (mean forecast)
# ---------------------------------------------------------------------------


def train_tree_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    max_iter: int = 400,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        loss="squared_error",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_tree_model(
    model: HistGradientBoostingRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelMetrics:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, train_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    rmse_train = mean_squared_error(y_train, train_pred) ** 0.5
    rmse_test = mean_squared_error(y_test, test_pred) ** 0.5
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    return ModelMetrics(
        mae_train=float(mae_train),
        mae_test=float(mae_test),
        rmse_train=float(rmse_train),
        rmse_test=float(rmse_test),
        r2_train=float(r2_train),
        r2_test=float(r2_test),
    )


# ---------------------------------------------------------------------------
# Quantile regression forest (probabilistic forecast)
# ---------------------------------------------------------------------------


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    learning_rate: float = 0.05,
    max_depth: int = 3,
    n_estimators: int = 600,
) -> Dict[float, GradientBoostingRegressor]:
    models: Dict[float, GradientBoostingRegressor] = {}
    for q in quantiles:
        reg = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=0.9,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        models[float(q)] = reg
    return models


def evaluate_quantile_models(
    models: Mapping[float, GradientBoostingRegressor],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelMetrics:
    if not models:
        raise ValueError("No quantile models provided for evaluation.")

    median_level = min(models.keys(), key=lambda q: abs(q - 0.5))
    median_model = models[median_level]

    train_pred = median_model.predict(X_train)
    test_pred = median_model.predict(X_test)

    mae_train = mean_absolute_error(y_train, train_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    rmse_train = mean_squared_error(y_train, train_pred) ** 0.5
    rmse_test = mean_squared_error(y_test, test_pred) ** 0.5
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)

    extra: Dict[str, float] = {}
    for q, model in models.items():
        extra[f"pinball_loss_train_{q:.2f}"] = _pinball_loss(
            y_train, model.predict(X_train), q
        )
        extra[f"pinball_loss_test_{q:.2f}"] = _pinball_loss(
            y_test, model.predict(X_test), q
        )

    sorted_levels = sorted(models.keys())
    if len(sorted_levels) >= 2:
        lower = sorted_levels[0]
        upper = sorted_levels[-1]
        lower_pred = models[lower].predict(X_test)
        upper_pred = models[upper].predict(X_test)
        coverage = np.mean((y_test >= lower_pred) & (y_test <= upper_pred))
        extra[f"interval_coverage_test_{lower:.2f}_{upper:.2f}"] = float(coverage)
        extra["interval_mean_width_test"] = float(np.mean(upper_pred - lower_pred))

    return ModelMetrics(
        mae_train=float(mae_train),
        mae_test=float(mae_test),
        rmse_train=float(rmse_train),
        rmse_test=float(rmse_test),
        r2_train=float(r2_train),
        r2_test=float(r2_test),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def persist_model(
    artifact: VolatilityModel,
    *,
    metrics: ModelMetrics,
    model_path: Path,
) -> Path:
    """
    Persist the trained model artifact and structured metrics.
    """

    metrics_payload = asdict(metrics)
    extra = metrics_payload.get("extra") or {}
    metrics_payload["extra"] = {k: float(v) for k, v in extra.items()}

    payload: MutableMapping[str, Any] = {
        "type": artifact.model_type,
        "model": artifact,
        "metrics": metrics_payload,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    return model_path

