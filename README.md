# Forex-Volatility-Helper

Forecast the next-week (or multi-week) pip range for a forex pair using hourly data.
The toolkit downloads hourly candles, engineers weekly volatility features, and offers
multiple model families—gradient boosted trees, quantile regression, and classical
GARCH—to estimate both expected volatility and the probability that the coming move
exceeds user-defined pip thresholds.

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download hourly data (Yahoo Finance symbol)**
   ```bash
   ./data/download_data.py --pair EURUSD=X --start 2018-01-01 --output data/raw
   ```

3. **Build the weekly modelling dataset**
   ```bash
   ./data/make_dataset.py --pair EURUSD=X --horizon 1 --weekly-lags 1 2 3 4
   ```
   Add `--min-hours 72` (or similar) if the download has sparse weeks you still want to keep.
   When `data/preload/` contains Tick Data Suite exports (GMT+0), they are normalised
   (once) and merged automatically with any fresh downloads in `data/raw/`.
   Override directories via `--preload-dir` and `--raw-dir`, or pass a specific CSV
   with `--input`.

4. **Train a volatility model (tree-based mean forecast)**
   ```bash
   ./models/train_model.py \
       --dataset data/processed/EURUSD_X_h1.csv \
       --pair EURUSD=X \
       --horizon 1 \
       --model-type tree
   ```

   Use `--model-type quantile --quantiles 0.1 0.5 0.9` for probabilistic intervals,
   or `./models/train_garch.py --dataset ...` to fit a GARCH volatility model.

5. **Generate a forecast and exceedance probabilities**
   ```bash
  ./predict.py \
      --pair EURUSD=X \
      --model models/EURUSD_X_h1.joblib \
      --horizon 1 \
      --thresholds 350 400 500 \
      --min-hours 96 \
      --weekly-lags 1 2 3 4
   ```

Example output (tree model):
```
=== Volatility Forecast ===
Pair: EURUSD=X
Horizon: next 1 week(s)
Model type: tree
Predicted pip range (mean/median): 412.58

=== Exceedance Probabilities ===
P(range > 350 pips) = 73.4%
P(range > 400 pips) = 52.1%
P(range > 500 pips) = 21.7%
```

## Project Structure

- `data/download_data.py` – fetch hourly OHLCV candles from Yahoo Finance.
- `data/make_dataset.py` – convert hourly candles into weekly volatility features.
- `models/train_model.py` – train tree or quantile models, evaluate, and save artifacts.
- `models/train_garch.py` – fit a GARCH(p, q) volatility process on weekly closes.
- `predict.py` – load trained models (tree, quantile, or GARCH) and produce forecasts.
- `src/forex_helper/` – reusable utilities for data prep and modelling.

## Model Families

- **tree**: HistGradientBoostingRegressor that minimises squared error and reports mean
  pip range plus exceedance probabilities using a Gaussian residual approximation.
- **quantile**: GradientBoostingRegressor trained at user-defined quantiles (default
  10/50/90th) to provide predictive intervals and probability estimates.
- **garch**: Classical GARCH(p, q) process fit on weekly log returns; forecasts future
  return volatility and converts it to pip magnitudes using the latest close.

## How the Volatility Target Is Defined

- Hourly candles are aggregated into ISO weeks that start on Mondays.
- For each week, the realized pip range is `(weekly high - weekly low) / pip_size`.
- Forecasts aim at the sum of the next *h* weeks' pip ranges (default `h=1`).
- For tree and quantile models, exceedance probabilities assume residuals follow a
  normal distribution with variance estimated from training residuals or quantile spreads.
- For GARCH models, exceedance probabilities reflect a zero-mean normal assumption on
  the forecasted return volatility.

## Extending the Helper

- Tune feature engineering (technical indicators, macro inputs, different horizons).
- Experiment with additional quantile levels or alternative volatility processes.
- Backtest exceedance strategies, calibrate probability forecasts, or integrate with
  risk dashboards and deployment pipelines.
