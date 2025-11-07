"""
Data acquisition helpers for hourly forex data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The `yfinance` package is required. Install it with `pip install yfinance`."
    ) from exc


@dataclass
class DownloadConfig:
    """Configuration for pulling historical data."""

    pair: str
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1h"
    auto_adjust: bool = True
    back_adjust: bool = False


def download_hourly_data(config: DownloadConfig) -> pd.DataFrame:
    """
    Download hourly OHLCV data for a forex pair using Yahoo Finance.

    Parameters
    ----------
    config:
        Download configuration.

    Returns
    -------
    pandas.DataFrame
        Hourly OHLCV data indexed by timestamp.
    """

    ticker = yf.Ticker(config.pair)
    data = ticker.history(
        start=config.start,
        end=config.end,
        interval=config.interval,
        auto_adjust=config.auto_adjust,
        back_adjust=config.back_adjust,
    )
    if data.empty:
        raise ValueError(
            f"No data returned for {config.pair}. "
            "Check the symbol or relax the date range."
        )
    data = data.rename(columns=str.lower)
    data.index.name = "timestamp"
    return data


def _normalise_column_names(columns: Iterable[str]) -> List[str]:
    return [col.strip().lower().replace(" ", "_") for col in columns]


def _normalise_pair_symbol(pair: str) -> str:
    return pair.upper().replace("=X", "").replace("/", "")


def _convert_preload_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = _normalise_column_names(df.columns)

    if "date" not in df.columns:
        raise ValueError(f"Preload file {csv_path} missing 'Date' column.")

    if "time" not in df.columns:
        df["time"] = "00:00:00"

    timestamp = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        utc=True,
        errors="coerce",
    )
    df = df.assign(timestamp=timestamp).dropna(subset=["timestamp"])

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Preload file {csv_path} missing '{col}'.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tick_volume" in df.columns:
        df["volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0.0)
    elif "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def ensure_preload_processed(
    pair: str,
    preload_dir: Optional[Path],
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Ensure Tick Data Suite preload CSVs are normalised to the canonical format.
    """

    if preload_dir is None:
        return None, None

    preload_dir = Path(preload_dir)
    if not preload_dir.exists():
        return None, None

    normalized = _normalise_pair_symbol(pair)
    raw_files = sorted(
        p
        for p in preload_dir.glob(f"{normalized}*_H1*.csv")
        if "processed" not in p.stem.lower()
    )
    if not raw_files:
        return None, None

    processed_dir = preload_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{normalized}_H1_preprocessed.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path, parse_dates=["timestamp"], index_col="timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df, processed_path

    frames = [_convert_preload_csv(csv_path) for csv_path in raw_files]
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_csv(processed_path)
    return combined, processed_path


def _load_raw_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = _normalise_column_names(df.columns)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "date" in df.columns:
        time_component = df["time"] if "time" in df.columns else "00:00:00"
        ts = pd.to_datetime(
            df["date"].astype(str) + " " + time_component.astype(str),
            utc=True,
            errors="coerce",
        )
    else:
        raise ValueError(f"Raw file {path} missing timestamp information.")

    df = df.assign(timestamp=ts).dropna(subset=["timestamp"])

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Raw file {path} missing '{col}'.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    elif "tick_volume" in df.columns:
        volume = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0.0)
    else:
        volume = 0.0
    df["volume"] = volume

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def _matches_pair(path: Path, normalized_pair: str) -> bool:
    stem = re.sub(r"[^A-Z]", "", path.stem.upper())
    return normalized_pair.upper() in stem


def get_hourly_history(
    pair: str,
    preload_dir: Optional[Path] = Path("data/preload"),
    raw_dir: Optional[Path] = Path("data/raw"),
    extra_sources: Optional[Iterable[Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[int]]]:
    """
    Load hourly history for a forex pair from preprocessed preload data and
    optional raw downloads.
    """

    metadata: Dict[str, Optional[int]] = {
        "preload_rows": None,
        "raw_rows": 0,
        "raw_files": 0,
        "final_rows": None,
        "preload_path": None,
    }

    frames: List[pd.DataFrame] = []
    normalized = _normalise_pair_symbol(pair)

    preload_df, processed_path = ensure_preload_processed(pair, preload_dir)
    if preload_df is not None:
        frames.append(preload_df)
        metadata["preload_rows"] = len(preload_df)
        metadata["preload_path"] = str(processed_path) if processed_path else None

    candidate_paths: List[Path] = []
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        if raw_dir.exists():
            candidate_paths.extend(
                p
                for p in raw_dir.glob("*.csv")
                if _matches_pair(p, normalized)
            )
    if extra_sources:
        candidate_paths.extend(Path(p) for p in extra_sources)

    seen_paths = set()
    for path in candidate_paths:
        if path in seen_paths or not path.exists():
            continue
        raw_df = _load_raw_history(path)
        frames.append(raw_df)
        metadata["raw_files"] = (metadata["raw_files"] or 0) + 1
        metadata["raw_rows"] = (metadata["raw_rows"] or 0) + len(raw_df)
        seen_paths.add(path)

    if not frames:
        raise ValueError(
            f"No data found for {pair}. Ensure preload files exist in {preload_dir} "
            "or provide raw CSV downloads."
        )

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    metadata["final_rows"] = len(combined)
    return combined, metadata


def persist_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Persist a DataFrame as CSV.

    Parameters
    ----------
    df:
        DataFrame to save.
    output_path:
        Destination path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return output_path
