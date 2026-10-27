"""Feature Engineering Module for SupplySense AI.
Provides rigorous, leakage-free feature transformations for:
1. Calendar & temporal attributes (order year, month, day, day-of-week, weekend indicator, quarter).
2. Time-series demand aggregation and historical lag / rolling window construction for forecasting.
3. Pre-dispatch feature matrices for shipment delay risk classification without future target leakage.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_date_features(
    df: pd.DataFrame,
    date_col: str = "order_date_parsed",
) -> pd.DataFrame:
    """Extract temporal features from parsed order timestamp.

    Features created:
        - order_year: Calendar year
        - order_month: Month of order (1-12)
        - order_day: Day of month (1-31)
        - order_day_of_week: Day of week (0=Mon, 6=Sun)
        - is_weekend: Binary indicator (1 if Sat/Sun, else 0)
        - order_quarter: Calendar quarter (1-4)
    """
    data = df.copy()
    if date_col not in data.columns or data[date_col].isnull().all():
        raise ValueError(f"Column '{date_col}' must exist and contain valid datetime values.")

    dt_series = data[date_col]
    data["order_year"] = dt_series.dt.year
    data["order_month"] = dt_series.dt.month
    data["order_day"] = dt_series.dt.day
    data["order_day_of_week"] = dt_series.dt.dayofweek
    data["is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
    data["order_quarter"] = dt_series.dt.quarter

    return data


def aggregate_demand_by_date(
    df: pd.DataFrame,
    category: Optional[str] = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Aggregate transactional order quantities into continuous time-series demand.

    Args:
        df: Cleaned supply chain dataframe
        category: Specific category name or None for all
        freq: Frequency for continuous timeline resampling ('D' for daily, 'W' for weekly)

    Returns:
        pd.DataFrame with 'date' and 'demand' columns.
    """
    if "order_date_parsed" not in df.columns or df["order_date_parsed"].isnull().all():
        raise ValueError("Valid 'order_date_parsed' column is required for demand aggregation.")

    data = df.dropna(subset=["order_date_parsed"]).copy()
    if category and category != "All Categories" and "category_clean" in data.columns:
        data = data[data["category_clean"] == category]

    if data.empty:
        return pd.DataFrame(columns=["date", "demand"])

    data["date"] = data["order_date_parsed"].dt.floor("D")
    qty_col = "order_item_quantity_clean" if "order_item_quantity_clean" in data.columns else "order_quantity"
    daily = data.groupby("date")[qty_col].sum().reset_index()
    daily = daily.rename(columns={qty_col: "demand"})
    daily = daily.sort_values("date").reset_index(drop=True)

    # Resample to ensure unbroken chronology without missing dates
    daily = daily.set_index("date")
    daily = daily.resample(freq).sum().fillna(0.0)
    daily = daily.reset_index()
    return daily


def build_time_series_features(
    ts_df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 7, 14],
    rolling_windows: List[int] = [7, 14],
) -> pd.DataFrame:
    """Construct strictly backward-looking lag and rolling statistical features.

    To strictly prevent future-data leakage:
        - Lag t-1 represents yesterday's observed demand.
        - Rolling means at time t are computed over shifted series (t-1, t-2, ...),
          ensuring today's demand (y_t) is NEVER included in any predictor.
    """
    df = ts_df.copy()
    if "demand" not in df.columns:
        raise ValueError("Dataframe must contain 'demand' column.")

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    # Backward lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # Backward rolling statistics on shifted series
    shifted = df["demand"].shift(1)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w, min_periods=1).std().fillna(0.0)

    return df


def prepare_delay_classification_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Extract strictly pre-dispatch features and binary target for shipment delay risk.

    Strict Leakage Prevention:
        Excludes actual shipment duration ('days_for_shipping_real'), delivery timestamp,
        and post-dispatch status codes. Only order-time features are retained.
    """
    if "late_delivery_risk_clean" not in df.columns:
        raise ValueError("Target 'late_delivery_risk_clean' is missing from dataframe.")

    y = df["late_delivery_risk_clean"].astype(int)

    numerical_candidates = [
        "days_scheduled_clean",
        "order_item_quantity_clean",
        "sales_clean",
        "order_month",
        "order_day_of_week",
    ]
    categorical_candidates = [
        "shipping_mode_clean",
        "region_clean",
        "category_clean",
        "customer_segment",
        "market",
    ]

    leakage_blacklist = {
        "days_for_shipping_real",
        "days_real",
        "days_real_clean",
        "shipping_date_dateorders",
        "shipping_date_parsed",
        "delivery_status",
        "delivery_status_clean",
        "order_status",
    }

    num_features = [c for c in numerical_candidates if c in df.columns and c not in leakage_blacklist]
    cat_features = [c for c in categorical_candidates if c in df.columns and c not in leakage_blacklist]

    all_features = num_features + cat_features
    X = df[all_features].copy()

    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X, y, num_features, cat_features
