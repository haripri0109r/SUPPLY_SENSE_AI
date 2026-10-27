"""Demand Forecasting Module for SupplySense AI.
Provides chronological time-series aggregation, feature engineering with strictly past observations,
model training (Moving Average Baseline, Linear Regression, Random Forest),
rigorous held-out evaluation (MAE, RMSE, MAPE), and recursive future multi-step forecasting.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def aggregate_demand_by_date(
    df: pd.DataFrame,
    category: Optional[str] = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Aggregate demand (quantity ordered) chronologically by date and optionally category.

    Args:
        df: Cleaned supply chain dataframe
        category: Specific category name or None for all
        freq: Frequency for resampling ('D' for daily, 'W' for weekly)

    Returns:
        pd.DataFrame with 'date' index and 'demand' column.
    """
    if "order_date_parsed" not in df.columns or df["order_date_parsed"].isnull().all():
        raise ValueError("Valid 'order_date_parsed' column is required for demand forecasting.")

    data = df.dropna(subset=["order_date_parsed"]).copy()
    if category and category != "All Categories" and "category_clean" in data.columns:
        data = data[data["category_clean"] == category]

    if data.empty:
        return pd.DataFrame(columns=["demand"])

    # Aggregate by date
    data["date"] = data["order_date_parsed"].dt.floor("D")
    daily = data.groupby("date")["order_item_quantity_clean"].sum().reset_index()
    daily = daily.rename(columns={"order_item_quantity_clean": "demand"})
    daily = daily.sort_values("date").reset_index(drop=True)

    # Set datetime index and reindex/resample to guarantee continuous timeline
    daily = daily.set_index("date")
    daily = daily.resample(freq).sum().fillna(0.0)
    daily = daily.reset_index()
    return daily


def build_time_series_features(
    ts_df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 7, 14],
    rolling_windows: List[int] = [7, 14],
) -> pd.DataFrame:
    """Construct lag and rolling-average features using past observations only.

    Guarantees no future-data leakage by shifting before calculating rolling stats.
    """
    df = ts_df.copy()
    if "demand" not in df.columns:
        raise ValueError("Dataframe must contain 'demand' column.")

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    # Lag features (strictly past values)
    for lag in lags:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # Rolling window features on shifted series (strictly past values)
    shifted = df["demand"].shift(1)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w, min_periods=1).std().fillna(0)

    return df


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Calculate Mean Absolute Percentage Error (MAPE) strictly on non-zero actual demand."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero_mask = y_true > 0
    if not np.any(non_zero_mask):
        return None
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    return float(round(mape, 2))


def train_and_evaluate_forecasting_models(
    ts_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_observations: int = 20,
) -> Dict[str, Any]:
    """Train Moving Average, Linear Regression, and Random Forest models on chronological split.

    Returns:
        Evaluation results, test predictions, trained models, feature names, and metadata.
    """
    if len(ts_df) < min_observations:
        return {
            "success": False,
            "message": f"Insufficient historical data ({len(ts_df)} periods). Minimum {min_observations} required.",
            "metrics": {},
            "best_model_name": None,
        }

    # Generate features
    featured_df = build_time_series_features(ts_df)
    # Drop rows with NaNs resulting from lags
    max_lag = max([1, 2, 3, 7, 14])
    clean_featured = featured_df.iloc[max_lag:].copy().reset_index(drop=True)

    if len(clean_featured) < 10:
        return {
            "success": False,
            "message": "Insufficient data after generating lag features.",
            "metrics": {},
            "best_model_name": None,
        }

    feature_cols = [
        c
        for c in clean_featured.columns
        if c.startswith("lag_")
        or c.startswith("rolling_")
        or c in ["day_of_week", "month", "day_of_month"]
    ]

    # Chronological Split (NEVER randomly shuffled)
    n_total = len(clean_featured)
    n_train = int(n_total * (1.0 - test_ratio))
    n_train = max(n_train, 5)

    train_df = clean_featured.iloc[:n_train].copy()
    test_df = clean_featured.iloc[n_train:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["demand"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["demand"].values

    test_dates = test_df["date"].values

    models: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}
    metrics_summary: List[Dict[str, Any]] = []

    # 1. Moving Average Baseline (7-period moving average of past demand)
    ma_pred = test_df["rolling_mean_7"].values
    predictions["Moving Average (7-day)"] = ma_pred
    ma_mae = mean_absolute_error(y_test, ma_pred)
    ma_rmse = float(np.sqrt(mean_squared_error(y_test, ma_pred)))
    ma_mape = calculate_mape(y_test, ma_pred)
    metrics_summary.append({
        "Model": "Moving Average (7-day)",
        "MAE": round(float(ma_mae), 2),
        "RMSE": round(float(ma_rmse), 2),
        "MAPE (%)": ma_mape if ma_mape is not None else "N/A",
    })

    # 2. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = np.clip(lr.predict(X_test), 0, None)
    models["Linear Regression"] = lr
    predictions["Linear Regression"] = lr_pred
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
    lr_mape = calculate_mape(y_test, lr_pred)
    metrics_summary.append({
        "Model": "Linear Regression",
        "MAE": round(float(lr_mae), 2),
        "RMSE": round(float(lr_rmse), 2),
        "MAPE (%)": lr_mape if lr_mape is not None else "N/A",
    })

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = np.clip(rf.predict(X_test), 0, None)
    models["Random Forest"] = rf
    predictions["Random Forest"] = rf_pred
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
    rf_mape = calculate_mape(y_test, rf_pred)
    metrics_summary.append({
        "Model": "Random Forest",
        "MAE": round(float(rf_mae), 2),
        "RMSE": round(float(rf_rmse), 2),
        "MAPE (%)": rf_mape if rf_mape is not None else "N/A",
    })

    # Select best model based on held-out RMSE
    metrics_df = pd.DataFrame(metrics_summary)
    best_row = metrics_df.sort_values("RMSE").iloc[0]
    best_model_name = best_row["Model"]

    return {
        "success": True,
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "models": models,
        "predictions": predictions,
        "y_train": y_train,
        "y_test": y_test,
        "train_dates": train_df["date"].values,
        "test_dates": test_dates,
        "feature_cols": feature_cols,
        "full_featured_df": clean_featured,
    }


def recursive_future_forecast(
    ts_df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    horizon: int = 14,
    freq: str = "D",
) -> pd.DataFrame:
    """Generate recursive multi-step future demand forecast.

    Each future step updates lags and rolling averages using previous predictions.
    """
    history_dates = list(ts_df["date"].values)
    history_demand = list(ts_df["demand"].values)

    future_dates = []
    future_preds = []

    last_date = pd.to_datetime(history_dates[-1])

    for i in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=i if freq == "D" else i * 7)
        future_dates.append(next_date)

        # Build feature dictionary for next step
        feat_dict = {
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "day_of_month": next_date.day,
        }

        # Lags
        combined_demand = history_demand + future_preds
        n = len(combined_demand)
        for lag in [1, 2, 3, 7, 14]:
            feat_dict[f"lag_{lag}"] = combined_demand[n - lag] if n >= lag else combined_demand[-1]

        # Rolling means on past demand
        for w in [7, 14]:
            slice_w = combined_demand[max(0, n - w):]
            feat_dict[f"rolling_mean_{w}"] = float(np.mean(slice_w))
            feat_dict[f"rolling_std_{w}"] = float(np.std(slice_w)) if len(slice_w) > 1 else 0.0

        feat_vector = np.array([[feat_dict[c] for c in feature_cols]])
        pred = float(np.clip(model.predict(feat_vector)[0], 0, None))
        future_preds.append(round(pred, 1))

    return pd.DataFrame({
        "date": future_dates,
        "predicted_demand": future_preds,
    })
