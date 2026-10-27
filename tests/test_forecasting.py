"""Unit tests for SupplySense AI Demand Forecasting Pipeline.
Covers:
- Demand aggregation by date and continuous timeline resampling
- Temporal and calendar feature engineering
- Backward-looking lag and rolling statistics (leakage prevention)
- Chronological train/test splitting
- Model training (Moving Average Baseline, Linear Regression, Random Forest)
- Error metric calculations (MAE, RMSE, MAPE on non-zero demand)
- Recursive future multi-step forecasting
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    extract_date_features,
    aggregate_demand_by_date,
    build_time_series_features,
)
from src.forecasting import (
    calculate_mape,
    train_and_evaluate_forecasting_models,
    recursive_future_forecast,
)


@pytest.fixture
def synthetic_time_series_data() -> pd.DataFrame:
    """Generate 60 days of continuous daily demand data."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    np.random.seed(42)
    demand = 20 + 5 * np.sin(np.arange(60) * (2 * np.pi / 7)) + np.random.normal(0, 2, 60)
    demand = np.clip(demand, 1, None).round().astype(int)

    return pd.DataFrame({
        "date": dates,
        "demand": demand,
    })


@pytest.fixture
def synthetic_transactions_df() -> pd.DataFrame:
    """Generate transactional records with order dates and quantities."""
    dates = pd.date_range("2023-01-01", periods=45, freq="D")
    records = []
    np.random.seed(42)
    categories = ["Apparel", "Footwear"]

    for dt in dates:
        for cat in categories:
            qty = int(np.random.randint(1, 10))
            records.append({
                "order_date_parsed": dt,
                "category_clean": cat,
                "order_item_quantity_clean": qty,
                "sales_clean": qty * 25.0,
            })
    return pd.DataFrame(records)


class TestForecastingPipeline:
    def test_extract_date_features(self, synthetic_transactions_df):
        df_feat = extract_date_features(synthetic_transactions_df, "order_date_parsed")
        assert "order_year" in df_feat.columns
        assert "order_month" in df_feat.columns
        assert "order_day" in df_feat.columns
        assert "order_day_of_week" in df_feat.columns
        assert "is_weekend" in df_feat.columns
        assert "order_quarter" in df_feat.columns
        assert set(df_feat["is_weekend"].unique()).issubset({0, 1})

    def test_aggregate_demand_by_date(self, synthetic_transactions_df):
        daily = aggregate_demand_by_date(synthetic_transactions_df, freq="D")
        assert len(daily) == 45
        assert "date" in daily.columns
        assert "demand" in daily.columns
        assert daily["demand"].sum() == synthetic_transactions_df["order_item_quantity_clean"].sum()

        # Category-filtered aggregation
        apparel_daily = aggregate_demand_by_date(synthetic_transactions_df, category="Apparel", freq="D")
        assert len(apparel_daily) == 45
        assert apparel_daily["demand"].sum() == synthetic_transactions_df[synthetic_transactions_df["category_clean"] == "Apparel"]["order_item_quantity_clean"].sum()

    def test_build_time_series_features_no_leakage(self, synthetic_time_series_data):
        feat_df = build_time_series_features(synthetic_time_series_data)
        assert "lag_1" in feat_df.columns
        assert "lag_7" in feat_df.columns
        assert "rolling_mean_7" in feat_df.columns
        assert "rolling_std_7" in feat_df.columns

        # Verify that lag_1 strictly equals demand at t-1
        assert feat_df.loc[1, "lag_1"] == synthetic_time_series_data.loc[0, "demand"]
        assert feat_df.loc[7, "lag_7"] == synthetic_time_series_data.loc[0, "demand"]

        # Verify rolling_mean_7 does NOT include demand at time t
        # Row 7 rolling_mean_7 must be the mean of demand from row 0 to 6
        expected_mean = synthetic_time_series_data.loc[0:6, "demand"].mean()
        np.testing.assert_almost_equal(feat_df.loc[7, "rolling_mean_7"], expected_mean)

    def test_calculate_mape(self):
        y_true = np.array([10.0, 20.0, 0.0, 50.0])
        y_pred = np.array([11.0, 19.0, 5.0, 45.0])
        mape = calculate_mape(y_true, y_pred)
        assert mape is not None
        assert 0.0 < mape < 50.0

        # All zeros case
        assert calculate_mape(np.array([0.0, 0.0]), np.array([5.0, 10.0])) is None

    def test_train_and_evaluate_forecasting_models(self, synthetic_time_series_data):
        results = train_and_evaluate_forecasting_models(synthetic_time_series_data, test_ratio=0.2)
        assert results["success"] is True
        assert "metrics_df" in results
        assert "best_model_name" in results
        assert len(results["models"]) >= 2
        assert len(results["predictions"]) >= 3

        metrics_df = results["metrics_df"]
        assert "Moving Average (7-day)" in metrics_df["Model"].values
        assert "Linear Regression" in metrics_df["Model"].values
        assert "Random Forest" in metrics_df["Model"].values
        for col in ["MAE", "RMSE"]:
            assert col in metrics_df.columns
            assert (metrics_df[col] >= 0).all()

    def test_recursive_future_forecast(self, synthetic_time_series_data):
        results = train_and_evaluate_forecasting_models(synthetic_time_series_data, test_ratio=0.2)
        best_model = results["models"].get("Random Forest") or results["models"]["Linear Regression"]
        future_df = recursive_future_forecast(
            synthetic_time_series_data,
            best_model,
            results["feature_cols"],
            horizon=7,
            freq="D",
        )
        assert len(future_df) == 7
        assert "date" in future_df.columns
        assert "predicted_demand" in future_df.columns
        assert (future_df["predicted_demand"] >= 0).all()
