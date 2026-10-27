"""Unit and Integration Tests for SupplySense AI Pipeline.
Verifies data loading, column normalization, data cleaning, date parsing,
demand aggregation, leakage-free time-series features, chronological splitting,
delay classification, metric calculations, and inventory risk calculations.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    apply_dashboard_filters,
    clean_supply_chain_data,
    inspect_dataset_quality,
    load_dataset,
    normalize_column_name,
    resolve_canonical_columns,
)
from src.delay_prediction import (
    compute_historical_delay_breakdowns,
    predict_single_order_delay,
    prepare_delay_features,
    train_delay_classification_models,
)
from src.forecasting import (
    aggregate_demand_by_date,
    build_time_series_features,
    calculate_mape,
    recursive_future_forecast,
    train_and_evaluate_forecasting_models,
)
from src.recommendations import (
    generate_supply_chain_recommendations,
    recommendations_to_dataframe,
)
from src.risk_analysis import (
    calculate_category_inventory_risk,
    calculate_regional_performance_benchmarks,
)


@pytest.fixture
def synthetic_raw_dataframe():
    """Create a synthetic supply chain dataframe mimicking real DataCo schema."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    data = []
    np.random.seed(42)
    categories = ["Sporting Goods", "Electronics", "Apparel"]
    regions = ["Southeast Asia", "North America", "Western Europe"]
    modes = ["Standard Class", "First Class", "Second Class"]

    for i, dt in enumerate(dates):
        cat = categories[i % len(categories)]
        reg = regions[i % len(regions)]
        mode = modes[i % len(modes)]
        late = 1 if np.random.rand() > 0.45 else 0
        qty = int(np.random.randint(1, 6))
        sales = float(qty * np.random.uniform(20.0, 150.0))
        sched_days = int(np.random.choice([2, 3, 4]))
        real_days = sched_days + (np.random.randint(1, 3) if late else 0)

        data.append({
            "order date (DateOrders)": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "shipping date (DateOrders)": (dt + pd.Timedelta(days=real_days)).strftime("%Y-%m-%d %H:%M:%S"),
            "Order Item Quantity": qty,
            "Sales": sales,
            "Category Name": cat,
            "Order Region": reg,
            "Shipping Mode": mode,
            "Days for shipment (scheduled)": sched_days,
            "Days for shipping (real)": real_days,
            "Late_delivery_risk": late,
            "Delivery Status": "Late delivery" if late else "Shipping on time",
        })

    # Add 2 duplicate rows and 1 invalid row
    df = pd.DataFrame(data)
    df = pd.concat([df, df.iloc[[0]], df.iloc[[1]]], ignore_index=True)
    return df


class TestDataProcessing:
    def test_column_normalization(self):
        assert normalize_column_name("Order Item Quantity") == "order_item_quantity"
        assert normalize_column_name("Days for shipping (real)") == "days_for_shipping_real"
        assert normalize_column_name("order date (DateOrders)") == "order_date_dateorders"
        assert normalize_column_name("  Category   Name  ") == "category_name"

    def test_canonical_column_resolution(self, synthetic_raw_dataframe):
        canonical = resolve_canonical_columns(synthetic_raw_dataframe.columns)
        assert canonical["quantity"] == "Order Item Quantity"
        assert canonical["late_delivery_risk"] == "Late_delivery_risk"
        assert canonical["order_date"] == "order date (DateOrders)"

    def test_clean_supply_chain_data(self, synthetic_raw_dataframe):
        cleaned, report = clean_supply_chain_data(synthetic_raw_dataframe)
        assert len(cleaned) == len(synthetic_raw_dataframe)
        assert "order_item_quantity_clean" in cleaned.columns
        assert "sales_clean" in cleaned.columns
        assert "late_delivery_risk_clean" in cleaned.columns
        assert "category_clean" in cleaned.columns
        assert cleaned["order_date_parsed"].notnull().all()

    def test_missing_and_duplicate_inspection(self, synthetic_raw_dataframe):
        quality = inspect_dataset_quality(synthetic_raw_dataframe)
        assert quality["duplicate_rows"] == 2
        assert quality["num_rows"] == len(synthetic_raw_dataframe)

    def test_dashboard_filters(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        filtered = apply_dashboard_filters(
            cleaned,
            regions=["Southeast Asia"],
            categories=["Sporting Goods"],
        )
        assert (filtered["region_clean"] == "Southeast Asia").all()
        assert (filtered["category_clean"] == "Sporting Goods").all()


class TestDemandForecasting:
    def test_demand_aggregation(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        daily_ts = aggregate_demand_by_date(cleaned, category="All Categories", freq="D")
        assert not daily_ts.empty
        assert "date" in daily_ts.columns
        assert "demand" in daily_ts.columns
        assert daily_ts["demand"].sum() > 0

    def test_leakage_free_feature_generation(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        daily_ts = aggregate_demand_by_date(cleaned, freq="D")
        featured = build_time_series_features(daily_ts, lags=[1, 2, 7])

        # Lag 1 at index 1 must equal demand at index 0
        assert featured["lag_1"].iloc[1] == daily_ts["demand"].iloc[0]
        # First row lag must be NaN (no past data)
        assert np.isnan(featured["lag_1"].iloc[0])

    def test_chronological_forecasting_models(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        daily_ts = aggregate_demand_by_date(cleaned, freq="D")
        results = train_and_evaluate_forecasting_models(daily_ts, test_ratio=0.25, min_observations=15)

        assert results["success"] is True
        assert results["best_model_name"] in ["Moving Average (7-day)", "Linear Regression", "Random Forest"]
        assert "metrics_df" in results
        assert len(results["metrics_df"]) == 3
        # Check RMSE is a non-negative float
        for rmse in results["metrics_df"]["RMSE"]:
            assert float(rmse) >= 0.0

    def test_recursive_future_forecast(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        daily_ts = aggregate_demand_by_date(cleaned, freq="D")
        train_res = train_and_evaluate_forecasting_models(daily_ts, test_ratio=0.2, min_observations=15)
        model = train_res["models"]["Linear Regression"]
        future = recursive_future_forecast(daily_ts, model, train_res["feature_cols"], horizon=7)

        assert len(future) == 7
        assert "predicted_demand" in future.columns
        assert (future["predicted_demand"] >= 0).all()

    def test_mape_calculation_safe_zero(self):
        y_true = np.array([0, 10, 20])
        y_pred = np.array([5, 12, 18])
        mape = calculate_mape(y_true, y_pred)
        # Should only evaluate on [10, 20] -> |10-12|/10 = 0.2, |20-18|/20 = 0.1 -> avg 0.15 = 15.0%
        assert mape == 15.0


class TestDelayPrediction:
    def test_leakage_columns_excluded(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        X, y, num_cols, cat_cols = prepare_delay_features(cleaned)

        for leaked in ["days_for_shipping_real", "days_real_clean", "delivery_status", "shipping_date_parsed"]:
            assert leaked not in X.columns
            assert leaked not in num_cols
            assert leaked not in cat_cols

    def test_model_training_and_metrics(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        results = train_delay_classification_models(cleaned, test_size=0.25)

        assert results["success"] is True
        rf_metrics = results["metrics"]["Random Forest"]
        assert 0.0 <= rf_metrics["Accuracy"] <= 1.0
        assert 0.0 <= rf_metrics["Precision"] <= 1.0
        assert 0.0 <= rf_metrics["Recall"] <= 1.0
        assert 0.0 <= rf_metrics["F1-Score"] <= 1.0
        assert 0.0 <= rf_metrics["ROC-AUC"] <= 1.0

    def test_single_order_prediction(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        results = train_delay_classification_models(cleaned, test_size=0.25)
        pipeline = results["best_pipeline"]

        order_dict = {
            "days_scheduled_clean": 4,
            "order_item_quantity_clean": 2,
            "sales_clean": 150.0,
            "shipping_mode_clean": "Standard Class",
            "region_clean": "Southeast Asia",
            "category_clean": "Sporting Goods",
        }
        pred_res = predict_single_order_delay(pipeline, order_dict)
        assert "is_delayed_predicted" in pred_res
        assert 0.0 <= pred_res["delay_probability"] <= 1.0
        assert pred_res["risk_label"] in ["Low Delay Risk", "Moderate Risk", "High Delay Risk"]


class TestRiskAndRecommendations:
    def test_category_inventory_risk_rule_based(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        risk_df = calculate_category_inventory_risk(cleaned)

        assert not risk_df.empty
        assert "inventory_risk_level" in risk_df.columns
        assert set(risk_df["inventory_risk_level"].unique()).issubset({"Low", "Medium", "High"})
        assert "demand_cv" in risk_df.columns
        assert "recommended_action" in risk_df.columns

    def test_regional_performance(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        benchmarks = calculate_regional_performance_benchmarks(cleaned)

        assert "regional_summary" in benchmarks
        assert "shipping_mode_summary" in benchmarks
        assert "efficiency_rank" in benchmarks["regional_summary"].columns

    def test_recommendations_generation(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        risk_df = calculate_category_inventory_risk(cleaned)
        delay_res = train_delay_classification_models(cleaned, test_size=0.25)
        daily_ts = aggregate_demand_by_date(cleaned)
        fc_res = train_and_evaluate_forecasting_models(daily_ts, test_ratio=0.25, min_observations=15)

        recs = generate_supply_chain_recommendations(
            cleaned,
            inventory_risk_df=risk_df,
            delay_model_results=delay_res,
            forecast_results=fc_res,
        )
        assert len(recs) >= 3
        df_recs = recommendations_to_dataframe(recs)
        assert not df_recs.empty
        assert "Priority" in df_recs.columns
        assert "Analytical Type" in df_recs.columns


class TestMissingColumnsHandling:
    def test_handles_missing_optional_columns(self):
        # Dataframe with bare minimal columns
        minimal_df = pd.DataFrame({
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "quantity": [1, 2, 3, 4],
            "sales": [10.0, 20.0, 30.0, 40.0],
        })
        cleaned, report = clean_supply_chain_data(minimal_df)
        assert cleaned["category_clean"].iloc[0] == "General Category"
        assert cleaned["region_clean"].iloc[0] == "Global"
        assert cleaned["shipping_mode_clean"].iloc[0] == "Standard Class"
        assert cleaned["late_delivery_risk_clean"].iloc[0] == 0
