"""Unit tests for SupplySense AI Risk Analysis, Delay Prediction, and Recommendation Engine.
Covers:
- Delay classification feature extraction & target leakage prevention
- Pre-dispatch classification model training and evaluation
- Single order real-time prediction
- Category inventory risk scoring (Demand CV, Lead Time Gap, Low/Medium/High Tiers)
- Regional & shipping mode performance benchmarking
- Recommendation generation and priority scoring
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import prepare_delay_classification_features
from src.delay_prediction import (
    train_delay_classification_models,
    predict_single_order_delay,
    LEAKAGE_COLUMNS,
)
from src.inventory_risk import (
    calculate_category_inventory_risk,
    calculate_regional_performance_benchmarks,
)
from src.recommendations import (
    generate_supply_chain_recommendations,
    recommendations_to_dataframe,
)


@pytest.fixture
def synthetic_risk_df() -> pd.DataFrame:
    """Create a diverse synthetic dataset for classification and risk scoring."""
    np.random.seed(42)
    n = 100
    categories = ["Cleats", "Water Sports", "Cardio Equipment", "Camping & Hiking"]
    regions = ["Western Europe", "Central America", "South Asia", "Pacific Asia"]
    modes = ["Standard Class", "First Class", "Second Class", "Same Day"]

    data = []
    for i in range(n):
        cat = categories[i % len(categories)]
        reg = regions[i % len(regions)]
        mode = modes[i % len(modes)]
        sched_days = int(np.random.choice([2, 4, 6]))
        late = 1 if (mode == "First Class" or np.random.rand() > 0.4) else 0
        real_days = sched_days + (2 if late else -1)
        qty = int(np.random.randint(1, 8))
        sales = float(qty * np.random.uniform(30.0, 200.0))

        data.append({
            "order_date_parsed": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
            "category_clean": cat,
            "region_clean": reg,
            "shipping_mode_clean": mode,
            "days_scheduled_clean": sched_days,
            "days_real_clean": real_days,
            "late_delivery_risk_clean": late,
            "order_item_quantity_clean": qty,
            "sales_clean": sales,
            "order_month": 1,
            "order_day_of_week": i % 7,
            # Leakage columns to test filtration
            "days_for_shipping_real": real_days,
            "delivery_status": "Late delivery" if late else "Shipping on time",
        })
    return pd.DataFrame(data)


class TestRecommendationsAndRisk:
    def test_prepare_delay_classification_features_leakage_free(self, synthetic_risk_df):
        X, y, num_cols, cat_cols = prepare_delay_classification_features(synthetic_risk_df)
        assert len(X) == len(synthetic_risk_df)
        assert len(y) == len(synthetic_risk_df)

        # Strictly check that no leakage columns exist in feature set
        for col in X.columns:
            assert col not in LEAKAGE_COLUMNS
            assert "real" not in col.lower()
            assert "delivery_status" not in col.lower()

    def test_train_delay_classification_models(self, synthetic_risk_df):
        results = train_delay_classification_models(synthetic_risk_df, test_size=0.25)
        assert results["success"] is True
        assert "metrics" in results
        assert "Random Forest" in results["metrics"]
        assert "Baseline (Most Frequent)" in results["metrics"]

        rf_metrics = results["metrics"]["Random Forest"]
        assert 0.0 <= rf_metrics["Accuracy"] <= 1.0
        assert 0.0 <= rf_metrics["Precision"] <= 1.0
        assert 0.0 <= rf_metrics["Recall"] <= 1.0
        assert 0.0 <= rf_metrics["F1-Score"] <= 1.0
        assert "confusion_matrix" in results
        assert "ROC-AUC" in rf_metrics

    def test_predict_single_order_delay(self, synthetic_risk_df):
        results = train_delay_classification_models(synthetic_risk_df, test_size=0.25)
        rf_pipeline = results["pipelines"]["Random Forest"]

        pred = predict_single_order_delay(
            rf_pipeline,
            shipping_mode="First Class",
            region="Western Europe",
            category="Cleats",
            days_scheduled=2,
            order_item_quantity=2,
            sales=120.0,
        )
        assert "predicted_class" in pred
        assert pred["predicted_class"] in [0, 1]
        assert "delay_probability" in pred
        assert 0.0 <= pred["delay_probability"] <= 1.0
        assert pred["risk_label"] in ["High Delay Risk", "Low Delay Risk"]

    def test_calculate_category_inventory_risk(self, synthetic_risk_df):
        risk_df = calculate_category_inventory_risk(synthetic_risk_df)
        assert not risk_df.empty
        assert "inventory_risk_level" in risk_df.columns
        assert "demand_cv" in risk_df.columns
        assert "lead_time_gap" in risk_df.columns
        assert "recommended_action" in risk_df.columns
        assert set(risk_df["inventory_risk_level"].unique()).issubset({"Low", "Medium", "High"})

    def test_calculate_regional_performance_benchmarks(self, synthetic_risk_df):
        benchmarks = calculate_regional_performance_benchmarks(synthetic_risk_df)
        assert "regional_summary" in benchmarks
        assert "shipping_mode_summary" in benchmarks

        reg_df = benchmarks["regional_summary"]
        assert "late_rate_pct" in reg_df.columns
        assert "efficiency_rank" in reg_df.columns

        mode_df = benchmarks["shipping_mode_summary"]
        assert "fulfillment_gap" in mode_df.columns

    def test_generate_supply_chain_recommendations(self, synthetic_risk_df):
        risk_df = calculate_category_inventory_risk(synthetic_risk_df)
        results = train_delay_classification_models(synthetic_risk_df, test_size=0.25)

        recs = generate_supply_chain_recommendations(
            synthetic_risk_df,
            inventory_risk_df=risk_df,
            delay_model_results=results,
        )
        assert len(recs) >= 2
        for rec in recs:
            assert "priority" in rec
            assert rec["priority"] in ["High", "Medium", "Low"]
            assert "domain" in rec
            assert "finding" in rec
            assert "action" in rec
            assert "evidence" in rec

        recs_df = recommendations_to_dataframe(recs)
        assert isinstance(recs_df, pd.DataFrame)
        assert len(recs_df) == len(recs)
