"""Unit tests for SupplySense AI Data Processing Pipeline.
Covers:
- Column normalization
- Canonical column resolution
- Missing value handling
- Date parsing
- Duplicate handling
- Data loading and quality inspection
"""

from __future__ import annotations
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data_loader import inspect_dataset_quality, load_raw_data, load_dataset
from src.data_cleaning import (
    normalize_column_name,
    resolve_canonical_columns,
    clean_supply_chain_data,
    apply_dashboard_filters,
)


@pytest.fixture
def synthetic_raw_dataframe() -> pd.DataFrame:
    """Create a synthetic supply chain dataframe mimicking real DataCo schema."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
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

    # Add 2 duplicate rows to verify duplicate detection without blind deletion
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
        assert canonical["sales"] == "Sales"

    def test_clean_supply_chain_data(self, synthetic_raw_dataframe):
        cleaned, report = clean_supply_chain_data(synthetic_raw_dataframe)
        # Verify row preservation
        assert len(cleaned) == len(synthetic_raw_dataframe)
        assert "order_item_quantity_clean" in cleaned.columns
        assert "sales_clean" in cleaned.columns
        assert "order_date_parsed" in cleaned.columns
        assert "shipping_date_parsed" in cleaned.columns
        assert "days_real_clean" in cleaned.columns
        assert "days_scheduled_clean" in cleaned.columns
        assert "late_delivery_risk_clean" in cleaned.columns

        # Verify parsing was successful
        assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date_parsed"])
        assert cleaned["order_item_quantity_clean"].min() >= 1

    def test_inspect_dataset_quality(self, synthetic_raw_dataframe):
        quality = inspect_dataset_quality(synthetic_raw_dataframe)
        assert quality["total_rows"] == 32
        assert quality["total_columns"] == 11
        assert quality["duplicate_rows"] == 2
        assert "missing_by_column" in quality

    def test_load_dataset_csv(self, synthetic_raw_dataframe):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            temp_path = f.name
            synthetic_raw_dataframe.to_csv(temp_path, index=False)

        try:
            loaded_df = load_raw_data(temp_path)
            assert len(loaded_df) == len(synthetic_raw_dataframe)
            assert "Sales" in loaded_df.columns

            raw_df, clean_df, report = load_dataset(temp_path)
            assert len(raw_df) == len(clean_df)
            assert "sales_clean" in clean_df.columns
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_apply_dashboard_filters(self, synthetic_raw_dataframe):
        cleaned, _ = clean_supply_chain_data(synthetic_raw_dataframe)
        filtered = apply_dashboard_filters(
            cleaned,
            selected_categories=["Electronics"],
            selected_regions=["North America"],
        )
        assert all(filtered["category_clean"] == "Electronics")
        assert all(filtered["region_clean"] == "North America")
