"""Data Cleaning Module for SupplySense AI.
Handles schema normalization, canonical column resolution, type coercion,
date parsing, missing value imputation, and dashboard filtering.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def normalize_column_name(col: str) -> str:
    """Normalize a column name into a safe, snake_case identifier."""
    if not isinstance(col, str):
        col = str(col)
    clean = col.strip().lower()
    # Replace non-alphanumeric characters with underscores
    clean = re.sub(r"[^\w\s]", "_", clean)
    clean = re.sub(r"[\s_]+", "_", clean)
    return clean.strip("_")


CANONICAL_COLUMN_CANDIDATES = {
    "order_date": [
        "order_date_dateorders",
        "order_date",
        "orderdate",
        "date_orders",
        "order_date__dateorders_",
    ],
    "shipping_date": [
        "shipping_date_dateorders",
        "shipping_date",
        "shippingdate",
        "shipping_date__dateorders_",
    ],
    "quantity": [
        "order_item_quantity",
        "quantity",
        "order_quantity",
        "item_quantity",
    ],
    "sales": [
        "sales",
        "order_item_total",
        "sales_per_customer",
        "order_total",
        "order_item_product_price",
    ],
    "category": [
        "category_name",
        "category",
        "product_category",
        "category_id",
    ],
    "product_name": [
        "product_name",
        "product",
        "product_card_id",
        "item_name",
    ],
    "region": [
        "order_region",
        "region",
        "market",
        "order_country",
        "order_city",
    ],
    "shipping_mode": [
        "shipping_mode",
        "delivery_mode",
        "ship_mode",
    ],
    "days_scheduled": [
        "days_for_shipment_scheduled",
        "scheduled_shipping_days",
        "days_for_shipment__scheduled_",
        "scheduled_days",
    ],
    "days_real": [
        "days_for_shipping_real",
        "actual_shipping_days",
        "days_for_shipping__real_",
        "real_days",
    ],
    "late_delivery_risk": [
        "late_delivery_risk",
        "late_risk",
        "is_late",
        "delay_risk",
    ],
    "delivery_status": [
        "delivery_status",
        "shipping_status",
        "delivery_state",
    ],
    "order_status": [
        "order_status",
        "status",
        "state",
    ],
    "customer_segment": [
        "customer_segment",
        "segment",
    ],
    "department_name": [
        "department_name",
        "department",
    ],
}


def resolve_canonical_columns(df_columns: List[str]) -> Dict[str, Optional[str]]:
    """Map canonical semantic keys to the actual column name present in the dataframe."""
    normalized_cols = {col: normalize_column_name(col) for col in df_columns}
    norm_to_actual = {norm: orig for orig, norm in normalized_cols.items()}

    resolved = {}
    for canonical, candidates in CANONICAL_COLUMN_CANDIDATES.items():
        matched = None
        for candidate in candidates:
            cand_norm = normalize_column_name(candidate)
            if cand_norm in norm_to_actual:
                matched = norm_to_actual[cand_norm]
                break
        resolved[canonical] = matched
    return resolved


def clean_supply_chain_data(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and normalize the supply chain dataset safely without massive row dropping.

    Returns:
        (cleaned_df, cleaning_report)
    """
    df = raw_df.copy()
    initial_rows = len(df)
    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "dropped_rows": 0,
        "drop_reasons": [],
        "imputed_fields": [],
    }

    # 1. Normalize all column names
    col_mapping = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=col_mapping)

    # 2. Identify canonical columns on normalized dataframe
    canonical = resolve_canonical_columns(list(df.columns))
    report["canonical_columns"] = canonical

    # 3. Clean and parse date columns
    order_date_col = canonical.get("order_date")
    if order_date_col and order_date_col in df.columns:
        df["order_date_parsed"] = pd.to_datetime(df[order_date_col], errors="coerce")
        invalid_dates = int(df["order_date_parsed"].isnull().sum())
        if invalid_dates > 0:
            report["drop_reasons"].append(
                f"Dropped {invalid_dates} rows where order date was unparseable."
            )
            df = df.dropna(subset=["order_date_parsed"]).copy()
            report["dropped_rows"] += invalid_dates
        df["order_year"] = df["order_date_parsed"].dt.year
        df["order_month"] = df["order_date_parsed"].dt.month
        df["order_date_only"] = df["order_date_parsed"].dt.date
    else:
        df["order_date_parsed"] = pd.NaT

    shipping_date_col = canonical.get("shipping_date")
    if shipping_date_col and shipping_date_col in df.columns:
        df["shipping_date_parsed"] = pd.to_datetime(df[shipping_date_col], errors="coerce")
    else:
        df["shipping_date_parsed"] = pd.NaT

    # 4. Standardize numeric columns
    qty_col = canonical.get("quantity")
    if qty_col and qty_col in df.columns:
        df["order_item_quantity_clean"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1)
        df["order_item_quantity_clean"] = df["order_item_quantity_clean"].clip(lower=0)
    else:
        df["order_item_quantity_clean"] = 1.0
        report["imputed_fields"].append("order_item_quantity (defaulted to 1)")

    sales_col = canonical.get("sales")
    if sales_col and sales_col in df.columns:
        df["sales_clean"] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0.0)
        df["sales_clean"] = df["sales_clean"].clip(lower=0)
    else:
        df["sales_clean"] = 0.0
        report["imputed_fields"].append("sales (defaulted to 0.0)")

    # 5. Standardize categorical columns
    cat_col = canonical.get("category")
    if cat_col and cat_col in df.columns:
        df["category_clean"] = df[cat_col].fillna("Unknown Category").astype(str).str.strip()
    else:
        df["category_clean"] = "General Category"
        report["imputed_fields"].append("category (defaulted to 'General Category')")

    region_col = canonical.get("region")
    if region_col and region_col in df.columns:
        df["region_clean"] = df[region_col].fillna("Unknown Region").astype(str).str.strip()
    else:
        df["region_clean"] = "Global"
        report["imputed_fields"].append("region (defaulted to 'Global')")

    mode_col = canonical.get("shipping_mode")
    if mode_col and mode_col in df.columns:
        df["shipping_mode_clean"] = df[mode_col].fillna("Standard Class").astype(str).str.strip()
    else:
        df["shipping_mode_clean"] = "Standard Class"
        report["imputed_fields"].append("shipping_mode (defaulted to 'Standard Class')")

    # 6. Standardize target and shipping duration features
    late_col = canonical.get("late_delivery_risk")
    if late_col and late_col in df.columns:
        df["late_delivery_risk_clean"] = (
            pd.to_numeric(df[late_col], errors="coerce").fillna(0).astype(int)
        )
        df["late_delivery_risk_clean"] = df["late_delivery_risk_clean"].apply(
            lambda x: 1 if x == 1 else 0
        )
    else:
        df["late_delivery_risk_clean"] = 0
        report["imputed_fields"].append("late_delivery_risk (unavailable)")

    sched_col = canonical.get("days_scheduled")
    if sched_col and sched_col in df.columns:
        df["days_scheduled_clean"] = pd.to_numeric(df[sched_col], errors="coerce").fillna(4.0)
    else:
        df["days_scheduled_clean"] = 4.0
        report["imputed_fields"].append("days_scheduled (defaulted to 4.0)")

    real_days_col = canonical.get("days_real")
    if real_days_col and real_days_col in df.columns:
        df["days_real_clean"] = pd.to_numeric(df[real_days_col], errors="coerce").fillna(
            df["days_scheduled_clean"]
        )
    else:
        df["days_real_clean"] = df["days_scheduled_clean"]
        report["imputed_fields"].append("days_real (defaulted to scheduled)")

    deliv_status_col = canonical.get("delivery_status")
    if deliv_status_col and deliv_status_col in df.columns:
        df["delivery_status_clean"] = df[deliv_status_col].fillna("Unknown").astype(str).str.strip()
    else:
        df["delivery_status_clean"] = df["late_delivery_risk_clean"].apply(
            lambda x: "Late delivery" if x == 1 else "Shipping on time"
        )

    cust_seg_col = canonical.get("customer_segment")
    if cust_seg_col and cust_seg_col in df.columns:
        df["customer_segment_clean"] = df[cust_seg_col].fillna("Consumer").astype(str).str.strip()
    else:
        df["customer_segment_clean"] = "Consumer"

    report["final_rows"] = len(df)
    report["retention_rate_pct"] = round(len(df) / max(initial_rows, 1) * 100, 2)
    report["duplicate_rows_detected"] = int(raw_df.duplicated().sum())
    report["canonical_mappings"] = canonical
    if "order_date_parsed" in df.columns and not df["order_date_parsed"].isnull().all():
        valid_dates = df["order_date_parsed"].dropna()
        report["date_range"] = {
            "start": str(valid_dates.min().date()),
            "end": str(valid_dates.max().date()),
        }
    else:
        report["date_range"] = {"start": "N/A", "end": "N/A"}

    return df, report


def apply_dashboard_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[Any, Any]] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    shipping_modes: Optional[List[str]] = None,
    risk_levels: Optional[List[str]] = None,
    selected_regions: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Filter the cleaned supply chain dataframe based on UI sidebar selections."""
    filtered = df.copy()

    # Resolve aliases
    if selected_regions is not None and regions is None:
        regions = selected_regions
    if selected_categories is not None and categories is None:
        categories = selected_categories

    # Filter date range
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if "order_date_parsed" in filtered.columns and not filtered["order_date_parsed"].isnull().all():
            filtered = filtered[
                (filtered["order_date_parsed"] >= start_date)
                & (filtered["order_date_parsed"] <= end_date + pd.Timedelta(days=1))
            ]

    # Filter regions
    if regions and "region_clean" in filtered.columns:
        filtered = filtered[filtered["region_clean"].isin(regions)]

    # Filter product categories
    if categories and "category_clean" in filtered.columns:
        filtered = filtered[filtered["category_clean"].isin(categories)]

    # Filter shipping modes
    if shipping_modes and "shipping_mode_clean" in filtered.columns:
        filtered = filtered[filtered["shipping_mode_clean"].isin(shipping_modes)]

    # Filter risk levels if present
    if risk_levels and "inventory_risk_level" in filtered.columns:
        filtered = filtered[filtered["inventory_risk_level"].isin(risk_levels)]

    return filtered
