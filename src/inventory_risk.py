"""Inventory and Regional Risk Analysis Module for SupplySense AI.
Calculates transparent, rule-based inventory risk estimates from demand variability
and lead-time uncertainty, without fabricating current stock-on-hand or replenishment levels.
Also calculates regional and shipping-mode performance benchmarks.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


INVENTORY_DATA_LIMITATION_DISCLAIMER = (
    "Note on Inventory Limitations: The DataCo Supply Chain dataset records historical customer transactions, "
    "orders, and shipment durations, but does not provide real-time stock-on-hand, minimum batch sizes, "
    "or warehouse replenishment intervals. The metrics below represent an empirical, rule-based Demand & Lead Time "
    "Exposure Assessment derived strictly from observed transaction volatility, not fabricated stock counts."
)

SUPPLIER_DATA_LIMITATION_DISCLAIMER = (
    "Note on Supplier Identification: The dataset does not contain dedicated supplier IDs or manufacturer vendor codes. "
    "To preserve analytical integrity, this analysis evaluates regional fulfillment centers, geographic markets, "
    "and carrier shipping modes rather than misrepresenting customer or product records as suppliers."
)


def calculate_category_inventory_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rule-based inventory risk for each product category based on demand and shipping volatility.

    Metrics:
        - Total Demand Volume
        - Average Order Quantity
        - Demand Standard Deviation & Coefficient of Variation (CV)
        - Average Scheduled Days & Real Shipping Days
        - Shipping Delay Gap (Real - Scheduled)
        - Late Delivery Rate (%)
        - Composite Demand Risk Level (Low, Medium, High)
        - Recommended Action
    """
    if "category_clean" not in df.columns:
        raise ValueError("'category_clean' column is required for inventory risk analysis.")

    # Group by product category
    grouped = df.groupby("category_clean").agg(
        total_orders=("order_item_quantity_clean", "count"),
        total_demand=("order_item_quantity_clean", "sum"),
        avg_quantity=("order_item_quantity_clean", "mean"),
        std_quantity=("order_item_quantity_clean", "std"),
        total_sales=("sales_clean", "sum"),
        avg_scheduled_days=("days_scheduled_clean", "mean"),
        avg_real_days=("days_real_clean", "mean"),
        late_rate=("late_delivery_risk_clean", "mean"),
    ).reset_index()

    # Fill NaNs for single-item categories
    grouped["std_quantity"] = grouped["std_quantity"].fillna(0.0)
    # Coefficient of Variation = std / mean
    grouped["demand_cv"] = np.where(
        grouped["avg_quantity"] > 0,
        grouped["std_quantity"] / grouped["avg_quantity"],
        0.0,
    )

    # Lead time gap
    grouped["lead_time_gap"] = grouped["avg_real_days"] - grouped["avg_scheduled_days"]

    # Rule-Based Risk Scoring:
    # High Risk if:
    # - Late delivery rate > 55% AND demand_cv > 0.35
    # - OR lead_time_gap > 0.5 days
    # Medium Risk if:
    # - Late delivery rate > 45% OR demand_cv > 0.25
    # Low Risk:
    # - All other categories
    def determine_risk(row: pd.Series) -> Tuple[str, str]:
        late_pct = row["late_rate"] * 100
        cv = row["demand_cv"]
        gap = row["lead_time_gap"]

        if (late_pct >= 55.0 and cv >= 0.35) or gap > 0.5:
            return (
                "High",
                "Buffer Stock Review: High demand volatility combined with frequent fulfillment delays. "
                "Audit carrier SLAs and calculate dynamic safety stock once on-hand inventory feeds become available.",
            )
        elif late_pct >= 45.0 or cv >= 0.25:
            return (
                "Medium",
                "Monitoring Required: Moderate demand swings or intermittent shipping delays. "
                "Monitor replenishment lead-times and re-evaluate shipping mode allocation.",
            )
        else:
            return (
                "Low",
                "Stable Performance: Predictable demand profile and consistent on-time fulfillment. "
                "Maintain standard replenishment cycle.",
            )

    risk_results = grouped.apply(determine_risk, axis=1)
    grouped["inventory_risk_level"] = [r[0] for r in risk_results]
    grouped["recommended_action"] = [r[1] for r in risk_results]

    # Format numeric columns for clean presentation
    grouped["total_demand"] = grouped["total_demand"].astype(int)
    grouped["avg_quantity"] = grouped["avg_quantity"].round(2)
    grouped["demand_cv"] = grouped["demand_cv"].round(3)
    grouped["total_sales"] = grouped["total_sales"].round(2)
    grouped["avg_scheduled_days"] = grouped["avg_scheduled_days"].round(1)
    grouped["avg_real_days"] = grouped["avg_real_days"].round(1)
    grouped["lead_time_gap"] = grouped["lead_time_gap"].round(2)
    grouped["late_delivery_rate_pct"] = (grouped["late_rate"] * 100).round(1)

    return grouped.sort_values(by=["late_delivery_rate_pct", "total_demand"], ascending=[False, False])


def calculate_regional_performance_benchmarks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate regional and shipping-mode performance metrics and efficiency rankings."""
    results = {}

    # 1. Regional Performance
    if "region_clean" in df.columns:
        reg = df.groupby("region_clean").agg(
            order_volume=("order_item_quantity_clean", "count"),
            total_sales=("sales_clean", "sum"),
            avg_order_value=("sales_clean", "mean"),
            late_rate=("late_delivery_risk_clean", "mean"),
            avg_real_days=("days_real_clean", "mean"),
            avg_scheduled_days=("days_scheduled_clean", "mean"),
        ).reset_index()

        reg["late_rate_pct"] = (reg["late_rate"] * 100).round(1)
        reg["total_sales"] = reg["total_sales"].round(2)
        reg["avg_order_value"] = reg["avg_order_value"].round(2)
        reg["avg_real_days"] = reg["avg_real_days"].round(1)
        reg["avg_scheduled_days"] = reg["avg_scheduled_days"].round(1)
        reg["lead_time_variance"] = (reg["avg_real_days"] - reg["avg_scheduled_days"]).round(2)

        # Rank regions by fulfillment efficiency (lowest late rate, highest volume)
        reg["efficiency_rank"] = reg["late_rate"].rank(ascending=True, method="min").astype(int)
        results["regional_summary"] = reg.sort_values("order_volume", ascending=False)

    # 2. Shipping Mode Performance
    if "shipping_mode_clean" in df.columns:
        mode = df.groupby("shipping_mode_clean").agg(
            order_volume=("order_item_quantity_clean", "count"),
            total_sales=("sales_clean", "sum"),
            late_rate=("late_delivery_risk_clean", "mean"),
            avg_real_days=("days_real_clean", "mean"),
            avg_scheduled_days=("days_scheduled_clean", "mean"),
        ).reset_index()

        mode["late_rate_pct"] = (mode["late_rate"] * 100).round(1)
        mode["total_sales"] = mode["total_sales"].round(2)
        mode["avg_real_days"] = mode["avg_real_days"].round(1)
        mode["avg_scheduled_days"] = mode["avg_scheduled_days"].round(1)
        mode["fulfillment_gap"] = (mode["avg_real_days"] - mode["avg_scheduled_days"]).round(2)
        results["shipping_mode_summary"] = mode.sort_values("order_volume", ascending=False)

    return results
