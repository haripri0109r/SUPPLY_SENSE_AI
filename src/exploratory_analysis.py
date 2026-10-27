"""Exploratory Data Analysis (EDA) Module for SupplySense AI.
Provides summary metrics and publication-quality static visualization functions
covering orders, sales, product categories, shipping modes, delivery status, and lead times.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configure aesthetic plotting defaults
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def calculate_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive overview KPIs."""
    total_orders = len(df)
    qty_col = "order_item_quantity_clean" if "order_item_quantity_clean" in df.columns else "order_item_quantity"
    total_quantity = int(df[qty_col].sum()) if qty_col in df.columns else 0
    sales_col = "sales_clean" if "sales_clean" in df.columns else "sales"
    total_sales = float(df[sales_col].sum()) if sales_col in df.columns else 0.0
    late_col = "late_delivery_risk_clean" if "late_delivery_risk_clean" in df.columns else "late_delivery_risk"
    late_rate = float(df[late_col].mean() * 100) if late_col in df.columns else 0.0
    real_days_col = "days_real_clean" if "days_real_clean" in df.columns else "days_for_shipping_real"
    avg_shipping_days = float(df[real_days_col].mean()) if real_days_col in df.columns else 0.0

    return {
        "total_orders": total_orders,
        "total_sales": round(total_sales, 2),
        "total_quantity": total_quantity,
        "late_delivery_rate": round(late_rate, 2),
        "avg_shipping_days": round(avg_shipping_days, 2),
    }


def compute_order_quantity_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistical distribution metrics for order item quantities."""
    col = "order_item_quantity_clean" if "order_item_quantity_clean" in df.columns else "order_item_quantity"
    if col not in df.columns:
        return {"error": "Quantity column not found."}
    s = df[col].dropna()
    return {
        "count": int(s.count()),
        "mean": float(round(s.mean(), 2)),
        "std": float(round(s.std(), 2)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def compute_sales_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistical distribution metrics for transaction sales."""
    col = "sales_clean" if "sales_clean" in df.columns else "sales"
    if col not in df.columns:
        return {"error": "Sales column not found."}
    s = df[col].dropna()
    return {
        "count": int(s.count()),
        "sum": float(round(s.sum(), 2)),
        "mean": float(round(s.mean(), 2)),
        "std": float(round(s.std(), 2)),
        "min": float(s.min()),
        "median": float(s.median()),
        "p95": float(round(s.quantile(0.95), 2)),
        "max": float(s.max()),
    }


def compute_category_demand_breakdown(
    df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Aggregate total orders, units sold, and late rate by product category."""
    cat_col = "category_clean" if "category_clean" in df.columns else "category_name"
    qty_col = "order_item_quantity_clean" if "order_item_quantity_clean" in df.columns else "order_item_quantity"
    sales_col = "sales_clean" if "sales_clean" in df.columns else "sales"
    late_col = "late_delivery_risk_clean" if "late_delivery_risk_clean" in df.columns else "late_delivery_risk"

    if cat_col not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        qty_col: "sum",
        sales_col: "sum",
    }
    if late_col in df.columns:
        agg_dict[late_col] = ["count", "mean"]

    grouped = df.groupby(cat_col).agg(agg_dict)
    if late_col in df.columns:
        grouped.columns = ["total_quantity", "total_sales", "order_count", "late_rate"]
        grouped["late_rate_pct"] = (grouped["late_rate"] * 100).round(1)
    else:
        grouped.columns = ["total_quantity", "total_sales"]
        grouped["order_count"] = df.groupby(cat_col).size()

    grouped = grouped.sort_values(by="total_quantity", ascending=False).reset_index()
    return grouped.head(top_n)


def compute_monthly_sales_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales and order quantities by month."""
    date_col = "order_date_parsed" if "order_date_parsed" in df.columns else "order_date"
    sales_col = "sales_clean" if "sales_clean" in df.columns else "sales"
    qty_col = "order_item_quantity_clean" if "order_item_quantity_clean" in df.columns else "order_item_quantity"

    if date_col not in df.columns or df[date_col].isnull().all():
        return pd.DataFrame()

    df_temp = df.dropna(subset=[date_col]).copy()
    df_temp["year_month"] = df_temp[date_col].dt.to_period("M").dt.to_timestamp()
    trend = (
        df_temp.groupby("year_month")
        .agg(
            total_sales=(sales_col, "sum"),
            total_quantity=(qty_col, "sum"),
            order_count=(sales_col, "count"),
        )
        .reset_index()
    )
    return trend.sort_values(by="year_month")


def compute_shipping_mode_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze order distribution and late delivery frequency across shipping modes."""
    mode_col = "shipping_mode_clean" if "shipping_mode_clean" in df.columns else "shipping_mode"
    late_col = "late_delivery_risk_clean" if "late_delivery_risk_clean" in df.columns else "late_delivery_risk"

    if mode_col not in df.columns:
        return pd.DataFrame()

    if late_col in df.columns:
        mode_df = (
            df.groupby(mode_col)
            .agg(
                order_count=(mode_col, "count"),
                late_orders=(late_col, "sum"),
                late_rate=(late_col, "mean"),
            )
            .reset_index()
        )
        mode_df["late_rate_pct"] = (mode_df["late_rate"] * 100).round(1)
        mode_df["share_of_orders_pct"] = (
            mode_df["order_count"] / mode_df["order_count"].sum() * 100
        ).round(1)
    else:
        mode_df = df[mode_col].value_counts().reset_index()
        mode_df.columns = [mode_col, "order_count"]
        mode_df["share_of_orders_pct"] = (
            mode_df["order_count"] / mode_df["order_count"].sum() * 100
        ).round(1)

    return mode_df.sort_values(by="order_count", ascending=False)


def compute_delivery_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution of delivery fulfillment states."""
    deliv_col = "delivery_status_clean" if "delivery_status_clean" in df.columns else "delivery_status"
    if deliv_col not in df.columns:
        return pd.DataFrame()

    counts = df[deliv_col].value_counts().reset_index()
    counts.columns = ["delivery_status", "order_count"]
    counts["percentage"] = (counts["order_count"] / counts["order_count"].sum() * 100).round(1)
    return counts


def compute_regional_volume_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Summarize regional fulfillment volume and late delivery rate."""
    reg_col = "region_clean" if "region_clean" in df.columns else "order_region"
    sales_col = "sales_clean" if "sales_clean" in df.columns else "sales"
    late_col = "late_delivery_risk_clean" if "late_delivery_risk_clean" in df.columns else "late_delivery_risk"

    if reg_col not in df.columns:
        return pd.DataFrame()

    if late_col in df.columns:
        reg_df = (
            df.groupby(reg_col)
            .agg(
                order_count=(reg_col, "count"),
                total_sales=(sales_col, "sum"),
                late_deliveries=(late_col, "sum"),
                late_rate=(late_col, "mean"),
            )
            .reset_index()
        )
        reg_df["late_rate_pct"] = (reg_df["late_rate"] * 100).round(1)
    else:
        reg_df = (
            df.groupby(reg_col)
            .agg(order_count=(reg_col, "count"), total_sales=(sales_col, "sum"))
            .reset_index()
        )

    return reg_df.sort_values(by="order_count", ascending=False).head(top_n)


def compute_shipping_duration_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare scheduled versus actual shipping days."""
    sched_col = "days_scheduled_clean" if "days_scheduled_clean" in df.columns else "days_for_shipment_scheduled"
    real_col = "days_real_clean" if "days_real_clean" in df.columns else "days_for_shipping_real"

    if sched_col not in df.columns or real_col not in df.columns:
        return {"error": "Shipping duration columns not found."}

    avg_sched = float(round(df[sched_col].mean(), 2))
    avg_real = float(round(df[real_col].mean(), 2))
    diff = float(round(avg_real - avg_sched, 2))

    return {
        "avg_scheduled_days": avg_sched,
        "avg_real_days": avg_real,
        "avg_variance_days": diff,
        "max_real_days": float(df[real_col].max()),
    }


# =====================================================================
# Publication-Quality Static Plots (Exportable to outputs/figures/)
# =====================================================================

def plot_order_quantity_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot histogram and KDE of order item quantities."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    else:
        fig = ax.figure

    col = "order_item_quantity_clean" if "order_item_quantity_clean" in df.columns else "order_item_quantity"
    data = df[col].dropna()

    sns.histplot(data, bins=range(int(data.min()), int(data.max()) + 2), kde=False, color="#2563EB", ax=ax, edgecolor="#1E293B", alpha=0.85)
    ax.set_title("Order Item Quantity Distribution", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Order Item Quantity (Units)", fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency (Order Count)", fontsize=11, labelpad=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_sales_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot distribution of sales per order transaction (up to 99th percentile)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    else:
        fig = ax.figure

    col = "sales_clean" if "sales_clean" in df.columns else "sales"
    data = df[col].dropna()
    p99 = data.quantile(0.99)

    sns.histplot(data[data <= p99], bins=40, kde=True, color="#059669", ax=ax, edgecolor="#064E3B", alpha=0.75)
    ax.axvline(data.median(), color="#DC2626", linestyle="--", linewidth=1.5, label=f"Median (${data.median():.1f})")
    ax.axvline(data.mean(), color="#D97706", linestyle="-", linewidth=1.5, label=f"Mean (${data.mean():.1f})")
    ax.set_title("Transaction Sales Distribution (Trimmed at 99th Percentile)", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Sales Value (USD)", fontsize=11, labelpad=8)
    ax.set_ylabel("Order Count", fontsize=11, labelpad=8)
    ax.legend(frameon=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_top_product_categories(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of top product categories by quantity."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    else:
        fig = ax.figure

    cat_df = compute_category_demand_breakdown(df, top_n=top_n)

    if cat_df.empty:
        return fig

    cat_col = cat_df.columns[0]
    sns.barplot(data=cat_df, y=cat_col, x="total_quantity", palette="Blues_r", ax=ax)
    ax.set_title(f"Top {top_n} Product Categories by Total Order Quantity", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Total Units Ordered", fontsize=11, labelpad=8)
    ax.set_ylabel("Product Category", fontsize=11, labelpad=8)
    
    # Value annotations on bars
    for p in ax.patches:
        val = int(p.get_width())
        ax.annotate(f"{val:,}", (val, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color="#1E293B")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_monthly_sales_trends(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot chronological monthly revenue trend curve."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    else:
        fig = ax.figure

    trend = compute_monthly_sales_trends(df)

    if trend.empty:
        return fig

    ax.plot(trend["year_month"], trend["total_sales"] / 1000.0, marker="o", color="#3B82F6", linewidth=2.2, label="Monthly Revenue ($K)")
    ax.set_title("Chronological Monthly Sales Revenue Trend (2015 – 2018)", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Order Month", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Sales (Thousand USD)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_delivery_status_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot donut chart of delivery fulfillment status."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    else:
        fig = ax.figure

    deliv_df = compute_delivery_status_distribution(df)

    if deliv_df.empty:
        return fig

    colors = ["#EF4444", "#10B981", "#3B82F6", "#F59E0B"]
    wedges, texts, autotexts = ax.pie(
        deliv_df["order_count"],
        labels=deliv_df["delivery_status"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[: len(deliv_df)],
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.set_title("Overall Shipment Delivery Status Distribution", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def generate_all_eda_figures(
    df: pd.DataFrame, output_dir: str = "outputs/figures"
) -> Dict[str, str]:
    """Generate and save all core EDA plots into the outputs directory.

    Returns:
        Dictionary mapping figure name to absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    p1 = os.path.join(output_dir, "eda_order_quantity_distribution.png")
    plot_order_quantity_distribution(df, save_path=p1)
    paths["order_quantity"] = p1
    plt.close()

    p2 = os.path.join(output_dir, "eda_sales_distribution.png")
    plot_sales_distribution(df, save_path=p2)
    paths["sales"] = p2
    plt.close()

    p3 = os.path.join(output_dir, "eda_top_product_categories.png")
    plot_top_product_categories(df, top_n=10, save_path=p3)
    paths["top_categories"] = p3
    plt.close()

    p4 = os.path.join(output_dir, "eda_monthly_sales_trends.png")
    plot_monthly_sales_trends(df, save_path=p4)
    paths["monthly_sales"] = p4
    plt.close()

    p5 = os.path.join(output_dir, "eda_delivery_status_distribution.png")
    plot_delivery_status_distribution(df, save_path=p5)
    paths["delivery_status"] = p5
    plt.close()

    return paths
