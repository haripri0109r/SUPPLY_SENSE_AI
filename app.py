"""SupplySense AI — Predictive Supply Chain Intelligence Platform.
Streamlit Application Entry Point.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_processing import (
    apply_dashboard_filters,
    clean_supply_chain_data,
    inspect_dataset_quality,
    load_dataset,
    resolve_canonical_columns,
)
from src.delay_prediction import (
    compute_historical_delay_breakdowns,
    predict_single_order_delay,
    train_delay_classification_models,
)
from src.forecasting import (
    aggregate_demand_by_date,
    recursive_future_forecast,
    train_and_evaluate_forecasting_models,
)
from src.recommendations import (
    generate_supply_chain_recommendations,
    recommendations_to_dataframe,
)
from src.risk_analysis import (
    INVENTORY_DATA_LIMITATION_DISCLAIMER,
    SUPPLIER_DATA_LIMITATION_DISCLAIMER,
    calculate_category_inventory_risk,
    calculate_regional_performance_benchmarks,
)

# Page configuration
st.set_page_config(
    page_title="SupplySense AI — Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS-style Dark Navy + Soft Blue CSS Theme
st.markdown(
    """
    <style>
    /* Global Base */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    
    /* Branding Header */
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #3B82F6, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.8rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
    }
    
    /* Card Component */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-card-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-card-sub {
        font-size: 0.78rem;
        margin-top: 6px;
    }
    
    /* Status Badges */
    .badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-orange {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-blue {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    
    /* Notice Banners */
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #1E3A8A;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #92400E;
    }
    
    /* Streamlit UI Polish */
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 8px 16px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached data loading
@st.cache_data(show_spinner=False)
def get_cached_dataset(filepath_or_bytes: Any) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    return load_dataset(filepath_or_bytes)


# Cached model training
@st.cache_data(show_spinner=False)
def get_cached_forecasting_results(ts_df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    return train_and_evaluate_forecasting_models(ts_df)


@st.cache_resource(show_spinner=False)
def get_cached_delay_models(df: pd.DataFrame) -> Dict[str, Any]:
    return train_delay_classification_models(df)


# SIDEBAR: Branding, Dataset Selection & Universal Filters
st.sidebar.markdown('<div class="brand-title">SupplySense AI</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-subtitle">Predictive Supply Chain Platform</div>', unsafe_allow_html=True)

# Dataset source selector
dataset_mode = st.sidebar.radio(
    "Data Source",
    ["Local Workspace Dataset", "Upload Custom CSV"],
    index=0,
)

raw_df, cleaned_df, metadata = None, None, None
default_dataset_paths = [
    os.path.join("data", "raw", "DataCoSupplyChainDataset.csv"),
    os.path.join("data", "DataCoSupplyChainDataset.csv"),
    os.path.join("data", "SupplySense_cleaned.csv"),
]

found_local_path = None
for p in default_dataset_paths:
    if os.path.exists(p):
        found_local_path = p
        break

if dataset_mode == "Local Workspace Dataset":
    local_path = st.sidebar.text_input("Dataset File Path", value=found_local_path or "data/DataCoSupplyChainDataset.csv")
    if os.path.exists(local_path):
        try:
            with st.spinner("Loading supply chain dataset..."):
                raw_df, cleaned_df, metadata = get_cached_dataset(local_path)
            st.sidebar.success(f"✓ Loaded {len(cleaned_df):,} records")
        except Exception as e:
            st.sidebar.error(f"Error reading dataset: {e}")
    else:
        st.sidebar.warning(f"File not found: {local_path}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload DataCo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded dataset..."):
                raw_df, cleaned_df, metadata = get_cached_dataset(uploaded_file)
            st.sidebar.success(f"✓ Loaded {len(cleaned_df):,} records")
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded file: {e}")

# Page Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard Overview",
        "📈 Demand Forecast",
        "🛡️ Inventory Risk",
        "🚚 Shipment Delay Risk",
        "🌐 Supplier & Regional Performance",
        "📋 Recommendations & Filters",
    ],
    index=0,
)

# Empty State Warning
if cleaned_df is None or cleaned_df.empty:
    st.title("📦 Welcome to SupplySense AI")
    st.markdown(
        """
        <div class="warning-box">
            <strong>Dataset Required:</strong> No valid dataset is currently loaded. 
            Please select the local workspace dataset or upload a DataCo Smart Supply Chain CSV in the sidebar to begin analysis.
            Never will synthetic placeholders be generated in place of real supply chain data.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if found_local_path:
        st.info(f"Detected default dataset at `{found_local_path}`. Select 'Local Workspace Dataset' in the sidebar.")
    st.stop()


# SIDEBAR FILTERS (Only active when dataset is loaded)
st.sidebar.markdown("---")
st.sidebar.markdown("### Global Filters")

# 1. Date Range Filter
min_date = cleaned_df["order_date_parsed"].min()
max_date = cleaned_df["order_date_parsed"].max()

date_filter = None
if pd.notnull(min_date) and pd.notnull(max_date):
    date_filter = st.sidebar.date_input(
        "Order Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

# 2. Region Filter
all_regions = sorted(cleaned_df["region_clean"].unique().tolist())
selected_regions = st.sidebar.multiselect("Regions", all_regions, default=[])

# 3. Category Filter
all_categories = sorted(cleaned_df["category_clean"].unique().tolist())
selected_categories = st.sidebar.multiselect("Product Categories", all_categories, default=[])

# 4. Shipping Mode Filter
all_modes = sorted(cleaned_df["shipping_mode_clean"].unique().tolist())
selected_modes = st.sidebar.multiselect("Shipping Modes", all_modes, default=[])

# Apply filters
filtered_df = apply_dashboard_filters(
    cleaned_df,
    date_range=date_filter if isinstance(date_filter, (tuple, list)) and len(date_filter) == 2 else None,
    regions=selected_regions if selected_regions else None,
    categories=selected_categories if selected_categories else None,
    shipping_modes=selected_modes if selected_modes else None,
)

# Quick stats in sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Records:** `{len(filtered_df):,}` / `{len(cleaned_df):,}`")
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()


# ==============================================================================
# PAGE 1 — DASHBOARD OVERVIEW
# ==============================================================================
if page == "📊 Dashboard Overview":
    st.title("📊 Supply Chain Dashboard Overview")
    st.markdown("Real-time operational health, shipment fulfillment metrics, and demand throughput.")

    # High-level KPIs
    total_orders = len(filtered_df)
    total_sales = filtered_df["sales_clean"].sum()
    total_quantity = filtered_df["order_item_quantity_clean"].sum()
    late_rate = filtered_df["late_delivery_risk_clean"].mean() * 100 if "late_delivery_risk_clean" in filtered_df else 0
    avg_real_days = filtered_df["days_real_clean"].mean() if "days_real_clean" in filtered_df else 0
    avg_sched_days = filtered_df["days_scheduled_clean"].mean() if "days_scheduled_clean" in filtered_df else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Total Orders</div>
                <div class="metric-card-value">{total_orders:,}</div>
                <div class="metric-card-sub text-muted">Items: {total_quantity:,.0f} units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Total Revenue</div>
                <div class="metric-card-value">${total_sales:,.2f}</div>
                <div class="metric-card-sub text-muted">Avg Order: ${total_sales/max(total_orders,1):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        badge_cls = "badge-red" if late_rate > 50 else ("badge-orange" if late_rate > 30 else "badge-green")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Late Delivery Rate</div>
                <div class="metric-card-value">{late_rate:.1f}%</div>
                <div class="metric-card-sub"><span class="{badge_cls}">Fulfillment Exposure</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        gap = avg_real_days - avg_sched_days
        gap_sign = "+" if gap > 0 else ""
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Avg Actual Shipping Days</div>
                <div class="metric-card-value">{avg_real_days:.1f} <span style="font-size:1rem;color:#64748B;">days</span></div>
                <div class="metric-card-sub text-muted">Scheduled: {avg_sched_days:.1f}d ({gap_sign}{gap:.1f}d gap)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1: Order Trend & Delivery Status
    row1_c1, row1_c2 = st.columns([7, 5])
    with row1_c1:
        st.subheader("📈 Order Demand Trend Over Time")
        daily_trend = (
            filtered_df.dropna(subset=["order_date_parsed"])
            .groupby(pd.Grouper(key="order_date_parsed", freq="W"))
            .agg(quantity=("order_item_quantity_clean", "sum"), revenue=("sales_clean", "sum"))
            .reset_index()
        )
        if not daily_trend.empty:
            fig_trend = px.line(
                daily_trend,
                x="order_date_parsed",
                y="quantity",
                title="Weekly Ordered Quantity Throughput",
                labels={"order_date_parsed": "Date", "quantity": "Units Ordered"},
                color_discrete_sequence=["#2563EB"],
            )
            fig_trend.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=40, b=20),
                height=340,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Insufficient timestamped records for trend plotting.")

    with row1_c2:
        st.subheader("🚚 Delivery Status Breakdown")
        deliv_dist = filtered_df["delivery_status_clean"].value_counts().reset_index()
        deliv_dist.columns = ["Status", "Orders"]
        color_map = {
            "Late delivery": "#EF4444",
            "Advance shipping": "#10B981",
            "Shipping on time": "#3B82F6",
            "Shipping canceled": "#94A3B8",
        }
        fig_donut = px.pie(
            deliv_dist,
            names="Status",
            values="Orders",
            hole=0.45,
            color="Status",
            color_discrete_map=color_map,
        )
        fig_donut.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Charts Row 2: Top Categories & Regional Performance
    row2_c1, row2_c2 = st.columns([6, 6])
    with row2_c1:
        st.subheader("🏷️ Top Product Categories by Demand")
        cat_demand = (
            filtered_df.groupby("category_clean")["order_item_quantity_clean"]
            .sum()
            .reset_index()
            .sort_values("order_item_quantity_clean", ascending=True)
            .tail(10)
        )
        fig_cat = px.bar(
            cat_demand,
            x="order_item_quantity_clean",
            y="category_clean",
            orientation="h",
            labels={"order_item_quantity_clean": "Total Units", "category_clean": "Category"},
            color="order_item_quantity_clean",
            color_continuous_scale="Blues",
        )
        fig_cat.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with row2_c2:
        st.subheader("🌍 Regional Volume & Late Rate")
        reg_summary = (
            filtered_df.groupby("region_clean")
            .agg(orders=("order_item_quantity_clean", "count"), late_rate=("late_delivery_risk_clean", "mean"))
            .reset_index()
            .sort_values("orders", ascending=False)
            .head(10)
        )
        reg_summary["late_rate_pct"] = (reg_summary["late_rate"] * 100).round(1)

        fig_reg = px.bar(
            reg_summary,
            x="region_clean",
            y="orders",
            color="late_rate_pct",
            color_continuous_scale="Reds",
            labels={"orders": "Total Orders", "region_clean": "Region", "late_rate_pct": "Late Rate (%)"},
        )
        fig_reg.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    # Data Quality Inspection Card
    with st.expander("🔍 Dataset Health & Schema Quality Summary"):
        q = metadata.get("quality_summary", {})
        c_q1, c_q2, c_q3 = st.columns(3)
        c_q1.write(f"**Total Records (Cleaned):** {len(cleaned_df):,}")
        c_q2.write(f"**Duplicate Rows Identified:** {q.get('duplicate_rows', 0):,}")
        c_q3.write(f"**Data Quality Warnings:** {len(q.get('warnings', []))}")
        if q.get("warnings"):
            for w in q["warnings"]:
                st.warning(w)


# ==============================================================================
# PAGE 2 — DEMAND FORECASTING
# ==============================================================================
elif page == "📈 Demand Forecast":
    st.title("📈 Chronological Demand Forecasting")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Methodology:</strong> Aggregates historical order volume chronologically, 
            builds lag and rolling-average features strictly from past observations to prevent future data leakage, 
            and evaluates baseline vs Linear Regression vs Random Forest on held-out test data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fc_col1, fc_col2, fc_col3 = st.columns([5, 4, 3])
    with fc_col1:
        selected_cat = st.selectbox(
            "Select Product Category",
            ["All Categories"] + sorted(cleaned_df["category_clean"].unique().tolist()),
            index=0,
        )
    with fc_col2:
        freq_choice = st.radio("Aggregation Interval", ["Daily", "Weekly"], horizontal=True, index=0)
        freq_code = "D" if freq_choice == "Daily" else "W"
    with fc_col3:
        horizon = st.slider("Forecast Horizon", min_value=7, max_value=60, value=14, step=7)

    # Aggregate time series
    try:
        ts_df = aggregate_demand_by_date(cleaned_df, category=selected_cat, freq=freq_code)
    except Exception as e:
        st.error(f"Failed to aggregate demand: {e}")
        st.stop()

    if len(ts_df) < 20:
        st.warning(
            f"Insufficient historical data ({len(ts_df)} periods) for '{selected_cat}'. "
            "At least 20 continuous chronological observations are required to train lag features safely."
        )
        st.stop()

    # Train and evaluate models
    with st.spinner("Training time-series forecasting models without data leakage..."):
        fc_res = train_and_evaluate_forecasting_models(ts_df, test_ratio=0.25, min_observations=20)

    if not fc_res.get("success"):
        st.error(fc_res.get("message", "Model training failed."))
        st.stop()

    best_model_name = fc_res["best_model_name"]
    metrics_df = fc_res["metrics_df"]

    # Model Comparison Metrics Cards
    st.subheader("Model Benchmark (Held-Out Chronological Test Set)")
    st.markdown(f"**Top Performing Forecaster:** <span class='badge-green'>{best_model_name}</span>", unsafe_allow_html=True)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Plot Historical vs Held-Out Test Predictions
    fig_eval = go.Figure()
    train_dates = pd.to_datetime(fc_res["train_dates"])
    test_dates = pd.to_datetime(fc_res["test_dates"])

    fig_eval.add_trace(go.Scatter(
        x=train_dates,
        y=fc_res["y_train"],
        mode="lines",
        name="Historical Train Demand",
        line=dict(color="#94A3B8", width=1.5),
    ))
    fig_eval.add_trace(go.Scatter(
        x=test_dates,
        y=fc_res["y_test"],
        mode="lines+markers",
        name="Actual Test Demand",
        line=dict(color="#0F172A", width=2.5),
        marker=dict(size=4),
    ))

    # Add model predictions
    colors = {"Moving Average (7-day)": "#F59E0B", "Linear Regression": "#10B981", "Random Forest": "#2563EB"}
    for m_name, pred_vals in fc_res["predictions"].items():
        is_best = (m_name == best_model_name)
        fig_eval.add_trace(go.Scatter(
            x=test_dates,
            y=pred_vals,
            mode="lines",
            name=f"{m_name} (Test Pred)",
            line=dict(color=colors.get(m_name, "#8B5CF6"), width=2.5 if is_best else 1.5, dash="dash" if not is_best else "solid"),
        ))

    fig_eval.update_layout(
        title=f"Held-Out Demand Validation for '{selected_cat}'",
        xaxis_title="Date",
        yaxis_title="Demand Quantity (Units)",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        height=420,
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # Future Multi-step Forecast
    st.subheader(f"🔮 {horizon}-Step Future Recursive Forecast")
    st.caption("Forecast iterates recursively, generating past lag values from preceding model predictions.")

    best_model_obj = fc_res["models"].get(best_model_name)
    if best_model_obj is None and "Linear Regression" in fc_res["models"]:
        best_model_obj = fc_res["models"]["Linear Regression"]

    future_df = recursive_future_forecast(
        ts_df,
        model=best_model_obj,
        feature_cols=fc_res["feature_cols"],
        horizon=horizon,
        freq=freq_code,
    )

    # Plot Future Projection
    fig_future = go.Figure()
    # Recent history
    recent_ts = ts_df.tail(45)
    fig_future.add_trace(go.Scatter(
        x=recent_ts["date"],
        y=recent_ts["demand"],
        mode="lines",
        name="Recent Actual Demand",
        line=dict(color="#64748B", width=2),
    ))
    fig_future.add_trace(go.Scatter(
        x=future_df["date"],
        y=future_df["predicted_demand"],
        mode="lines+markers",
        name=f"Future Forecast ({best_model_name})",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=6, color="#1D4ED8"),
    ))

    fig_future.update_layout(
        title=f"Future Demand Projection ({selected_cat})",
        xaxis_title="Date",
        yaxis_title="Predicted Demand (Units)",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        height=380,
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Forecast Export
    csv_fc = future_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Demand Forecast to CSV",
        data=csv_fc,
        file_name=f"supply_sense_demand_forecast_{selected_cat.replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ==============================================================================
# PAGE 3 — INVENTORY RISK
# ==============================================================================
elif page == "🛡️ Inventory Risk":
    st.title("🛡️ Inventory Risk Assessment")
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>Rule-Based Risk Model:</strong> {INVENTORY_DATA_LIMITATION_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calculate inventory risk
    risk_df = calculate_category_inventory_risk(cleaned_df)

    high_risk_count = int((risk_df["inventory_risk_level"] == "High").sum())
    med_risk_count = int((risk_df["inventory_risk_level"] == "Medium").sum())
    low_risk_count = int((risk_df["inventory_risk_level"] == "Low").sum())
    avg_cv = float(risk_df["demand_cv"].mean())

    # KPI summary
    ik1, ik2, ik3, ik4 = st.columns(4)
    with ik1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">High-Risk Categories</div>
                <div class="metric-card-value" style="color:#DC2626;">{high_risk_count}</div>
                <div class="metric-card-sub"><span class="badge-red">Urgent Action Required</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ik2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Medium-Risk Categories</div>
                <div class="metric-card-value" style="color:#D97706;">{med_risk_count}</div>
                <div class="metric-card-sub"><span class="badge-orange">Close Monitoring</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ik3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Low-Risk Categories</div>
                <div class="metric-card-value" style="color:#059669;">{low_risk_count}</div>
                <div class="metric-card-sub"><span class="badge-green">Stable Fulfillment</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ik4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Avg Demand Volatility (CV)</div>
                <div class="metric-card-value">{avg_cv:.2f}</div>
                <div class="metric-card-sub text-muted">Coefficient of Variation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk Matrix Scatter Plot: Demand Volatility (CV) vs Fulfillment Delay Rate
    st.subheader("🎯 Demand Volatility vs. Fulfillment Delay Exposure")
    fig_scatter = px.scatter(
        risk_df,
        x="demand_cv",
        y="late_delivery_rate_pct",
        size="total_demand",
        color="inventory_risk_level",
        hover_name="category_clean",
        color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
        labels={
            "demand_cv": "Demand Volatility (Coefficient of Variation)",
            "late_delivery_rate_pct": "Late Delivery Rate (%)",
            "inventory_risk_level": "Risk Category",
            "total_demand": "Total Units",
        },
    )
    fig_scatter.add_hline(y=55, line_dash="dash", line_color="#EF4444", annotation_text="Critical Delay Threshold (55%)")
    fig_scatter.add_vline(x=0.35, line_dash="dash", line_color="#F59E0B", annotation_text="High Volatility Threshold (0.35)")
    fig_scatter.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Risk Filter & Table
    st.subheader("📋 Product Category Inventory Risk Scorecard")
    risk_filter = st.selectbox("Filter Risk Level", ["All Levels", "High", "Medium", "Low"], index=0)

    display_risk = risk_df.copy()
    if risk_filter != "All Levels":
        display_risk = display_risk[display_risk["inventory_risk_level"] == risk_filter]

    st.dataframe(
        display_risk[[
            "category_clean",
            "inventory_risk_level",
            "total_demand",
            "demand_cv",
            "late_delivery_rate_pct",
            "avg_real_days",
            "avg_scheduled_days",
            "lead_time_gap",
            "recommended_action",
        ]].rename(columns={
            "category_clean": "Category",
            "inventory_risk_level": "Risk Level",
            "total_demand": "Total Demand (Units)",
            "demand_cv": "Demand Volatility (CV)",
            "late_delivery_rate_pct": "Late Rate (%)",
            "avg_real_days": "Avg Actual Days",
            "avg_scheduled_days": "Avg Scheduled Days",
            "lead_time_gap": "Lead Time Gap",
            "recommended_action": "Recommended Action",
        }),
        use_container_width=True,
        hide_index=True,
    )

    csv_risk = display_risk.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Category Risk Analysis (CSV)",
        data=csv_risk,
        file_name="supply_sense_inventory_risk.csv",
        mime="text/csv",
    )


# ==============================================================================
# PAGE 4 — SHIPMENT DELAY RISK
# ==============================================================================
elif page == "🚚 Shipment Delay Risk":
    st.title("🚚 Predictive Shipment Delay Classification")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Leakage-Free Predictive Machine Learning:</strong> 
            Trained exclusively on features known at order checkout (Scheduled Days, Shipping Mode, Region, Product Category, Order Value). 
            Post-delivery variables such as <code>days_for_shipping_real</code>, <code>delivery_status</code>, and <code>shipping_date</code> 
            are strictly excluded to guarantee genuine predictive validity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Training classification pipelines and evaluating models..."):
        delay_res = get_cached_delay_models(cleaned_df)

    if not delay_res.get("success"):
        st.error(delay_res.get("message", "Model training failed."))
        st.stop()

    # Model Evaluation Metrics
    st.subheader("Model Performance Comparison (Unobserved Test Set)")
    metrics_comp = pd.DataFrame(delay_res["metrics"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(metrics_comp, use_container_width=True, hide_index=True)

    rf_metrics = delay_res["metrics"]["Random Forest"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{rf_metrics['Accuracy']*100:.1f}%")
    m2.metric("Precision", f"{rf_metrics['Precision']*100:.1f}%")
    m3.metric("Recall", f"{rf_metrics['Recall']*100:.1f}%")
    m4.metric("F1-Score", f"{rf_metrics['F1-Score']*100:.1f}%")
    m5.metric("ROC-AUC", f"{rf_metrics['ROC-AUC']:.3f}")

    # Confusion Matrix & Feature Importances
    deliv_c1, deliv_c2 = st.columns([5, 7])
    with deliv_c1:
        st.subheader("Confusion Matrix (Random Forest)")
        cm = np.array(delay_res["confusion_matrices"]["Random Forest"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["On-Time (0)", "Late (1)"],
            y=["On-Time (0)", "Late (1)"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with deliv_c2:
        st.subheader("Key Pre-Shipment Delay Drivers")
        feat_imp_df = delay_res.get("feature_importances", pd.DataFrame())
        if not feat_imp_df.empty:
            fig_imp = px.bar(
                feat_imp_df.head(10).sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal",
            )
            fig_imp.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # Interactive What-If Single Order Delay Risk Estimator
    st.subheader("⚡ Live Order 'What-If' Delay Risk Predictor")
    st.markdown("Test hypothetical order consignments before dispatch to preview calculated late delivery risk.")

    wi_col1, wi_col2, wi_col3, wi_col4 = st.columns(4)
    with wi_col1:
        wi_mode = st.selectbox("Shipping Mode", sorted(cleaned_df["shipping_mode_clean"].unique().tolist()))
        wi_sched = st.number_input("Scheduled Days", min_value=1, max_value=10, value=4)
    with wi_col2:
        wi_region = st.selectbox("Destination Region", sorted(cleaned_df["region_clean"].unique().tolist()))
        wi_cat = st.selectbox("Product Category", sorted(cleaned_df["category_clean"].unique().tolist()))
    with wi_col3:
        wi_qty = st.number_input("Order Quantity", min_value=1, max_value=50, value=2)
        wi_sales = st.number_input("Order Value ($)", min_value=1.0, max_value=5000.0, value=120.0, step=10.0)
    with wi_col4:
        st.write("**Evaluation Action**")
        st.write("")
        predict_clicked = st.button("🔮 Predict Consignment Delay Risk")

    if predict_clicked:
        order_dict = {
            "days_scheduled_clean": wi_sched,
            "order_item_quantity_clean": wi_qty,
            "sales_clean": wi_sales,
            "shipping_mode_clean": wi_mode,
            "region_clean": wi_region,
            "category_clean": wi_cat,
        }
        res_pred = predict_single_order_delay(delay_res["best_pipeline"], order_dict)
        prob = res_pred["delay_probability"]
        badge = "badge-red" if prob >= 0.6 else ("badge-orange" if prob >= 0.4 else "badge-green")

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 6px solid {'#EF4444' if prob>=0.6 else ('#F59E0B' if prob>=0.4 else '#10B981')};">
                <div class="metric-card-label">Consignment Risk Evaluation</div>
                <div class="metric-card-value">{res_pred['risk_label']} ({prob*100:.1f}%)</div>
                <div class="metric-card-sub">
                    <span class="{badge}">Estimated Delay Probability: {prob:.4f}</span> | 
                    Routing Mode: <strong>{wi_mode}</strong> to <strong>{wi_region}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 5 — SUPPLIER AND REGIONAL PERFORMANCE
# ==============================================================================
elif page == "🌐 Supplier & Regional Performance":
    st.title("🌐 Regional & Fulfillment Channel Performance")
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>Fulfillment Context:</strong> {SUPPLIER_DATA_LIMITATION_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    benchmarks = calculate_regional_performance_benchmarks(cleaned_df)
    reg_df = benchmarks.get("regional_summary", pd.DataFrame())
    mode_df = benchmarks.get("shipping_mode_summary", pd.DataFrame())

    # Regional Performance Table
    st.subheader("🌍 Regional Fulfillment Efficiency Scorecard")
    st.dataframe(
        reg_df[[
            "efficiency_rank",
            "region_clean",
            "order_volume",
            "total_sales",
            "late_rate_pct",
            "avg_real_days",
            "avg_scheduled_days",
            "lead_time_variance",
        ]].rename(columns={
            "efficiency_rank": "Rank",
            "region_clean": "Region",
            "order_volume": "Total Orders",
            "total_sales": "Total Sales ($)",
            "late_rate_pct": "Late Rate (%)",
            "avg_real_days": "Avg Actual Days",
            "avg_scheduled_days": "Avg Scheduled Days",
            "lead_time_variance": "Delay Variance (d)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        st.subheader("🚚 Carrier Shipping Mode Comparison")
        fig_mode = px.bar(
            mode_df,
            x="shipping_mode_clean",
            y=["avg_scheduled_days", "avg_real_days"],
            barmode="group",
            labels={"value": "Days", "shipping_mode_clean": "Shipping Mode", "variable": "Duration"},
            color_discrete_map={"avg_scheduled_days": "#3B82F6", "avg_real_days": "#EF4444"},
        )
        fig_mode.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with col_bench2:
        st.subheader("📊 Carrier Delay Rate by Shipping Channel")
        fig_delay_mode = px.bar(
            mode_df,
            x="shipping_mode_clean",
            y="late_rate_pct",
            color="late_rate_pct",
            color_continuous_scale="Reds",
            labels={"shipping_mode_clean": "Shipping Mode", "late_rate_pct": "Late Delivery Rate (%)"},
        )
        fig_delay_mode.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_delay_mode, use_container_width=True)


# ==============================================================================
# PAGE 6 — RECOMMENDATIONS AND FILTERS
# ==============================================================================
elif page == "📋 Recommendations & Filters":
    st.title("📋 Actionable Supply Chain Recommendations")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Evidence-Based Synthesis:</strong> 
            Operational recommendations synthesized strictly from empirical calculations, demand forecasting, 
            and machine learning models. Rule-based recommendations are clearly differentiated from ML predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calculate underlying components
    risk_df = calculate_category_inventory_risk(filtered_df)
    delay_res = get_cached_delay_models(cleaned_df)

    # Optional demand forecast results for top category
    ts_df = aggregate_demand_by_date(cleaned_df, freq="D")
    fc_res = train_and_evaluate_forecasting_models(ts_df, min_observations=20) if len(ts_df) >= 20 else None

    # Generate recommendations
    recs = generate_supply_chain_recommendations(
        df=filtered_df,
        inventory_risk_df=risk_df,
        delay_model_results=delay_res,
        forecast_results=fc_res,
    )

    # Filter recommendations by priority
    priority_filter = st.radio("Filter by Priority", ["All", "High", "Medium", "Low"], horizontal=True, index=0)
    displayed_recs = [r for r in recs if priority_filter == "All" or r["priority"] == priority_filter]

    # Render Recommendation Cards
    for r in displayed_recs:
        p_badge = "badge-red" if r["priority"] == "High" else ("badge-orange" if r["priority"] == "Medium" else "badge-green")
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 5px solid {'#EF4444' if r['priority']=='High' else ('#F59E0B' if r['priority']=='Medium' else '#10B981')};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span class="{p_badge}">{r['priority']} Priority</span>
                        <span class="badge-blue" style="margin-left:8px;">{r['domain']}</span>
                    </div>
                    <span style="font-size:0.75rem; color:#64748B; font-weight:600;">{r['type']}</span>
                </div>
                <div style="font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:6px;">
                    {r['finding']}
                </div>
                <div style="font-size:0.92rem; color:#334155; margin-bottom:8px;">
                    <strong>Action:</strong> {r['action']}
                </div>
                <div style="font-size:0.82rem; color:#64748B;">
                    <strong>Evidence / Basis:</strong> {r['evidence']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📑 Operational Action Summary Matrix")
    action_table = recommendations_to_dataframe(displayed_recs)
    st.dataframe(action_table, use_container_width=True, hide_index=True)

    # CSV Download Options
    st.markdown("### 📥 Data & Action Plan Export")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        csv_actions = action_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Action Summary Matrix (CSV)",
            data=csv_actions,
            file_name="supplysense_action_recommendations.csv",
            mime="text/csv",
        )
    with export_col2:
        csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=csv_filtered,
            file_name="supplysense_filtered_data.csv",
            mime="text/csv",
        )
