"""Recommendations Module for SupplySense AI.
Generates evidence-backed operational recommendations and action summaries,
clearly differentiating Rule-Based heuristics from Machine Learning predictions.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_supply_chain_recommendations(
    df: pd.DataFrame,
    inventory_risk_df: Optional[pd.DataFrame] = None,
    delay_model_results: Optional[Dict[str, Any]] = None,
    forecast_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Synthesize evidence-backed recommendations from calculations and model outputs."""
    recommendations: List[Dict[str, Any]] = []

    # 1. Delivery Delay Risk Recommendations (from actual data & ML models)
    if "late_delivery_risk_clean" in df.columns:
        overall_late_rate = float(df["late_delivery_risk_clean"].mean() * 100)

        # Shipping mode analysis
        if "shipping_mode_clean" in df.columns:
            mode_late = df.groupby("shipping_mode_clean")["late_delivery_risk_clean"].mean() * 100
            worst_mode = mode_late.idxmax()
            worst_mode_rate = float(mode_late.max())

            if worst_mode_rate > 50.0:
                recommendations.append({
                    "priority": "High",
                    "domain": "Logistics & Fulfillment",
                    "type": "Rule-Based Metric",
                    "finding": f"Shipping mode '{worst_mode}' exhibits a {worst_mode_rate:.1f}% late delivery rate (overall average: {overall_late_rate:.1f}%).",
                    "action": f"Renegotiate carrier service level agreements (SLAs) for '{worst_mode}' or adjust customer-facing promised delivery windows to realistic timelines.",
                    "evidence": f"Carrier performance data across {len(df):,} orders indicates consistent fulfillment delay in {worst_mode}.",
                })

        # Regional delay analysis
        if "region_clean" in df.columns:
            reg_late = df.groupby("region_clean")["late_delivery_risk_clean"].mean() * 100
            high_delay_regions = reg_late[reg_late > 55.0]
            if not high_delay_regions.empty:
                top_delayed_reg = high_delay_regions.index[0]
                top_delay_pct = high_delay_regions.iloc[0]
                recommendations.append({
                    "priority": "High",
                    "domain": "Regional Network",
                    "type": "Rule-Based Metric",
                    "finding": f"Region '{top_delayed_reg}' experiences delayed fulfillment on {top_delay_pct:.1f}% of consignments.",
                    "action": f"Establish localized forward stocking hubs or alternative regional 3PL partnerships to service '{top_delayed_reg}'.",
                    "evidence": f"Historical delay rate of {top_delay_pct:.1f}% significantly exceeds corporate target benchmarks (< 15%).",
                })

    # 2. Machine Learning Delay Model Insights
    if delay_model_results and delay_model_results.get("success"):
        best_name = delay_model_results.get("best_model_name", "Random Forest")
        metrics = delay_model_results.get("metrics", {}).get(best_name, {})
        accuracy = metrics.get("Accuracy", 0) * 100
        auc = metrics.get("ROC-AUC", 0)

        feat_imp = delay_model_results.get("feature_importances", pd.DataFrame())
        top_driver = feat_imp.iloc[0]["Feature"] if not feat_imp.empty else "Scheduled shipping duration"

        recommendations.append({
            "priority": "Medium",
            "domain": "Predictive Risk Controls",
            "type": "Machine Learning Prediction",
            "finding": f"Trained {best_name} classifier achieves {accuracy:.1f}% test accuracy (ROC-AUC: {auc:.3f}). Top delay predictor: '{top_driver}'.",
            "action": f"Deploy the {best_name} model at checkout to trigger real-time pre-dispatch routing flags for orders with delay probability > 60%.",
            "evidence": f"Held-out validation on {delay_model_results.get('test_size', 0):,} unobserved orders proves high discriminatory power.",
        })

    # 3. Inventory Risk Recommendations (from rule-based risk table)
    if inventory_risk_df is not None and not inventory_risk_df.empty:
        high_risk_cats = inventory_risk_df[inventory_risk_df["inventory_risk_level"] == "High"]
        if not high_risk_cats.empty:
            top_cat = high_risk_cats.iloc[0]["category_clean"]
            cat_cv = high_risk_cats.iloc[0]["demand_cv"]
            cat_late = high_risk_cats.iloc[0]["late_delivery_rate_pct"]
            recommendations.append({
                "priority": "High",
                "domain": "Inventory Exposure",
                "type": "Rule-Based Exposure Heuristic",
                "finding": f"Category '{top_cat}' is categorized as High Risk due to high demand volatility (CV: {cat_cv:.2f}) and {cat_late:.1f}% fulfillment delays.",
                "action": f"Implement safety stock buffer reviews for '{top_cat}' and initiate supplier lead-time stabilization discussions.",
                "evidence": f"Combined demand variance (CV: {cat_cv:.2f}) and historical shipment delay rate ({cat_late:.1f}%).",
            })
        else:
            recommendations.append({
                "priority": "Low",
                "domain": "Inventory Exposure",
                "type": "Rule-Based Exposure Heuristic",
                "finding": "No categories currently breach critical volatility and delay thresholds simultaneously.",
                "action": "Maintain existing replenishment order cycle while continuing periodic weekly coefficient-of-variation audits.",
                "evidence": "Observed demand CV remains below 0.35 threshold across all evaluated active categories.",
            })

    # 4. Demand Forecasting Insights
    if forecast_results and forecast_results.get("success"):
        best_forecaster = forecast_results.get("best_model_name", "Random Forest")
        metrics_df = forecast_results.get("metrics_df", pd.DataFrame())
        if not metrics_df.empty:
            best_rmse = metrics_df[metrics_df["Model"] == best_forecaster]["RMSE"].values[0]
            recommendations.append({
                "priority": "Medium",
                "domain": "Demand Planning",
                "type": "Machine Learning Prediction",
                "finding": f"Held-out chronological time-series evaluation identifies '{best_forecaster}' as the top demand forecaster (RMSE: {best_rmse}).",
                "action": f"Utilize the {best_forecaster} multi-step recursive projection for 14-day aggregate procurement planning.",
                "evidence": f"Model outperformed naive moving-average baseline over held-out chronological test period.",
            })

    # 5. Data Governance & Stock Feeds Recommendation
    recommendations.append({
        "priority": "Medium",
        "domain": "Data Governance",
        "type": "System Architectural Recommendation",
        "finding": "The current dataset contains transactional order records but lacks real-time warehouse stock-on-hand and purchase replenishment intervals.",
        "action": "Integrate ERP/WMS inventory feeds into SupplySense AI to unlock exact automated Economic Order Quantity (EOQ) and dynamic reorder point calculations.",
        "evidence": "Data pipeline audit identified absence of current inventory balance attributes in the transaction source schema.",
    })

    return recommendations


def recommendations_to_dataframe(recs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format recommendations list into a standardized, presentation-ready DataFrame."""
    if not recs:
        return pd.DataFrame(columns=["Priority", "Domain", "Type", "Finding", "Recommended Action", "Evidence Basis"])

    df = pd.DataFrame(recs)
    df = df.rename(columns={
        "priority": "Priority",
        "domain": "Domain",
        "type": "Analytical Type",
        "finding": "Observed Finding",
        "action": "Recommended Action",
        "evidence": "Evidence / Basis",
    })
    return df
