"""SupplySense AI: Predictive Supply Chain Intelligence Platform.

Package providing modular pipelines for:
- Data loading and inspection (src.data_loader)
- Data cleaning and canonical normalization (src.data_cleaning)
- Exploratory data analysis (src.exploratory_analysis)
- Temporal and time-series feature engineering (src.feature_engineering)
- Demand forecasting with recursive multi-step forecasting (src.forecasting)
- Pre-dispatch shipment delay risk classification (src.delay_prediction)
- Empirical inventory and regional performance benchmarking (src.inventory_risk)
- Actionable decision recommendations (src.recommendations)
"""

__version__ = "1.0.0"

from src.data_loader import load_dataset, load_raw_data, inspect_dataset_quality
from src.data_cleaning import clean_supply_chain_data, apply_dashboard_filters
from src.exploratory_analysis import (
    calculate_summary_kpis,
    plot_order_quantity_distribution,
    plot_sales_distribution,
    plot_top_product_categories,
    plot_monthly_sales_trends,
    plot_delivery_status_distribution,
    generate_all_eda_figures,
)
from src.feature_engineering import (
    extract_date_features,
    aggregate_demand_by_date,
    build_time_series_features,
    prepare_delay_classification_features,
)
from src.forecasting import (
    train_and_evaluate_forecasting_models,
    recursive_future_forecast,
    calculate_mape,
)
from src.delay_prediction import (
    train_delay_classification_models,
    predict_single_order_delay_risk,
)
from src.inventory_risk import (
    calculate_category_inventory_risk,
    calculate_regional_performance_benchmarks,
    INVENTORY_DATA_LIMITATION_DISCLAIMER,
    SUPPLIER_DATA_LIMITATION_DISCLAIMER,
)
from src.recommendations import generate_supply_chain_recommendations

__all__ = [
    "load_dataset",
    "load_raw_data",
    "inspect_dataset_quality",
    "clean_supply_chain_data",
    "apply_dashboard_filters",
    "calculate_summary_kpis",
    "plot_order_quantity_distribution",
    "plot_sales_distribution",
    "plot_top_product_categories",
    "plot_monthly_sales_trends",
    "plot_delivery_status_distribution",
    "generate_all_eda_figures",
    "extract_date_features",
    "aggregate_demand_by_date",
    "build_time_series_features",
    "prepare_delay_classification_features",
    "train_and_evaluate_forecasting_models",
    "recursive_future_forecast",
    "calculate_mape",
    "train_delay_classification_models",
    "predict_single_order_delay_risk",
    "calculate_category_inventory_risk",
    "calculate_regional_performance_benchmarks",
    "INVENTORY_DATA_LIMITATION_DISCLAIMER",
    "SUPPLIER_DATA_LIMITATION_DISCLAIMER",
    "generate_supply_chain_recommendations",
]
