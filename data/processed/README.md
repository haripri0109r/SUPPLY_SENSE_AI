# Processed Data Directory

This directory stores cleaned, normalized, and engineered datasets produced by the SupplySense AI data engineering pipeline (`src/data_cleaning.py` and `src/feature_engineering.py`).

## Generated Files

- **`SupplySense_cleaned.csv`**: Full normalized transaction dataset containing standardized `snake_case` column headers, parsed ISO-8601 timestamps (`order_date_parsed`, `shipping_date_parsed`), validated numeric values, imputed missing categories, and normalized target flags (`late_delivery_risk_clean`).
- **`daily_demand_aggregated.csv`**: Time-series demand aggregated by date and product category, used for chronological demand forecasting.

## Processing Guarantees

1. **Non-destructive Processing:** Raw data in `data/raw/` is never modified or overwritten.
2. **Deterministic Imputation:** Known sentinel values (`Unknown`) are utilized for sparse fields (e.g., customer zip codes) to avoid arbitrary row discarding.
3. **Leakage Prevention:** Engineered lag and rolling features are derived strictly from antecedent historical time steps.
