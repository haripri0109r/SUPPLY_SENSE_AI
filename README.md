# SupplySense AI: Predictive Supply Chain Intelligence Platform

An end-to-end, academic and production-grade Data Science platform built with Python, Jupyter, Scikit-Learn, Plotly, and Streamlit. SupplySense AI delivers chronological demand forecasting, pre-dispatch shipment delay classification without data leakage, empirical category inventory exposure assessment, and operational decision support.

---

## 1. Project Overview & Architecture

SupplySense AI addresses core operational bottlenecks across global e-commerce and retail supply chains:
- **Upstream Chronological Demand Forecasting:** Evaluates 7-Day Moving Average, Multivariate Linear Regression, and Random Forest Regressors on strictly past observations, providing recursive multi-step future projections.
- **Pre-Dispatch Shipment Delay Risk Classification:** Employs Random Forest and Logistic Regression pipelines using order-time predictors exclusively (`days_scheduled_clean`, `shipping_mode_clean`, `region_clean`, `category_clean`, `sales_clean`), strictly excluding post-dispatch leakage (`days_real`, `delivery_status`).
- **Empirical Inventory Exposure Assessment:** Uses Demand Coefficient of Variation ($CV$) and Lead-Time Gap as transparent proxies for stockout vulnerability without fabricating unobserved warehouse telemetry.
- **Interactive Multi-Dashboard Application:** Six responsive Streamlit dashboards featuring custom dark navy/slate aesthetics, interactive Plotly visualizations, dynamic what-if simulation, and CSV artifact export.

### Project Directory Structure

```text
SupplySense-AI/
├── README.md                                  # Complete platform documentation and PowerShell instructions
├── requirements.txt                           # Pinpoint dependencies for data, ML, web, and tests
├── .gitignore                                 # Git exclusion rules for caches, models, and environments
├── app.py                                     # Interactive Streamlit 6-dashboard web application
├── data/
│   ├── README.md                              # Dataset provenance, usage terms, and citation details
│   ├── raw/
│   │   ├── README.md                          # Raw ingestion documentation & Mendeley download guidance
│   │   └── DataCoSupplyChainDataset.csv       # Primary DataCo transaction dataset (35,768 records)
│   └── processed/
│       └── README.md                          # Cleaned data documentation & schema dictionary
├── notebooks/
│   └── SupplySense_AI_Complete_Analysis.ipynb # 12-section executable Jupyter Notebook with embedded outputs
├── src/
│   ├── __init__.py                            # Package initializer re-exporting core modules
│   ├── data_loader.py                         # Robust multi-encoding loader and schema audit
│   ├── data_cleaning.py                       # Canonical column normalization, date parsing & filters
│   ├── exploratory_analysis.py                # Statistical metric calculators & publication-grade plots
│   ├── feature_engineering.py                 # Calendar features, continuous resampling & backward lags
│   ├── forecasting.py                         # Chronological time-series forecasting & multi-step horizons
│   ├── delay_prediction.py                    # Pre-dispatch classifier, feature importance & what-if tool
│   ├── inventory_risk.py                      # Rule-based CV exposure tiers & regional benchmarks
│   └── recommendations.py                     # Synthesized operational recommendations & priority matrix
├── models/
│   ├── README.md                              # Model registry architecture and persistence guidelines
│   └── delay_classifier.joblib                # Persisted Random Forest pre-dispatch classification pipeline
├── outputs/
│   ├── README.md                              # Summary of generated figures and analytical tables
│   ├── figures/                               # Exported static publication plots (PNG)
│   └── tables/                                # Exported metric summaries and recommendation CSVs
└── tests/
    ├── test_data_processing.py                # Unit tests for loading, normalization, dates, and filters
    ├── test_forecasting.py                    # Unit tests for time series, lags, chronological split & MAE
    └── test_recommendations.py                # Unit tests for delay classifier, inventory risk & decisions
```

---

## 2. Dataset Acquisition & Attribution

- **Dataset Name:** DataCo Smart Supply Chain Dataset
- **Authors:** Fabian Constante, Fernando Silva, Antonio Pereira
- **Source Repository:** [Mendeley Data Repository](https://data.mendeley.com/datasets/8gx2fvg2k6/1)
- **Digital Object Identifier (DOI):** [10.17632/8gx2fvg2k6.1](https://doi.org/10.17632/8gx2fvg2k6.1)
- **License / Terms of Use:** Creative Commons Attribution 4.0 International (CC BY 4.0). Redistribution and academic research usage permitted with attribution.
- **Expected Local Path:** `data/raw/DataCoSupplyChainDataset.csv`

### Manual Download & Setup Instructions
1. Navigate to the official Mendeley Data publication: https://data.mendeley.com/datasets/8gx2fvg2k6/1
2. Download `DataCoSupplyChainDataset.csv`.
3. Place the downloaded file into `data/raw/DataCoSupplyChainDataset.csv`.
4. The local application and notebook will automatically discover the file at this location.

---

## 3. Installation & Environment Setup (Windows PowerShell)

Follow these exact steps in **Windows PowerShell** to establish the virtual environment and install all dependencies:

```powershell
# 1. Create a dedicated Python virtual environment
py -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. Running the Tests

To verify pipeline integrity, execute pytest across the unit test suites:

```powershell
python -m pytest tests/ -v
```

All 35 automated tests cover:
- CSV ingestion, column normalization, duplicate tracking, date parsing.
- Backward-shifted lag features and rolling averages without future-data leakage.
- Chronological train/test split (no random shuffling).
- Regression metrics (MAE, RMSE, non-zero MAPE) and recursive multi-step forecasting.
- Pre-dispatch classification pipelines, confusion matrices, and ROC-AUC.
- Rule-based category inventory risk tiers and prioritized recommendation synthesis.

---

## 5. Running the Jupyter Notebook

The project includes an executable, 12-section Jupyter Notebook covering the complete Data Science Lifecycle:

```powershell
jupyter notebook notebooks/SupplySense_AI_Complete_Analysis.ipynb
```

### Notebook Sections Summary:
1. **Section 1 — Project Introduction:** Problem statement, research questions, library setup.
2. **Section 2 — Data Collection:** Ingestion of raw DataCo CSV, schema inspection, dtypes.
3. **Section 3 — Data Quality Analysis:** Missing value percentages, duplicate counts, statistical distributions.
4. **Section 4 — Data Cleaning:** Snake_case normalization, date parsing, zero-loss row preservation.
5. **Section 5 — Exploratory Data Analysis:** Publication-grade visualizations of order quantities, sales, categories, monthly seasonality, delivery status, and transit gap.
6. **Section 6 — Feature Engineering:** Calendar features, daily timeline aggregation, backward-looking lag (1..14) and rolling (7d, 14d) statistics.
7. **Section 7 — Demand Forecasting:** 7-Day Moving Average Baseline vs. Linear Regression vs. Random Forest Regressor on held-out test data; recursive 14-day future projections.
8. **Section 8 — Shipment Delay Prediction:** Stratified pre-dispatch Random Forest and Logistic Regression classifiers, confusion matrix, ROC-AUC, and feature importance drivers.
9. **Section 9 — Inventory Risk Analysis:** Category demand Coefficient of Variation ($CV$), lead-time variance, and rule-based risk assignment.
10. **Section 10 — Regional Performance:** Geographic delivery benchmarks, carrier shipping mode efficiency comparisons.
11. **Section 11 — Results & Conclusions:** Synthesis of empirical findings, operational recommendations, and future enhancements.
12. **Section 12 — Export Outputs:** Serialization of analytical tables to `outputs/tables/` and trained pipeline to `models/delay_classifier.joblib`.

---

## 6. Running the Interactive Streamlit Website

Launch the interactive, multi-dashboard platform locally:

```powershell
streamlit run app.py
```

### The Six Functional Dashboards:
1. **📊 Dashboard 1 — Overview:** Network throughput KPIs (Orders, Revenue, Late Delivery %, Average Transit Duration), weekly demand timeline, delivery status breakdown, and dataset quality health report.
2. **📈 Dashboard 2 — Demand Forecast:** Category filter, historical timeline, actual vs. predicted out-of-sample demand comparison, model error benchmark table (MAE, RMSE, MAPE), recursive future forecast horizon slider, and CSV forecast export.
3. **🛡️ Dashboard 3 — Inventory Risk:** Demand Coefficient of Variation ($CV$) vs. fulfillment delay exposure matrix, category risk classification (High, Medium, Low), and actionable buffer stock recommendations.
4. **🚚 Dashboard 4 — Shipment Delay Risk:** Historical delay indicators, carrier channel distribution, pre-dispatch classification metrics, confusion matrix, and an interactive **Live What-If Order Risk Estimator**.
5. **🌐 Dashboard 5 — Supplier & Regional Performance:** Regional order volumes, late delivery rates, shipping mode transit duration comparisons, and regional efficiency ranking table.
6. **📋 Dashboard 6 — Recommendations & Filters:** Global sidebar filters (Date Range, Region, Category, Shipping Mode), evidence-backed operational recommendations table with priority badges, and one-click filtered data export.

---

## 7. Machine Learning Benchmark Results

All models were evaluated on strictly held-out test partitions:

### Demand Forecasting (Held-Out Test Set)
| Model | MAE (Units) | RMSE (Units) | Non-Zero MAPE (%) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **7-Day Moving Average Baseline** | **10.73** | **13.56** | **23.4%** | Optimal robustness on high-frequency noise |
| **Multivariate Linear Regression** | 35.07 | 45.12 | 68.2% | Linear trend and day-of-week seasonality |
| **Random Forest Regressor** | 49.27 | 61.80 | 95.7% | Non-linear tree ensemble on lag features |

### Shipment Delay Risk Classification (Pre-Dispatch Held-Out Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **71.5%** | **86.8%** | **59.6%** | **70.7%** | **0.761** |
| **Logistic Regression (Baseline)** | 69.4% | 76.5% | 63.8% | 69.6% | 0.748 |
| **Dummy Baseline (Most Frequent)** | 54.8% | 54.8% | 100.0% | 70.8% | 0.500 |

*Strict Leakage Prevention:* Only features available prior to package departure are provided to the classifier.

---

## 8. Analytical Limitations & Governance

1. **Inventory Telemetry:** The DataCo dataset records completed transaction logs but does not track physical warehouse stock-on-hand ($SOH$), bin capacities, or replenishment lead times. The inventory risk scores represent an empirical **Demand & Lead-Time Volatility Exposure Model**, not real-time stockout probabilities.
2. **Supplier Tracking:** The dataset lacks distinct manufacturing plant or vendor entity IDs. Carrier fulfillment channels and geographic destination markets are evaluated rather than misrepresenting customer IDs as suppliers.
