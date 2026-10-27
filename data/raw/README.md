# DataCo Smart Supply Chain Dataset — Raw Data Directory

This directory contains the original, unaltered transaction dataset for the **SupplySense AI: Predictive Supply Chain Intelligence Platform**.

## Dataset Source & Attribution

- **Dataset Name:** DataCo Smart Supply Chain for Big Data Analysis
- **Authors:** Fabian Constante, Fernando Silva, and António Pereira
- **Repository:** Mendeley Data, V1
- **DOI:** [10.17632/8gx2fvg2k6.1](https://doi.org/10.17632/8gx2fvg2k6.1)
- **Publication Citation:** 
  > Constante, F., Silva, F., & Pereira, A. (2019). "DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS." *Data in Brief*, 25, 104083. https://doi.org/10.1016/j.dib.2019.104083
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset Specifications

- **Expected Filename:** `DataCoSupplyChainDataset.csv`
- **File Format:** Comma-Separated Values (CSV)
- **Encoding:** UTF-8 or ISO-8859-1 (Latin-1)
- **Dimensions:** 35,768 order transactions × 53 raw operational features
- **Temporal Coverage:** January 2015 – January 2018
- **Primary Operational Areas:**
  - Order details (Sales, Quantity, Discounts, Profit, Product Price)
  - Logistics & Shipping (Days for shipping scheduled vs. real, Shipping Mode, Delivery Status)
  - Customer Information (Segment, City, State, Country, Market, Region)
  - Product Categorization (Category Name, Department, Product Name)
  - Risk Target (`Late_delivery_risk`: 1 if shipment was delayed, 0 otherwise)

## Manual Acquisition Instructions

If acquiring the dataset manually:
1. Navigate to the official Mendeley Data repository: https://data.mendeley.com/datasets/8gx2fvg2k6/1
2. Download `DataCoSupplyChainDataset.csv`.
3. Place the downloaded CSV file into this directory: `data/raw/DataCoSupplyChainDataset.csv`.
4. Run `python -c "import pandas as pd; df=pd.read_csv('data/raw/DataCoSupplyChainDataset.csv', encoding='latin1'); print(df.shape)"` to verify successful loading.
