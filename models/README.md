# Serialized Machine Learning Models

This directory stores persisted Scikit-learn pipelines and model metadata trained by SupplySense AI.

## Persisted Models

- **`delay_classifier.joblib`**: A production-ready Scikit-learn `Pipeline` encapsulating a `ColumnTransformer` (standardizing numeric predictors and one-hot encoding categorical variables) and an optimized `RandomForestClassifier` (100 estimators, max depth 12).
  - **Inputs:** Order-time features only (Scheduled Days, Item Quantity, Sales Volume, Shipping Mode, Customer Segment, Order Region).
  - **Target:** `late_delivery_risk` (Binary: 1 = Late, 0 = On Time/Early).
  - **Verification:** Built with zero post-shipment target leakage.

## Loading Example

```python
import joblib

pipeline = joblib.load("models/delay_classifier.joblib")
print("Pipeline loaded successfully:", type(pipeline))
```
