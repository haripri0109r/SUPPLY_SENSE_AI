"""Shipment Delay Risk Module for SupplySense AI.
Builds and evaluates machine learning classification models to predict Late Delivery Risk
strictly using pre-shipment features, preventing data leakage.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# STRICT EXCLUSION: Fields that leak actual delivery outcome or occur post-order
LEAKAGE_COLUMNS = [
    "days_for_shipping_real",
    "days_real",
    "days_real_clean",
    "shipping_date_dateorders",
    "shipping_date_parsed",
    "delivery_status",
    "delivery_status_clean",
    "order_status",
]


def prepare_delay_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Extract valid pre-shipment features and binary target, preventing data leakage."""
    if "late_delivery_risk_clean" not in df.columns:
        raise ValueError("Target 'late_delivery_risk_clean' is missing from dataframe.")

    y = df["late_delivery_risk_clean"].astype(int)

    # Candidate feature columns available at order placement
    numerical_candidates = [
        "days_scheduled_clean",
        "order_item_quantity_clean",
        "sales_clean",
    ]
    categorical_candidates = [
        "shipping_mode_clean",
        "region_clean",
        "category_clean",
    ]

    # Optional additional features if present in raw/cleaned dataframe
    if "customer_segment" in df.columns:
        categorical_candidates.append("customer_segment")
    if "market" in df.columns:
        categorical_candidates.append("market")
    if "order_month" in df.columns:
        numerical_candidates.append("order_month")

    # Filter to existing columns and ensure none are in LEAKAGE_COLUMNS
    num_features = [c for c in numerical_candidates if c in df.columns and c not in LEAKAGE_COLUMNS]
    cat_features = [c for c in categorical_candidates if c in df.columns and c not in LEAKAGE_COLUMNS]

    all_features = num_features + cat_features
    X = df[all_features].copy()

    # Fill categorical missing values
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    # Fill numeric missing values
    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X, y, num_features, cat_features


def train_delay_classification_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Baseline and Random Forest delay risk classifiers with a leakage-free pipeline.

    Returns:
        Evaluation metrics, confusion matrices, ROC-AUC, trained pipelines, and test sets.
    """
    X, y, num_features, cat_features = prepare_delay_features(df)

    if len(y.unique()) < 2:
        return {
            "success": False,
            "message": "Delay risk target contains only one unique class; classification requires at least two classes.",
        }

    # Split into train and test sets (stratified by class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build preprocessing ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )

    results: Dict[str, Any] = {
        "success": True,
        "models": {},
        "metrics": {},
        "confusion_matrices": {},
        "roc_curves": {},
        "feature_names": {"numeric": num_features, "categorical": cat_features},
        "test_size": len(y_test),
        "train_size": len(y_train),
        "class_distribution": y.value_counts(normalize=True).to_dict(),
    }

    # 1. Baseline Model: Most Frequent Dummy Classifier
    dummy_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    dummy_pipeline.fit(X_train, y_train)
    dummy_preds = dummy_pipeline.predict(X_test)
    results["models"]["Baseline (Most Frequent)"] = dummy_pipeline
    results["metrics"]["Baseline (Most Frequent)"] = {
        "Accuracy": round(float(accuracy_score(y_test, dummy_preds)), 4),
        "Precision": round(float(precision_score(y_test, dummy_preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, dummy_preds, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_test, dummy_preds, zero_division=0)), 4),
        "ROC-AUC": 0.5,
    }
    results["confusion_matrices"]["Baseline (Most Frequent)"] = confusion_matrix(y_test, dummy_preds).tolist()

    # 2. Linear Baseline Model: Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=500, random_state=random_state)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1] if hasattr(lr_pipeline, "predict_proba") else lr_preds

    results["models"]["Logistic Regression (Baseline)"] = lr_pipeline
    results["metrics"]["Logistic Regression (Baseline)"] = {
        "Accuracy": round(float(accuracy_score(y_test, lr_preds)), 4),
        "Precision": round(float(precision_score(y_test, lr_preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, lr_preds, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_test, lr_preds, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, lr_probs)), 4),
    }
    results["confusion_matrices"]["Logistic Regression (Baseline)"] = confusion_matrix(y_test, lr_preds).tolist()

    # 3. Random Forest Classifier
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

    results["models"]["Random Forest"] = rf_pipeline
    results["metrics"]["Random Forest"] = {
        "Accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
        "Precision": round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, rf_probs)), 4),
    }
    results["confusion_matrices"]["Random Forest"] = confusion_matrix(y_test, rf_preds).tolist()
    results["confusion_matrix"] = results["confusion_matrices"]["Random Forest"]
    results["pipelines"] = results["models"]

    results["confusion_matrices"]["Random Forest"] = confusion_matrix(y_test, rf_preds).tolist()

    # Feature Importance for Random Forest
    try:
        rf_model = rf_pipeline.named_steps["classifier"]
        fitted_pre = rf_pipeline.named_steps["preprocessor"]
        cat_names = list(fitted_pre.named_transformers_["cat"].get_feature_names_out(cat_features))
        all_feat_names = num_features + cat_names
        importances = rf_model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": all_feat_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(15)
        results["feature_importances"] = feat_imp_df
    except Exception:
        results["feature_importances"] = pd.DataFrame()

    results["best_model_name"] = "Random Forest"
    results["best_pipeline"] = rf_pipeline
    return results


def predict_single_order_delay(
    pipeline: Pipeline,
    order_data: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Predict delay risk for an individual order scenario."""
    payload = {}
    if order_data is not None:
        payload.update(order_data)
    payload.update(kwargs)

    # Map standard aliases
    alias_map = {
        "shipping_mode": "shipping_mode_clean",
        "region": "region_clean",
        "category": "category_clean",
        "days_scheduled": "days_scheduled_clean",
        "order_item_quantity": "order_item_quantity_clean",
        "sales": "sales_clean",
    }
    for old_k, new_k in alias_map.items():
        if old_k in payload and new_k not in payload:
            payload[new_k] = payload[old_k]

    input_df = pd.DataFrame([payload])

    # Ensure all columns expected by the trained pipeline are present
    if hasattr(pipeline, "feature_names_in_"):
        expected_cols = pipeline.feature_names_in_
    elif hasattr(pipeline.named_steps.get("preprocessor"), "feature_names_in_"):
        expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_
    else:
        expected_cols = []

    for col in expected_cols:
        if col not in input_df.columns:
            if "month" in col:
                input_df[col] = 1
            elif "days" in col or "quantity" in col or "sales" in col:
                input_df[col] = 1.0
            else:
                input_df[col] = "Unknown"

    pred = int(pipeline.predict(input_df)[0])
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    return {
        "predicted_class": pred,
        "is_delayed_predicted": bool(pred == 1),
        "delay_probability": round(prob, 4),
        "risk_label": "High Delay Risk" if prob >= 0.6 else ("Moderate Risk" if prob >= 0.4 else "Low Delay Risk"),
    }


predict_single_order_delay_risk = predict_single_order_delay


def compute_historical_delay_breakdowns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate actual historical delay statistics by Region and Shipping Mode."""
    results = {}
    if "region_clean" in df.columns and "late_delivery_risk_clean" in df.columns:
        region_stats = df.groupby("region_clean").agg(
            total_orders=("late_delivery_risk_clean", "count"),
            late_orders=("late_delivery_risk_clean", "sum"),
            late_rate=("late_delivery_risk_clean", "mean"),
            avg_scheduled_days=("days_scheduled_clean", "mean"),
            avg_real_days=("days_real_clean", "mean"),
        ).reset_index()
        region_stats["late_rate_pct"] = (region_stats["late_rate"] * 100).round(1)
        region_stats["avg_scheduled_days"] = region_stats["avg_scheduled_days"].round(1)
        region_stats["avg_real_days"] = region_stats["avg_real_days"].round(1)
        results["by_region"] = region_stats.sort_values("late_rate", ascending=False)

    if "shipping_mode_clean" in df.columns and "late_delivery_risk_clean" in df.columns:
        mode_stats = df.groupby("shipping_mode_clean").agg(
            total_orders=("late_delivery_risk_clean", "count"),
            late_orders=("late_delivery_risk_clean", "sum"),
            late_rate=("late_delivery_risk_clean", "mean"),
            avg_scheduled_days=("days_scheduled_clean", "mean"),
            avg_real_days=("days_real_clean", "mean"),
        ).reset_index()
        mode_stats["late_rate_pct"] = (mode_stats["late_rate"] * 100).round(1)
        mode_stats["avg_scheduled_days"] = mode_stats["avg_scheduled_days"].round(1)
        mode_stats["avg_real_days"] = mode_stats["avg_real_days"].round(1)
        results["by_shipping_mode"] = mode_stats.sort_values("late_rate", ascending=False)

    return results
