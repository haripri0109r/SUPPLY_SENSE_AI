"""Data Ingestion and Loading Module for SupplySense AI.
Handles CSV discovery, robust reading across encodings, schema inspection,
and data quality reporting.
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


DEFAULT_DATASET_PATHS = [
    os.path.join("data", "raw", "DataCoSupplyChainDataset.csv"),
    os.path.join("data", "DataCoSupplyChainDataset.csv"),
    "DataCoSupplyChainDataset.csv",
]


def find_dataset_path(preferred_path: Optional[str] = None) -> Optional[str]:
    """Locate the dataset file across common project locations."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    for candidate in DEFAULT_DATASET_PATHS:
        if os.path.exists(candidate):
            return candidate

    return None


def load_raw_data(
    filepath_or_buffer: Optional[Union[str, Any]] = None,
) -> pd.DataFrame:
    """Load the raw CSV dataset with fallback encoding detection.

    Args:
        filepath_or_buffer: Filepath string, Path object, or file-like buffer.

    Returns:
        pd.DataFrame containing the raw, unmodified data.
    """
    target = filepath_or_buffer
    if target is None:
        target = find_dataset_path()
        if target is None:
            raise FileNotFoundError(
                "DataCo Supply Chain dataset not found. Please place 'DataCoSupplyChainDataset.csv' "
                "in 'data/raw/' or specify a valid file path."
            )

    if isinstance(target, str):
        try:
            return pd.read_csv(target, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(target, encoding="latin1")
    else:
        # Buffer / upload object
        try:
            return pd.read_csv(target, encoding="utf-8")
        except UnicodeDecodeError:
            if hasattr(target, "seek"):
                target.seek(0)
            return pd.read_csv(target, encoding="latin1")


def inspect_dataset_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality statistics and warnings.

    Args:
        df: Input DataFrame to evaluate.

    Returns:
        Dictionary with dimensions, missingness, duplicate rates, and warning flags.
    """
    warnings: List[str] = []
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / max(len(df), 1) * 100).round(2)
    duplicates = int(df.duplicated().sum())

    if duplicates > 0:
        warnings.append(
            f"Detected {duplicates:,} duplicate rows ({duplicates/max(len(df),1)*100:.1f}%)."
        )

    high_missing = missing_pct[missing_pct > 20]
    for col, pct in high_missing.items():
        warnings.append(f"Column '{col}' has {pct}% missing values.")

    # Check for negative numbers in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if "quantity" in col.lower() and (df[col] < 0).any():
            warnings.append(f"Negative values detected in quantity column '{col}'.")
        if "sales" in col.lower() and (df[col] < 0).any():
            warnings.append(f"Negative sales values detected in '{col}'.")

    return {
        "num_rows": len(df),
        "total_rows": len(df),
        "num_cols": df.shape[1],
        "total_columns": df.shape[1],
        "missing_counts": missing_counts.to_dict(),
        "missing_by_column": missing_counts.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "duplicate_rows": duplicates,
        "duplicate_pct": round(duplicates / max(len(df), 1) * 100, 2),
        "warnings": warnings,
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "column_names": list(df.columns),
    }


def load_dataset(
    filepath_or_buffer: Optional[Union[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Complete ingestion entry point: loads raw data, computes quality profile,
    and returns cleaned dataframe using data_cleaning pipeline.

    Returns:
        (raw_df, cleaned_df, metadata_dict)
    """
    # Lazy import to avoid circular dependency
    from src.data_cleaning import clean_supply_chain_data

    raw_df = load_raw_data(filepath_or_buffer)
    quality_summary = inspect_dataset_quality(raw_df)
    cleaned_df, cleaning_report = clean_supply_chain_data(raw_df)

    metadata = {
        "raw_shape": raw_df.shape,
        "cleaned_shape": cleaned_df.shape,
        "quality_summary": quality_summary,
        "cleaning_report": cleaning_report,
    }
    return raw_df, cleaned_df, metadata
