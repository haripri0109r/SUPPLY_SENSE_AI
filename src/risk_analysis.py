"""Backward-compatibility module forwarding to src.inventory_risk."""

from src.inventory_risk import (
    INVENTORY_DATA_LIMITATION_DISCLAIMER,
    SUPPLIER_DATA_LIMITATION_DISCLAIMER,
    calculate_category_inventory_risk,
    calculate_regional_performance_benchmarks,
)

__all__ = [
    "INVENTORY_DATA_LIMITATION_DISCLAIMER",
    "SUPPLIER_DATA_LIMITATION_DISCLAIMER",
    "calculate_category_inventory_risk",
    "calculate_regional_performance_benchmarks",
]
