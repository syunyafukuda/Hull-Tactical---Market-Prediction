"""Metrics package for model evaluation.

This package provides evaluation metrics for the Hull Tactical competition:
- Hull Competition Sharpe (official LB metric)
- RMSE (reference metric)
- MSR proxy (legacy)
"""

from src.metrics.hull_sharpe import (
    HullSharpeResult,
    compute_hull_sharpe,
    evaluate_hull_metric,
    hull_sharpe_score,
    validate_prediction,
)

__all__ = [
    "HullSharpeResult",
    "compute_hull_sharpe",
    "evaluate_hull_metric",
    "hull_sharpe_score",
    "validate_prediction",
]
