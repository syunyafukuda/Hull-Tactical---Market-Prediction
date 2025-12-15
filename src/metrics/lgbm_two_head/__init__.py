"""Two-Head LightGBM module for Hull Competition.

This module provides separate training and prediction scripts for the
two-head learning approach (forward_returns + risk_free_rate prediction).
"""

from __future__ import annotations

__all__ = [
    "train_lgbm_two_head",
    "predict_lgbm_two_head",
]
