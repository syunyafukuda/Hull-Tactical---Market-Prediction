"""CatBoost model training and evaluation.

This module provides training and inference for CatBoost models using the FS_compact
feature set and unified preprocessing pipeline.

Key features:
- Ordered Boosting for overfitting resistance
- Compatible with TimeSeriesSplit cross-validation
- Produces artifacts consistent with other model types
"""
