"""Model selection and training modules.

This package provides a unified framework for training and evaluating
multiple model types (LGBM, XGBoost, CatBoost, Ridge) on the FS_compact
feature set.
"""

from src.models.common.cv_utils import (
    create_cv_splits,
    evaluate_oof_predictions,
    compute_fold_metrics,
)
from src.models.common.feature_loader import (
    load_fs_compact_features,
    get_excluded_features,
    apply_feature_exclusion,
)

__all__ = [
    "create_cv_splits",
    "evaluate_oof_predictions",
    "compute_fold_metrics",
    "load_fs_compact_features",
    "get_excluded_features",
    "apply_feature_exclusion",
]
