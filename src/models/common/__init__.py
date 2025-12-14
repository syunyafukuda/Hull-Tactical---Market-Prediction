"""Common utilities for model training and evaluation."""

from src.models.common.cv_utils import (
    compute_fold_metrics,
    create_cv_splits,
    evaluate_oof_predictions,
)
from src.models.common.feature_loader import (
    apply_feature_exclusion,
    get_excluded_features,
    load_fs_compact_features,
)
from src.models.common.signals import (
    PositionMapperConfig,
    analyze_position_distribution,
    calibrate_position_mapper,
    map_to_position,
    map_to_position_from_config,
)
from src.models.common.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    estimate_fold_count,
    make_walk_forward_splits,
    make_walk_forward_splits_df,
)

__all__ = [
    # cv_utils
    "create_cv_splits",
    "evaluate_oof_predictions",
    "compute_fold_metrics",
    # feature_loader
    "load_fs_compact_features",
    "get_excluded_features",
    "apply_feature_exclusion",
    # signals
    "PositionMapperConfig",
    "map_to_position",
    "map_to_position_from_config",
    "calibrate_position_mapper",
    "analyze_position_distribution",
    # walk_forward
    "WalkForwardConfig",
    "WalkForwardFold",
    "make_walk_forward_splits",
    "make_walk_forward_splits_df",
    "estimate_fold_count",
]
