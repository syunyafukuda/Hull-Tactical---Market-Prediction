"""SU10 (外部レジーム特徴量) モジュール。

本モジュールは、SPY Historical Data から算出したボラティリティ・トレンドレジーム特徴を生成する。
train/test 内部データとは独立した情報軸を提供する。
"""

from __future__ import annotations

from src.feature_generation.su10.feature_su10 import (
    SU10Config,
    SU10FeatureGenerator,
    SU10_FEATURE_COLUMNS,
    load_su10_config,
)

__all__ = [
    "SU10Config",
    "SU10FeatureGenerator",
    "SU10_FEATURE_COLUMNS",
    "load_su10_config",
]
