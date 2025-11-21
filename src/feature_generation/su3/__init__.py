"""SU3（欠損構造三次特徴量）の生成ロジック。

本モジュールは ``docs/feature_generation/SU3.md`` に記載された方針を実装し、
SU1の出力を入力として、遷移・再出現・代入影響を捕捉する特徴量を生成する。
"""

from __future__ import annotations

from src.feature_generation.su3.feature_su3 import (
    SU3Config,
    SU3FeatureGenerator,
    load_su3_config,
)

__all__ = [
    "SU3Config",
    "SU3FeatureGenerator",
    "load_su3_config",
]
