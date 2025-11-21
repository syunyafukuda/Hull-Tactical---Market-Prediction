"""SU2（欠損構造二次特徴量）の生成モジュール。

本モジュールは SU1 出力を入力として、ローリング統計、EWMA、遷移統計、正規化を
過去のみで算出する二次特徴量を生成する。
"""

from __future__ import annotations

from src.feature_generation.su2.feature_su2 import (
	SU2Config,
	SU2FeatureGenerator,
	generate_su2_features,
	load_su2_config,
)

__all__ = [
	"SU2Config",
	"SU2FeatureGenerator",
	"generate_su2_features",
	"load_su2_config",
]
