"""SU11 (メタモデル・スタッキング) モジュール。

本モジュールは、Level-1 モデル（SU1+SU5+LGBM）の OOF 予測値を特徴量として利用し、
Level-2 モデル（Ridge または浅い LGBM）で最終予測を行う 2 段階スタッキングを実装する。
"""

from __future__ import annotations

from src.feature_generation.su11.feature_su11 import (
	SU11Config,
	SU11MetaFeatureBuilder,
	load_su11_config,
)

__all__ = [
	"SU11Config",
	"SU11MetaFeatureBuilder",
	"load_su11_config",
]
