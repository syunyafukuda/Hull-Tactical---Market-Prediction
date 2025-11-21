#!/usr/bin/env python
"""SU3 特徴量バンドルの学習エントリーポイント。

本スクリプトは生データから欠損構造を表現する SU1 + SU3 特徴量を生成し、
軽量な前処理パイプラインを通し、LightGBM 回帰器を学習する。

Note: This is a placeholder implementation demonstrating the SU3 integration concept.
For actual training, the full SU1 training infrastructure should be adapted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Add project paths
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))

from src.feature_generation.su1.feature_su1 import SU1Config, SU1FeatureGenerator  # noqa: E402
from src.feature_generation.su3.feature_su3 import SU3Config, SU3FeatureGenerator  # noqa: E402


class SU3FeatureAugmenter(BaseEstimator, TransformerMixin):
	"""SU1 と SU3 特徴量を統合して入力フレームへ追加するトランスフォーマー。

	SU1 特徴を生成した後に、その出力を使って SU3 特徴を追加生成する。
	"""

	def __init__(
		self,
		su1_config: SU1Config,
		su3_config: SU3Config,
		fill_value: float | None = 0.0,
	) -> None:
		self.su1_config = su1_config
		self.su3_config = su3_config
		self.fill_value = fill_value

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU3FeatureAugmenter":
		"""SU1 と SU3 の生成器を初期化し、特徴名を記録する。

		Args:
			X: 生データの DataFrame
			y: ターゲット（未使用）

		Returns:
			self
		"""
		frame = self._ensure_dataframe(X)

		# SU1 特徴生成
		su1_generator = SU1FeatureGenerator(self.su1_config)
		su1_generator.fit(frame)
		su1_features = su1_generator.transform(frame)
		if self.fill_value is not None:
			su1_features = su1_features.fillna(self.fill_value)

		# SU1特徴を元データに結合
		combined = pd.concat([frame, su1_features], axis=1)

		# SU3 特徴生成
		su3_generator = SU3FeatureGenerator(self.su3_config)
		su3_generator.fit(combined)
		su3_features = su3_generator.transform(combined)
		if self.fill_value is not None:
			su3_features = su3_features.fillna(self.fill_value)

		# 内部状態を保存
		self.su1_generator_ = su1_generator
		self.su3_generator_ = su3_generator
		self.su1_feature_names_ = list(su1_features.columns)
		self.su3_feature_names_ = list(su3_features.columns)
		self.input_columns_ = list(frame.columns)

		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""SU1 と SU3 特徴を生成し、入力データに追加する。

		Args:
			X: 生データの DataFrame

		Returns:
			SU1 と SU3 特徴を追加した DataFrame
		"""
		if not hasattr(self, "su1_generator_") or not hasattr(self, "su3_generator_"):
			raise RuntimeError("SU3FeatureAugmenter must be fitted before transform().")

		frame = self._ensure_dataframe(X)

		# SU1 特徴生成
		su1_features = self.su1_generator_.transform(frame)
		su1_features = su1_features.reindex(columns=self.su1_feature_names_, copy=True)
		if self.fill_value is not None:
			su1_features = su1_features.fillna(self.fill_value)

		# SU1特徴を元データに結合
		combined = pd.concat([frame, su1_features], axis=1)

		# SU3 特徴生成
		su3_features = self.su3_generator_.transform(combined)
		su3_features = su3_features.reindex(columns=self.su3_feature_names_, copy=True)
		if self.fill_value is not None:
			su3_features = su3_features.fillna(self.fill_value)

		# 全特徴を結合
		augmented = pd.concat([frame, su1_features, su3_features], axis=1)
		augmented.index = frame.index

		return augmented

	@staticmethod
	def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
		"""入力が DataFrame であることを確認する。"""
		if not isinstance(X, pd.DataFrame):  # pragma: no cover
			raise TypeError("SU3FeatureAugmenter expects a pandas.DataFrame input")
		return X.copy()


def main() -> None:
	"""メインエントリーポイント。

	Note: This is a placeholder. For actual training, the full training logic
	from train_su1.py should be adapted to use SU3FeatureAugmenter instead of
	SU1FeatureAugmenter.
	"""
	print("SU3 training script placeholder.")
	print("To perform actual training, adapt the full training logic from train_su1.py")
	print("and replace SU1FeatureAugmenter with SU3FeatureAugmenter.")
	print()
	print("Key changes needed:")
	print("1. Load both SU1Config and SU3Config from feature_generation.yaml")
	print("2. Use SU3FeatureAugmenter(su1_config, su3_config) instead of SU1FeatureAugmenter")
	print("3. Add regularization parameters: reg_alpha=0.1, reg_lambda=0.1 to LGBMRegressor")
	print("4. Output artifacts to artifacts/SU3/ directory")


if __name__ == "__main__":
	main()
