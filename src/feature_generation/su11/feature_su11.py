"""SU11 メタ特徴量ビルダー / 設定クラス。

Level-1 モデル（SU1+SU5+LGBM）の OOF 予測値を特徴量とし、
Level-2 モデルで最終予測を行う 2 段階スタッキングの基盤となるクラス群を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
	from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SU11Config:
	"""SU11 スタッキング設定。

	Attributes
	----------
	level1_artifacts_dir : str
		Level-1 モデルのアーティファクト格納ディレクトリ（例: artifacts/SU5）。
	level2_model_type : str
		Level-2 モデルの種類。"ridge" | "lgbm" | "identity" のいずれか。
		"identity" は Level-1 出力をそのまま利用（デバッグ用）。
	use_extra_features : bool
		Level-2 で y_pred_L1 以外の追加特徴を使用するかどうか。
	extra_feature_names : list[str]
		使用する追加特徴名（use_extra_features=True の場合のみ有効）。
	ridge_alpha : float
		Ridge 回帰の正則化強度（level2_model_type="ridge" の場合）。
	lgbm_n_estimators : int
		LGBM のブースティング回数（level2_model_type="lgbm" の場合）。
	lgbm_max_depth : int
		LGBM の最大深さ（level2_model_type="lgbm" の場合）。
	n_splits : int
		Level-2 の CV 分割数。
	random_state : int
		乱数シード。
	"""

	level1_artifacts_dir: str = "artifacts/SU5"
	level2_model_type: str = "ridge"
	use_extra_features: bool = False
	extra_feature_names: list[str] = field(default_factory=list)
	ridge_alpha: float = 0.001
	lgbm_n_estimators: int = 50
	lgbm_max_depth: int = 3
	n_splits: int = 5
	random_state: int = 42

	def __post_init__(self) -> None:
		valid_types = {"ridge", "lgbm", "identity"}
		if self.level2_model_type not in valid_types:
			msg = f"level2_model_type must be one of {valid_types}, got {self.level2_model_type!r}"
			raise ValueError(msg)


def load_su11_config(config_path: str | Path, *, section: str = "su11") -> SU11Config:
	"""YAML から SU11Config をロードする。

	Parameters
	----------
	config_path : str | Path
		設定ファイルのパス。
	section : str
		読み込む YAML セクション名。

	Returns
	-------
	SU11Config
		設定オブジェクト。
	"""
	with Path(config_path).open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)

	if section not in data:
		return SU11Config()

	sec = data[section]
	return SU11Config(
		level1_artifacts_dir=sec.get("level1_artifacts_dir", "artifacts/SU5"),
		level2_model_type=sec.get("level2_model_type", "ridge"),
		use_extra_features=sec.get("use_extra_features", False),
		extra_feature_names=sec.get("extra_feature_names", []),
		ridge_alpha=sec.get("ridge_alpha", 0.001),
		lgbm_n_estimators=sec.get("lgbm_n_estimators", 50),
		lgbm_max_depth=sec.get("lgbm_max_depth", 3),
		n_splits=sec.get("n_splits", 5),
		random_state=sec.get("random_state", 42),
	)


# ---------------------------------------------------------------------------
# Meta Feature Builder
# ---------------------------------------------------------------------------
class SU11MetaFeatureBuilder:
	"""Level-2 用データセット構築クラス。

	Level-1 モデルの OOF/test 予測値から Level-2 モデル学習用のデータセットを構築する。

	Parameters
	----------
	config : SU11Config
		設定オブジェクト。
	"""

	def __init__(self, config: SU11Config) -> None:
		self.config = config
		self._level1_oof: pd.DataFrame | None = None
		self._level1_test_pred: pd.Series | None = None

	def load_level1_artifacts(self) -> None:
		"""Level-1 アーティファクトから OOF 予測とテスト予測を読み込む。"""
		artifacts_dir = Path(self.config.level1_artifacts_dir)

		# OOF predictions
		oof_path = artifacts_dir / "oof_predictions.csv"
		if not oof_path.exists():
			msg = f"OOF predictions not found: {oof_path}"
			raise FileNotFoundError(msg)
		self._level1_oof = pd.read_csv(oof_path)

		# Test predictions (from submission.csv)
		sub_path = artifacts_dir / "submission.csv"
		if sub_path.exists():
			sub_df = pd.read_csv(sub_path)
			if "prediction" in sub_df.columns:
				self._level1_test_pred = pd.Series(sub_df["prediction"])
			elif "market_forward_excess_returns" in sub_df.columns:
				self._level1_test_pred = pd.Series(sub_df["market_forward_excess_returns"])

	@property
	def level1_oof(self) -> pd.DataFrame:
		"""Level-1 OOF 予測データフレーム。"""
		if self._level1_oof is None:
			self.load_level1_artifacts()
		assert self._level1_oof is not None
		return self._level1_oof

	@property
	def level1_test_pred(self) -> pd.Series | None:
		"""Level-1 テスト予測（利用可能な場合）。"""
		if self._level1_test_pred is None and self._level1_oof is None:
			self.load_level1_artifacts()
		return self._level1_test_pred

	def build_level2_train(
		self,
		*,
		X_extra: pd.DataFrame | None = None,
	) -> tuple[pd.DataFrame, pd.Series]:
		"""Level-2 学習用データセットを構築する。

		Parameters
		----------
		X_extra : pd.DataFrame | None
			追加特徴量（オプション）。インデックスは OOF と揃っている必要がある。

		Returns
		-------
		X_level2 : pd.DataFrame
			Level-2 用の特徴量データフレーム。
		y_level2 : pd.Series
			Level-2 用の目的変数（y_true）。
		"""
		oof_df = self.level1_oof

		# 基本特徴: y_pred_L1
		X_level2 = pd.DataFrame({"y_pred_L1": oof_df["y_pred"].values})

		# 追加特徴（オプション）
		if self.config.use_extra_features and X_extra is not None:
			# OOF のインデックスに基づいてサブセット化
			row_indices = oof_df["row_index"].values
			extra_subset = X_extra.iloc[row_indices].reset_index(drop=True)
			for col in self.config.extra_feature_names:
				if col in extra_subset.columns:
					X_level2[col] = extra_subset[col].values

		# 目的変数
		y_level2 = pd.Series(oof_df["y_true"].values, name="y_true")

		return X_level2, y_level2

	def build_level2_test(
		self,
		test_pred_L1: pd.Series | NDArray[np.floating[Any]] | None = None,
		*,
		X_extra: pd.DataFrame | None = None,
	) -> pd.DataFrame:
		"""Level-2 推論用データセットを構築する。

		Parameters
		----------
		test_pred_L1 : pd.Series | NDArray | None
			Level-1 のテスト予測値。None の場合はロード済みの値を使用。
		X_extra : pd.DataFrame | None
			追加特徴量（オプション）。

		Returns
		-------
		X_level2 : pd.DataFrame
			Level-2 用の特徴量データフレーム。
		"""
		if test_pred_L1 is None:
			test_pred_L1 = self.level1_test_pred
		if test_pred_L1 is None:
			msg = "Level-1 test predictions not available."
			raise ValueError(msg)

		# numpy 配列の場合は Series に変換
		if isinstance(test_pred_L1, np.ndarray):
			test_pred_L1 = pd.Series(test_pred_L1)

		X_level2 = pd.DataFrame({"y_pred_L1": test_pred_L1.values})

		# 追加特徴（オプション）
		if self.config.use_extra_features and X_extra is not None:
			for col in self.config.extra_feature_names:
				if col in X_extra.columns:
					X_level2[col] = X_extra[col].values

		return X_level2

	def get_fold_mapping(self) -> dict[int, list[int]]:
		"""OOF データから fold ごとのインデックスマッピングを取得。

		Returns
		-------
		dict[int, list[int]]
			fold 番号 → 行インデックスのリスト。
		"""
		oof_df = self.level1_oof
		mapping: dict[int, list[int]] = {}
		for _, row in oof_df.iterrows():
			fold_val = row.get("fold")
			fold = int(fold_val) if fold_val is not None and pd.notna(fold_val) else -1
			if fold not in mapping:
				mapping[fold] = []
			mapping[fold].append(int(row["row_index"]))
		return mapping
