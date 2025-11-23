"""SU4（代入影響トレース特徴量）の生成ロジック。

本モジュールは欠損補完の副作用を明示的に特徴化する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


def _infer_group(column_name: str) -> str | None:
	"""列名の接頭辞から特徴グループを推定する。"""
	prefix_chars: list[str] = []
	for char in column_name:
		if char.isalpha() and char.isupper():
			prefix_chars.append(char)
			continue
		if char.isdigit():
			break
		# 英数字以外が出現した場合は規約外列とみなす。
		return None

	if not prefix_chars:
		return None

	# 複数文字の接頭辞もそのままグループ識別子とする。
	return "".join(prefix_chars)


@dataclass(frozen=True)
class SU4Config:
	"""SU4（代入影響トレース）特徴生成の設定を保持するデータクラス。"""

	id_column: str
	output_prefix: str

	# 列数制御
	top_k_imp_delta: int
	top_k_holiday_cross: int

	# Winsorization
	winsor_p: float

	# 代入手法リスト
	imp_methods: Tuple[str, ...]

	# fold 境界でリセットするか
	reset_each_fold: bool

	# 型
	dtype_flag: np.dtype
	dtype_int: np.dtype
	dtype_float: np.dtype

	@classmethod
	def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU4Config":
		"""YAML設定から SU4Config を生成する。"""
		id_column = mapping.get("id_column", "date_id")
		output_prefix = mapping.get("output_prefix", "su4")

		# 列数制御
		top_k_imp_delta = int(mapping.get("top_k_imp_delta", 25))
		top_k_holiday_cross = int(mapping.get("top_k_holiday_cross", 10))

		# Winsorization
		winsor_p = float(mapping.get("winsor_p", 0.99))
		if not 0.0 < winsor_p < 1.0:
			raise ValueError(f"winsor_p must be in (0, 1), got {winsor_p}")

		# 代入手法リスト
		imp_methods_list = mapping.get("imp_methods", ["ffill", "mice", "missforest", "ridge_stack", "holiday_bridge", "other"])
		imp_methods = tuple(imp_methods_list)

		# fold境界リセット
		reset_each_fold = bool(mapping.get("reset_each_fold", True))

		# データ型
		dtype_section = mapping.get("dtype", {})
		dtype_flag = np.dtype(dtype_section.get("flag", "uint8"))
		dtype_int = np.dtype(dtype_section.get("int", "int16"))
		dtype_float = np.dtype(dtype_section.get("float", "float32"))

		return cls(
			id_column=id_column,
			output_prefix=output_prefix,
			top_k_imp_delta=top_k_imp_delta,
			top_k_holiday_cross=top_k_holiday_cross,
			winsor_p=winsor_p,
			imp_methods=imp_methods,
			reset_each_fold=reset_each_fold,
			dtype_flag=dtype_flag,
			dtype_int=dtype_int,
			dtype_float=dtype_float,
		)


def load_su4_config(config_path: str | Path) -> SU4Config:
	"""SU4 設定 YAML を読み込み :class:`SU4Config` を生成する。"""
	path = Path(config_path).resolve()
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

	try:
		su4_section = full_cfg["su4"]
	except KeyError as exc:
		raise KeyError("'su4' section is required in feature_generation.yaml") from exc

	return SU4Config.from_mapping(su4_section)


class SU4FeatureGenerator(BaseEstimator, TransformerMixin):
	"""SU4 代入影響トレース特徴量生成器。

	入力: 生データ（raw_data）と補完済みデータ（imputed_data）
	出力: imp_used, imp_delta, imp_method, holiday_cross などの特徴
	"""

	def __init__(self, config: SU4Config):
		self.config = config
		self.target_columns_: Optional[List[str]] = None
		self.top_k_delta_cols_: Optional[List[str]] = None
		self.top_k_holiday_cols_: Optional[List[str]] = None
		self.group_policies_: Optional[Dict[str, str]] = None
		self.feature_names_: Optional[List[str]] = None

	def fit(self, raw_data: pd.DataFrame, imputed_data: pd.DataFrame, y: Any = None) -> "SU4FeatureGenerator":
		"""特徴名の抽出とtop-k列の選択。

		Args:
			raw_data: 補完前の生データ
			imputed_data: 補完済みデータ
			y: unused (sklearn互換のため)

		Returns:
			self
		"""
		if not isinstance(raw_data, pd.DataFrame) or not isinstance(imputed_data, pd.DataFrame):
			raise TypeError("SU4FeatureGenerator expects pandas.DataFrame inputs")

		# 1. 対象列の抽出（M/E/I/P/Sグループ）
		self.target_columns_ = self._extract_target_columns(raw_data)

		# 2. 補完頻度の計算
		imputation_rates = self._compute_imputation_rates(raw_data, imputed_data)

		# 3. top-k 列の選択
		self.top_k_delta_cols_ = self._select_top_k_delta_cols(imputation_rates)
		self.top_k_holiday_cols_ = self._select_top_k_holiday_cols(imputation_rates)

		# 4. グループ別ポリシーの読み込み（preprocess.yamlから）
		self.group_policies_ = self._load_group_policies()

		# 5. feature_names_ を組み立て
		self.feature_names_ = self._build_feature_names()

		return self

	def transform(self, raw_data: pd.DataFrame, imputed_data: pd.DataFrame, su1_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
		"""SU4特徴を生成。

		Args:
			raw_data: 補完前の生データ
			imputed_data: 補完済みデータ
			su1_features: SU1特徴（m/<col>を含む）、holiday_cross用

		Returns:
			SU4特徴のDataFrame
		"""
		if self.target_columns_ is None or self.top_k_delta_cols_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		features: Dict[str, np.ndarray] = {}

		# A. 代入実施フラグ
		imp_used = self._compute_imp_used(raw_data, imputed_data)
		features.update(imp_used)

		# B. 代入差分
		imp_delta = self._compute_imp_delta(raw_data, imputed_data, imp_used)
		features.update(imp_delta)

		# C. 代入絶対差分
		imp_absdelta = self._compute_imp_absdelta(imp_delta)
		features.update(imp_absdelta)

		# D. 代入手法 One-hot
		imp_method = self._compute_imp_method_onehot(raw_data)
		features.update(imp_method)

		# E. 交差特徴（holiday_bridge × m/<col>）
		if su1_features is not None:
			holiday_cross = self._compute_holiday_cross(imp_method, su1_features)
			features.update(holiday_cross)

		return pd.DataFrame(features, index=imputed_data.index)

	def _extract_target_columns(self, raw_data: pd.DataFrame) -> List[str]:
		"""M/E/I/P/Sグループの列を抽出する。"""
		target_groups = {"M", "E", "I", "P", "S"}
		columns = []

		for col in raw_data.columns:
			if col == self.config.id_column:
				continue
			group = _infer_group(col)
			if group in target_groups:
				columns.append(col)

		return sorted(columns)

	def _compute_imputation_rates(self, raw_data: pd.DataFrame, imputed_data: pd.DataFrame) -> pd.Series:
		"""各列の補完実施率を計算する。"""
		rates = {}

		for col in self.target_columns_ or []:
			if col not in raw_data.columns or col not in imputed_data.columns:
				continue

			raw_na = raw_data[col].isna()
			imputed_not_na = ~imputed_data[col].isna()
			imputed = (raw_na & imputed_not_na).sum()
			total = len(raw_data)

			rates[col] = imputed / total if total > 0 else 0.0

		return pd.Series(rates)

	def _select_top_k_delta_cols(self, imputation_rates: pd.Series) -> List[str]:
		"""補完頻度上位k列を選択する（imp_delta用）。"""
		if len(imputation_rates) == 0:
			return []

		k = min(self.config.top_k_imp_delta, len(imputation_rates))
		return list(imputation_rates.nlargest(k).index)

	def _select_top_k_holiday_cols(self, imputation_rates: pd.Series) -> List[str]:
		"""補完頻度上位k列を選択する（holiday_cross用）。"""
		if len(imputation_rates) == 0:
			return []

		k = min(self.config.top_k_holiday_cross, len(imputation_rates))
		return list(imputation_rates.nlargest(k).index)

	def _load_group_policies(self) -> Dict[str, str]:
		"""preprocess.yamlからグループ別ポリシーを読み込む。"""
		# デフォルトのポリシーマッピング（preprocess.yaml に基づく）
		default_policies = {
			"M": "ridge_stack",
			"E": "ridge_stack",
			"I": "ridge_stack",
			"P": "mice",
			"S": "missforest",
		}

		# 実際のpreprocess.yamlを読み込む試みも可能だが、
		# ここではデフォルト値を使用する（エラーハンドリング簡略化のため）
		try:
			config_path = Path("configs/preprocess.yaml")
			if config_path.exists():
				with config_path.open("r", encoding="utf-8") as fh:
					preprocess_cfg = yaml.safe_load(fh) or {}

				# 各グループのポリシーを抽出
				for group_key in ["m_group", "e_group", "i_group", "p_group", "s_group"]:
					if group_key in preprocess_cfg:
						group_cfg = preprocess_cfg[group_key]
						policy = group_cfg.get("policy")
						if policy:
							group_name = group_key.split("_")[0].upper()
							default_policies[group_name] = policy
		except Exception:
			# エラー時はデフォルトを使用
			pass

		return default_policies

	def _build_feature_names(self) -> List[str]:
		"""生成される特徴名のリストを作成。"""
		names: List[str] = []

		# A. 代入実施フラグ
		for col in self.target_columns_ or []:
			names.append(f"imp_used/{col}")

		# B. 代入差分
		for col in self.top_k_delta_cols_ or []:
			names.append(f"imp_delta/{col}")

		# C. 代入絶対差分
		for col in self.top_k_delta_cols_ or []:
			names.append(f"imp_absdelta/{col}")

		# D. 代入手法 One-hot
		for method in self.config.imp_methods:
			names.append(f"imp_method/{method}")

		# E. 交差特徴
		for col in self.top_k_holiday_cols_ or []:
			names.append(f"holiday_bridge_x_m/{col}")

		return names

	def _compute_imp_used(self, raw_data: pd.DataFrame, imputed_data: pd.DataFrame) -> Dict[str, np.ndarray]:
		"""代入実施フラグを生成する。"""
		features = {}

		for col in self.target_columns_ or []:
			if col not in raw_data.columns or col not in imputed_data.columns:
				continue

			raw_na = np.asarray(raw_data[col].isna())
			imputed_not_na = ~np.asarray(imputed_data[col].isna())
			imp_used = (raw_na & imputed_not_na).astype(self.config.dtype_flag)

			features[f"imp_used/{col}"] = imp_used

		return features

	def _compute_imp_delta(
		self, raw_data: pd.DataFrame, imputed_data: pd.DataFrame, imp_used: Dict[str, np.ndarray]
	) -> Dict[str, np.ndarray]:
		"""代入差分を生成する（winsorize含む）。"""
		features = {}

		for col in self.top_k_delta_cols_ or []:
			if col not in raw_data.columns or col not in imputed_data.columns:
				continue

			# delta = imputed - raw (補完された箇所のみ)
			raw_vals = np.asarray(raw_data[col].values)
			imputed_vals = np.asarray(imputed_data[col].values)
			imp_used_key = f"imp_used/{col}"

			if imp_used_key not in imp_used:
				continue

			# 補完された箇所のデルタを計算（NaNを0で置き換える）
			delta_full = imputed_vals - raw_vals
			# NaNを0で置き換え（補完されていない箇所は0にする）
			delta = np.where(
				imp_used[imp_used_key] == 1,
				np.nan_to_num(delta_full, nan=0.0),
				0.0
			)

			# Winsorize（±p%）
			delta = self._winsorize(delta, self.config.winsor_p)

			features[f"imp_delta/{col}"] = delta.astype(self.config.dtype_float)

		return features

	def _compute_imp_absdelta(self, imp_delta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
		"""代入絶対差分を生成する。"""
		features = {}

		for key, values in imp_delta.items():
			if not key.startswith("imp_delta/"):
				continue

			col = key.replace("imp_delta/", "")
			abs_delta = np.abs(values)
			features[f"imp_absdelta/{col}"] = abs_delta.astype(self.config.dtype_float)

		return features

	def _compute_imp_method_onehot(self, raw_data: pd.DataFrame) -> Dict[str, np.ndarray]:
		"""代入手法のOne-hotエンコーディングを生成する。"""
		features = {}
		n_rows = len(raw_data)

		# 初期化（全て0）
		for method in self.config.imp_methods:
			features[f"imp_method/{method}"] = np.zeros(n_rows, dtype=self.config.dtype_flag)

		# 各列のグループに基づいてポリシーを決定
		if self.group_policies_ is None or self.target_columns_ is None:
			return features

		# 各行でどの手法が使われたかをカウント
		method_counts = {method: np.zeros(n_rows, dtype=int) for method in self.config.imp_methods}

		for col in self.target_columns_:
			group = _infer_group(col)
			if group is None:
				continue

			policy = self.group_policies_.get(group, "other")
			if policy not in self.config.imp_methods:
				policy = "other"

			# この列があれば、そのポリシーを使用
			method_counts[policy] += 1

		# 最も使われた手法を各行に割り当て
		for i in range(n_rows):
			max_count = 0
			max_method = "other"
			for method, counts in method_counts.items():
				if counts[i] > max_count:
					max_count = counts[i]
					max_method = method

			if max_count > 0:
				features[f"imp_method/{max_method}"][i] = 1

		return features

	def _compute_holiday_cross(
		self, imp_method: Dict[str, np.ndarray], su1_features: pd.DataFrame
	) -> Dict[str, np.ndarray]:
		"""holiday_bridge × m/<col> の交差特徴を生成する。"""
		features = {}

		# holiday_bridge フラグ
		holiday_flag = imp_method.get("imp_method/holiday_bridge")
		if holiday_flag is None:
			return features

		for col in self.top_k_holiday_cols_ or []:
			m_col = f"m/{col}"
			if m_col not in su1_features.columns:
				continue

			m_flag = su1_features[m_col].values
			cross = (holiday_flag == 1) & (m_flag == 1)
			features[f"holiday_bridge_x_m/{col}"] = cross.astype(self.config.dtype_flag)

		return features

	def _winsorize(self, values: np.ndarray, p: float) -> np.ndarray:
		"""配列をwinsorize（±p%でクリップ）する。"""
		# NaNを除外してパーセンタイルを計算
		valid_mask = ~np.isnan(values)
		valid_values = values[valid_mask]

		if len(valid_values) == 0:
			return values

		lower_p = (1 - p) * 100
		upper_p = p * 100

		lower_bound = np.percentile(valid_values, lower_p)
		upper_bound = np.percentile(valid_values, upper_p)

		return np.clip(values, lower_bound, upper_bound)


class SU4FeatureAugmenter(BaseEstimator, TransformerMixin):
	"""SU4特徴をパイプラインに統合するためのTransformer。

	補完前の生データを内部保持し、補完後のデータと比較してSU4特徴を生成する。
	"""

	def __init__(self, config: SU4Config, raw_data: pd.DataFrame):
		self.config = config
		self.raw_data_ = raw_data
		self.generator_: Optional[SU4FeatureGenerator] = None

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU4FeatureAugmenter":
		"""SU4FeatureGeneratorを初期化・fit。

		Args:
			X: 補完済みデータ
			y: unused

		Returns:
			self
		"""
		self.generator_ = SU4FeatureGenerator(self.config)
		self.generator_.fit(self.raw_data_, X, y)
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""SU4特徴を生成してXに結合。

		Args:
			X: 補完済みデータ（SU1特徴を含む可能性あり）

		Returns:
			X + SU4特徴
		"""
		if self.generator_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		# SU1特徴があれば抽出（holiday_cross用）
		su1_features: Optional[pd.DataFrame] = None
		m_cols = [c for c in X.columns if c.startswith("m/")]
		if m_cols:
			su1_df = X[m_cols]
			if isinstance(su1_df, pd.DataFrame):
				su1_features = su1_df

		su4_features = self.generator_.transform(self.raw_data_, X, su1_features)
		return pd.concat([X, su4_features], axis=1)
