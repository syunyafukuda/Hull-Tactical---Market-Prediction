"""SU3（欠損構造三次特徴量）の生成ロジック。

本モジュールは SU1 の出力を入力として、欠損パターンの時間的変化を捕捉する。
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
class SU3Config:
	"""SU3特徴量生成に必要な設定を保持するデータクラス。"""

	# 基本設定
	id_column: str
	output_prefix: str

	# 遷移フラグ
	include_transitions: bool
	transition_group_agg: bool  # True: 群集約のみ、False: 列単位も含む

	# 再出現パターン
	include_reappearance: bool
	reappear_clip: int
	reappear_top_k: int

	# 代入影響トレース
	include_imputation_trace: bool
	imp_delta_winsorize_p: float
	imp_delta_top_k: int
	imp_policy_group_level: bool

	# 曜日・月次パターン
	include_temporal_bias: bool
	temporal_burn_in: int
	temporal_top_k: int

	# 祝日交差
	include_holiday_interaction: bool
	holiday_top_k: int

	# データ型
	dtype_flag: np.dtype
	dtype_int: np.dtype
	dtype_float: np.dtype

	# fold境界リセット
	reset_each_fold: bool

	@classmethod
	def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU3Config":
		"""YAML設定から SU3Config を生成する。"""
		id_column = mapping.get("id_column", "date_id")
		output_prefix = mapping.get("output_prefix", "su3")

		# 遷移フラグ
		include_transitions = bool(mapping.get("include_transitions", True))
		transition_group_agg = bool(mapping.get("transition_group_agg", True))

		# 再出現パターン
		include_reappearance = bool(mapping.get("include_reappearance", True))
		reappear_clip = int(mapping.get("reappear_clip", 60))
		reappear_top_k = int(mapping.get("reappear_top_k", 20))

		# 代入影響トレース
		include_imputation_trace = bool(mapping.get("include_imputation_trace", False))
		imp_delta_winsorize_p = float(mapping.get("imp_delta_winsorize_p", 0.99))
		imp_delta_top_k = int(mapping.get("imp_delta_top_k", 20))
		imp_policy_group_level = bool(mapping.get("imp_policy_group_level", True))

		# 曜日・月次パターン
		include_temporal_bias = bool(mapping.get("include_temporal_bias", True))
		temporal_burn_in = int(mapping.get("temporal_burn_in", 3))
		temporal_top_k = int(mapping.get("temporal_top_k", 20))

		# 祝日交差
		include_holiday_interaction = bool(mapping.get("include_holiday_interaction", True))
		holiday_top_k = int(mapping.get("holiday_top_k", 20))

		# データ型
		dtype_section = mapping.get("dtype", {})
		dtype_flag = np.dtype(dtype_section.get("flag", "uint8"))
		dtype_int = np.dtype(dtype_section.get("int", "int16"))
		dtype_float = np.dtype(dtype_section.get("float", "float32"))

		# fold境界リセット
		reset_each_fold = bool(mapping.get("reset_each_fold", True))

		return cls(
			id_column=id_column,
			output_prefix=output_prefix,
			include_transitions=include_transitions,
			transition_group_agg=transition_group_agg,
			include_reappearance=include_reappearance,
			reappear_clip=reappear_clip,
			reappear_top_k=reappear_top_k,
			include_imputation_trace=include_imputation_trace,
			imp_delta_winsorize_p=imp_delta_winsorize_p,
			imp_delta_top_k=imp_delta_top_k,
			imp_policy_group_level=imp_policy_group_level,
			include_temporal_bias=include_temporal_bias,
			temporal_burn_in=temporal_burn_in,
			temporal_top_k=temporal_top_k,
			include_holiday_interaction=include_holiday_interaction,
			holiday_top_k=holiday_top_k,
			dtype_flag=dtype_flag,
			dtype_int=dtype_int,
			dtype_float=dtype_float,
			reset_each_fold=reset_each_fold,
		)


def load_su3_config(config_path: str | Path) -> SU3Config:
	"""SU3 設定 YAML を読み込み :class:`SU3Config` を生成する。"""
	path = Path(config_path).resolve()
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

	try:
		su3_section = full_cfg["su3"]
	except KeyError as exc:
		raise KeyError("'su3' section is required in feature_generation.yaml") from exc

	return SU3Config.from_mapping(su3_section)


class SU3FeatureGenerator(BaseEstimator, TransformerMixin):
	"""SU3特徴量生成器。
	
	SU1の出力（m/<col>, gap_ffill/<col>, run_na/<col>, run_obs/<col>）
	を入力として、遷移・再出現・代入影響・曜日月次・祝日交差を生成する。
	"""

	def __init__(self, config: SU3Config):
		self.config = config
		self.m_columns_: Optional[List[str]] = None
		self.gap_ffill_columns_: Optional[List[str]] = None
		self.run_na_columns_: Optional[List[str]] = None
		self.run_obs_columns_: Optional[List[str]] = None
		self.groups_: Optional[Dict[str, List[str]]] = None
		self.feature_names_: Optional[List[str]] = None

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU3FeatureGenerator":
		"""特徴名の抽出とメタデータの保存。"""
		if not isinstance(X, pd.DataFrame):
			raise TypeError("SU3FeatureGenerator expects a pandas.DataFrame input")

		# SU1特徴列を識別
		self.m_columns_ = sorted([c for c in X.columns if c.startswith("m/")])
		self.gap_ffill_columns_ = sorted([c for c in X.columns if c.startswith("gap_ffill/")])
		self.run_na_columns_ = sorted([c for c in X.columns if c.startswith("run_na/")])
		self.run_obs_columns_ = sorted([c for c in X.columns if c.startswith("run_obs/")])

		if not self.m_columns_:
			raise ValueError("No 'm/' columns found in input. SU3 requires SU1 features as input.")

		# グループマッピングを構築
		self.groups_ = self._extract_groups()

		# 特徴名リストを生成（詳細は実装に依存）
		self.feature_names_ = self._generate_feature_names()

		return self

	def transform(self, X: pd.DataFrame, fold_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
		"""SU3特徴を生成。
		
		Args:
			X: SU1特徴を含むDataFrame
			fold_indices: CV用のfoldインデックス（fold境界でリセット）
		
		Returns:
			SU3特徴のDataFrame
		"""
		if self.m_columns_ is None or self.groups_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		# fold境界の準備
		fold_boundaries = self._compute_fold_boundaries(len(X), fold_indices)

		features: Dict[str, np.ndarray] = {}

		# A. 遷移フラグ
		if self.config.include_transitions:
			trans_features = self._generate_transition_features(X, fold_boundaries)
			features.update(trans_features)

		# B. 再出現パターン
		if self.config.include_reappearance:
			reappear_features = self._generate_reappearance_features(X, fold_boundaries)
			features.update(reappear_features)

		# C. 代入影響トレース（今回はスキップ - Stage 1ではオフ）
		# if self.config.include_imputation_trace:
		#     imp_features = self._generate_imputation_features(X, fold_boundaries)
		#     features.update(imp_features)

		# D. 曜日・月次パターン
		if self.config.include_temporal_bias:
			temporal_features = self._generate_temporal_features(X, fold_boundaries)
			features.update(temporal_features)

		# E. 祝日交差
		if self.config.include_holiday_interaction:
			holiday_features = self._generate_holiday_features(X, fold_boundaries)
			features.update(holiday_features)

		result_df = pd.DataFrame(features, index=X.index)
		return result_df

	def _extract_groups(self) -> Dict[str, List[str]]:
		"""m/<col>列からグループマッピングを構築する。"""
		groups: Dict[str, List[str]] = {}

		if self.m_columns_ is None:
			return groups

		for col in self.m_columns_:
			# "m/M1" -> "M1"
			base_col = col[2:]
			group = _infer_group(base_col)
			if group:
				if group not in groups:
					groups[group] = []
				groups[group].append(base_col)
		return groups

	def _generate_feature_names(self) -> List[str]:
		"""生成される特徴名のリストを作成（メタデータ用）。"""
		names = []

		if self.groups_ is None:
			return names

		if self.config.include_transitions and self.config.transition_group_agg:
			for grp in sorted(self.groups_.keys()):
				names.append(f"{self.config.output_prefix}/trans_rate/{grp}")

		if self.config.include_reappearance and self.m_columns_ is not None:
			# top-kの選択ロジックは実装時に決定
			# ここでは全列を仮定
			for col in self.m_columns_[:self.config.reappear_top_k]:
				base_col = col[2:]
				names.append(f"{self.config.output_prefix}/reappear_gap/{base_col}")
				names.append(f"{self.config.output_prefix}/pos_since_reappear/{base_col}")

		if self.config.include_temporal_bias and self.m_columns_ is not None:
			for col in self.m_columns_[:self.config.temporal_top_k]:
				base_col = col[2:]
				names.append(f"{self.config.output_prefix}/dow_m_rate/{base_col}")
				names.append(f"{self.config.output_prefix}/month_m_rate/{base_col}")

		if self.config.include_holiday_interaction and self.m_columns_ is not None:
			for col in self.m_columns_[:self.config.holiday_top_k]:
				base_col = col[2:]
				names.append(f"{self.config.output_prefix}/holiday_bridge_m/{base_col}")

		return names

	def _compute_fold_boundaries(
		self, n_rows: int, fold_indices: Optional[np.ndarray]
	) -> List[Tuple[int, int]]:
		"""fold境界のリストを計算する。"""
		if fold_indices is None or not self.config.reset_each_fold:
			return [(0, n_rows)]

		boundaries = []
		unique_folds = np.unique(fold_indices)
		for fold_id in unique_folds:
			fold_mask = fold_indices == fold_id
			indices = np.where(fold_mask)[0]
			if len(indices) > 0:
				boundaries.append((int(indices[0]), int(indices[-1]) + 1))

		return boundaries if boundaries else [(0, n_rows)]

	def _generate_transition_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""遷移フラグの生成。"""
		features = {}

		if self.groups_ is None or self.m_columns_ is None:
			return features

		if self.config.transition_group_agg:
			# 群集約のみ（6列程度）
			for grp, grp_cols in self.groups_.items():
				trans_rate = self._compute_group_trans_rate(X, grp_cols, fold_boundaries)
				features[f"{self.config.output_prefix}/trans_rate/{grp}"] = trans_rate
		else:
			# 列単位（初期検証用 - 今回は使用しない）
			for col in self.m_columns_:
				base_col = col[2:]
				na_to_obs, obs_to_na = self._compute_transitions(
					np.asarray(X[col].values), fold_boundaries
				)
				features[f"{self.config.output_prefix}/na_to_obs/{base_col}"] = na_to_obs
				features[f"{self.config.output_prefix}/obs_to_na/{base_col}"] = obs_to_na

		return features

	def _compute_transitions(
		self, m_values: np.ndarray, fold_boundaries: List[Tuple[int, int]]
	) -> Tuple[np.ndarray, np.ndarray]:
		"""1列の遷移フラグを計算。"""
		n = len(m_values)
		na_to_obs = np.zeros(n, dtype=self.config.dtype_flag)
		obs_to_na = np.zeros(n, dtype=self.config.dtype_flag)

		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx + 1, end_idx):
				prev_val = m_values[i - 1]
				curr_val = m_values[i]

				if prev_val == 1 and curr_val == 0:  # NaN → 観測
					na_to_obs[i] = 1
				elif prev_val == 0 and curr_val == 1:  # 観測 → NaN
					obs_to_na[i] = 1

		return na_to_obs, obs_to_na

	def _compute_group_trans_rate(
		self, X: pd.DataFrame, grp_cols: List[str], fold_boundaries: List[Tuple[int, int]]
	) -> np.ndarray:
		"""群内遷移率を計算。"""
		n = len(X)
		trans_rate = np.zeros(n, dtype=self.config.dtype_float)

		if not grp_cols:
			return trans_rate

		# m/<col>列名に変換
		m_cols = [f"m/{col}" for col in grp_cols]
		
		# 有効な列のみを使用
		valid_m_cols = [col for col in m_cols if col in X.columns]
		if not valid_m_cols:
			return trans_rate

		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx + 1, end_idx):
				trans_count = 0
				for col in valid_m_cols:
					prev_val = X[col].iloc[i - 1]
					curr_val = X[col].iloc[i]
					if prev_val != curr_val:
						trans_count += 1
				if valid_m_cols:
					trans_rate[i] = trans_count / len(valid_m_cols)
				else:
					trans_rate[i] = 0.0

		return trans_rate

	def _generate_reappearance_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""再出現パターンの生成。"""
		features = {}

		if self.m_columns_ is None:
			return features

		# top-k列を選択（簡易実装: 最初のk列）
		selected_cols = self.m_columns_[: self.config.reappear_top_k]

		for col in selected_cols:
			base_col = col[2:]
			m_col = col
			run_na_col = f"run_na/{base_col}"

			if m_col not in X.columns or run_na_col not in X.columns:
				continue

			reappear_gap, pos_since_reappear = self._compute_reappearance(
				np.asarray(X[m_col].values), np.asarray(X[run_na_col].values), fold_boundaries
			)

			features[f"{self.config.output_prefix}/reappear_gap/{base_col}"] = reappear_gap
			features[f"{self.config.output_prefix}/pos_since_reappear/{base_col}"] = pos_since_reappear

		return features

	def _compute_reappearance(
		self, m_values: np.ndarray, run_na_values: np.ndarray, fold_boundaries: List[Tuple[int, int]]
	) -> Tuple[np.ndarray, np.ndarray]:
		"""再出現間隔と位置の正規化を計算。"""
		n = len(m_values)
		reappear_gap = np.zeros(n, dtype=self.config.dtype_int)
		pos_since_reappear = np.zeros(n, dtype=self.config.dtype_float)

		for start_idx, end_idx in fold_boundaries:
			days_since_reappear = 0
			for i in range(start_idx, end_idx):
				# 再出現間隔: 今日観測かつ昨日NaN
				if i > start_idx and m_values[i] == 0 and m_values[i - 1] == 1:
					gap = min(int(run_na_values[i - 1]), self.config.reappear_clip)
					reappear_gap[i] = gap
					days_since_reappear = 0  # 復帰時点を0としてスタート
				
				# 観測継続中はインクリメント
				if m_values[i] == 0:
					# 位置の正規化 (復帰時点が0、その後1, 2, 3...とインクリメント)
					normalized_pos = days_since_reappear / self.config.reappear_clip
					pos_since_reappear[i] = min(max(normalized_pos, 0.0), 1.0)
					days_since_reappear += 1
				else:  # NaN中
					days_since_reappear = 0
					pos_since_reappear[i] = 0.0

		return reappear_gap, pos_since_reappear

	def _generate_temporal_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""曜日・月次パターンの生成。"""
		features = {}

		if self.m_columns_ is None:
			return features

		# date_idから曜日・月を抽出（date_id % 7, date_id // 30 % 12 などの簡易実装）
		# 実際にはdate_idがインデックスにある場合はそれを使用
		if self.config.id_column in X.columns:
			date_ids = X[self.config.id_column].values
		elif self.config.id_column == X.index.name:
			date_ids = X.index.values
		else:
			# date_idがない場合は行番号を使用
			date_ids = np.arange(len(X))

		# top-k列を選択
		selected_cols = self.m_columns_[: self.config.temporal_top_k]

		for col in selected_cols:
			base_col = col[2:]
			if col not in X.columns:
				continue

			dow_m_rate, month_m_rate = self._compute_temporal_bias(
				np.asarray(X[col].values), np.asarray(date_ids), fold_boundaries
			)

			features[f"{self.config.output_prefix}/dow_m_rate/{base_col}"] = dow_m_rate
			features[f"{self.config.output_prefix}/month_m_rate/{base_col}"] = month_m_rate

		return features

	def _compute_temporal_bias(
		self, m_values: np.ndarray, date_ids: np.ndarray, fold_boundaries: List[Tuple[int, int]]
	) -> Tuple[np.ndarray, np.ndarray]:
		"""曜日別・月次別の欠損率を計算（expanding平均）。
		
		Note: date_idの曜日・月計算は簡易的な近似を使用。
		      曜日: date_id % 7 (0-6)
		      月: (date_id // 30) % 12 (0-11として扱う)
		"""
		n = len(m_values)
		dow_m_rate = np.zeros(n, dtype=self.config.dtype_float)
		month_m_rate = np.zeros(n, dtype=self.config.dtype_float)

		for start_idx, end_idx in fold_boundaries:
			# 曜日ごとのカウンタ（0-6）
			dow_na_count = np.zeros(7, dtype=np.int32)
			dow_total_count = np.zeros(7, dtype=np.int32)

			# 月ごとのカウンタ（0-11 として扱う）
			month_na_count = np.zeros(12, dtype=np.int32)
			month_total_count = np.zeros(12, dtype=np.int32)

			for i in range(start_idx, end_idx):
				dow = int(date_ids[i]) % 7
				month = (int(date_ids[i]) // 30) % 12

				# 更新
				dow_na_count[dow] += int(m_values[i])
				dow_total_count[dow] += 1
				month_na_count[month] += int(m_values[i])
				month_total_count[month] += 1

				# burn-in期間後に比率を計算 (division by zero protection)
				if dow_total_count[dow] >= self.config.temporal_burn_in and dow_total_count[dow] > 0:
					dow_m_rate[i] = dow_na_count[dow] / dow_total_count[dow]
				else:
					dow_m_rate[i] = 0.0

				if month_total_count[month] >= self.config.temporal_burn_in and month_total_count[month] > 0:
					month_m_rate[i] = month_na_count[month] / month_total_count[month]
				else:
					month_m_rate[i] = 0.0

		return dow_m_rate, month_m_rate

	def _generate_holiday_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""祝日交差の生成。"""
		features = {}

		if self.m_columns_ is None:
			return features

		# holiday_bridge列が存在するか確認
		if "holiday_bridge" not in X.columns:
			# 存在しない場合は全て0
			selected_cols = self.m_columns_[: self.config.holiday_top_k]
			for col in selected_cols:
				base_col = col[2:]
				features[f"{self.config.output_prefix}/holiday_bridge_m/{base_col}"] = np.zeros(
					len(X), dtype=self.config.dtype_flag
				)
			return features

		holiday_bridge = np.asarray(X["holiday_bridge"].values)

		# top-k列を選択
		selected_cols = self.m_columns_[: self.config.holiday_top_k]

		for col in selected_cols:
			base_col = col[2:]
			if col not in X.columns:
				continue

			# holiday_bridge * m (両方が1のときのみ1)
			m_vals = np.asarray(X[col].values)
			holiday_m = (holiday_bridge * m_vals).astype(self.config.dtype_flag)
			features[f"{self.config.output_prefix}/holiday_bridge_m/{base_col}"] = holiday_m

		return features


__all__ = [
	"SU3Config",
	"SU3FeatureGenerator",
	"load_su3_config",
]
