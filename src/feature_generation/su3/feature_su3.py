"""SU3（欠損構造三次特徴量）の生成ロジック。

本モジュールは ``docs/feature_generation/SU3.md`` に記載された方針を実装し、
SU1出力から遷移・再出現・時間的パターンを捕捉する特徴を生成する
トランスフォーマー ``SU3FeatureGenerator`` と、設定 YAML を読み込むための
ヘルパー関数を提供する。
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


def _coerce_dtype(dtype_like: str) -> np.dtype:
	"""dtype 指定文字列を ``numpy.dtype`` に変換する。"""
	try:
		dtype = np.dtype(dtype_like)
	except TypeError as exc:  # pragma: no cover - 防御的分岐
		raise ValueError(f"Invalid dtype specification: {dtype_like!r}") from exc
	return dtype


@dataclass(frozen=True)
class SU3Config:
	"""SU3 特徴量生成に必要な設定を保持するデータクラス。"""

	id_column: str
	output_prefix: str

	# 遷移フラグ
	include_transitions: bool
	transition_group_agg: bool

	# 再出現パターン
	include_reappearance: bool
	reappear_clip: int
	reappear_top_k: int

	# 代入影響トレース
	include_imputation_trace: bool
	imp_delta_winsorize_p: float
	imp_delta_top_k: int
	imp_policy_group_level: bool

	# 曜日・月次
	include_temporal_bias: bool
	temporal_burn_in: int
	temporal_top_k: int

	# 祝日交差
	include_holiday_interaction: bool
	holiday_top_k: int

	# データ型
	flag_dtype: np.dtype
	int_dtype: np.dtype
	float_dtype: np.dtype

	# fold境界リセット
	reset_each_fold: bool

	@classmethod
	def from_mapping(cls, mapping: Mapping[str, Any], *, base_dir: Path) -> "SU3Config":
		"""YAML設定マッピングからSU3Configを生成する。"""
		id_column = mapping.get("id_column", "date_id")
		output_prefix = mapping.get("output_prefix", "su3")

		include_transitions = bool(mapping.get("include_transitions", True))
		transition_group_agg = bool(mapping.get("transition_group_agg", True))

		include_reappearance = bool(mapping.get("include_reappearance", True))
		reappear_clip = int(mapping.get("reappear_clip", 60))
		if reappear_clip <= 0:
			raise ValueError(f"reappear_clip must be positive, got {reappear_clip}")
		reappear_top_k = int(mapping.get("reappear_top_k", 20))

		include_imputation_trace = bool(mapping.get("include_imputation_trace", False))
		imp_delta_winsorize_p = float(mapping.get("imp_delta_winsorize_p", 0.99))
		imp_delta_top_k = int(mapping.get("imp_delta_top_k", 20))
		imp_policy_group_level = bool(mapping.get("imp_policy_group_level", True))

		include_temporal_bias = bool(mapping.get("include_temporal_bias", True))
		temporal_burn_in = int(mapping.get("temporal_burn_in", 3))
		temporal_top_k = int(mapping.get("temporal_top_k", 20))

		include_holiday_interaction = bool(mapping.get("include_holiday_interaction", True))
		holiday_top_k = int(mapping.get("holiday_top_k", 20))

		dtype_section = mapping.get("dtype", {})
		flag_dtype = _coerce_dtype(dtype_section.get("flag", "uint8"))
		int_dtype = _coerce_dtype(dtype_section.get("int", "int16"))
		float_dtype = _coerce_dtype(dtype_section.get("float", "float32"))

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
			flag_dtype=flag_dtype,
			int_dtype=int_dtype,
			float_dtype=float_dtype,
			reset_each_fold=reset_each_fold,
		)


def load_su3_config(config_path: str | Path) -> SU3Config:
	"""SU3 設定 YAML を読み込み :class:`SU3Config` を生成する。"""
	path = Path(config_path).resolve()
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

	try:
		su3_section = full_cfg["su3"]
	except KeyError as exc:  # pragma: no cover - 防御的分岐
		raise KeyError("'su3' section is required in feature_generation.yaml") from exc

	return SU3Config.from_mapping(su3_section, base_dir=path.parent)


class SU3FeatureGenerator(BaseEstimator, TransformerMixin):
	"""SU1の出力から SU3 遷移・再出現特徴量を生成するトランスフォーマー。

	SU1が生成した欠損フラグ（m/<col>）、ギャップ（gap_ffill/<col>）、
	ラン長（run_na/<col>, run_obs/<col>）を入力として、
	時間的変化（遷移、再出現）や時間的バイアス（曜日、月次）を捕捉する。
	"""

	def __init__(self, config: SU3Config):
		self.config = config
		self.m_columns_: List[str] | None = None
		self.base_columns_: List[str] | None = None
		self.group_columns_: Dict[str, List[str]] | None = None
		self.feature_names_: List[str] | None = None

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU3FeatureGenerator":
		"""特徴名の抽出とメタデータの保存。

		Args:
			X: SU1特徴を含むDataFrame
			y: ターゲット（未使用）

		Returns:
			self
		"""
		df = self._ensure_dataframe(X)

		# SU1特徴列を識別
		self.m_columns_ = sorted([c for c in df.columns if c.startswith("m/") and "/" in c and c.count("/") == 1])
		self.base_columns_ = [c.split("/", 1)[1] for c in self.m_columns_]

		# グループマッピング
		self.group_columns_ = self._build_group_columns()

		# 特徴名リストを生成（メタデータ用）
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
		if self.m_columns_ is None or self.group_columns_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		df = self._ensure_dataframe(X)

		# fold境界の準備
		fold_boundaries = self._compute_fold_boundaries(len(df), fold_indices)

		features: Dict[str, np.ndarray] = {}

		# A. 遷移フラグ
		if self.config.include_transitions:
			trans_features = self._generate_transition_features(df, fold_boundaries)
			features.update(trans_features)

		# B. 再出現パターン
		if self.config.include_reappearance:
			reappear_features = self._generate_reappearance_features(df, fold_boundaries)
			features.update(reappear_features)

		# D. 曜日・月次
		if self.config.include_temporal_bias:
			temporal_features = self._generate_temporal_features(df, fold_boundaries)
			features.update(temporal_features)

		# E. 祝日交差
		if self.config.include_holiday_interaction:
			holiday_features = self._generate_holiday_features(df, fold_boundaries)
			features.update(holiday_features)

		result_df = pd.DataFrame(features, index=df.index)
		return result_df

	# ------------------------------------------------------------------
	# 内部ヘルパー
	# ------------------------------------------------------------------
	def _ensure_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
		"""入力がDataFrameであることを確認する。"""
		if not isinstance(X, pd.DataFrame):  # pragma: no cover - 防御的分岐
			raise TypeError("SU3FeatureGenerator expects a pandas.DataFrame input")
		return X.copy()

	def _build_group_columns(self) -> Dict[str, List[str]]:
		"""ベース列をグループごとに分類する。"""
		if self.base_columns_ is None:
			return {}

		group_map: Dict[str, List[str]] = {}
		for col in self.base_columns_:
			grp = _infer_group(col)
			if grp is not None:
				if grp not in group_map:
					group_map[grp] = []
				group_map[grp].append(col)
		return group_map

	def _compute_fold_boundaries(
		self, n_rows: int, fold_indices: Optional[np.ndarray]
	) -> List[Tuple[int, int]]:
		"""fold境界を計算する。

		Args:
			n_rows: データの行数
			fold_indices: fold番号の配列（None の場合は全データを1つのfoldとして扱う）

		Returns:
			(start_idx, end_idx)のタプルのリスト

		Note:
			fold_indicesは連続した行に対して同じfold番号が付与されていることを想定。
			非連続なfoldインデックスの場合、各連続ブロックを別のfoldとして扱う。
		"""
		if fold_indices is None or not self.config.reset_each_fold:
			return [(0, n_rows)]

		boundaries: List[Tuple[int, int]] = []
		unique_folds = np.unique(fold_indices)
		for fold in unique_folds:
			mask = fold_indices == fold
			indices = np.where(mask)[0]
			if len(indices) > 0:
				# 連続ブロックごとに境界を作成
				blocks: List[List[int]] = []
				current_block = [int(indices[0])]
				for i in range(1, len(indices)):
					if indices[i] == indices[i-1] + 1:
						current_block.append(int(indices[i]))
					else:
						blocks.append(current_block)
						current_block = [int(indices[i])]
				blocks.append(current_block)
				
				# 各ブロックを境界として追加
				for block in blocks:
					boundaries.append((block[0], block[-1] + 1))

		return boundaries

	def _generate_feature_names(self) -> List[str]:
		"""生成される特徴名のリストを返す（メタデータ用）。"""
		names: List[str] = []

		if self.group_columns_ is None or self.base_columns_ is None:
			return names

		if self.config.include_transitions and self.config.transition_group_agg:
			for grp in sorted(self.group_columns_.keys()):
				names.append(f"{self.config.output_prefix}/trans_rate/{grp}")

		if self.config.include_reappearance:
			# top-kなので実際の列名は動的に決まるが、ここでは全列を列挙
			for col in self.base_columns_[:self.config.reappear_top_k]:
				names.append(f"{self.config.output_prefix}/reappear_gap/{col}")
				names.append(f"{self.config.output_prefix}/pos_since_reappear/{col}")

		if self.config.include_temporal_bias:
			for col in self.base_columns_[:self.config.temporal_top_k]:
				names.append(f"{self.config.output_prefix}/dow_m_rate/{col}")
				names.append(f"{self.config.output_prefix}/month_m_rate/{col}")

		if self.config.include_holiday_interaction:
			for col in self.base_columns_[:self.config.holiday_top_k]:
				names.append(f"{self.config.output_prefix}/holiday_bridge_m/{col}")

		return names

	# ------------------------------------------------------------------
	# A. 遷移フラグ
	# ------------------------------------------------------------------
	def _generate_transition_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""遷移フラグの生成（群集約のみ）。"""
		features: Dict[str, np.ndarray] = {}

		if self.group_columns_ is None:
			return features

		if self.config.transition_group_agg:
			# 群集約のみ
			for grp in sorted(self.group_columns_.keys()):
				grp_cols = [f"m/{c}" for c in self.group_columns_[grp] if f"m/{c}" in X.columns]
				if grp_cols:
					trans_rate = self._compute_group_trans_rate(X, grp_cols, fold_boundaries)
					features[f"{self.config.output_prefix}/trans_rate/{grp}"] = trans_rate

		return features

	def _compute_group_trans_rate(
		self, X: pd.DataFrame, grp_cols: List[str], fold_boundaries: List[Tuple[int, int]]
	) -> np.ndarray:
		"""群内遷移率を計算する。

		Args:
			X: 入力DataFrame
			grp_cols: 群に属する列名のリスト
			fold_boundaries: fold境界のリスト

		Returns:
			遷移率の配列
		"""
		n = len(X)
		trans_rate = np.zeros(n, dtype=self.config.float_dtype)

		if not grp_cols:
			return trans_rate

		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx + 1, end_idx):
				trans_count = 0
				for col in grp_cols:
					prev_val = X[col].iloc[i - 1]
					curr_val = X[col].iloc[i]
					if prev_val != curr_val:
						trans_count += 1
				trans_rate[i] = trans_count / len(grp_cols)

		return trans_rate

	# ------------------------------------------------------------------
	# B. 再出現パターン
	# ------------------------------------------------------------------
	def _generate_reappearance_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""再出現パターンの生成。"""
		features: Dict[str, np.ndarray] = {}

		if self.base_columns_ is None:
			return features

		# top-k列のみを処理
		selected_cols = self.base_columns_[:self.config.reappear_top_k]

		for col in selected_cols:
			m_col = f"m/{col}"
			run_na_col = f"run_na/{col}"

			if m_col not in X.columns or run_na_col not in X.columns:
				continue

			reappear_gap, pos_since_reappear = self._compute_reappearance(
				np.asarray(X[m_col].values),
				np.asarray(X[run_na_col].values),
				fold_boundaries,
			)

			features[f"{self.config.output_prefix}/reappear_gap/{col}"] = reappear_gap
			features[f"{self.config.output_prefix}/pos_since_reappear/{col}"] = pos_since_reappear

		return features

	def _compute_reappearance(
		self,
		m_values: np.ndarray,
		run_na_values: np.ndarray,
		fold_boundaries: List[Tuple[int, int]],
	) -> Tuple[np.ndarray, np.ndarray]:
		"""再出現間隔と再出現位置を計算する。

		Args:
			m_values: 欠損フラグ (1=NaN, 0=観測)
			run_na_values: NaN連続長
			fold_boundaries: fold境界のリスト

		Returns:
			(reappear_gap, pos_since_reappear)のタプル
		"""
		n = len(m_values)
		reappear_gap = np.zeros(n, dtype=self.config.int_dtype)
		pos_since_reappear = np.zeros(n, dtype=self.config.float_dtype)

		for start_idx, end_idx in fold_boundaries:
			days_since_reappear = 0

			for i in range(start_idx, end_idx):
				if i == start_idx:
					# 初回はリセット
					days_since_reappear = 0
					reappear_gap[i] = 0
					pos_since_reappear[i] = 0.0
					continue

				prev_m = m_values[i - 1]
				curr_m = m_values[i]

				if curr_m == 0 and prev_m == 1:
					# 再出現点（NaN → 観測）
					gap = min(int(run_na_values[i - 1]), self.config.reappear_clip)
					reappear_gap[i] = gap
					days_since_reappear = 0
					pos_since_reappear[i] = 0.0
				elif curr_m == 0:
					# 観測継続中
					days_since_reappear += 1
					reappear_gap[i] = 0
					pos_since_reappear[i] = min(days_since_reappear / 60.0, 1.0)
				else:
					# NaN中
					days_since_reappear = 0
					reappear_gap[i] = 0
					pos_since_reappear[i] = 0.0

		return reappear_gap, pos_since_reappear

	# ------------------------------------------------------------------
	# D. 曜日・月次
	# ------------------------------------------------------------------
	def _generate_temporal_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""曜日・月次の欠損率を生成。"""
		features: Dict[str, np.ndarray] = {}

		if self.base_columns_ is None:
			return features

		# date_idから曜日と月を推定
		if self.config.id_column not in X.columns and self.config.id_column not in X.index.names:
			# date_idがない場合はスキップ
			return features

		if self.config.id_column in X.columns:
			date_ids = np.asarray(X[self.config.id_column].values)
		else:
			date_ids = np.asarray(X.index.values)

		# top-k列のみを処理
		selected_cols = self.base_columns_[:self.config.temporal_top_k]

		for col in selected_cols:
			m_col = f"m/{col}"
			if m_col not in X.columns:
				continue

			dow_m_rate = self._compute_temporal_bias(
				np.asarray(X[m_col].values),
				date_ids,
				fold_boundaries,
				period_type="dow",
			)
			month_m_rate = self._compute_temporal_bias(
				np.asarray(X[m_col].values),
				date_ids,
				fold_boundaries,
				period_type="month",
			)

			features[f"{self.config.output_prefix}/dow_m_rate/{col}"] = dow_m_rate
			features[f"{self.config.output_prefix}/month_m_rate/{col}"] = month_m_rate

		return features

	def _compute_temporal_bias(
		self,
		m_values: np.ndarray,
		date_ids: np.ndarray,
		fold_boundaries: List[Tuple[int, int]],
		period_type: str,
	) -> np.ndarray:
		"""曜日または月次の欠損率を計算する（expanding平均）。

		Args:
			m_values: 欠損フラグ (1=NaN, 0=観測)
			date_ids: date_idの配列
			fold_boundaries: fold境界のリスト
			period_type: "dow" または "month"

		Returns:
			period別欠損率の配列

		Note:
			月次計算は簡易的に date_id // 30 % 12 を使用。
			実際の月の日数は異なるため、近似値として扱う。
		"""
		n = len(m_values)
		bias_rate = np.zeros(n, dtype=self.config.float_dtype)

		for start_idx, end_idx in fold_boundaries:
			# period ごとのカウンタを初期化
			if period_type == "dow":
				n_periods = 7
				periods = date_ids[start_idx:end_idx] % 7
			else:  # month
				n_periods = 12
				# 簡易的な月計算（date_id を30日周期と仮定）
				# Note: 実際の月の日数は28-31日だが、ここでは近似値として30を使用
				periods = (date_ids[start_idx:end_idx] // 30) % 12

			period_na_count = np.zeros(n_periods, dtype=np.int32)
			period_total_count = np.zeros(n_periods, dtype=np.int32)

			for i in range(start_idx, end_idx):
				period = int(periods[i - start_idx])
				period_na_count[period] += m_values[i]
				period_total_count[period] += 1

				# burn-in期間を経過したら率を計算
				if period_total_count[period] >= self.config.temporal_burn_in:
					bias_rate[i] = period_na_count[period] / period_total_count[period]
				else:
					bias_rate[i] = 0.0

		return bias_rate

	# ------------------------------------------------------------------
	# E. 祝日交差
	# ------------------------------------------------------------------
	def _generate_holiday_features(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""祝日交差特徴の生成。"""
		features: Dict[str, np.ndarray] = {}

		if self.base_columns_ is None:
			return features

		# holiday_bridge列が存在するか確認
		holiday_col = None
		for candidate in ["holiday_bridge", "holiday", "is_holiday"]:
			if candidate in X.columns:
				holiday_col = candidate
				break

		if holiday_col is None:
			# 祝日列が存在しない場合はスキップ
			return features

		holiday_values = np.asarray(X[holiday_col].values)

		# top-k列のみを処理
		selected_cols = self.base_columns_[:self.config.holiday_top_k]

		for col in selected_cols:
			m_col = f"m/{col}"
			if m_col not in X.columns:
				continue

			m_values = np.asarray(X[m_col].values)
			holiday_bridge_m = (holiday_values * m_values).astype(self.config.flag_dtype)
			features[f"{self.config.output_prefix}/holiday_bridge_m/{col}"] = holiday_bridge_m

		return features
