"""SU5（共欠損構造特徴量）の生成ロジック。

本モジュールは SU1 の出力 (m/<col>) を入力として、列間の共欠損構造を捕捉する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


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
class SU5Config:
	"""SU5（共欠損構造）特徴生成の設定を保持するデータクラス。"""

	id_column: str
	output_prefix: str

	# ペア選択
	top_k_pairs: int
	top_k_pairs_per_group: Optional[int]

	# ローリング共欠損率
	windows: Tuple[int, ...]

	# fold 境界でローリング統計をリセットするか
	reset_each_fold: bool

	# 型
	dtype_flag: np.dtype
	dtype_int: np.dtype
	dtype_float: np.dtype
	
	# brushup 設定
	brushup_enabled: bool
	brushup_n_clusters: int
	brushup_random_state: int
	brushup_include_density: bool
	brushup_include_deg_stats: bool
	brushup_include_centrality: bool

	@classmethod
	def from_mapping(cls, mapping: Mapping[str, Any]) -> "SU5Config":
		"""YAML設定から SU5Config を生成する。"""
		id_column = mapping.get("id_column", "date_id")
		output_prefix = mapping.get("output_prefix", "su5")

		# ペア選択
		top_k_pairs = int(mapping.get("top_k_pairs", 10))
		top_k_pairs_per_group = mapping.get("top_k_pairs_per_group")
		if top_k_pairs_per_group is not None:
			top_k_pairs_per_group = int(top_k_pairs_per_group)

		# ローリング共欠損率
		windows_list = mapping.get("windows", [5, 20])
		windows = tuple(windows_list)

		# fold境界リセット
		reset_each_fold = bool(mapping.get("reset_each_fold", True))

		# データ型
		dtype_section = mapping.get("dtype", {})
		dtype_flag = np.dtype(dtype_section.get("flag", "uint8"))
		dtype_int = np.dtype(dtype_section.get("int", "int16"))
		dtype_float = np.dtype(dtype_section.get("float", "float32"))

		# brushup 設定
		brushup_section = mapping.get("brushup", {})
		brushup_enabled = bool(brushup_section.get("enabled", False))
		cluster_section = brushup_section.get("cluster", {})
		brushup_n_clusters = int(cluster_section.get("n_clusters", 6))
		brushup_random_state = int(cluster_section.get("random_state", 42))
		brushup_include_density = bool(brushup_section.get("include_density", True))
		brushup_include_deg_stats = bool(brushup_section.get("include_deg_stats", True))
		brushup_include_centrality = bool(brushup_section.get("include_centrality", True))

		return cls(
			id_column=id_column,
			output_prefix=output_prefix,
			top_k_pairs=top_k_pairs,
			top_k_pairs_per_group=top_k_pairs_per_group,
			windows=windows,
			reset_each_fold=reset_each_fold,
			dtype_flag=dtype_flag,
			dtype_int=dtype_int,
			dtype_float=dtype_float,
			brushup_enabled=brushup_enabled,
			brushup_n_clusters=brushup_n_clusters,
			brushup_random_state=brushup_random_state,
			brushup_include_density=brushup_include_density,
			brushup_include_deg_stats=brushup_include_deg_stats,
			brushup_include_centrality=brushup_include_centrality,
		)


def load_su5_config(config_path: str | Path) -> SU5Config:
	"""SU5 設定 YAML を読み込み :class:`SU5Config` を生成する。"""
	path = Path(config_path).resolve()
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

	try:
		su5_section = full_cfg["su5"]
	except KeyError as exc:
		raise KeyError("'su5' section is required in feature_generation.yaml") from exc

	return SU5Config.from_mapping(su5_section)


class SU5FeatureGenerator(BaseEstimator, TransformerMixin):
	"""SU5 共欠損特徴量生成器。

	入力: SU1 で生成された `m/<col>` 列を含む DataFrame
	出力: co-miss フラグ・ローリング共欠損率・degree などの特徴
	"""

	def __init__(self, config: SU5Config):
		self.config = config
		self.m_columns_: Optional[List[str]] = None
		self.groups_: Optional[Dict[str, List[str]]] = None
		self.top_pairs_: Optional[List[Tuple[str, str]]] = None
		self.feature_names_: Optional[List[str]] = None
		self.kmeans_model_: Optional[Any] = None  # k-means for brushup

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU5FeatureGenerator":
		"""特徴名の抽出とtop-kペアの選択。

		Args:
			X: SU1特徴を含むDataFrame (m/<col> を含む)
			y: unused (sklearn互換のため)

		Returns:
			self
		"""
		if not isinstance(X, pd.DataFrame):
			raise TypeError("SU5FeatureGenerator expects a pandas.DataFrame input")

		# 1. SU1 の m 列抽出
		self.m_columns_ = sorted([c for c in X.columns if c.startswith("m/")])

		if not self.m_columns_:
			raise ValueError("No 'm/' columns found in input. SU5 requires SU1 features as input.")

		# 2. 列プレフィクスからグループを抽出（M/E/I/P/S/V）
		self.groups_ = self._extract_groups()

		# 3. 学習期間で共欠損スコアを集計し、top-k ペアを決定
		self.top_pairs_ = self._select_top_k_pairs(X)

		# 4. feature_names_ を組み立て（transform 出力列の順序を固定）
		self.feature_names_ = self._build_feature_names()

		# 5. brushup: k-means fit (if enabled)
		if self.config.brushup_enabled:
			miss_matrix = X[[col for col in self.m_columns_]].values
			self.kmeans_model_ = KMeans(
				n_clusters=self.config.brushup_n_clusters,
				random_state=self.config.brushup_random_state,
				n_init=10
			)
			self.kmeans_model_.fit(miss_matrix)

		return self

	def transform(self, X: pd.DataFrame, fold_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
		"""SU5特徴を生成。

		Args:
			X: SU1特徴を含むDataFrame
			fold_indices: CV用のfoldインデックス（fold境界でリセット）

		Returns:
			SU5特徴のDataFrame
		"""
		if self.m_columns_ is None or self.top_pairs_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		n = len(X)
		features: Dict[str, np.ndarray] = {}

		# fold 境界決定
		fold_boundaries = self._compute_fold_boundaries(n, fold_indices)

		# A. 単日共欠損フラグ
		co_now = self._compute_co_miss_now(X, fold_boundaries)
		features.update(co_now)

		# B. ローリング共欠損率
		co_roll = self._compute_co_miss_rollrate(features, fold_boundaries)
		features.update(co_roll)

		# C. degree（列ごとの共欠損次数）
		co_deg = self._compute_co_miss_degree(X)
		features.update(co_deg)

		# D. brushup features (if enabled)
		if self.config.brushup_enabled:
			brushup_feats = self._compute_brushup_features(X, features)
			features.update(brushup_feats)

		return pd.DataFrame(features, index=X.index)

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

	def _select_top_k_pairs(self, X: pd.DataFrame) -> List[Tuple[str, str]]:
		"""学習期間で共欠損スコアを計算し、top-kペアを選択する。

		Args:
			X: SU1特徴を含むDataFrame

		Returns:
			選択された列ペアのリスト [(col_a, col_b), ...]
		"""
		if self.m_columns_ is None:
			return []

		# 全ペアの共欠損スコアを計算
		pair_scores: List[Tuple[float, str, str]] = []

		for i, col_a in enumerate(self.m_columns_):
			for col_b in self.m_columns_[i + 1 :]:
				# 同時NaN比を共欠損スコアとする
				m_a = np.asarray(X[col_a].values)
				m_b = np.asarray(X[col_b].values)

				both_na = int(np.sum((m_a == 1) & (m_b == 1)))
				either_na = int(np.sum((m_a == 1) | (m_b == 1)))

				# Jaccard風のスコア
				if either_na > 0:
					score = float(both_na) / float(either_na)
				else:
					score = 0.0

				# ベース列名を取得（"m/M1" -> "M1"）
				base_a = col_a[2:]
				base_b = col_b[2:]
				pair_scores.append((score, base_a, base_b))

		# スコアでソート（降順）
		pair_scores.sort(reverse=True)

		# top-k選択
		if self.config.top_k_pairs_per_group is not None:
			# グループ単位で選択
			selected_pairs = self._select_top_k_per_group(pair_scores)
		else:
			# グローバルで選択
			selected_pairs = [(a, b) for _, a, b in pair_scores[: self.config.top_k_pairs]]

		return selected_pairs

	def _select_top_k_per_group(self, pair_scores: List[Tuple[float, str, str]]) -> List[Tuple[str, str]]:
		"""グループ単位でtop-kペアを選択する。"""
		if self.groups_ is None or self.config.top_k_pairs_per_group is None:
			return []

		selected: List[Tuple[str, str]] = []
		group_counts: Dict[str, int] = {grp: 0 for grp in self.groups_.keys()}

		for score, col_a, col_b in pair_scores:
			group_a = _infer_group(col_a)
			group_b = _infer_group(col_b)

			# 同一グループ内のペアのみ選択
			if group_a == group_b and group_a is not None:
				if group_counts[group_a] < self.config.top_k_pairs_per_group:
					selected.append((col_a, col_b))
					group_counts[group_a] += 1

		return selected

	def _build_feature_names(self) -> List[str]:
		"""生成される特徴名のリストを作成（メタデータ用）。"""
		names: List[str] = []

		if self.top_pairs_ is None:
			return names

		# A. 単日共欠損フラグ
		for col_a, col_b in self.top_pairs_:
			names.append(f"co_miss_now/{col_a}__{col_b}")

		# B. ローリング共欠損率
		for window in self.config.windows:
			for col_a, col_b in self.top_pairs_:
				names.append(f"co_miss_rollrate_{window}/{col_a}__{col_b}")

		# C. degree（列ごとの共欠損次数）
		if self.m_columns_ is not None:
			for col in self.m_columns_:
				base_col = col[2:]
				names.append(f"co_miss_deg/{base_col}")

		return names

	def _compute_fold_boundaries(
		self, n_rows: int, fold_indices: Optional[np.ndarray]
	) -> List[Tuple[int, int]]:
		"""fold_indices からローリング統計の境界を計算する。

		fold_indices の想定:
		  - 0: 全 fold の train 部分（共通 prefix）
		  - 1..K: 各 fold の validation 区間
		reset_each_fold=True の場合、各 fold 区間の境界でローリング統計をリセットする。

		Args:
			n_rows: 行数
			fold_indices: fold 番号の配列（None の場合は全体を 1 区間とする）

		Returns:
			(start_idx, end_idx) のリスト（各区間で連続したローリング統計を計算）
		"""
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

	def _compute_co_miss_now(
		self, X: pd.DataFrame, fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""単日共欠損フラグを計算する。"""
		features: Dict[str, np.ndarray] = {}

		if self.top_pairs_ is None:
			return features

		for col_a, col_b in self.top_pairs_:
			m_a = np.asarray(X[f"m/{col_a}"].values)
			m_b = np.asarray(X[f"m/{col_b}"].values)

			# 両方が1（NaN）のときに1を立てる
			co_miss = ((m_a == 1) & (m_b == 1)).astype(self.config.dtype_flag)

			features[f"co_miss_now/{col_a}__{col_b}"] = co_miss

		return features

	def _compute_co_miss_rollrate(
		self, features: Dict[str, np.ndarray], fold_boundaries: List[Tuple[int, int]]
	) -> Dict[str, np.ndarray]:
		"""ローリング共欠損率を計算する。"""
		rollrate_features: Dict[str, np.ndarray] = {}

		if self.top_pairs_ is None:
			return rollrate_features

		for window in self.config.windows:
			for col_a, col_b in self.top_pairs_:
				co_miss_now = features[f"co_miss_now/{col_a}__{col_b}"]
				n = len(co_miss_now)

				# ローリング平均を計算（fold境界を考慮）
				rollrate = np.full(n, np.nan, dtype=self.config.dtype_float)

				for start_idx, end_idx in fold_boundaries:
					for i in range(start_idx, end_idx):
						window_start = max(start_idx, i - window + 1)
						window_end = i + 1

						if window_end - window_start >= window:
							# 窓が十分なサイズの場合のみ計算
							window_values = co_miss_now[window_start:window_end]
							rollrate[i] = np.mean(window_values)

				rollrate_features[f"co_miss_rollrate_{window}/{col_a}__{col_b}"] = rollrate

		return rollrate_features

	def _compute_co_miss_degree(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
		"""列ごとの共欠損次数（degree）を計算する。

		各列が top_pairs_ に何回出現するかをカウントする。
		"""
		degree_features: Dict[str, np.ndarray] = {}

		if self.m_columns_ is None or self.top_pairs_ is None:
			return degree_features

		# 各列の出現回数をカウント
		degree_counts: Dict[str, int] = {col[2:]: 0 for col in self.m_columns_}

		for col_a, col_b in self.top_pairs_:
			degree_counts[col_a] += 1
			degree_counts[col_b] += 1

		# 全行で同じ値を持つ特徴として追加
		n = len(X)
		for col in self.m_columns_:
			base_col = col[2:]
			degree_value = degree_counts[base_col]
			degree_features[f"co_miss_deg/{base_col}"] = np.full(
				n, degree_value, dtype=self.config.dtype_int
			)

		return degree_features

	def _compute_brushup_features(
		self, X: pd.DataFrame, existing_features: Dict[str, np.ndarray]
	) -> Dict[str, np.ndarray]:
		"""Generate SU5 brushup features (4-5 new columns)."""
		brushup_features: Dict[str, np.ndarray] = {}
		
		if self.m_columns_ is None or self.top_pairs_ is None:
			return brushup_features
		
		n = len(X)
		
		# 1. miss_pattern_cluster (k-means)
		if self.kmeans_model_ is not None:
			miss_matrix = X[[col for col in self.m_columns_]].values
			cluster_labels = self.kmeans_model_.predict(miss_matrix)
			brushup_features["miss_pattern_cluster"] = cluster_labels.astype(np.int8)
		
		# 2. co_miss_density
		if self.config.brushup_include_density:
			co_miss_now_cols = [f"co_miss_now/{a}__{b}" for a, b in self.top_pairs_]
			co_miss_flags = np.column_stack([existing_features[col] for col in co_miss_now_cols])
			co_miss_density = co_miss_flags.mean(axis=1).astype(self.config.dtype_float)
			brushup_features["co_miss_density"] = co_miss_density
		
		# 3. co_miss_deg_sum and co_miss_deg_mean
		if self.config.brushup_include_deg_stats:
			# Get degree values for each column
			degree_values = {}
			for col in self.m_columns_:
				base_col = col[2:]
				degree_col = f"co_miss_deg/{base_col}"
				if degree_col in existing_features:
					# co_miss_deg values are constant across all rows (static graph property)
					# so we can extract from first element
					degree_values[base_col] = existing_features[degree_col][0]
			
			# Calculate deg sum and mean for missing columns in each row
			deg_sum = np.zeros(n, dtype=self.config.dtype_float)
			deg_mean = np.zeros(n, dtype=self.config.dtype_float)
			
			for i in range(n):
				missing_cols = []
				for col in self.m_columns_:
					if X.iloc[i][col] == 1:  # missing
						base_col = col[2:]
						if base_col in degree_values:
							missing_cols.append(degree_values[base_col])
				
				if missing_cols:
					deg_sum[i] = sum(missing_cols)
					deg_mean[i] = sum(missing_cols) / len(missing_cols)
			
			brushup_features["co_miss_deg_sum"] = deg_sum
			brushup_features["co_miss_deg_mean"] = deg_mean
		
		# 4. miss_graph_centrality (optional)
		if self.config.brushup_include_centrality:
			# Build set of columns in top-k pairs
			pair_cols = set()
			for a, b in self.top_pairs_:
				pair_cols.add(a)
				pair_cols.add(b)
			
			# Count how many missing columns are in top-k pairs
			centrality = np.zeros(n, dtype=np.int8)
			for i in range(n):
				for col in self.m_columns_:
					base_col = col[2:]
					if X.iloc[i][col] == 1 and base_col in pair_cols:  # missing and in top-k
						centrality[i] += 1
			
			brushup_features["miss_graph_centrality"] = centrality
		
		return brushup_features


# Note: SU5FeatureAugmenter is defined in train_su5.py
# (removed duplicate class definition to avoid confusion)
