"""SU2（欠損構造二次特徴量）の生成ロジック。

本モジュールは ``docs/feature_generation/SU2.md`` に記載された方針を実装し、
SU1出力から二次派生特徴を生成するトランスフォーマー ``SU2FeatureGenerator`` と、
設定 YAML や SU1 データを読み込むためのヘルパー関数を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

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


def _path_from_config(base_dir: Path, value: str) -> Path:
	"""YAML 設定内で指定されたパスを解決する。"""
	raw_path = Path(value)
	return raw_path if raw_path.is_absolute() else (base_dir / raw_path).resolve()


@dataclass(frozen=True)
class SU2Config:
	"""SU2 特徴量生成に必要な設定を保持するデータクラス。"""

	id_column: str
	input_sources: tuple[str, ...]
	target_groups: tuple[str, ...]
	rolling_windows: tuple[int, ...]
	ewma_alpha: tuple[float, ...]
	recovery_clip: int
	clip_max: int
	flag_dtype: np.dtype
	run_dtype: np.dtype
	float_dtype: np.dtype
	output_prefix: str
	drop_constant_columns: bool
	include_rolling: bool
	include_ewma: bool
	include_transitions: bool
	include_normalization: bool
	fill_missing_with_zero: bool
	epsilon: float

	@classmethod
	def from_mapping(cls, mapping: Mapping[str, Any], *, base_dir: Path) -> "SU2Config":
		id_column = mapping.get("id_column", "date_id")
		
		input_sources = tuple(mapping.get("input_sources", ["m", "gap_ffill", "run_na", "run_obs"]))
		
		groups_mapping = mapping.get("target_groups", {})
		if isinstance(groups_mapping, dict):
			include_groups = set(groups_mapping.get("include", []))
			exclude_groups = set(groups_mapping.get("exclude", []))
			target_groups = tuple(sorted(include_groups.difference(exclude_groups)))
		else:
			# Fallback to su1 target_groups if not specified
			target_groups = tuple(["D", "M", "E", "I", "P", "S", "V"])
		
		if not target_groups:
			raise ValueError("SU2 configuration must specify at least one target group.")
		
		rolling_windows = tuple(mapping.get("rolling_windows", [5, 10, 20, 60]))
		ewma_alpha = tuple(mapping.get("ewma_alpha", [0.1, 0.3, 0.5]))
		recovery_clip = int(mapping.get("recovery_clip", 60))
		clip_max = int(mapping.get("clip_max", 60))
		
		dtype_section = mapping.get("dtype", {})
		flag_dtype = _coerce_dtype(dtype_section.get("flag", "uint8"))
		run_dtype = _coerce_dtype(dtype_section.get("run", "int16"))
		float_dtype = _coerce_dtype(dtype_section.get("float", "float32"))
		
		output_prefix = str(mapping.get("output_prefix", "su2"))
		drop_constant_columns = bool(mapping.get("drop_constant_columns", True))
		
		features_section = mapping.get("features", {})
		
		rolling_section = features_section.get("rolling", {})
		include_rolling = len(rolling_section.get("include_metrics", [])) > 0 if isinstance(rolling_section, dict) else True
		
		ewma_section = features_section.get("ewma", {})
		include_ewma = len(ewma_section.get("signals", [])) > 0 if isinstance(ewma_section, dict) else True
		
		transitions_section = features_section.get("transitions", {})
		include_transitions = bool(transitions_section) if isinstance(transitions_section, dict) else True
		
		normalization_section = features_section.get("normalization", {})
		include_normalization = bool(normalization_section) if isinstance(normalization_section, dict) else True
		
		fill_missing_with_zero = bool(rolling_section.get("fill_missing_with_zero", True)) if isinstance(rolling_section, dict) else True
		
		epsilon = float(normalization_section.get("epsilon", 1.0e-6)) if isinstance(normalization_section, dict) else 1.0e-6
		
		return cls(
			id_column=id_column,
			input_sources=input_sources,
			target_groups=target_groups,
			rolling_windows=rolling_windows,
			ewma_alpha=ewma_alpha,
			recovery_clip=recovery_clip,
			clip_max=clip_max,
			flag_dtype=flag_dtype,
			run_dtype=run_dtype,
			float_dtype=float_dtype,
			output_prefix=output_prefix,
			drop_constant_columns=drop_constant_columns,
			include_rolling=include_rolling,
			include_ewma=include_ewma,
			include_transitions=include_transitions,
			include_normalization=include_normalization,
			fill_missing_with_zero=fill_missing_with_zero,
			epsilon=epsilon,
		)


def load_su2_config(config_path: str | Path) -> SU2Config:
	"""SU2 設定 YAML を読み込み :class:`SU2Config` を生成する。"""
	path = Path(config_path).resolve()
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

	try:
		su2_section = full_cfg["su2"]
	except KeyError as exc:  # pragma: no cover - 防御的分岐
		raise KeyError("'su2' section is required in feature_generation.yaml") from exc

	return SU2Config.from_mapping(su2_section, base_dir=path.parent)


def _clip_array(values: np.ndarray, clip_value: int) -> np.ndarray:
	"""配列を上限値でクリップし、その参照を返す。"""
	np.clip(values, None, clip_value, out=values)
	return values


class SU2FeatureGenerator(BaseEstimator, TransformerMixin):
	"""SU1 出力から SU2 二次特徴量を生成するトランスフォーマー。"""

	def __init__(self, config: SU2Config):
		self.config = config
		self.feature_columns_: Dict[str, list[str]] | None = None

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU2FeatureGenerator":
		df = self._ensure_dataframe(X)
		self.feature_columns_ = self._select_feature_columns(df.columns)
		return self

	def transform(self, X: pd.DataFrame, *, fold_indices: np.ndarray | None = None) -> pd.DataFrame:
		if self.feature_columns_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		df = self._ensure_dataframe(X)
		
		# Generate features with fold boundary reset if provided
		feature_df = self._generate_features(df, fold_indices=fold_indices)
		feature_df.index = df.index
		return feature_df

	# ------------------------------------------------------------------
	# 内部ヘルパー
	# ------------------------------------------------------------------
	def _ensure_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
		if not isinstance(X, pd.DataFrame):  # pragma: no cover - 防御的分岐
			raise TypeError("SU2FeatureGenerator expects a pandas.DataFrame input")
		return X.copy()

	def _select_feature_columns(self, columns: Iterable[str]) -> Dict[str, list[str]]:
		"""入力列をソース別（m, gap_ffill, run_na, run_obs）に分類する。"""
		selected: Dict[str, list[str]] = {source: [] for source in self.config.input_sources}
		
		for column in columns:
			for source in self.config.input_sources:
				prefix = f"{source}/"
				if column.startswith(prefix):
					# Extract base column name and check group
					base_col = column[len(prefix):]
					group = _infer_group(base_col)
					if group and group in self.config.target_groups:
						selected[source].append(column)
					break
		
		return selected

	def _generate_features(self, df: pd.DataFrame, *, fold_indices: np.ndarray | None = None) -> pd.DataFrame:
		"""SU2 特徴量を生成する。"""
		feature_columns = self.feature_columns_
		if feature_columns is None:
			raise RuntimeError("The transformer must be fitted before generating features.")

		all_features: list[pd.DataFrame] = []
		
		# Determine fold boundaries
		if fold_indices is not None:
			fold_boundaries = self._get_fold_boundaries(fold_indices)
		else:
			fold_boundaries = [(0, len(df))]
		
		# Generate rolling statistics
		if self.config.include_rolling:
			rolling_features = self._generate_rolling_features(df, feature_columns, fold_boundaries)
			if rolling_features is not None and not rolling_features.empty:
				all_features.append(rolling_features)
		
		# Generate EWMA features
		if self.config.include_ewma:
			ewma_features = self._generate_ewma_features(df, feature_columns, fold_boundaries)
			if ewma_features is not None and not ewma_features.empty:
				all_features.append(ewma_features)
		
		# Generate transition features
		if self.config.include_transitions:
			transition_features = self._generate_transition_features(df, feature_columns, fold_boundaries)
			if transition_features is not None and not transition_features.empty:
				all_features.append(transition_features)
		
		# Generate normalization features
		if self.config.include_normalization:
			norm_features = self._generate_normalization_features(df, feature_columns, fold_boundaries)
			if norm_features is not None and not norm_features.empty:
				all_features.append(norm_features)
		
		if not all_features:
			return pd.DataFrame(index=df.index)
		
		result = pd.concat(all_features, axis=1)
		
		# Drop constant columns if requested
		if self.config.drop_constant_columns:
			result = self._drop_constant_columns(result)
		
		return result

	def _get_fold_boundaries(self, fold_indices: np.ndarray) -> list[tuple[int, int]]:
		"""折ごとの境界インデックスを取得する。"""
		boundaries: list[tuple[int, int]] = []
		unique_folds = np.unique(fold_indices)
		for fold in unique_folds:
			mask = fold_indices == fold
			indices = np.where(mask)[0]
			if len(indices) > 0:
				boundaries.append((int(indices[0]), int(indices[-1]) + 1))
		return boundaries

	def _generate_rolling_features(
		self, df: pd.DataFrame, feature_columns: Dict[str, list[str]], fold_boundaries: list[tuple[int, int]]
	) -> pd.DataFrame | None:
		"""ローリング統計特徴を生成する。"""
		features: MutableMapping[str, np.ndarray] = {}
		
		# For m/ columns: mean, std, zscore
		m_cols = feature_columns.get("m", [])
		for col in m_cols:
			base_col = col[len("m/"):]  # Remove "m/" prefix
			values = df[col].to_numpy(dtype=self.config.float_dtype)
			
			for window in self.config.rolling_windows:
				# Rolling mean (excluding current)
				mean_values = self._rolling_stat(values, window, stat="mean", fold_boundaries=fold_boundaries)
				features[f"{self.config.output_prefix}/roll_mean[{window}]/m/{base_col}"] = mean_values
				
				# Rolling std (excluding current)
				std_values = self._rolling_stat(values, window, stat="std", fold_boundaries=fold_boundaries)
				features[f"{self.config.output_prefix}/roll_std[{window}]/m/{base_col}"] = std_values
				
				# Z-score: (current - mean) / std
				zscore_values = np.zeros_like(values, dtype=self.config.float_dtype)
				for i in range(len(values)):
					if std_values[i] > self.config.epsilon:
						zscore_values[i] = (values[i] - mean_values[i]) / std_values[i]
				features[f"{self.config.output_prefix}/roll_zscore[{window}]/m/{base_col}"] = zscore_values
		
		# For run_na/ columns: max
		run_na_cols = feature_columns.get("run_na", [])
		for col in run_na_cols:
			base_col = col[len("run_na/"):]  # Remove "run_na/" prefix
			values = df[col].to_numpy(dtype=self.config.run_dtype)
			
			for window in self.config.rolling_windows:
				max_values = self._rolling_stat(values, window, stat="max", fold_boundaries=fold_boundaries)
				features[f"{self.config.output_prefix}/roll_max[{window}]/run_na/{base_col}"] = max_values
		
		# For run_obs/ columns: max
		run_obs_cols = feature_columns.get("run_obs", [])
		for col in run_obs_cols:
			base_col = col[len("run_obs/"):]  # Remove "run_obs/" prefix
			values = df[col].to_numpy(dtype=self.config.run_dtype)
			
			for window in self.config.rolling_windows:
				max_values = self._rolling_stat(values, window, stat="max", fold_boundaries=fold_boundaries)
				features[f"{self.config.output_prefix}/roll_max[{window}]/run_obs/{base_col}"] = max_values
		
		if not features:
			return None
		
		return pd.DataFrame(features, index=df.index)

	def _rolling_stat(
		self, values: np.ndarray, window: int, stat: str, fold_boundaries: list[tuple[int, int]]
	) -> np.ndarray:
		"""過去のみのローリング統計を計算する（現在値を含まない）。"""
		result = np.zeros(len(values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx, end_idx):
				# Calculate stat over past window (excluding current position i)
				hist_start = max(start_idx, i - window)
				hist_end = i  # Exclude current
				
				if hist_end <= hist_start:
					# No history available
					if self.config.fill_missing_with_zero:
						result[i] = 0.0
					else:
						result[i] = np.nan
					continue
				
				window_values = values[hist_start:hist_end]
				
				if stat == "mean":
					result[i] = float(np.mean(window_values))
				elif stat == "std":
					if len(window_values) > 1:
						result[i] = float(np.std(window_values, ddof=1))  # Unbiased std
					else:
						result[i] = 0.0
				elif stat == "max":
					result[i] = float(np.max(window_values))
				else:
					result[i] = 0.0
		
		return result

	def _generate_ewma_features(
		self, df: pd.DataFrame, feature_columns: Dict[str, list[str]], fold_boundaries: list[tuple[int, int]]
	) -> pd.DataFrame | None:
		"""EWMA/EWSTD 特徴を生成する。"""
		features: MutableMapping[str, np.ndarray] = {}
		
		# For m/ columns
		m_cols = feature_columns.get("m", [])
		for col in m_cols:
			base_col = col[len("m/"):]  # Remove "m/" prefix
			values = df[col].to_numpy(dtype=self.config.float_dtype)
			
			for alpha in self.config.ewma_alpha:
				ewma_values = self._ewma(values, alpha, fold_boundaries)
				features[f"{self.config.output_prefix}/ewma[{alpha}]/m/{base_col}"] = ewma_values
				
				ewstd_values = self._ewstd(values, alpha, fold_boundaries)
				features[f"{self.config.output_prefix}/ewstd[{alpha}]/m/{base_col}"] = ewstd_values
		
		# For gap_ffill/ columns
		gap_cols = feature_columns.get("gap_ffill", [])
		for col in gap_cols:
			base_col = col[len("gap_ffill/"):]  # Remove "gap_ffill/" prefix
			values = df[col].to_numpy(dtype=self.config.float_dtype)
			
			for alpha in self.config.ewma_alpha:
				ewma_values = self._ewma(values, alpha, fold_boundaries)
				features[f"{self.config.output_prefix}/ewma[{alpha}]/gap/{base_col}"] = ewma_values
		
		if not features:
			return None
		
		return pd.DataFrame(features, index=df.index)

	def _ewma(self, values: np.ndarray, alpha: float, fold_boundaries: list[tuple[int, int]]) -> np.ndarray:
		"""指数平滑移動平均を計算する（折境界でリセット）。"""
		result = np.zeros(len(values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			ewma = 0.0
			initialized = False
			
			for i in range(start_idx, end_idx):
				if not initialized:
					ewma = values[i]
					initialized = True
				else:
					ewma = alpha * values[i] + (1 - alpha) * ewma
				result[i] = ewma
		
		return result

	def _ewstd(self, values: np.ndarray, alpha: float, fold_boundaries: list[tuple[int, int]]) -> np.ndarray:
		"""指数平滑標準偏差を計算する（Welford型）。"""
		result = np.zeros(len(values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			mean = 0.0
			var = 0.0
			initialized = False
			
			for i in range(start_idx, end_idx):
				if not initialized:
					mean = values[i]
					var = 0.0
					initialized = True
				else:
					delta = values[i] - mean
					mean = alpha * values[i] + (1 - alpha) * mean
					var = (1 - alpha) * (var + alpha * delta * delta)
				result[i] = float(np.sqrt(max(0.0, var)))
		
		return result

	def _generate_transition_features(
		self, df: pd.DataFrame, feature_columns: Dict[str, list[str]], fold_boundaries: list[tuple[int, int]]
	) -> pd.DataFrame | None:
		"""遷移・レジーム統計特徴を生成する。"""
		features: MutableMapping[str, np.ndarray] = {}
		
		# Flip rate for m/ columns
		m_cols = feature_columns.get("m", [])
		for col in m_cols:
			base_col = col[len("m/"):]  # Remove "m/" prefix
			values = df[col].to_numpy(dtype=self.config.flag_dtype)
			
			for window in self.config.rolling_windows:
				flip_rate = self._flip_rate(values, window, fold_boundaries)
				features[f"{self.config.output_prefix}/flip_rate[{window}]/m/{base_col}"] = flip_rate
		
		# Burst score
		run_na_cols = feature_columns.get("run_na", [])
		run_obs_cols = feature_columns.get("run_obs", [])
		
		# Match run_na and run_obs columns
		for run_na_col in run_na_cols:
			base_col = run_na_col[len("run_na/"):]  # Remove "run_na/" prefix
			run_obs_col = f"run_obs/{base_col}"
			
			if run_obs_col in run_obs_cols:
				run_na_values = df[run_na_col].to_numpy(dtype=self.config.run_dtype)
				run_obs_values = df[run_obs_col].to_numpy(dtype=self.config.run_dtype)
				
				for window in self.config.rolling_windows:
					burst_score = self._burst_score(run_na_values, run_obs_values, window, fold_boundaries)
					features[f"{self.config.output_prefix}/burst_score[{window}]/{base_col}"] = burst_score
		
		# Recovery lag
		for run_na_col in run_na_cols:
			base_col = run_na_col[len("run_na/"):]  # Remove "run_na/" prefix
			run_obs_col = f"run_obs/{base_col}"
			
			if run_obs_col in run_obs_cols:
				run_na_values = df[run_na_col].to_numpy(dtype=self.config.run_dtype)
				run_obs_values = df[run_obs_col].to_numpy(dtype=self.config.run_dtype)
				
				recovery_lag = self._recovery_lag(run_na_values, run_obs_values, fold_boundaries)
				features[f"{self.config.output_prefix}/recovery_lag/{base_col}"] = recovery_lag
		
		if not features:
			return None
		
		return pd.DataFrame(features, index=df.index)

	def _flip_rate(self, values: np.ndarray, window: int, fold_boundaries: list[tuple[int, int]]) -> np.ndarray:
		"""0↔1 遷移回数 / W を計算する。"""
		result = np.zeros(len(values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx, end_idx):
				hist_start = max(start_idx, i - window + 1)
				hist_end = i + 1
				
				if hist_end - hist_start < 2:
					result[i] = 0.0
					continue
				
				window_values = values[hist_start:hist_end]
				flips = 0
				for j in range(1, len(window_values)):
					if window_values[j] != window_values[j - 1]:
						flips += 1
				
				result[i] = float(flips) / window
		
		return result

	def _burst_score(
		self, run_na_values: np.ndarray, run_obs_values: np.ndarray, window: int, fold_boundaries: list[tuple[int, int]]
	) -> np.ndarray:
		"""max_roll(run_na) / (max_roll(run_na) + max_roll(run_obs) + eps) を計算する。"""
		result = np.zeros(len(run_na_values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx, end_idx):
				hist_start = max(start_idx, i - window)
				hist_end = i
				
				if hist_end <= hist_start:
					result[i] = 0.0
					continue
				
				max_na = float(np.max(run_na_values[hist_start:hist_end]))
				max_obs = float(np.max(run_obs_values[hist_start:hist_end]))
				
				denominator = max_na + max_obs + self.config.epsilon
				result[i] = max_na / denominator
		
		return result

	def _recovery_lag(
		self, run_na_values: np.ndarray, run_obs_values: np.ndarray, fold_boundaries: list[tuple[int, int]]
	) -> np.ndarray:
		"""直近で run_na>0 から run_obs>0 に変化した時点からの経過日数を計算する。"""
		result = np.zeros(len(run_na_values), dtype=self.config.run_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			lag = 0
			last_recovery_idx = -1
			
			for i in range(start_idx, end_idx):
				# Detect recovery: transition from run_na>0 to run_obs>0
				if i > start_idx:
					prev_na = run_na_values[i - 1] > 0
					curr_obs = run_obs_values[i] > 0
					
					if prev_na and curr_obs:
						last_recovery_idx = i
						lag = 0
				
				if last_recovery_idx >= 0:
					lag = i - last_recovery_idx
					if lag > self.config.recovery_clip:
						lag = self.config.recovery_clip
				
				result[i] = lag
		
		return result

	def _generate_normalization_features(
		self, df: pd.DataFrame, feature_columns: Dict[str, list[str]], fold_boundaries: list[tuple[int, int]]
	) -> pd.DataFrame | None:
		"""正規化・スケーリング特徴を生成する。"""
		features: MutableMapping[str, np.ndarray] = {}
		
		# For gap_ffill/ columns: minmax and rank
		gap_cols = feature_columns.get("gap_ffill", [])
		for col in gap_cols:
			base_col = col[len("gap_ffill/"):]  # Remove "gap_ffill/" prefix
			values = df[col].to_numpy(dtype=self.config.float_dtype)
			
			for window in self.config.rolling_windows:
				minmax_values = self._minmax_normalize(values, window, fold_boundaries)
				features[f"{self.config.output_prefix}/minmax[{window}]/gap/{base_col}"] = minmax_values
				
				rank_values = self._rank_normalize(values, window, fold_boundaries)
				features[f"{self.config.output_prefix}/rank[{window}]/gap/{base_col}"] = rank_values
		
		if not features:
			return None
		
		return pd.DataFrame(features, index=df.index)

	def _minmax_normalize(self, values: np.ndarray, window: int, fold_boundaries: list[tuple[int, int]]) -> np.ndarray:
		"""過去Wに対するmin-max正規化を計算する（現在値を含まない過去のみ）。"""
		result = np.zeros(len(values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx, end_idx):
				hist_start = max(start_idx, i - window)
				hist_end = i  # Exclude current (past only)
				
				if hist_end <= hist_start:
					result[i] = 0.0
					continue
				
				window_values = values[hist_start:hist_end]
				min_val = float(np.min(window_values))
				max_val = float(np.max(window_values))
				
				denominator = max_val - min_val
				if denominator < self.config.epsilon:
					result[i] = 0.0
				else:
					result[i] = (values[i] - min_val) / denominator
		
		return result

	def _rank_normalize(self, values: np.ndarray, window: int, fold_boundaries: list[tuple[int, int]]) -> np.ndarray:
		"""過去W内ランク / W を計算する（0-1）（現在値を含まない過去のみ）。"""
		result = np.zeros(len(values), dtype=self.config.float_dtype)
		
		for start_idx, end_idx in fold_boundaries:
			for i in range(start_idx, end_idx):
				hist_start = max(start_idx, i - window)
				hist_end = i  # Exclude current (past only)
				
				if hist_end <= hist_start:
					result[i] = 0.0
					continue
				
				window_values = values[hist_start:hist_end]
				current_val = values[i]
				
				# Count how many values are less than current
				rank = int(np.sum(window_values < current_val))
				result[i] = float(rank) / window
		
		return result

	def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
		"""定数列（全欠損または全て同じ値）を削除する。"""
		to_drop = []
		for col in df.columns:
			values = df[col].dropna()
			if len(values) == 0 or values.nunique() <= 1:
				to_drop.append(col)
		
		if to_drop:
			df = df.drop(columns=to_drop)
		
		return df


def generate_su2_features(
	config_path: str | Path,
	su1_features: pd.DataFrame,
	*,
	fold_indices: np.ndarray | None = None,
) -> pd.DataFrame:
	"""High-level helper to produce SU2 features from SU1 output."""
	config = load_su2_config(config_path)
	generator = SU2FeatureGenerator(config)
	generator.fit(su1_features)
	return generator.transform(su1_features, fold_indices=fold_indices)


__all__ = [
	"SU2Config",
	"SU2FeatureGenerator",
	"generate_su2_features",
	"load_su2_config",
]
