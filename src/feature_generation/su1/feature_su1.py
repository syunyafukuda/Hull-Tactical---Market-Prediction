"""SU1（欠損構造コア特徴量）の生成ロジック。

本モジュールは ``docs/feature_generation/SU1.md`` に記載された方針を実装し、
scikit-learn 互換のトランスフォーマー ``SU1FeatureGenerator`` と、設定 YAML や
生データを読み込むためのヘルパー関数を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Sequence, cast

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
class SU1Config:
	"""SU1 特徴量生成に必要な設定を保持するデータクラス。"""

	id_column: str
	exclude_columns: tuple[str, ...]
	target_groups: tuple[str, ...]
	gap_clip: int
	run_clip: int
	flag_dtype: np.dtype
	run_dtype: np.dtype
	include_avg_gap: bool
	include_avg_run: bool
	exclude_all_nan_for_means: bool
	raw_dir: Path
	train_filename: str
	test_filename: str
	brushup_enabled: bool
	brushup_miss_count_window: int
	brushup_streak_threshold: int
	brushup_regime_recent_window: int
	brushup_regime_past_window: int
	brushup_regime_recent_threshold: float
	brushup_regime_past_threshold: float

	@classmethod
	def from_mapping(cls, mapping: Mapping[str, Any], *, base_dir: Path) -> "SU1Config":
		data_section = mapping.get("data", {})
		exclude_columns = tuple(mapping.get("exclude_columns", ()))

		groups_mapping = mapping.get("groups", {})
		include_groups = set(groups_mapping.get("include", []))
		exclude_groups = set(groups_mapping.get("exclude", []))
		target_groups = tuple(sorted(include_groups.difference(exclude_groups)))

		if not target_groups:
			raise ValueError("SU1 configuration must specify at least one target group.")

		id_column = mapping.get("id_column", "date_id")
		gap_clip = int(mapping.get("gap_clip", 60))
		run_clip = int(mapping.get("run_clip", gap_clip))

		dtype_section = mapping.get("dtype", {})
		flag_dtype = _coerce_dtype(dtype_section.get("flag", "uint8"))
		run_dtype = _coerce_dtype(dtype_section.get("run", "int16"))

		include_group_means = mapping.get("include_group_means", {})
		include_avg_gap = bool(include_group_means.get("gap_ffill", True))
		include_avg_run = bool(include_group_means.get("run_na", True))
		exclude_all_nan_for_means = bool(include_group_means.get("exclude_all_nan", False))

		raw_dir = _path_from_config(base_dir, data_section.get("raw_dir", "data/raw"))
		train_filename = data_section.get("train_filename", "train.csv")
		test_filename = data_section.get("test_filename", "test.csv")

		brushup_section = mapping.get("brushup", {})
		brushup_enabled = bool(brushup_section.get("enabled", False))
		brushup_miss_count_window = int(brushup_section.get("miss_count_window", 5))
		brushup_streak_threshold = int(brushup_section.get("streak_threshold", 3))
		regime_change = brushup_section.get("regime_change", {})
		brushup_regime_recent_window = int(regime_change.get("recent_window", 5))
		brushup_regime_past_window = int(regime_change.get("past_window", 30))
		brushup_regime_recent_threshold = float(regime_change.get("recent_threshold", 0.5))
		brushup_regime_past_threshold = float(regime_change.get("past_threshold", 0.1))

		return cls(
			id_column=id_column,
			exclude_columns=exclude_columns,
			target_groups=target_groups,
			gap_clip=gap_clip,
			run_clip=run_clip,
			flag_dtype=flag_dtype,
			run_dtype=run_dtype,
			include_avg_gap=include_avg_gap,
			include_avg_run=include_avg_run,
			exclude_all_nan_for_means=exclude_all_nan_for_means,
			raw_dir=raw_dir,
			train_filename=train_filename,
			test_filename=test_filename,
			brushup_enabled=brushup_enabled,
			brushup_miss_count_window=brushup_miss_count_window,
			brushup_streak_threshold=brushup_streak_threshold,
			brushup_regime_recent_window=brushup_regime_recent_window,
			brushup_regime_past_window=brushup_regime_past_window,
			brushup_regime_recent_threshold=brushup_regime_recent_threshold,
			brushup_regime_past_threshold=brushup_regime_past_threshold,
		)

	@property
	def train_path(self) -> Path:
		"""学習データ CSV への絶対パスを返す。"""

		return (self.raw_dir / self.train_filename).resolve()

	@property
	def test_path(self) -> Path:
		"""テストデータ CSV への絶対パスを返す。"""

		return (self.raw_dir / self.test_filename).resolve()


def load_su1_config(config_path: str | Path) -> SU1Config:
	"""SU1 設定 YAML を読み込み :class:`SU1Config` を生成する。"""

	path = Path(config_path).resolve()
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Mapping[str, Any] = yaml.safe_load(fh) or {}

	try:
		su1_section = full_cfg["su1"]
	except KeyError as exc:  # pragma: no cover - 防御的分岐
		raise KeyError("'su1' section is required in feature_generation.yaml") from exc

	return SU1Config.from_mapping(su1_section, base_dir=path.parent)


def load_raw_dataset(config: SU1Config, *, dataset: Literal["train", "test"] = "train") -> pd.DataFrame:
	"""SU1 特徴量生成用に生データセットを読み込む。"""

	if dataset not in {"train", "test"}:  # pragma: no cover - 防御的分岐
		raise ValueError("dataset must be 'train' or 'test'")

	csv_path = config.train_path if dataset == "train" else config.test_path
	if not csv_path.exists():
		raise FileNotFoundError(f"Raw data file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if config.id_column in df.columns:
		df = df.set_index(config.id_column)
	return df


def _clip_array(values: np.ndarray, clip_value: int) -> np.ndarray:
	"""配列を上限値でクリップし、その参照を返す。"""

	np.clip(values, None, clip_value, out=values)
	return values


def _distance_from_last_observation(mask: np.ndarray, clip: int, dtype: np.dtype) -> np.ndarray:
	"""NaN マスクから直近観測までの距離を算出する。"""

	out = np.zeros(mask.shape[0], dtype=dtype)
	last_obs_index = -1
	seen_obs = False
	for idx, is_missing in enumerate(mask):
		if is_missing:
			if not seen_obs:
				out[idx] = clip
			else:
				distance = idx - last_obs_index
				out[idx] = distance if distance <= clip else clip
		else:
			out[idx] = 0
			last_obs_index = idx
			seen_obs = True

	if not seen_obs:
		out.fill(0)
	return out


def _run_length(mask: np.ndarray, clip: int, dtype: np.dtype, *, target_missing: bool) -> np.ndarray:
	"""NaN または観測が連続する長さを算出する。"""

	out = np.zeros(mask.shape[0], dtype=dtype)
	counter = 0
	for idx, is_missing in enumerate(mask):
		condition = is_missing if target_missing else not is_missing
		if condition:
			counter = counter + 1 if counter < clip else clip
			out[idx] = counter
		else:
			counter = 0
			out[idx] = 0
	return out


class SU1FeatureGenerator(BaseEstimator, TransformerMixin):
	"""生データから SU1 欠損構造特徴量を生成するトランスフォーマー。"""

	def __init__(self, config: SU1Config):
		self.config = config
		self.feature_columns_: list[str] | None = None
		self.group_columns_: Dict[str, list[str]] | None = None

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU1FeatureGenerator":
		df = self._ensure_dataframe(X)
		self.feature_columns_ = self._select_feature_columns(df.columns)
		self.group_columns_ = self._build_group_columns(self.feature_columns_)
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		if self.feature_columns_ is None or self.group_columns_ is None:
			raise RuntimeError("The transformer must be fitted before calling transform().")

		df = self._ensure_dataframe(X)
		missing_columns = [col for col in self.feature_columns_ if col not in df.columns]
		if missing_columns:
			raise KeyError(f"Input dataframe is missing columns required for SU1: {missing_columns}")

		feature_df = self._generate_features(df)
		feature_df.index = df.index
		return feature_df

	# ------------------------------------------------------------------
	# 内部ヘルパー
	# ------------------------------------------------------------------
	def _ensure_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
		if not isinstance(X, pd.DataFrame):  # pragma: no cover - 防御的分岐
			raise TypeError("SU1FeatureGenerator expects a pandas.DataFrame input")
		return X.copy()

	def _select_feature_columns(self, columns: Iterable[str]) -> list[str]:
		selected: list[str] = []
		for column in columns:
			if column in self.config.exclude_columns:
				continue
			group = _infer_group(column)
			if group and group in self.config.target_groups:
				selected.append(column)
		if not selected:
			raise ValueError("No columns matched the SU1 configuration criteria.")
		return selected

	def _build_group_columns(self, columns: Sequence[str]) -> Dict[str, list[str]]:
		group_map: Dict[str, list[str]] = {group: [] for group in self.config.target_groups}
		for column in columns:
			group = _infer_group(column)
			if group in group_map:
				group_map[group].append(column)
		return group_map

	def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
		feature_columns = self.feature_columns_
		group_columns = self.group_columns_
		if feature_columns is None or group_columns is None:
			raise RuntimeError("The transformer must be fitted before generating features.")

		data = df[feature_columns]
		mask = data.isna()
		all_nan_series = cast(pd.Series, mask.all(axis=0))
		all_nan_lookup = cast(dict[str, bool], all_nan_series.to_dict())

		flag_df = mask.astype(self.config.flag_dtype)
		flag_df.columns = [f"m/{col}" for col in feature_columns]

		gap_data: MutableMapping[str, np.ndarray] = {}
		run_na_data: MutableMapping[str, np.ndarray] = {}
		run_obs_data: MutableMapping[str, np.ndarray] = {}

		for column in feature_columns:
			column_mask_source = mask[column]
			if isinstance(column_mask_source, pd.Series):
				column_mask = column_mask_source.to_numpy(dtype=bool)
			else:
				column_mask = np.asarray(column_mask_source, dtype=bool)
			gap_values = _distance_from_last_observation(
				column_mask, self.config.gap_clip, self.config.run_dtype
			)
			run_na_values = _run_length(
				column_mask,
				self.config.run_clip,
				self.config.run_dtype,
				target_missing=True,
			)
			run_obs_values = _run_length(column_mask, self.config.run_clip, self.config.run_dtype, target_missing=False)

			gap_data[f"gap_ffill/{column}"] = _clip_array(gap_values, self.config.gap_clip)
			run_na_data[f"run_na/{column}"] = _clip_array(run_na_values, self.config.run_clip)
			run_obs_data[f"run_obs/{column}"] = _clip_array(run_obs_values, self.config.run_clip)

		gap_df = pd.DataFrame(gap_data, index=data.index)
		run_na_df = pd.DataFrame(run_na_data, index=data.index)
		run_obs_df = pd.DataFrame(run_obs_data, index=data.index)

		m_any_day = flag_df.sum(axis=1).astype(self.config.run_dtype)
		m_rate_day = (m_any_day / len(feature_columns)).astype(np.float32)

		group_features: Dict[str, pd.Series] = {
			"m_any_day": m_any_day,
			"m_rate_day": m_rate_day,
			"m_cnt/ALL": m_any_day,
			"m_rate/ALL": m_rate_day,
		}

		for group, columns in group_columns.items():
			if not columns:
				continue
			flag_cols = [f"m/{col}" for col in columns]
			group_count = flag_df[flag_cols].sum(axis=1).astype(self.config.run_dtype)
			group_rate = (group_count / len(columns)).astype(np.float32)

			group_features[f"m_cnt/{group}"] = group_count
			group_features[f"m_rate/{group}"] = group_rate

			if self.config.include_avg_gap:
				gap_cols = [f"gap_ffill/{col}" for col in columns]
				gap_values = gap_df[gap_cols].copy()
				if self.config.exclude_all_nan_for_means:
					for orig_col, gap_col in zip(columns, gap_cols):
						if all_nan_lookup.get(orig_col, False):
							gap_values[gap_col] = np.nan
				gap_mean = cast(pd.Series, gap_values.mean(axis=1))
				group_features[f"avg_gapff/{group}"] = gap_mean.astype(np.float32)
			if self.config.include_avg_run:
				run_cols = [f"run_na/{col}" for col in columns]
				run_values = run_na_df[run_cols].copy()
				if self.config.exclude_all_nan_for_means:
					for orig_col, run_col in zip(columns, run_cols):
						if all_nan_lookup.get(orig_col, False):
							run_values[run_col] = np.nan
				run_mean = cast(pd.Series, run_values.mean(axis=1))
				group_features[f"avg_run_na/{group}"] = run_mean.astype(np.float32)

		aggregated_df = pd.DataFrame(group_features, index=data.index)

		output_frames = [flag_df, gap_df, run_na_df, run_obs_df, aggregated_df]
		
		# SU1 brushup features
		if self.config.brushup_enabled:
			brushup_features = self._generate_brushup_features(flag_df, run_na_df, feature_columns)
			output_frames.append(brushup_features)
		
		return pd.concat(output_frames, axis=1)

	def _generate_brushup_features(
		self, flag_df: pd.DataFrame, run_na_df: pd.DataFrame, feature_columns: list[str]
	) -> pd.DataFrame:
		"""Generate SU1 brushup features (5 new columns)."""
		brushup_data: Dict[str, pd.Series] = {}
		n_cols = len(feature_columns)
		
		# 1. miss_count_last_5d & miss_ratio_last_5d
		m_cols = [f"m/{col}" for col in feature_columns]
		daily_miss_count = flag_df[m_cols].sum(axis=1)
		miss_count_last_5d = daily_miss_count.rolling(
			window=self.config.brushup_miss_count_window, 
			min_periods=self.config.brushup_miss_count_window
		).sum()
		# Fill NaN with 0 before converting to int16
		miss_count_last_5d = miss_count_last_5d.fillna(0)
		brushup_data["miss_count_last_5d"] = miss_count_last_5d.astype(self.config.run_dtype)
		
		miss_ratio_last_5d = miss_count_last_5d / (self.config.brushup_miss_count_window * n_cols)
		brushup_data["miss_ratio_last_5d"] = miss_ratio_last_5d.astype(np.float32)
		
		# 2. is_long_missing_streak & long_streak_col_count
		run_na_cols = [f"run_na/{col}" for col in feature_columns]
		is_long_streak = (run_na_df[run_na_cols].max(axis=1) >= self.config.brushup_streak_threshold).astype(self.config.flag_dtype)
		brushup_data["is_long_missing_streak"] = is_long_streak
		
		long_streak_count = (run_na_df[run_na_cols] >= self.config.brushup_streak_threshold).sum(axis=1)
		brushup_data["long_streak_col_count"] = long_streak_count.astype(self.config.run_dtype)
		
		# 3. miss_regime_change
		regime_change_flags = []
		for col in feature_columns:
			m_col = f"m/{col}"
			# Recent 5-day missingness rate
			recent_miss_rate = flag_df[m_col].rolling(
				window=self.config.brushup_regime_recent_window
			).mean()
			# Past 30-day missingness rate (shifted by 5 to avoid overlap)
			past_miss_rate = flag_df[m_col].shift(
				self.config.brushup_regime_recent_window
			).rolling(
				window=self.config.brushup_regime_past_window
			).mean()
			
			# Regime change: recent > threshold AND past < threshold
			is_change = (
				(recent_miss_rate > self.config.brushup_regime_recent_threshold) & 
				(past_miss_rate < self.config.brushup_regime_past_threshold)
			)
			regime_change_flags.append(is_change)
		
		# Any column has regime change
		regime_change_df = pd.DataFrame(regime_change_flags).T
		miss_regime_change = regime_change_df.any(axis=1).fillna(False).astype(self.config.flag_dtype)
		brushup_data["miss_regime_change"] = miss_regime_change
		
		return pd.DataFrame(brushup_data, index=flag_df.index)


def generate_su1_features(
	config_path: str | Path,
	*,
	dataset: Literal["train", "test"] = "train",
) -> pd.DataFrame:
	"""High-level helper to produce SU1 features from raw data."""

	config = load_su1_config(config_path)
	raw_df = load_raw_dataset(config, dataset=dataset)
	generator = SU1FeatureGenerator(config)
	generator.fit(raw_df)
	return generator.transform(raw_df)


__all__ = [
	"SU1Config",
	"SU1FeatureGenerator",
	"generate_su1_features",
	"load_raw_dataset",
	"load_su1_config",
]

