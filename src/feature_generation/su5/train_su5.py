#!/usr/bin/env python
"""SU5 特徴量バンドルの学習エントリーポイント。

本スクリプトは生データから SU1 特徴量を生成し、その上に SU5 共欠損特徴量を追加して
軽量な前処理パイプラインを通し、LightGBM 回帰器を学習する。生成された
``sklearn.Pipeline``（特徴量生成＋前処理＋モデル）は
``artifacts/SU5/inference_bundle.pkl`` に保存され、推論時に同じ処理フローを再利用できる。

主な役割
--------
* SU1/SU5 用 YAML 設定を読み込む。
* :class:`SU1FeatureGenerator` + :class:`SU5FeatureGenerator` で特徴量を作成し、生データの説明変数へ連結する。
* 時系列分割で OOF 指標を算出し、挙動を記録する。
* 全学習データで再学習したパイプラインやメタ情報、特徴量リストを成果物として出力する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import sklearn
import yaml

try:
	import lightgbm as lgb  # type: ignore
	from lightgbm import LGBMRegressor  # type: ignore
	HAS_LGBM = True
except Exception:
	LGBMRegressor = None  # type: ignore
	lgb = None  # type: ignore
	HAS_LGBM = False

import joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))


def _ensure_numpy_bitgenerator_aliases() -> None:
	"""Register legacy numpy BitGenerator names so joblib pickles stay portable."""
	try:
		import numpy as _np
	except Exception:
		return

	generator_cls = getattr(_np.random, "MT19937", None)
	if generator_cls is None:
		return

	alias_keys = (
		"<class 'numpy.random._mt19937.MT19937'>",
		"numpy.random._mt19937.MT19937",
		"numpy.random.bit_generator.MT19937",
		"MT19937",
	)

	registries: list[dict[str, Any]] = []
	try:
		import numpy.random._pickle as _np_random_pickle  # type: ignore
	except Exception:
		_np_random_pickle = None  # type: ignore

	if _np_random_pickle is not None:
		registry = getattr(_np_random_pickle, "BitGenerators", None)
		if isinstance(registry, dict):
			registries.append(registry)

	try:
		import numpy.random.bit_generator as _np_bitgen  # type: ignore
	except Exception:
		_np_bitgen = None  # type: ignore

	if _np_bitgen is not None:
		for attr_name in ("_bit_generators", "BitGenerators"):
			registry = getattr(_np_bitgen, attr_name, None)
			if isinstance(registry, dict):
				registries.append(registry)

	seen: set[int] = set()
	unique_registries: list[dict[str, Any]] = []
	for registry in registries:
		registry_id = id(registry)
		if registry_id not in seen:
			unique_registries.append(registry)
			seen.add(registry_id)

	for registry in unique_registries:
		for key in alias_keys:
			registry.setdefault(key, generator_cls)

	if not unique_registries:
		try:
			import numpy.random._pickle as _np_random_pickle  # type: ignore
		except Exception:
			return
		registry = getattr(_np_random_pickle, "BitGenerators", None)
		if isinstance(registry, dict):
			for key in alias_keys:
				registry.setdefault(key, generator_cls)


_ensure_numpy_bitgenerator_aliases()

from preprocess.E_group.e_group import EGroupImputer  # noqa: E402
from preprocess.I_group.i_group import IGroupImputer  # noqa: E402
from preprocess.M_group.m_group import MGroupImputer  # noqa: E402
from preprocess.P_group.p_group import PGroupImputer  # noqa: E402
from preprocess.S_group.s_group import SGroupImputer  # noqa: E402
from scripts.utils_msr import (  # noqa: E402
	PostProcessParams,
	evaluate_msr_proxy,
	grid_search_msr,
)
from src.feature_generation.su1.feature_su1 import (  # noqa: E402
	SU1Config,
	SU1FeatureGenerator,
	load_su1_config,
)
from src.feature_generation.su5.feature_su5 import (  # noqa: E402
	SU5Config,
	SU5FeatureGenerator,
	load_su5_config,
)


class SU5FeatureAugmenter(BaseEstimator, TransformerMixin):
	"""SU1 + SU5 特徴量を入力フレームへ追加するトランスフォーマー。"""

	def __init__(
		self,
		su1_config: SU1Config,
		su5_config: SU5Config,
		fill_value: float | None = 0.0,
	) -> None:
		self.su1_config = su1_config
		self.su5_config = su5_config
		self.fill_value = fill_value

	def fit(self, X: pd.DataFrame, y: Any = None) -> "SU5FeatureAugmenter":
		frame = self._ensure_dataframe(X)
		# SU1 fit
		self.su1_generator_ = SU1FeatureGenerator(self.su1_config)
		self.su1_generator_.fit(frame)
		su1_features = self.su1_generator_.transform(frame)
		# SU5 fit (using SU1 features)
		# Note: fold_indices is not needed during fit() because SU5FeatureGenerator.fit()
		# only computes global co-missingness statistics for top-k pair selection.
		# fold_indices is only required during transform() for fold-aware rolling features.
		self.su5_generator_ = SU5FeatureGenerator(self.su5_config)
		self.su5_generator_.fit(su1_features)
		su5_features = self.su5_generator_.transform(su1_features)

		# Store feature names
		self.su1_feature_names_ = list(su1_features.columns)
		self.su5_feature_names_ = list(su5_features.columns)
		self.input_columns_ = list(frame.columns)

		return self

	def transform(
		self, X: pd.DataFrame, fold_indices: np.ndarray | None = None
	) -> pd.DataFrame:
		if not hasattr(self, "su1_generator_"):
			raise RuntimeError("SU5FeatureAugmenter must be fitted before transform().")

		frame = self._ensure_dataframe(X)
		# Generate SU1 features
		su1_features = self.su1_generator_.transform(frame)
		su1_features = su1_features.reindex(columns=self.su1_feature_names_, copy=True)
		if self.fill_value is not None:
			su1_features = su1_features.fillna(self.fill_value)

		# Generate SU5 features
		su5_features = self.su5_generator_.transform(su1_features, fold_indices=fold_indices)
		su5_features = su5_features.reindex(columns=self.su5_feature_names_, copy=True)
		if self.fill_value is not None:
			su5_features = su5_features.fillna(self.fill_value)

		# Concatenate: original + SU1 + SU5
		augmented = pd.concat([frame, su1_features, su5_features], axis=1)
		augmented.index = frame.index
		return augmented

	@staticmethod
	def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
		if not isinstance(X, pd.DataFrame):
			raise TypeError("SU5FeatureAugmenter expects a pandas.DataFrame input")
		return X.copy()


def infer_train_file(data_dir: Path, explicit: str | None) -> Path:
	if explicit:
		candidate = Path(explicit)
		if not candidate.exists():
			raise FileNotFoundError(f"--train-file not found: {candidate}")
		return candidate
	for name in ("train.parquet", "train.csv", "Train.parquet", "Train.csv"):
		candidate = data_dir / name
		if candidate.exists():
			return candidate
	for ext in ("parquet", "csv"):
		discovered = list(data_dir.glob(f"*.{ext}"))
		if discovered:
			return discovered[0]
	raise FileNotFoundError(f"No train file found under {data_dir}")


def infer_test_file(data_dir: Path, explicit: str | None) -> Path:
	if explicit:
		candidate = Path(explicit)
		if not candidate.exists():
			raise FileNotFoundError(f"--test-file not found: {candidate}")
		return candidate
	for name in ("test.parquet", "test.csv", "Test.parquet", "Test.csv"):
		candidate = data_dir / name
		if candidate.exists():
			return candidate
	for ext in ("parquet", "csv"):
		discovered = list(data_dir.glob(f"*.{ext}"))
		if not discovered:
			continue
		discovered.sort(key=lambda p: ("test" not in p.stem.lower(), p.name))
		return discovered[0]
	raise FileNotFoundError(f"No test file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
	suffix = path.suffix.lower()
	if suffix == ".parquet":
		return pd.read_parquet(path)
	if suffix == ".csv":
		return pd.read_csv(path)
	raise ValueError(f"Unsupported extension: {path.suffix}")


def _to_1d(pred: Any) -> np.ndarray:
	array = np.asarray(pred)
	if array.ndim > 1:
		array = array.ravel()
	return array.astype(float, copy=False)


def _normalise_scalar(value: Any) -> Any:
	if isinstance(value, (np.generic,)):
		return value.item()
	if isinstance(value, np.ndarray):
		return [_normalise_scalar(v) for v in value.tolist()]
	if value is pd.NA:  # type: ignore[attr-defined]
		return None
	if isinstance(value, (np.dtype, pd.api.extensions.ExtensionDtype)):
		return str(value)
	if isinstance(value, (pd.Series, pd.Index)):
		return [_normalise_scalar(v) for v in value.tolist()]
	if isinstance(value, (list, tuple, set)):
		return [_normalise_scalar(v) for v in value]
	if isinstance(value, dict):
		return {str(key): _normalise_scalar(val) for key, val in value.items()}
	if isinstance(value, Path):
		return str(value)
	return value


def load_preprocess_policies(config_path: str | Path) -> Dict[str, Dict[str, Any]]:
	path = Path(config_path).resolve()
	if not path.exists():
		raise FileNotFoundError(f"preprocess config not found: {path}")
	with path.open("r", encoding="utf-8") as fh:
		full_cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

	def _calendar_value(raw: Any) -> str | None:
		if raw is None:
			return None
		value = str(raw).strip()
		return value or None

	def _policy_params(section: Mapping[str, Any]) -> Dict[str, Any]:
		raw_params = section.get("policy_params", {}) if isinstance(section, Mapping) else {}
		if not isinstance(raw_params, Mapping):
			return {}
		return {str(k): _normalise_scalar(v) for k, v in raw_params.items()}

	groups = {key: full_cfg.get(key, {}) or {} for key in ("m_group", "e_group", "i_group", "p_group", "s_group")}
	settings: Dict[str, Dict[str, Any]] = {}
	for key, section in groups.items():
		if not isinstance(section, Mapping):
			section = {}
		common: Dict[str, Any] = {
			"policy": str(section.get("policy", "ffill_bfill")),
			"rolling_window": int(section.get("rolling_window", 5)),
			"ema_alpha": float(section.get("ema_alpha", 0.3)),
			"calendar_column": _calendar_value(section.get("calendar_column")),
			"policy_params": _policy_params(section),
		}
		settings[key] = common

		if key == "e_group":
			common["all_nan_strategy"] = str(section.get("all_nan_strategy", "keep_nan"))
			common["all_nan_fill"] = float(section.get("all_nan_fill", 0.0))
		elif key == "i_group":
			common["clip_quantile_low"] = float(section.get("clip_quantile_low", 0.001))
			common["clip_quantile_high"] = float(section.get("clip_quantile_high", 0.999))
			disable_flag = section.get("disable_quantile_clip")
			if disable_flag is not None:
				common["enable_quantile_clip"] = not bool(disable_flag)
			else:
				common["enable_quantile_clip"] = bool(section.get("enable_quantile_clip", True))
		elif key in {"p_group", "s_group"}:
			common["mad_clip_scale"] = float(section.get("mad_clip_scale", 4.0))
			common["mad_clip_min_samples"] = int(section.get("mad_clip_min_samples", 25))
			disable_flag = section.get("disable_mad_clip")
			if disable_flag is not None:
				common["enable_mad_clip"] = not bool(disable_flag)
			else:
				common["enable_mad_clip"] = bool(section.get("enable_mad_clip", True))
			common["fallback_quantile_low"] = float(section.get("fallback_quantile_low", 0.005))
			common["fallback_quantile_high"] = float(section.get("fallback_quantile_high", 0.995))
	return settings


def _collect_imputer_metadata(pipeline: Pipeline) -> Dict[str, Dict[str, Any]]:
	metadata: Dict[str, Dict[str, Any]] = {}
	for step_name in ("m_imputer", "e_imputer", "i_imputer", "p_imputer", "s_imputer"):
		imputer = pipeline.named_steps.get(step_name)
		if imputer is None:
			continue
		step_meta: Dict[str, Any] = {
			"policy_requested": getattr(imputer, "policy_requested", getattr(imputer, "policy", None)),
			"policy": getattr(imputer, "policy", None),
			"rolling_window": getattr(imputer, "rolling_window", None),
			"ema_alpha": getattr(imputer, "ema_alpha", None),
			"calendar_column": getattr(imputer, "_calendar_column_name_", getattr(imputer, "calendar_column", None)),
			"policy_params": _normalise_scalar(getattr(imputer, "_policy_params", getattr(imputer, "policy_params", {}))),
			"columns": list(getattr(imputer, "columns_", [])),
			"generated_columns": list(getattr(imputer, "extra_columns_", [])),
		}
		state = getattr(imputer, "_state_", None)
		if isinstance(state, dict):
			warnings = state.get("warnings")
			if isinstance(warnings, list):
				step_meta["warnings"] = [_normalise_scalar(msg) for msg in warnings]
			for key in ("clip_bounds", "mad_clip_bounds", "all_nan_columns"):
				if key in state:
					step_meta[key] = _normalise_scalar(state[key])
		metadata[step_name] = step_meta
	return metadata


def _build_preprocess(num_fill_value: float, *, handle_unknown: str = "ignore") -> ColumnTransformer:
	numeric_selector = make_column_selector(dtype_include=np.number)  # type: ignore[arg-type]
	categorical_selector = make_column_selector(dtype_exclude=np.number)  # type: ignore[arg-type]

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="constant", fill_value=num_fill_value)),
		]
	)

	encoder_kwargs: Dict[str, Any] = {"handle_unknown": handle_unknown}
	try:
		encoder = OneHotEncoder(sparse_output=False, **encoder_kwargs)
	except TypeError:
		encoder = OneHotEncoder(sparse=False, **encoder_kwargs)  # type: ignore[call-arg]

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", encoder),
		]
	)

	transformer = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_selector),
			("cat", categorical_pipeline, categorical_selector),
		],
		remainder="drop",
		verbose_feature_names_out=False,
	)
	return transformer


def build_pipeline(
	su1_config: SU1Config,
	su5_config: SU5Config,
	preprocess_settings: Mapping[str, Dict[str, Any]],
	*,
	numeric_fill_value: float,
	model_kwargs: Dict[str, Any],
	random_state: int,
) -> Pipeline:
	augmenter = SU5FeatureAugmenter(su1_config, su5_config, fill_value=numeric_fill_value)
	m_cfg = preprocess_settings.get("m_group", {})
	e_cfg = preprocess_settings.get("e_group", {})
	i_cfg = preprocess_settings.get("i_group", {})
	p_cfg = preprocess_settings.get("p_group", {})
	s_cfg = preprocess_settings.get("s_group", {})

	m_imputer = MGroupImputer(
		columns=None,
		policy=str(m_cfg.get("policy", "ffill_bfill")),
		rolling_window=int(m_cfg.get("rolling_window", 5)),
		ema_alpha=float(m_cfg.get("ema_alpha", 0.3)),
		calendar_column=m_cfg.get("calendar_column"),
		policy_params=m_cfg.get("policy_params", {}),
		random_state=random_state,
	)
	e_imputer = EGroupImputer(
		columns=None,
		policy=str(e_cfg.get("policy", "ffill_bfill")),
		rolling_window=int(e_cfg.get("rolling_window", 5)),
		ema_alpha=float(e_cfg.get("ema_alpha", 0.3)),
		calendar_column=e_cfg.get("calendar_column"),
		policy_params=e_cfg.get("policy_params", {}),
		random_state=random_state,
		all_nan_strategy=str(e_cfg.get("all_nan_strategy", "keep_nan")),
		all_nan_fill=float(e_cfg.get("all_nan_fill", 0.0)),
	)
	i_imputer = IGroupImputer(
		columns=None,
		policy=str(i_cfg.get("policy", "ffill_bfill")),
		rolling_window=int(i_cfg.get("rolling_window", 5)),
		ema_alpha=float(i_cfg.get("ema_alpha", 0.3)),
		calendar_column=i_cfg.get("calendar_column"),
		policy_params=i_cfg.get("policy_params", {}),
		random_state=random_state,
		clip_quantile_low=float(i_cfg.get("clip_quantile_low", 0.001)),
		clip_quantile_high=float(i_cfg.get("clip_quantile_high", 0.999)),
		enable_quantile_clip=bool(i_cfg.get("enable_quantile_clip", True)),
	)
	p_imputer = PGroupImputer(
		columns=None,
		policy=str(p_cfg.get("policy", "ffill_bfill")),
		rolling_window=int(p_cfg.get("rolling_window", 5)),
		ema_alpha=float(p_cfg.get("ema_alpha", 0.3)),
		calendar_column=p_cfg.get("calendar_column"),
		policy_params=p_cfg.get("policy_params", {}),
		random_state=random_state,
		mad_clip_scale=float(p_cfg.get("mad_clip_scale", 4.0)),
		mad_clip_min_samples=int(p_cfg.get("mad_clip_min_samples", 25)),
		enable_mad_clip=bool(p_cfg.get("enable_mad_clip", True)),
		fallback_quantile_low=float(p_cfg.get("fallback_quantile_low", 0.005)),
		fallback_quantile_high=float(p_cfg.get("fallback_quantile_high", 0.995)),
	)
	s_imputer = SGroupImputer(
		columns=None,
		policy=str(s_cfg.get("policy", "ffill_bfill")),
		rolling_window=int(s_cfg.get("rolling_window", 5)),
		ema_alpha=float(s_cfg.get("ema_alpha", 0.3)),
		calendar_column=s_cfg.get("calendar_column"),
		policy_params=s_cfg.get("policy_params", {}),
		random_state=random_state,
		mad_clip_scale=float(s_cfg.get("mad_clip_scale", 4.0)),
		mad_clip_min_samples=int(s_cfg.get("mad_clip_min_samples", 25)),
		enable_mad_clip=bool(s_cfg.get("enable_mad_clip", True)),
		fallback_quantile_low=float(s_cfg.get("fallback_quantile_low", 0.005)),
		fallback_quantile_high=float(s_cfg.get("fallback_quantile_high", 0.995)),
	)

	preprocess = _build_preprocess(numeric_fill_value)
	if not HAS_LGBM or LGBMRegressor is None:  # type: ignore[truthy-function]
		raise RuntimeError("LightGBM is required but not installed. Please install 'lightgbm'.")
	model = LGBMRegressor(**model_kwargs)  # type: ignore[arg-type]
	steps = [
		("augment", augmenter),
		("m_imputer", m_imputer),
		("e_imputer", e_imputer),
		("i_imputer", i_imputer),
		("p_imputer", p_imputer),
		("s_imputer", s_imputer),
		("preprocess", preprocess),
		("model", model),
	]
	return Pipeline(steps=steps)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Train a LightGBM model with SU5 features.")
	ap.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing train/test files")
	ap.add_argument("--train-file", type=str, default=None, help="Explicit path to the training file")
	ap.add_argument("--test-file", type=str, default=None, help="Explicit path to the test file")
	ap.add_argument("--config-path", type=str, default="configs/feature_generation.yaml", help="Path to feature_generation.yaml")
	ap.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml", help="Path to preprocess.yaml")
	ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
	ap.add_argument("--id-col", type=str, default="date_id")
	ap.add_argument("--out-dir", type=str, default="artifacts/SU5")
	ap.add_argument("--numeric-fill-value", type=float, default=0.0, help="Value used to fill numeric NaNs after feature generation")
	ap.add_argument("--n-splits", type=int, default=5, help="Number of folds for TimeSeriesSplit")
	ap.add_argument("--gap", type=int, default=0, help="Gap between train and validation indices in each fold")
	ap.add_argument("--min-val-size", type=int, default=0, help="Skip folds where validation after gap is smaller than this size")
	ap.add_argument("--learning-rate", type=float, default=0.05)
	ap.add_argument("--n-estimators", type=int, default=600)
	ap.add_argument("--num-leaves", type=int, default=63)
	ap.add_argument("--min-data-in-leaf", type=int, default=32)
	ap.add_argument("--feature-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-freq", type=int, default=1)
	ap.add_argument("--random-state", type=int, default=42)
	ap.add_argument("--verbosity", type=int, default=-1, help="LightGBM verbosity level")
	ap.add_argument("--no-artifacts", action="store_true", help="If set, do not write artifacts to disk")
	ap.add_argument(
		"--signal-optimize-for",
		type=str,
		choices=("msr", "msr_down", "vmsr"),
		default="msr",
		help="Metric used to select signal post-process parameters",
	)
	ap.add_argument(
		"--signal-mult-grid",
		type=float,
		nargs="+",
		default=(0.5, 0.75, 1.0, 1.25, 1.5),
		help="Grid of multipliers for signal scaling (mult)",
	)
	ap.add_argument(
		"--signal-lo-grid",
		type=float,
		nargs="+",
		default=(0.8, 0.9, 1.0),
		help="Grid of lower clip bounds for signal",
	)
	ap.add_argument(
		"--signal-hi-grid",
		type=float,
		nargs="+",
		default=(1.0, 1.1, 1.2),
		help="Grid of upper clip bounds for signal",
	)
	ap.add_argument(
		"--signal-lam-grid",
		type=float,
		nargs="+",
		default=(0.0,),
		help="Grid of lambda values when optimizing vMSR",
	)
	ap.add_argument(
		"--signal-eps",
		type=float,
		default=1e-8,
		help="Stability epsilon used in MSR calculations",
	)
	return ap.parse_args(argv)


def _prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    id_col: str,
    exclude_lagged: bool = True,
) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrices for training and testing.

    By default, columns starting with 'lagged_' are excluded from features.
    This is to avoid potential data leakage or unintended use of lagged features.
    Set exclude_lagged=False to include lagged features if desired.
    """
    drop_cols = {
        target_col,
        id_col,
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
        "is_scored",
    }

    y_series = cast(pd.Series, train_df[target_col])
    y = y_series.astype(float)

    candidate_cols = [c for c in train_df.columns if c not in drop_cols]
    test_cols = set(test_df.columns) - drop_cols
    if exclude_lagged:
        use_cols = [c for c in candidate_cols if c in test_cols and not c.startswith("lagged_")]
    else:
        use_cols = [c for c in candidate_cols if c in test_cols]
    if not use_cols:
        raise RuntimeError("No usable feature columns after intersecting train/test.")

    X = cast(pd.DataFrame, train_df[use_cols].copy())
    return X, y, use_cols
def _initialise_callbacks(model: Any) -> List[Any]:
	callbacks: List[Any] = []
	if not HAS_LGBM or lgb is None:
		return callbacks
	n_estimators = int(model.get_params().get("n_estimators", 100))
	log_period = max(1, n_estimators // 10)
	try:
		callbacks.append(lgb.log_evaluation(period=log_period))

		def _progress(env: Any) -> None:
			iteration = getattr(env, "iteration", 0)
			if iteration % log_period != 0 and iteration != n_estimators:
				return
			pct = 0.0 if n_estimators == 0 else 100.0 * iteration / n_estimators
			print(f"[progress] iteration {iteration}/{n_estimators} ({pct:5.1f}%)")

		callbacks.append(_progress)
	except Exception:
		callbacks = []
	return callbacks


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
	with path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)

	data_dir = Path(args.data_dir)
	out_dir = Path(args.out_dir)
	if not args.no_artifacts:
		out_dir.mkdir(parents=True, exist_ok=True)

	su1_config = load_su1_config(args.config_path)
	su5_config = load_su5_config(args.config_path)
	preprocess_settings = load_preprocess_policies(args.preprocess_config)

	train_path = infer_train_file(data_dir, args.train_file)
	test_path = infer_test_file(data_dir, args.test_file)
	print(f"[info] train file: {train_path}")
	print(f"[info] test file : {test_path}")

	train_df = load_table(train_path)
	test_df = load_table(test_path)

	if args.id_col in train_df.columns:
		train_df = train_df.sort_values(args.id_col).reset_index(drop=True)
	if args.id_col in test_df.columns:
		test_df = test_df.sort_values(args.id_col).reset_index(drop=True)

	if args.target_col not in train_df.columns:
		raise KeyError(f"Target column '{args.target_col}' was not found in train data.")

	X, y, feature_cols = _prepare_features(train_df, test_df, target_col=args.target_col, id_col=args.id_col)

	model_kwargs = {
		"learning_rate": args.learning_rate,
		"n_estimators": args.n_estimators,
		"num_leaves": args.num_leaves,
		"min_data_in_leaf": args.min_data_in_leaf,
		"feature_fraction": args.feature_fraction,
		"bagging_fraction": args.bagging_fraction,
		"bagging_freq": args.bagging_freq,
		"random_state": args.random_state,
		"n_jobs": -1,
		"verbosity": args.verbosity,
	}
	signal_optimize_for = args.signal_optimize_for
	signal_mult_grid = tuple(float(x) for x in (args.signal_mult_grid or (1.0,)))
	signal_lo_grid = tuple(float(x) for x in (args.signal_lo_grid or (0.0,)))
	signal_hi_grid = tuple(float(x) for x in (args.signal_hi_grid or (2.0,)))
	signal_lam_grid = tuple(float(x) for x in (args.signal_lam_grid or (0.0,)))
	signal_eps = float(args.signal_eps)
	if not signal_mult_grid or not signal_lo_grid or not signal_hi_grid:
		raise ValueError("Signal post-process grids must not be empty.")
	base_pipeline = build_pipeline(
		su1_config,
		su5_config,
		preprocess_settings,
		numeric_fill_value=args.numeric_fill_value,
		model_kwargs=model_kwargs,
		random_state=args.random_state,
	)
	callbacks = _initialise_callbacks(base_pipeline.named_steps["model"])

	# CV with fold_indices for SU5
	splitter = TimeSeriesSplit(n_splits=args.n_splits)
	X_np = X.reset_index(drop=True)
	y_np = y.reset_index(drop=True)
	y_np_array = y_np.to_numpy()

	# Pre-fit augmenter for CV
	su5_prefit = SU5FeatureAugmenter(su1_config, su5_config, fill_value=args.numeric_fill_value)
	su5_prefit.fit(X_np)

	# Build fold_indices array for entire dataset
	fold_indices_full = np.full(len(X_np), -1, dtype=int)
	# Rationale for fold_indices assignment:
	# Unlike the SU2-style approach (see sweep_oof.py lines 183-188), where only validation indices
	# are assigned the fold index and train indices are left as -1, here both train and validation
	# indices are assigned the same fold_idx value for each fold. This is required by SU5FeatureGenerator,
	# which uses the fold_indices array to generate features consistently for all samples in each fold.
	# The reset_each_fold logic in SU5FeatureGenerator expects fold labels for every sample, not just
	# validation samples, so that feature augmentation is performed in a way that is consistent within
	# each fold. This ensures that both train and validation data for a fold are processed with the same
	# fold context, which is necessary for reproducibility and correct feature generation in SU5.
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np)):
		fold_indices_full[train_idx] = fold_idx
		fold_indices_full[val_idx] = fold_idx

	# Transform with fold_indices
	X_augmented_all = su5_prefit.transform(X_np, fold_indices=fold_indices_full)
	core_pipeline_template = cast(Pipeline, Pipeline(base_pipeline.steps[1:]))

	oof_pred = np.full(len(X_np), np.nan, dtype=float)
	fold_logs: List[Dict[str, Any]] = []

	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np), start=1):
		train_idx = np.array(train_idx)
		val_idx = np.array(val_idx)

		if args.gap > 0:
			if len(train_idx) > args.gap:
				train_idx = train_idx[:-args.gap]
			if len(val_idx) > args.gap:
				val_idx = val_idx[args.gap:]
		if len(train_idx) == 0 or len(val_idx) == 0:
			print(f"[warn][fold {fold_idx}] skipped due to empty train/val after gap application")
			continue
		if args.min_val_size and len(val_idx) < args.min_val_size:
			print(f"[warn][fold {fold_idx}] skipped because validation size {len(val_idx)} < min-val-size")
			continue

		X_train = X_augmented_all.iloc[train_idx]
		y_train = y_np.iloc[train_idx]
		X_valid = X_augmented_all.iloc[val_idx]
		y_valid = y_np.iloc[val_idx]

		pipe = cast(Pipeline, clone(core_pipeline_template))
		fit_kwargs: Dict[str, Any] = {}
		if callbacks:
			fit_kwargs["model__callbacks"] = callbacks
			fit_kwargs["model__eval_set"] = [(X_valid, y_valid)]
			fit_kwargs["model__eval_metric"] = "rmse"

		pipe.fit(X_train, y_train, **fit_kwargs)
		pred = pipe.predict(X_valid)
		pred = _to_1d(pred)
		rmse = float(math.sqrt(mean_squared_error(y_valid, pred)))
		oof_pred[val_idx] = pred

		fold_params, fold_grid = grid_search_msr(
			y_pred=pred,
			y_true=y_valid.to_numpy(),
			mult_grid=signal_mult_grid,
			lo_grid=signal_lo_grid,
			hi_grid=signal_hi_grid,
			eps=signal_eps,
			optimize_for=signal_optimize_for,
			lam_grid=signal_lam_grid if signal_optimize_for == "vmsr" else (0.0,),
		)
		if signal_optimize_for == "vmsr":
			candidates = [
				row
				for row in fold_grid
				if row["mult"] == fold_params.mult and row["lo"] == fold_params.lo and row["hi"] == fold_params.hi
			]
			if candidates:
				best_row = max(candidates, key=lambda r: r.get("vmsr", float("-inf")))
				fold_lam = float(best_row.get("vmsr_lam", 0.0))
			else:
				fold_lam = float(signal_lam_grid[0]) if signal_lam_grid else 0.0
		else:
			fold_lam = 0.0
		fold_metrics = evaluate_msr_proxy(pred, y_valid.to_numpy(), fold_params, eps=signal_eps, lam=fold_lam)

		fold_logs.append(
			{
				"fold": fold_idx,
				"train_size": int(len(train_idx)),
				"val_size": int(len(val_idx)),
				"rmse_val": rmse,
				"val_start_index": int(val_idx[0]),
				"val_end_index": int(val_idx[-1]),
				"gap": int(args.gap),
				"best_mult": float(fold_params.mult),
				"best_lo": float(fold_params.lo),
				"best_hi": float(fold_params.hi),
				"best_lam": float(fold_lam),
				"best_rmse": float(fold_metrics["rmse"]),
				"best_msr": float(fold_metrics["msr"]),
				"best_msr_down": float(fold_metrics["msr_down"]),
				"best_vmsr": float(fold_metrics["vmsr"]),
				"best_vmsr_lam": float(fold_metrics["vmsr_lam"]),
				"best_mean": float(fold_metrics["mean"]),
				"best_std": float(fold_metrics["std"]),
				"best_std_down": float(fold_metrics["std_down"]),
			}
		)
		log_msg = (
			f"[metric][fold {fold_idx}] rmse={rmse:.6f} | signal mult={fold_params.mult} lo={fold_params.lo} hi={fold_params.hi}"
			f" | msr={fold_metrics['msr']:.6f} msr_down={fold_metrics['msr_down']:.6f}"
		)
		if signal_optimize_for == "vmsr":
			log_msg += f" vmsr={fold_metrics['vmsr']:.6f} lam={fold_lam:.2f}"
		print(log_msg)

	valid_mask = ~np.isnan(oof_pred)
	if valid_mask.any():
		overall_rmse = float(math.sqrt(mean_squared_error(y_np_array[valid_mask], oof_pred[valid_mask])))
	else:
		overall_rmse = float("nan")
	print(f"[metric][oof] rmse={overall_rmse:.6f}")

	coverage = float(np.mean(valid_mask)) if valid_mask.size else 0.0
	best_metrics_global: Dict[str, float]
	if valid_mask.any():
		best_params_global, grid_all = grid_search_msr(
			y_pred=oof_pred[valid_mask],
			y_true=y_np_array[valid_mask],
			mult_grid=signal_mult_grid,
			lo_grid=signal_lo_grid,
			hi_grid=signal_hi_grid,
			eps=signal_eps,
			optimize_for=signal_optimize_for,
			lam_grid=signal_lam_grid if signal_optimize_for == "vmsr" else (0.0,),
		)
		if signal_optimize_for == "vmsr":
			candidates = [
				row
				for row in grid_all
				if row["mult"] == best_params_global.mult and row["lo"] == best_params_global.lo and row["hi"] == best_params_global.hi
			]
			if candidates:
				best_row = max(candidates, key=lambda r: r.get("vmsr", float("-inf")))
				lam_best_global = float(best_row.get("vmsr_lam", 0.0))
			else:
				lam_best_global = float(signal_lam_grid[0]) if signal_lam_grid else 0.0
		else:
			lam_best_global = 0.0
		best_metrics_global = evaluate_msr_proxy(
			oof_pred[valid_mask],
			y_np_array[valid_mask],
			best_params_global,
			eps=signal_eps,
			lam=lam_best_global,
		)
		print(
			f"[metric][signal] best mult={best_params_global.mult} lo={best_params_global.lo} hi={best_params_global.hi}"
			f" | msr={best_metrics_global['msr']:.6f} msr_down={best_metrics_global['msr_down']:.6f}"
		)
		if signal_optimize_for == "vmsr":
			print(f"[metric][signal] vmsr={best_metrics_global['vmsr']:.6f} lam={lam_best_global:.2f}")
	else:
		best_params_global = PostProcessParams()
		lam_best_global = 0.0
		best_metrics_global = {
			"rmse": float("nan"),
			"msr": float("nan"),
			"msr_down": float("nan"),
			"vmsr": float("nan"),
			"vmsr_lam": 0.0,
			"mean": float("nan"),
			"std": float("nan"),
			"std_down": float("nan"),
		}

	# Final retraining on all data
	final_pipeline = cast(Pipeline, clone(base_pipeline))
	fit_kwargs_final: Dict[str, Any] = {}
	if callbacks:
		fit_kwargs_final["model__callbacks"] = callbacks
		fit_kwargs_final["model__eval_metric"] = "rmse"
	final_pipeline.fit(X_np, y_np, **fit_kwargs_final)

	named_steps = cast(Mapping[str, Any], final_pipeline.named_steps)
	augment_step = named_steps.get("augment")
	model_step = named_steps.get("model")
	if model_step is None:
		raise RuntimeError("Pipeline is missing a 'model' step.")

	su1_generated_columns = list(getattr(augment_step, "su1_feature_names_", [])) if augment_step is not None else []
	su5_generated_columns = list(getattr(augment_step, "su5_feature_names_", [])) if augment_step is not None else []
	feature_manifest: Dict[str, Any] = {
		"pipeline_input_columns": feature_cols,
		"su1_generated_columns": su1_generated_columns,
		"su5_generated_columns": su5_generated_columns,
	}
	feature_manifest["su1_generated_columns_cv"] = getattr(su5_prefit, "su1_feature_names_", [])
	feature_manifest["su5_generated_columns_cv"] = getattr(su5_prefit, "su5_feature_names_", [])
	preprocess_obj = named_steps.get("preprocess")
	if preprocess_obj is None:
		raise RuntimeError("Pipeline is missing a 'preprocess' step.")
	preprocess_step = cast(ColumnTransformer, preprocess_obj)
	try:
		model_feature_names = preprocess_step.get_feature_names_out()
		feature_manifest["model_feature_names"] = list(map(str, model_feature_names))
	except Exception:
		transformed_sample = preprocess_step.transform(X_np.head(1))
		sample_array = np.asarray(transformed_sample)
		if sample_array.ndim >= 2:
			feature_manifest["model_feature_count"] = int(sample_array.shape[1])
		else:
			feature_manifest["model_feature_count"] = int(sample_array.size)
	imputer_metadata = _collect_imputer_metadata(final_pipeline)
	model_params_serialized = {k: _normalise_scalar(v) for k, v in model_step.get_params().items()}
	su1_config_serialized = _normalise_scalar(asdict(su1_config))
	su5_config_serialized = _normalise_scalar(asdict(su5_config))
	preprocess_snapshot = _normalise_scalar(preprocess_settings)
	oof_best_params_serialized = {
		"mult": float(best_params_global.mult),
		"lo": float(best_params_global.lo),
		"hi": float(best_params_global.hi),
		"lam": float(lam_best_global),
	}
	best_metric_keys = ("rmse", "msr", "msr_down", "vmsr", "vmsr_lam", "mean", "std", "std_down")
	oof_best_metrics_serialized = {key: float(best_metrics_global[key]) for key in best_metric_keys}
	signal_grids_serialized = {
		"mult": [float(x) for x in signal_mult_grid],
		"lo": [float(x) for x in signal_lo_grid],
		"hi": [float(x) for x in signal_hi_grid],
		"lam": [float(x) for x in signal_lam_grid],
	}

	if not args.no_artifacts:
		bundle_path = out_dir / "inference_bundle.pkl"
		joblib.dump(final_pipeline, bundle_path)
		print(f"[ok] saved pipeline: {bundle_path}")

		meta = {
			"train_path": str(train_path),
			"test_path": str(test_path),
			"config_path": str(Path(args.config_path).resolve()),
			"preprocess_config_path": str(Path(args.preprocess_config).resolve()),
			"target_col": args.target_col,
			"id_col": args.id_col,
			"n_splits": args.n_splits,
			"gap": args.gap,
			"min_val_size": args.min_val_size,
			"numeric_fill_value": args.numeric_fill_value,
			"model_params": model_params_serialized,
			"oof_rmse": overall_rmse,
			"oof_coverage": coverage,
			"oof_best_params": oof_best_params_serialized,
			"oof_best_metrics": oof_best_metrics_serialized,
			"signal_optimize_for": signal_optimize_for,
			"signal_grids": signal_grids_serialized,
			"signal_eps": signal_eps,
			"fold_logs": fold_logs,
			"su1_config": su1_config_serialized,
			"su5_config": su5_config_serialized,
			"feature_count_before_su1": len(feature_cols),
			"su1_feature_count": len(feature_manifest.get("su1_generated_columns", [])),
			"su5_feature_count": len(feature_manifest.get("su5_generated_columns", [])),
			"preprocess_policy_snapshot": preprocess_snapshot,
			"imputer_metadata": imputer_metadata,
			"su5_cv_strategy": {
				"mode": "global_fit_with_fold_indices",
				"fill_value": args.numeric_fill_value,
			},
			"postprocess_defaults": {
				"clip_min": 0.0,
				"clip_max": 2.0,
				"use_post_process": True,
				"mult": 1.0,
				"lo": 0.0,
				"hi": 2.0,
			},
			"library_versions": {
				"numpy": np.__version__,
				"sklearn": sklearn.__version__,
				"joblib": joblib.__version__,
				"lightgbm": getattr(lgb, "__version__", None) if HAS_LGBM else None,
			},
		}
		meta_path = out_dir / "model_meta.json"
		with meta_path.open("w", encoding="utf-8") as fh:
			json.dump(meta, fh, indent=2, ensure_ascii=False)
		print(f"[ok] wrote meta: {meta_path}")

		feature_list_path = out_dir / "feature_list.json"
		with feature_list_path.open("w", encoding="utf-8") as fh:
			json.dump(feature_manifest, fh, indent=2, ensure_ascii=False)

		if fold_logs:
			cv_path = out_dir / "cv_fold_logs.csv"
			fieldnames = [
				"fold",
				"train_size",
				"val_size",
				"rmse_val",
				"val_start_index",
				"val_end_index",
				"gap",
				"best_mult",
				"best_lo",
				"best_hi",
				"best_lam",
				"best_rmse",
				"best_msr",
				"best_msr_down",
				"best_vmsr",
				"best_vmsr_lam",
				"best_mean",
				"best_std",
				"best_std_down",
			]
			_write_csv(cv_path, fold_logs, fieldnames=fieldnames)
		if valid_mask.any():
			oof_path = out_dir / "oof_predictions.csv"
			oof_records = (
				{
					"row_index": int(idx),
					"y_true": float(y_np.iloc[idx]),
					"y_pred": float(oof_pred[idx]),
					"fold": next((log["fold"] for log in fold_logs if log["val_start_index"] <= idx <= log["val_end_index"]), None),
				}
				for idx in np.where(valid_mask)[0]
			)
			_write_csv(oof_path, oof_records, fieldnames=["row_index", "y_true", "y_pred", "fold"])

	return 0


if __name__ == "__main__":
	sys.exit(main())
