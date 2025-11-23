#!/usr/bin/env python
"""SU4 モデルの推論エントリーポイント。

学習済みの ``artifacts/SU4/inference_bundle.pkl`` を読み込み、指定されたテストデータに
対して推論を実行し、提出形式の ``submission.csv``（および parquet 版）を出力する。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))


def _ensure_numpy_bitgenerator_aliases() -> None:
	"""Register legacy numpy BitGenerator names so joblib pickle loading works across versions."""
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
			import numpy.random._pickle as _np_random_pickle  # type: ignore[attr-defined]
		except Exception:
			return
		registry = getattr(_np_random_pickle, "BitGenerators", None)
		if isinstance(registry, dict):
			for key in alias_keys:
				registry.setdefault(key, generator_cls)


_ensure_numpy_bitgenerator_aliases()

# Import classes needed for unpickling
from preprocess.M_group.m_group import MGroupImputer  # noqa: E402,F401
from preprocess.E_group.e_group import EGroupImputer  # noqa: E402,F401
from preprocess.I_group.i_group import IGroupImputer  # noqa: E402,F401
from preprocess.P_group.p_group import PGroupImputer  # noqa: E402,F401
from preprocess.S_group.s_group import SGroupImputer  # noqa: E402,F401
from src.feature_generation.su4.train_su4 import SU1FeatureAugmenter, SU5FeatureAugmenter  # noqa: E402,F401
from src.feature_generation.su4.feature_su4 import SU4FeatureAugmenter  # noqa: E402,F401


def infer_test_file(data_dir: Path, explicit: str | None) -> Path:
	if explicit:
		candidate = Path(explicit)
		if not candidate.exists():
			raise FileNotFoundError(f"--test-file not found: {candidate}")
		return candidate
	candidates = [
		data_dir / "test.parquet",
		data_dir / "test.csv",
		data_dir / "Test.parquet",
		data_dir / "Test.csv",
	]
	for path in candidates:
		if path.exists():
			return path
	for ext in ("parquet", "csv"):
		found = sorted(
			(p for p in data_dir.glob(f"*.{ext}")),
			key=lambda p: ("test" not in p.stem.lower(), p.name),
		)
		if found:
			return found[0]
	raise FileNotFoundError(f"No test file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
	suffix = path.suffix.lower()
	if suffix == ".parquet":
		return pd.read_parquet(path)
	if suffix == ".csv":
		return pd.read_csv(path)
	raise ValueError(f"Unsupported extension: {path.suffix}")


def _load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as fh:
		return json.load(fh)


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str], *, max_missing: int | None = None) -> pd.DataFrame:
	"""Ensure test data has the same columns as training data.

	Args:
		frame: Test DataFrame to align
		columns: Expected column names from training
		max_missing: Maximum number of missing columns tolerated before raising an error.
			If None, any number of missing columns is allowed. Missing columns are filled with NaN.

	Returns:
		DataFrame with columns reindexed to match the training column order

	Raises:
		RuntimeError: If number of missing columns exceeds max_missing
	"""
	missing = [col for col in columns if col not in frame.columns]
	if max_missing is not None and len(missing) > max_missing:
		preview = ", ".join(str(col) for col in missing[:5])
		raise RuntimeError(
			f"Detected {len(missing)} missing feature columns > max-missing-columns={max_missing}. preview={preview}"
		)
	for col in missing:
		frame[col] = np.nan
	if missing:
		preview = ", ".join(str(col) for col in missing[:5])
		print(f"[warn] added {len(missing)} placeholder feature columns (preview: {preview})")
	return frame.reindex(columns=list(columns))


def _coerce_numeric_like_columns(frame: pd.DataFrame) -> list[str]:
	converted: list[str] = []
	object_cols = frame.select_dtypes(include="object").columns
	for col in object_cols:
		original = frame[col]
		if not isinstance(original, pd.Series):
			continue
		converted_series = pd.to_numeric(original, errors="coerce")
		if not isinstance(converted_series, pd.Series):
			continue
		if converted_series.notna().any() or original.notna().sum() == 0:
			frame[col] = converted_series
			converted.append(str(col))
	if converted:
		preview = ", ".join(converted[:5])
		print(f"[info] coerced {len(converted)} object columns to numeric (preview: {preview})")
	return converted


def _extract_required_calendar_columns(meta: Mapping[str, Any]) -> set[str]:
	required: set[str] = set()
	imputer_meta = meta.get("imputer_metadata") if isinstance(meta, Mapping) else None
	if isinstance(imputer_meta, Mapping):
		for info in imputer_meta.values():
			if not isinstance(info, Mapping):
				continue
			calendar_value = info.get("calendar_column")
			if isinstance(calendar_value, str) and calendar_value.strip():
				required.add(calendar_value.strip())
	return required


@dataclass(frozen=True)
class PostProcessParams:
	mult: float
	lo: float
	hi: float


DEFAULT_POSTPROCESS_PARAMS = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)


def _coerce_postprocess_params(mapping: Mapping[str, Any] | None) -> PostProcessParams | None:
	if not isinstance(mapping, Mapping):
		return None
	try:
		mult_val = float(mapping["mult"])
	except (KeyError, TypeError, ValueError):
		return None
	lo_val: float | None = None
	hi_val: float | None = None
	for lo_key in ("lo", "clip_min", "clip_lo"):
		value = mapping.get(lo_key)
		if value is not None:
			try:
				lo_val = float(value)
				break
			except (TypeError, ValueError):
				continue
	for hi_key in ("hi", "clip_max", "clip_hi"):
		value = mapping.get(hi_key)
		if value is not None:
			try:
				hi_val = float(value)
				break
			except (TypeError, ValueError):
				continue
	if lo_val is None or hi_val is None or not (lo_val < hi_val):
		return None
	return PostProcessParams(mult=mult_val, lo=lo_val, hi=hi_val)


def _resolve_postprocess_params(meta: Mapping[str, Any]) -> PostProcessParams:
	candidate_keys = (
		"post_process",
		"postprocess",
		"post_process_params",
		"postprocess_params",
		"signal_params",
		"oof_best_params",
	)
	for key in candidate_keys:
		params = _coerce_postprocess_params(meta.get(key))
		if params is not None:
			return params
	defaults = meta.get("postprocess_defaults")
	if isinstance(defaults, Mapping):
		candidate = _coerce_postprocess_params(
			{
				"mult": defaults.get("mult", DEFAULT_POSTPROCESS_PARAMS.mult),
				"lo": defaults.get("lo", defaults.get("clip_min", DEFAULT_POSTPROCESS_PARAMS.lo)),
				"hi": defaults.get("hi", defaults.get("clip_max", DEFAULT_POSTPROCESS_PARAMS.hi)),
			}
		)
		if candidate is not None:
			return candidate
	return DEFAULT_POSTPROCESS_PARAMS


def to_signal(pred: np.ndarray, params: PostProcessParams) -> np.ndarray:
	values = np.asarray(pred, dtype=float) * params.mult + 1.0
	return np.clip(values, params.lo, params.hi)


def _hash_file(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open("rb") as fh:
		for chunk in iter(lambda: fh.read(8192), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


def _check_bundle_compat(meta: Mapping[str, Any]) -> None:
	lib_versions = meta.get("library_versions") if isinstance(meta, Mapping) else None
	if isinstance(lib_versions, Mapping):
		for lib_name, expected_version in lib_versions.items():
			if not isinstance(lib_name, str):
				continue
			try:
				module = importlib.import_module(lib_name)
				actual_version = getattr(module, "__version__", None)
			except Exception:
				actual_version = None
			if expected_version is None or actual_version is None:
				print(f"[warn] unable to verify version for {lib_name}: expected={expected_version} actual={actual_version}")
			elif expected_version != actual_version:
				print(f"[warn] version mismatch for {lib_name}: expected={expected_version} actual={actual_version}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="SU4 inference pipeline")
	ap.add_argument("--artifact-dir", type=str, default="artifacts/SU4", help="Directory with inference_bundle.pkl and model_meta.json")
	ap.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing test data file")
	ap.add_argument("--test-file", type=str, default=None, help="Explicit path to test file")
	ap.add_argument("--output", type=str, default="submission.csv", help="Output submission CSV")
	ap.add_argument("--max-missing-columns", type=int, default=None, help="Max number of missing columns allowed in test data")
	return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)

	artifact_dir = Path(args.artifact_dir)
	data_dir = Path(args.data_dir)
	output_path = Path(args.output)

	bundle_path = artifact_dir / "inference_bundle.pkl"
	meta_path = artifact_dir / "model_meta.json"
	feature_list_path = artifact_dir / "feature_list.json"

	if not bundle_path.exists():
		print(f"[error] bundle not found: {bundle_path}", file=sys.stderr)
		return 1
	if not meta_path.exists():
		print(f"[error] metadata not found: {meta_path}", file=sys.stderr)
		return 1

	print(f"[info] loading metadata: {meta_path}")
	meta = _load_json(meta_path)
	_check_bundle_compat(meta)

	print(f"[info] loading feature list: {feature_list_path}")
	feature_manifest = _load_json(feature_list_path) if feature_list_path.exists() else {}
	pipeline_input_columns = feature_manifest.get("pipeline_input_columns", [])

	print(f"[info] loading bundle: {bundle_path}")
	bundle_hash = _hash_file(bundle_path)
	print(f"[info] bundle SHA256: {bundle_hash}")
	pipeline = joblib.load(bundle_path)

	test_path = infer_test_file(data_dir, args.test_file)
	print(f"[info] loading test data: {test_path}")
	test_df = load_table(test_path)
	print(f"[info] test rows: {len(test_df)}")

	_coerce_numeric_like_columns(test_df)

	if "row_id" not in test_df.columns:
		print("[warn] no row_id column in test data, creating sequential row_id")
		test_df["row_id"] = range(len(test_df))

	required_calendar_cols = _extract_required_calendar_columns(meta)
	for col in required_calendar_cols:
		if col not in test_df.columns:
			print(f"[warn] test data missing calendar column '{col}', filling with empty string")
			test_df[col] = ""

	if pipeline_input_columns:
		test_df = _ensure_columns(test_df, pipeline_input_columns, max_missing=args.max_missing_columns)
		X_test = test_df[pipeline_input_columns].copy()
	else:
		drop_cols = {"row_id", "date_id", "is_scored"}
		use_cols = [c for c in test_df.columns if c not in drop_cols]
		X_test = test_df[use_cols].copy()

	print(f"[info] predicting on {len(X_test)} test samples...")
	
	# SU4のraw_dataをtest環境のものに差し替え
	# テックリード指摘: 学習時のstate（winsor閾値等）はそのまま、raw_dataだけ差し替え
	su4_step = pipeline.named_steps.get("su4")
	if su4_step is not None and hasattr(su4_step, "raw_data"):
		print("[info] Updating SU4 raw_data for test inference...")
		# test環境のraw_dataを使用（X_testがそのままraw_dataとして使える）
		su4_step.raw_data = X_test.copy()
		print(f"[info] SU4 raw_data updated: {len(su4_step.raw_data)} rows")
	
	raw_predictions = pipeline.predict(X_test)

	postprocess_params = _resolve_postprocess_params(meta)
	print(f"[info] signal postprocess: mult={postprocess_params.mult} lo={postprocess_params.lo} hi={postprocess_params.hi}")
	predictions = to_signal(raw_predictions, postprocess_params)

	submission = pd.DataFrame({
		"row_id": test_df["row_id"],
		"prediction": predictions
	})

	output_path.parent.mkdir(parents=True, exist_ok=True)
	submission.to_csv(output_path, index=False)
	print(f"[ok] wrote submission: {output_path}")

	parquet_output = output_path.with_suffix(".parquet")
	submission.to_parquet(parquet_output, index=False)
	print(f"[ok] wrote submission: {parquet_output}")

	print("\n" + "="*80)
	print("SU4 Inference Complete")
	print("="*80)
	print(f"Predictions: {len(predictions)}")
	print(f"Signal range: [{predictions.min():.6f}, {predictions.max():.6f}]")
	print(f"Signal mean: {predictions.mean():.6f}")
	print(f"Signal std: {predictions.std():.6f}")
	print("="*80)

	return 0


if __name__ == "__main__":
	sys.exit(main())
