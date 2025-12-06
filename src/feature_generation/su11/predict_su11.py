#!/usr/bin/env python
"""SU11 推論スクリプト。

保存済みの SU11 アーティファクト（Level-1 + Level-2 統合バンドル）を読み込み、
新しいデータに対して予測を生成する。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import joblib


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

# SU5 クラスをインポート（pickle デシリアライズ用）
from src.feature_generation.lagged.feature_su5 import (  # noqa: F401, E402
	SU5Config,
	SU5FeatureGenerator,
)
from src.feature_generation.lagged.train_su5 import SU5FeatureAugmenter  # noqa: F401, E402
from src.feature_generation.su11.feature_su11 import SU11Config, SU11MetaFeatureBuilder  # noqa: E402


# ============================================================================
# ポストプロセス関数 (to_signal)
# ============================================================================
@dataclass(frozen=True)
class PostProcessParams:
	"""ポストプロセスパラメータ。"""
	mult: float
	lo: float
	hi: float


DEFAULT_POSTPROCESS_PARAMS = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)


def _coerce_postprocess_params(mapping: Mapping[str, Any] | None) -> PostProcessParams | None:
	"""辞書からポストプロセスパラメータを抽出する。"""
	if not isinstance(mapping, Mapping):
		return None
	try:
		mult_val = float(mapping.get("mult", 1.0))
		lo_val = float(mapping.get("lo", mapping.get("clip_min", 0.0)))
		hi_val = float(mapping.get("hi", mapping.get("clip_max", 2.0)))
	except (KeyError, TypeError, ValueError):
		return None
	if not (lo_val < hi_val):
		return None
	return PostProcessParams(mult=mult_val, lo=lo_val, hi=hi_val)


def _resolve_postprocess_params(meta: Mapping[str, Any]) -> PostProcessParams:
	"""メタデータからポストプロセスパラメータを解決する。"""
	candidate_keys = (
		"oof_best_params",
		"post_process",
		"postprocess",
		"post_process_params",
		"postprocess_params",
		"signal_params",
	)
	for key in candidate_keys:
		params = _coerce_postprocess_params(meta.get(key))
		if params is not None:
			return params
	defaults = meta.get("postprocess_defaults")
	if isinstance(defaults, Mapping):
		candidate = _coerce_postprocess_params(defaults)
		if candidate is not None:
			return candidate
	return DEFAULT_POSTPROCESS_PARAMS


def to_signal(pred: np.ndarray, params: PostProcessParams) -> np.ndarray:
	"""生の予測値をシグナル（1.0 付近）に変換する。"""
	values = np.asarray(pred, dtype=float) * params.mult + 1.0
	return np.clip(values, params.lo, params.hi)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""コマンドライン引数をパースする。"""
	ap = argparse.ArgumentParser(description="Run SU11 inference.")
	ap.add_argument(
		"--bundle-path",
		type=str,
		default="artifacts/SU11/inference_bundle.pkl",
		help="Path to SU11 inference bundle",
	)
	ap.add_argument(
		"--level1-bundle",
		type=str,
		default=None,
		help="Optional: Path to Level-1 inference bundle (defaults to bundle's level1_artifacts_dir)",
	)
	ap.add_argument(
		"--test-file",
		type=str,
		default="data/raw/test.csv",
		help="Path to test data file",
	)
	ap.add_argument(
		"--out-file",
		type=str,
		default="artifacts/SU11/submission.csv",
		help="Path to output submission file",
	)
	return ap.parse_args(argv)


def load_level1_bundle(bundle_path: str | Path) -> Any:
	"""Level-1 推論バンドルを読み込む。"""
	return joblib.load(bundle_path)


def predict_level1(
	bundle: Any,
	X: pd.DataFrame,
) -> np.ndarray:
	"""Level-1 モデルで予測を生成する。

	Level-1 バンドルには sklearn Pipeline が含まれているはず。
	"""
	pipeline = bundle.get("pipeline") or bundle.get("model")
	if pipeline is None:
		# SU5 バンドル形式に対応
		for key in bundle:
			if hasattr(bundle[key], "predict"):
				pipeline = bundle[key]
				break
	if pipeline is None:
		msg = "Could not find a predictable model in Level-1 bundle"
		raise ValueError(msg)
	return pipeline.predict(X)


def main(argv: Sequence[str] | None = None) -> int:
	"""メインエントリーポイント。"""
	args = parse_args(argv)

	# SU11 バンドル読み込み
	bundle_path = Path(args.bundle_path)
	if not bundle_path.exists():
		print(f"[error] SU11 bundle not found: {bundle_path}")
		return 1

	print(f"[info] Loading SU11 bundle: {bundle_path}")
	su11_bundle = joblib.load(bundle_path)

	config_dict = su11_bundle.get("config", {})
	config = SU11Config(**config_dict)
	level2_model = su11_bundle.get("level2_model")
	level1_dir = su11_bundle.get("level1_artifacts_dir", config.level1_artifacts_dir)

	print(f"[info] Level-2 model type: {config.level2_model_type}")
	print(f"[info] Level-1 artifacts: {level1_dir}")

	# Level-1 推論バンドル読み込み
	level1_bundle_path = args.level1_bundle or Path(level1_dir) / "inference_bundle.pkl"
	if not Path(level1_bundle_path).exists():
		print(f"[error] Level-1 bundle not found: {level1_bundle_path}")
		return 1

	print(f"[info] Loading Level-1 bundle: {level1_bundle_path}")
	level1_bundle = load_level1_bundle(level1_bundle_path)

	# テストデータ読み込み
	test_path = Path(args.test_file)
	if not test_path.exists():
		print(f"[error] Test file not found: {test_path}")
		return 1

	print(f"[info] Loading test data: {test_path}")
	test_df = pd.read_csv(test_path)
	print(f"[info] Test data shape: {test_df.shape}")

	# Level-1 予測
	print("[info] Running Level-1 inference...")
	# Level-1 バンドルから pipeline を取得
	if isinstance(level1_bundle, dict):
		level1_pipeline = level1_bundle.get("pipeline")
		if level1_pipeline is None:
			# 別形式のバンドルに対応
			for key, val in level1_bundle.items():
				if hasattr(val, "predict"):
					level1_pipeline = val
					break
	else:
		# Pipeline オブジェクトが直接保存されている場合
		level1_pipeline = level1_bundle

	if level1_pipeline is None or not hasattr(level1_pipeline, "predict"):
		print("[error] Could not find a predictable model in Level-1 bundle")
		return 1

	level1_pred = level1_pipeline.predict(test_df)
	print(f"[info] Level-1 predictions: {len(level1_pred)} samples")

	# Level-2 予測
	builder = SU11MetaFeatureBuilder(config)
	X_level2 = builder.build_level2_test(level1_pred)

	if level2_model is not None:
		print("[info] Running Level-2 inference...")
		level2_pred = level2_model.predict(X_level2)
	else:
		# identity モード
		print("[info] Using Level-1 predictions directly (identity mode)")
		level2_pred = X_level2["y_pred_L1"].values

	print(f"[info] Level-2 raw predictions: {len(level2_pred)} samples")

	# ポストプロセスパラメータを読み込み
	level1_meta_path = Path(level1_dir) / "model_meta.json"
	if level1_meta_path.exists():
		with level1_meta_path.open("r", encoding="utf-8") as f:
			level1_meta = json.load(f)
		pp_params = _resolve_postprocess_params(level1_meta)
	else:
		pp_params = DEFAULT_POSTPROCESS_PARAMS

	print(f"[info] Post-process params: mult={pp_params.mult}, lo={pp_params.lo}, hi={pp_params.hi}")

	# シグナルに変換
	final_pred = to_signal(np.asarray(level2_pred), pp_params)
	print(f"[info] Final predictions (signal): min={final_pred.min():.6f}, max={final_pred.max():.6f}")

	# 出力
	out_path = Path(args.out_file)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	# date_id があれば含める
	if "date_id" in test_df.columns:
		sub_df = pd.DataFrame({
			"date_id": test_df["date_id"],
			"prediction": final_pred,
		})
	else:
		sub_df = pd.DataFrame({
			"prediction": final_pred,
		})

	sub_df.to_csv(out_path, index=False)
	print(f"[info] Saved submission: {out_path}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
