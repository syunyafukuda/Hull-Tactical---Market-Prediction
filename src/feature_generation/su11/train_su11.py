#!/usr/bin/env python
"""SU11 スタッキング学習エントリーポイント。

このスクリプトは Level-1 モデル（SU5）の OOF 予測値を読み込み、
Level-2 モデル（Ridge / LGBM）を学習して artifacts/SU11 に保存する。

主な処理フロー:
1. Level-1 アーティファクト（OOF 予測）を読み込む
2. Level-2 用データセットを構築
3. Level-2 モデルを CV で学習・評価
4. 全データで再学習してアーティファクトを出力
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# sys.path にプロジェクトルートを追加
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

# SU5 クラスをインポート（pickle デシリアライズ用）
from src.feature_generation.su5.feature_su5 import (  # noqa: F401, E402
	SU5Config,
	SU5FeatureGenerator,
)
from src.feature_generation.su5.train_su5 import SU5FeatureAugmenter  # noqa: F401, E402

try:
	from lightgbm import LGBMRegressor  # type: ignore
	HAS_LGBM = True
except Exception:
	LGBMRegressor = None  # type: ignore
	HAS_LGBM = False

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

import joblib  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

from src.feature_generation.su11.feature_su11 import (  # noqa: E402
	SU11Config,
	SU11MetaFeatureBuilder,
)

# MSR 計算用ユーティリティ
def _sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
	"""Sharpe ratio 計算。"""
	returns = y_true * np.sign(y_pred)
	mean = float(np.mean(returns))
	std = float(np.std(returns)) + eps
	return mean / std


def _downside_sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
	"""Downside Sharpe ratio 計算。"""
	returns = y_true * np.sign(y_pred)
	mean = float(np.mean(returns))
	downside = returns[returns < 0]
	if len(downside) == 0:
		return float("inf")
	std = float(np.std(downside)) + eps
	return mean / std


# utils_msr からインポートを試行（あれば上書き）
try:
	from scripts.utils_msr import sharpe_ratio as _sharpe_ratio_imported  # type: ignore[import-not-found]
	from scripts.utils_msr import downside_sharpe_ratio as _downside_sharpe_ratio_imported  # type: ignore[import-not-found]

	def _sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:  # noqa: F811
		return float(_sharpe_ratio_imported(y_true, y_pred, eps=eps))

	def _downside_sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:  # noqa: F811
		return float(_downside_sharpe_ratio_imported(y_true, y_pred, eps=eps))
except ImportError:
	pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""コマンドライン引数をパースする。"""
	ap = argparse.ArgumentParser(description="Train Level-2 stacking model for SU11.")
	ap.add_argument(
		"--level1-dir",
		type=str,
		default="artifacts/SU5",
		help="Directory containing Level-1 artifacts (oof_predictions.csv, submission.csv)",
	)
	ap.add_argument(
		"--out-dir",
		type=str,
		default="artifacts/SU11",
		help="Output directory for SU11 artifacts",
	)
	ap.add_argument(
		"--config-path",
		type=str,
		default="configs/feature_generation.yaml",
		help="Path to feature_generation.yaml",
	)
	ap.add_argument(
		"--level2-model",
		type=str,
		choices=["ridge", "lgbm", "identity"],
		default="ridge",
		help="Level-2 model type",
	)
	ap.add_argument("--ridge-alpha", type=float, default=0.001, help="Ridge regularization strength")
	ap.add_argument("--lgbm-n-estimators", type=int, default=50, help="LGBM n_estimators for Level-2")
	ap.add_argument("--lgbm-max-depth", type=int, default=3, help="LGBM max_depth for Level-2")
	ap.add_argument("--n-splits", type=int, default=5, help="Number of CV folds for Level-2")
	ap.add_argument("--random-state", type=int, default=42, help="Random seed")
	ap.add_argument("--no-artifacts", action="store_true", help="Skip writing artifacts")
	return ap.parse_args(argv)


def build_level2_model(
	model_type: str,
	*,
	ridge_alpha: float = 1.0,
	lgbm_n_estimators: int = 50,
	lgbm_max_depth: int = 3,
	random_state: int = 42,
) -> Ridge | Any | None:
	"""Level-2 モデルを構築する。

	Returns
	-------
	Ridge | LGBMRegressor | None
		model_type に応じたモデル。"identity" の場合は None。
	"""
	if model_type == "ridge":
		return Ridge(alpha=ridge_alpha)
	elif model_type == "lgbm":
		if not HAS_LGBM:
			msg = "LightGBM is not installed. Cannot use lgbm as Level-2 model."
			raise ImportError(msg)
		assert LGBMRegressor is not None
		return LGBMRegressor(
			n_estimators=lgbm_n_estimators,
			max_depth=lgbm_max_depth,
			learning_rate=0.05,
			random_state=random_state,
			verbosity=-1,
		)
	elif model_type == "identity":
		# パススルー: Level-1 出力をそのまま使用
		return None
	else:
		msg = f"Unknown model_type: {model_type}"
		raise ValueError(msg)


def _write_csv(path: Path, rows: Any, *, fieldnames: list[str]) -> None:
	"""CSV ファイルを書き込む。"""
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
	"""メインエントリーポイント。"""
	args = parse_args(argv)

	# Config 構築
	config = SU11Config(
		level1_artifacts_dir=args.level1_dir,
		level2_model_type=args.level2_model,
		ridge_alpha=args.ridge_alpha,
		lgbm_n_estimators=args.lgbm_n_estimators,
		lgbm_max_depth=args.lgbm_max_depth,
		n_splits=args.n_splits,
		random_state=args.random_state,
	)

	print(f"[info] Level-1 artifacts: {config.level1_artifacts_dir}")
	print(f"[info] Level-2 model: {config.level2_model_type}")
	print(f"[info] Output directory: {args.out_dir}")

	# Level-1 アーティファクト読み込み
	builder = SU11MetaFeatureBuilder(config)
	try:
		builder.load_level1_artifacts()
	except FileNotFoundError as e:
		print(f"[error] {e}")
		return 1

	# Level-2 データセット構築
	X_L2, y_L2 = builder.build_level2_train()
	print(f"[info] Level-2 dataset: {len(X_L2)} samples, {len(X_L2.columns)} features")

	# CV
	splitter = TimeSeriesSplit(n_splits=config.n_splits)
	oof_pred = np.full(len(X_L2), np.nan, dtype=float)
	fold_logs: list[dict[str, Any]] = []

	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_L2), start=1):
		X_train = X_L2.iloc[train_idx]
		y_train = y_L2.iloc[train_idx]
		X_val = X_L2.iloc[val_idx]
		y_val = y_L2.iloc[val_idx]

		if config.level2_model_type == "identity":
			# パススルー: y_pred_L1 をそのまま使用
			pred = X_val["y_pred_L1"].values
		else:
			model = build_level2_model(
				config.level2_model_type,
				ridge_alpha=config.ridge_alpha,
				lgbm_n_estimators=config.lgbm_n_estimators,
				lgbm_max_depth=config.lgbm_max_depth,
				random_state=config.random_state,
			)
			assert model is not None, f"model should not be None for type={config.level2_model_type}"
			model.fit(X_train, y_train)
			pred = model.predict(X_val)

		oof_pred[val_idx] = pred
		rmse = float(math.sqrt(mean_squared_error(y_val, pred)))

		# MSR 計算
		y_val_np = y_val.to_numpy()
		msr = _sharpe_ratio(y_val_np, pred)
		msr_down = _downside_sharpe_ratio(y_val_np, pred)

		print(f"[metric][fold {fold_idx}] rmse={rmse:.6f} | msr={msr:.6f} | msr_down={msr_down:.6f}")

		fold_logs.append({
			"fold": fold_idx,
			"n_train": len(train_idx),
			"n_val": len(val_idx),
			"rmse_val": rmse,
			"msr": msr,
			"msr_down": msr_down,
			"val_start_index": int(val_idx[0]),
			"val_end_index": int(val_idx[-1]),
		})

	# 全体 OOF 評価
	valid_mask = ~np.isnan(oof_pred)
	if np.any(valid_mask):
		y_L2_masked = y_L2[valid_mask]
		oof_pred_masked = oof_pred[valid_mask]
		overall_rmse = float(math.sqrt(mean_squared_error(y_L2_masked, oof_pred_masked)))
		if isinstance(y_L2_masked, pd.Series):
			y_L2_np = y_L2_masked.to_numpy()
		else:
			y_L2_np = np.asarray(y_L2_masked)
		overall_msr = _sharpe_ratio(y_L2_np, oof_pred_masked)
		overall_msr_down = _downside_sharpe_ratio(y_L2_np, oof_pred_masked)
	else:
		overall_rmse = float("nan")
		overall_msr = float("nan")
		overall_msr_down = float("nan")

	coverage = float(valid_mask.sum() / len(oof_pred))
	print(f"[metric][oof] rmse={overall_rmse:.6f} | msr={overall_msr:.6f} | msr_down={overall_msr_down:.6f}")
	print(f"[metric][oof] coverage={coverage:.2%}")

	# Level-2 モデルを全データで再学習
	final_model: Ridge | Any | None
	if config.level2_model_type != "identity":
		final_model = build_level2_model(
			config.level2_model_type,
			ridge_alpha=config.ridge_alpha,
			lgbm_n_estimators=config.lgbm_n_estimators,
			lgbm_max_depth=config.lgbm_max_depth,
			random_state=config.random_state,
		)
		assert final_model is not None
		final_model.fit(X_L2, y_L2)
	else:
		final_model = None

	# アーティファクト出力
	if not args.no_artifacts:
		out_dir = Path(args.out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

		# Level-1 パイプラインをロード
		level1_bundle_path = Path(config.level1_artifacts_dir) / "inference_bundle.pkl"
		level1_pipeline = joblib.load(level1_bundle_path)

		# Level-2 モデル保存（Level-1 パイプラインも含む）
		bundle = {
			"config": asdict(config),
			"level2_model": final_model,
			"level1_artifacts_dir": str(config.level1_artifacts_dir),
			"level1_pipeline": level1_pipeline,
		}
		bundle_path = out_dir / "inference_bundle.pkl"
		joblib.dump(bundle, bundle_path)
		print(f"[info] Saved inference bundle: {bundle_path}")

		# Level-1 のメタ情報を読み込み（ポストプロセスパラメータ用）
		level1_meta_path = Path(config.level1_artifacts_dir) / "model_meta.json"
		level1_meta: dict[str, Any] = {}
		if level1_meta_path.exists():
			with level1_meta_path.open("r", encoding="utf-8") as f:
				level1_meta = json.load(f)

		# メタ情報
		meta: dict[str, Any] = {
			"su_id": "SU11",
			"level2_model_type": config.level2_model_type,
			"oof_rmse": overall_rmse,
			"oof_msr": overall_msr,
			"oof_msr_down": overall_msr_down,
			"oof_coverage": coverage,
			"n_splits": config.n_splits,
			"random_state": config.random_state,
			"level1_artifacts_dir": str(config.level1_artifacts_dir),
		}
		if config.level2_model_type == "ridge":
			meta["ridge_alpha"] = config.ridge_alpha
		elif config.level2_model_type == "lgbm":
			meta["lgbm_n_estimators"] = config.lgbm_n_estimators
			meta["lgbm_max_depth"] = config.lgbm_max_depth

		# Level-1 のポストプロセスパラメータをコピー
		for key in ("oof_best_params", "postprocess_defaults"):
			if key in level1_meta:
				meta[key] = level1_meta[key]

		meta_path = out_dir / "model_meta.json"
		with meta_path.open("w", encoding="utf-8") as f:
			json.dump(meta, f, indent=2)
		print(f"[info] Saved model meta: {meta_path}")

		# 特徴量リスト（Level-1 の feature_list.json をコピー）
		level1_feature_list_path = Path(config.level1_artifacts_dir) / "feature_list.json"
		if level1_feature_list_path.exists():
			import shutil
			fl_path = out_dir / "feature_list.json"
			shutil.copy(level1_feature_list_path, fl_path)
			print(f"[info] Copied Level-1 feature list: {fl_path}")
		else:
			# フォールバック: Level-2 特徴を保存
			feature_list = list(X_L2.columns)
			fl_path = out_dir / "feature_list.json"
			with fl_path.open("w", encoding="utf-8") as f:
				json.dump(feature_list, f, indent=2)
			print(f"[info] Saved Level-2 feature list: {fl_path}")

		# CV fold logs
		cv_path = out_dir / "cv_fold_logs.csv"
		_write_csv(
			cv_path,
			fold_logs,
			fieldnames=["fold", "n_train", "n_val", "rmse_val", "msr", "msr_down", "val_start_index", "val_end_index"],
		)
		print(f"[info] Saved CV fold logs: {cv_path}")

		# OOF predictions
		if valid_mask.any():
			oof_path = out_dir / "oof_predictions.csv"
			oof_records = [
				{
					"row_index": int(idx),
					"y_true": float(y_L2.iloc[idx]),
					"y_pred": float(oof_pred[idx]),
					"fold": next(
						(log["fold"] for log in fold_logs if log["val_start_index"] <= idx <= log["val_end_index"]),
						None,
					),
				}
				for idx in np.where(valid_mask)[0]
			]
			_write_csv(oof_path, oof_records, fieldnames=["row_index", "y_true", "y_pred", "fold"])
			print(f"[info] Saved OOF predictions: {oof_path}")

		# Test 予測（Level-1 test prediction を Level-2 で変換）
		test_pred_L1 = builder.level1_test_pred
		if test_pred_L1 is not None:
			X_test_L2 = builder.build_level2_test(test_pred_L1)
			if final_model is not None:
				test_pred_L2 = final_model.predict(X_test_L2)
			else:
				test_pred_L2 = X_test_L2["y_pred_L1"].values

			sub_path = out_dir / "submission.csv"
			sub_df = pd.DataFrame({
				"prediction": test_pred_L2,
			})
			sub_df.to_csv(sub_path, index=False)
			print(f"[info] Saved submission: {sub_path}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
