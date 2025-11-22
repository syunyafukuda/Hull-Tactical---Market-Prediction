#!/usr/bin/env python
"""SU3 特徴量のハイパーパラメータスイープスクリプト。

本スクリプトは SU3 の各種パラメータ（top-k 値など）をグリッドサーチし、
OOF 評価結果を保存する。LB 提出は行わず、ローカル評価のみ。

主な役割
--------
* configs/feature_generation.yaml の su3 セクションからスイープパラメータを読み込む
* グリッドサーチ実行:
  - reappear_top_k: [10, 20, 30, 50]
  - temporal_top_k: [10, 20, 30]
  - holiday_top_k: [10, 20, 30, 50]
  - include_imputation_trace: [true, false]
* 各構成で OOF 評価（RMSE, MSR, 特徴量数、学習時間）
* 結果出力:
  - results/ablation/SU3/sweep_YYYY-MM-DD_HHMMSS.json
  - results/ablation/SU3/sweep_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

try:
	from lightgbm import LGBMRegressor  # type: ignore
	import lightgbm as lgb  # type: ignore

	HAS_LGBM = True
except Exception:
	LGBMRegressor = None  # type: ignore
	lgb = None  # type: ignore
	HAS_LGBM = False


# Add project paths
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))

from src.feature_generation.su1.feature_su1 import SU1Config, load_su1_config  # noqa: E402
from src.feature_generation.su3.feature_su3 import SU3Config, load_su3_config  # noqa: E402
from src.feature_generation.su3.train_su3 import (  # noqa: E402
	build_pipeline,
	load_preprocess_policies,
	_to_1d,
)
from scripts.utils_msr import (  # noqa: E402
	PostProcessParams,
	evaluate_msr_proxy,
)


# Stage 1: パラメータグリッド定義（代入影響オフ）
PARAM_GRID = {
	'reappear_top_k': [10, 20, 30, 50],
	'temporal_top_k': [10, 20, 30],
	'holiday_top_k': [10, 20, 30, 50],
}

# Stage 2: 代入影響の評価（Phase 5で検討）
# 'include_imputation_trace': [False, True]
# 'imp_delta_top_k': [10, 20, 50]


def load_configs(
	config_path: str | Path,
	preprocess_config_path: str | Path
) -> Tuple[SU1Config, SU3Config, Dict[str, Dict[str, Any]]]:
	"""設定ファイルを読み込む。"""
	config_path = Path(config_path).resolve()
	preprocess_config_path = Path(preprocess_config_path).resolve()

	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	if not preprocess_config_path.exists():
		raise FileNotFoundError(f"Preprocess config file not found: {preprocess_config_path}")

	# Load SU1 and SU3 configs
	su1_config = load_su1_config(config_path)
	su3_config = load_su3_config(config_path)

	# Load preprocess settings
	preprocess_settings = load_preprocess_policies(preprocess_config_path)

	return su1_config, su3_config, preprocess_settings


def build_param_combinations(
	param_grid: Dict[str, List[Any]],
	include_imputation_trace: bool = False
) -> List[Dict[str, Any]]:
	"""パラメータグリッドから全組み合わせを生成。"""
	combinations = []

	# Extract parameter lists
	reappear_values = param_grid.get('reappear_top_k', [20])
	temporal_values = param_grid.get('temporal_top_k', [20])
	holiday_values = param_grid.get('holiday_top_k', [20])

	# Generate all combinations using itertools.product
	for reappear, temporal, holiday in itertools.product(
		reappear_values,
		temporal_values,
		holiday_values
	):
		config = {
			'reappear_top_k': int(reappear),
			'temporal_top_k': int(temporal),
			'holiday_top_k': int(holiday),
			'include_imputation_trace': include_imputation_trace,
		}
		combinations.append(config)

	return combinations


def infer_train_file(data_dir: str | Path) -> Path:
	"""学習データファイルのパスを推測する。"""
	data_dir = Path(data_dir).resolve()
	candidates = [
		data_dir / "train.csv",
		data_dir / "train.parquet",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"No train file found in {data_dir}")


def load_table(file_path: str | Path) -> pd.DataFrame:
	"""CSVまたはParquetファイルを読み込む。"""
	file_path = Path(file_path).resolve()
	if not file_path.exists():
		raise FileNotFoundError(f"File not found: {file_path}")

	if file_path.suffix == ".csv":
		return pd.read_csv(file_path)
	elif file_path.suffix in (".parquet", ".pq"):
		return pd.read_parquet(file_path)
	else:
		raise ValueError(f"Unsupported file format: {file_path.suffix}")


def evaluate_single_config(
	su1_config: SU1Config,
	su3_base_config: SU3Config,
	param_config: Dict[str, Any],
	preprocess_settings: Dict[str, Dict[str, Any]],
	train_data: pd.DataFrame,
	target: np.ndarray,
	id_col: str,
	*,
	n_splits: int,
	gap: int,
	min_val_size: int,
	model_kwargs: Dict[str, Any],
	numeric_fill_value: float,
	random_state: int
) -> Dict[str, Any]:
	"""単一パラメータ構成でOOF評価を実行。
	
	Returns:
		{
			'config': param_config,
			'oof_rmse': float,
			'oof_msr': float,
			'n_features': int,
			'train_time_sec': float,
			'fold_scores': [{'fold': int, 'rmse': float, 'msr': float}, ...]
		}
	"""
	start_time = time.time()

	# Update SU3 config with parameter values
	su3_config_dict = {
		'id_column': su3_base_config.id_column,
		'output_prefix': su3_base_config.output_prefix,
		'include_transitions': su3_base_config.include_transitions,
		'transition_group_agg': su3_base_config.transition_group_agg,
		'include_reappearance': su3_base_config.include_reappearance,
		'reappear_clip': su3_base_config.reappear_clip,
		'reappear_top_k': param_config['reappear_top_k'],
		'include_imputation_trace': param_config['include_imputation_trace'],
		'imp_delta_winsorize_p': su3_base_config.imp_delta_winsorize_p,
		'imp_delta_top_k': su3_base_config.imp_delta_top_k,
		'imp_policy_group_level': su3_base_config.imp_policy_group_level,
		'include_temporal_bias': su3_base_config.include_temporal_bias,
		'temporal_burn_in': su3_base_config.temporal_burn_in,
		'temporal_top_k': param_config['temporal_top_k'],
		'include_holiday_interaction': su3_base_config.include_holiday_interaction,
		'holiday_top_k': param_config['holiday_top_k'],
		'dtype': {
			'flag': str(su3_base_config.dtype_flag),
			'int': str(su3_base_config.dtype_int),
			'float': str(su3_base_config.dtype_float),
		},
		'reset_each_fold': su3_base_config.reset_each_fold,
	}
	su3_config = SU3Config.from_mapping(su3_config_dict)

	# Build pipeline
	pipeline = build_pipeline(
		su1_config,
		su3_config,
		preprocess_settings,
		numeric_fill_value=numeric_fill_value,
		model_kwargs=model_kwargs,
		random_state=random_state,
	)

	# Prepare for CV
	splitter = TimeSeriesSplit(n_splits=n_splits)
	X_reset = train_data.reset_index(drop=True)
	y_reset = pd.Series(target).reset_index(drop=True)

	oof_pred = np.full(len(X_reset), np.nan, dtype=float)
	fold_scores: List[Dict[str, Any]] = []

	# Run cross-validation
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_reset), start=1):
		train_idx = np.array(train_idx)
		val_idx = np.array(val_idx)

		# Apply gap
		if gap > 0:
			if len(train_idx) > gap:
				train_idx = train_idx[:-gap]
			if len(val_idx) > gap:
				val_idx = val_idx[gap:]

		if len(train_idx) == 0 or len(val_idx) == 0:
			continue
		if min_val_size > 0 and len(val_idx) < min_val_size:
			continue

		X_train = X_reset.iloc[train_idx]
		y_train = y_reset.iloc[train_idx]
		X_val = X_reset.iloc[val_idx]
		y_val = y_reset.iloc[val_idx]

		# Clone pipeline for this fold
		pipe = cast(Pipeline, clone(pipeline))

		# Fit without early stopping (simpler for sweep)
		# Note: We skip eval_set to avoid complexity with feature transformation
		fit_kwargs: Dict[str, Any] = {}

		pipe.fit(X_train, y_train, **fit_kwargs)

		# Predict
		pred = pipe.predict(X_val)
		pred = _to_1d(pred)
		oof_pred[val_idx] = pred

		# Calculate fold RMSE
		rmse = float(math.sqrt(mean_squared_error(y_val, pred)))

		# Calculate fold MSR (using default signal parameters)
		signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)
		metrics = evaluate_msr_proxy(pred, y_val.to_numpy(), signal_params, eps=1e-8, lam=0.0)
		msr = float(metrics['msr'])

		fold_scores.append({
			'fold': fold_idx,
			'rmse': rmse,
			'msr': msr,
		})

	# Calculate overall OOF metrics
	valid_mask = ~np.isnan(oof_pred)
	if valid_mask.any():
		oof_rmse = float(math.sqrt(mean_squared_error(target[valid_mask], oof_pred[valid_mask])))
		signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)
		oof_metrics = evaluate_msr_proxy(oof_pred[valid_mask], target[valid_mask], signal_params, eps=1e-8, lam=0.0)
		oof_msr = float(oof_metrics['msr'])
	else:
		oof_rmse = float('nan')
		oof_msr = float('nan')

	# Get feature count (from augmented features)
	# We need to fit the pipeline once to get feature names
	try:
		augmenter = pipeline.named_steps.get("augment")
		if augmenter and hasattr(augmenter, 'fit_transform'):
			sample_features = augmenter.fit_transform(X_reset.head(10))
			n_features = sample_features.shape[1]
		else:
			n_features = 0
	except Exception:
		n_features = 0

	train_time_sec = time.time() - start_time

	return {
		'config': param_config,
		'oof_rmse': oof_rmse,
		'oof_msr': oof_msr,
		'n_features': n_features,
		'train_time_sec': train_time_sec,
		'fold_scores': fold_scores,
	}


def save_results(
	results: List[Dict[str, Any]],
	output_dir: Path,
	timestamp: str,
	metadata: Dict[str, Any]
) -> None:
	"""結果をJSON/CSVで保存。"""
	output_dir.mkdir(parents=True, exist_ok=True)

	# Save JSON (detailed results with fold scores)
	json_file = output_dir / f'sweep_{timestamp}.json'
	json_data = {
		'metadata': metadata,
		'results': results,
	}
	with json_file.open('w', encoding='utf-8') as f:
		json.dump(json_data, f, indent=2, ensure_ascii=False)
	print(f"[save] Detailed results saved to: {json_file}")

	# Save CSV (summary)
	csv_file = output_dir / 'sweep_summary.csv'
	file_exists = csv_file.exists()

	with csv_file.open('a', encoding='utf-8', newline='') as f:
		fieldnames = [
			'timestamp',
			'config_id',
			'reappear_top_k',
			'temporal_top_k',
			'holiday_top_k',
			'include_imputation_trace',
			'oof_rmse',
			'oof_msr',
			'n_features',
			'train_time_sec',
		]
		writer = csv.DictWriter(f, fieldnames=fieldnames)

		if not file_exists:
			writer.writeheader()

		for idx, result in enumerate(results, start=1):
			row = {
				'timestamp': timestamp,
				'config_id': idx,
				'reappear_top_k': result['config']['reappear_top_k'],
				'temporal_top_k': result['config']['temporal_top_k'],
				'holiday_top_k': result['config']['holiday_top_k'],
				'include_imputation_trace': result['config']['include_imputation_trace'],
				'oof_rmse': result['oof_rmse'],
				'oof_msr': result['oof_msr'],
				'n_features': result['n_features'],
				'train_time_sec': result['train_time_sec'],
			}
			writer.writerow(row)

	print(f"[save] Summary results appended to: {csv_file}")


def main() -> int:
	"""エントリーポイント。"""
	ap = argparse.ArgumentParser(description="Sweep SU3 hyperparameters with OOF evaluation")
	ap.add_argument("--data-dir", type=str, default="data/raw")
	ap.add_argument("--train-file", type=str, default=None)
	ap.add_argument("--config-path", type=str, default="configs/feature_generation.yaml")
	ap.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml")
	ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
	ap.add_argument("--id-col", type=str, default="date_id")
	ap.add_argument("--output-dir", type=str, default="results/ablation/SU3")
	ap.add_argument("--n-splits", type=int, default=5)
	ap.add_argument("--gap", type=int, default=0)
	ap.add_argument("--min-val-size", type=int, default=0)
	ap.add_argument("--numeric-fill-value", type=float, default=0.0)
	ap.add_argument("--learning-rate", type=float, default=0.05)
	ap.add_argument("--n-estimators", type=int, default=600)
	ap.add_argument("--reg-alpha", type=float, default=0.1)
	ap.add_argument("--reg-lambda", type=float, default=0.1)
	ap.add_argument("--random-state", type=int, default=42)
	ap.add_argument("--verbosity", type=int, default=-1)
	ap.add_argument("--include-imputation-trace", action="store_true",
					help="Include imputation trace features (Stage 2)")

	args = ap.parse_args()

	print("=" * 80)
	print("SU3 Hyperparameter Sweep")
	print("=" * 80)
	print(f"Data directory: {args.data_dir}")
	print(f"Config path: {args.config_path}")
	print(f"Preprocess config: {args.preprocess_config}")
	print(f"Output directory: {args.output_dir}")
	print(f"N splits: {args.n_splits}")
	print(f"Gap: {args.gap}")
	print(f"Include imputation trace: {args.include_imputation_trace}")
	print()

	# Load configurations
	print("[load] Loading configurations...")
	su1_config, su3_config, preprocess_settings = load_configs(
		args.config_path,
		args.preprocess_config
	)
	print(f"[load] SU1 config loaded: {len(su1_config.target_groups)} groups")
	print(f"[load] SU3 config loaded: prefix={su3_config.output_prefix}")

	# Load training data
	print("[load] Loading training data...")
	if args.train_file:
		train_file = Path(args.train_file)
	else:
		train_file = infer_train_file(args.data_dir)
	
	print(f"[load] Reading: {train_file}")
	train_df = load_table(train_file)
	print(f"[load] Train shape: {train_df.shape}")

	# Prepare features and target
	if args.target_col not in train_df.columns:
		raise ValueError(f"Target column '{args.target_col}' not found in training data")

	# Extract features (simple approach - drop known non-feature columns)
	drop_cols = {
		args.target_col,
		args.id_col,
		"date_id",
		"forward_returns",
		"risk_free_rate",
		"market_forward_excess_returns",
		"is_scored",
	}
	feature_cols = [c for c in train_df.columns if c not in drop_cols and not c.startswith("lagged_")]
	X = cast(pd.DataFrame, train_df[feature_cols].copy())
	y = train_df[args.target_col].to_numpy()

	print(f"[prepare] Feature columns: {len(feature_cols)}")
	print(f"[prepare] Target shape: {y.shape}")

	# Generate parameter combinations
	print("[sweep] Generating parameter combinations...")
	param_combinations = build_param_combinations(
		PARAM_GRID,
		include_imputation_trace=args.include_imputation_trace
	)
	print(f"[sweep] Total configurations: {len(param_combinations)}")
	print()

	# Model configuration
	model_kwargs = {
		"learning_rate": args.learning_rate,
		"n_estimators": args.n_estimators,
		"reg_alpha": args.reg_alpha,
		"reg_lambda": args.reg_lambda,
		"random_state": args.random_state,
		"n_jobs": -1,
		"verbosity": args.verbosity,
	}

	# Evaluate each configuration
	results: List[Dict[str, Any]] = []
	for idx, param_config in enumerate(param_combinations, start=1):
		print(f"[sweep] Evaluating config {idx}/{len(param_combinations)}")
		print(f"[sweep] Parameters: {param_config}")

		try:
			result = evaluate_single_config(
				su1_config=su1_config,
				su3_base_config=su3_config,
				param_config=param_config,
				preprocess_settings=preprocess_settings,
				train_data=X,
				target=y,
				id_col=args.id_col,
				n_splits=args.n_splits,
				gap=args.gap,
				min_val_size=args.min_val_size,
				model_kwargs=model_kwargs,
				numeric_fill_value=args.numeric_fill_value,
				random_state=args.random_state,
			)
			results.append(result)

			print(f"[result] OOF RMSE: {result['oof_rmse']:.6f}")
			print(f"[result] OOF MSR: {result['oof_msr']:.6f}")
			print(f"[result] N features: {result['n_features']}")
			print(f"[result] Time: {result['train_time_sec']:.1f}s")
			print()

		except Exception as e:
			print(f"[error] Configuration {idx} failed: {e}")
			print("[error] Continuing with next configuration...")
			print()
			continue

	# Sort results by OOF MSR (ascending - lower is better)
	results.sort(key=lambda x: x['oof_msr'] if not math.isnan(x['oof_msr']) else float('inf'))

	# Save results
	timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
	metadata = {
		'timestamp': timestamp,
		'n_configs': len(results),
		'n_splits': args.n_splits,
		'gap': args.gap,
		'model_params': model_kwargs,
	}

	output_dir = Path(args.output_dir)
	save_results(results, output_dir, timestamp, metadata)

	# Display best configuration
	print()
	print("=" * 80)
	print("Best Configuration")
	print("=" * 80)
	if results:
		best = results[0]
		print(f"Config: {best['config']}")
		print(f"OOF RMSE: {best['oof_rmse']:.6f}")
		print(f"OOF MSR: {best['oof_msr']:.6f}")
		print(f"N features: {best['n_features']}")
		print(f"Time: {best['train_time_sec']:.1f}s")
	else:
		print("No successful configurations")

	print("=" * 80)
	return 0


if __name__ == "__main__":
	sys.exit(main())
