#!/usr/bin/env python
"""SU4 ハイパーパラメータスイープスクリプト。

異なるSU4設定でOOF評価を実施し、最適なパラメータを探索する。

Usage:
    python sweep_oof.py --config configs/feature_generation.yaml --data-dir data/raw
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import numpy as np
import pandas as pd

try:
	from lightgbm import LGBMRegressor  # type: ignore
	import lightgbm as lgb  # type: ignore
	HAS_LGBM = True
except Exception:
	LGBMRegressor = None  # type: ignore
	lgb = None  # type: ignore
	HAS_LGBM = False

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline



THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))

from src.feature_generation.su4.train_su4 import (  # noqa: E402
	build_pipeline,
	load_preprocess_policies,
	infer_train_file,
	infer_test_file,
	load_table,
	_prepare_features,
	_initialise_callbacks,
	_to_1d,
)
from src.feature_generation.su1.feature_su1 import (  # noqa: E402
	load_su1_config,
)
from src.feature_generation.su5.feature_su5 import (  # noqa: E402
	load_su5_config,
)
from src.feature_generation.su4.feature_su4 import (  # noqa: E402
	SU4Config,
	load_su4_config,
)
from scripts.utils_msr import (  # noqa: E402
	evaluate_msr_proxy,
	grid_search_msr,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="SU4 hyperparameter sweep")
	ap.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing train/test files")
	ap.add_argument("--train-file", type=str, default=None, help="Explicit path to the training file")
	ap.add_argument("--test-file", type=str, default=None, help="Explicit path to the test file")
	ap.add_argument("--config-path", type=str, default="configs/feature_generation.yaml", help="Path to feature_generation.yaml")
	ap.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml", help="Path to preprocess.yaml")
	ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
	ap.add_argument("--id-col", type=str, default="date_id")
	ap.add_argument("--output-dir", type=str, default="results/ablation/SU4")
	ap.add_argument("--numeric-fill-value", type=float, default=0.0)
	ap.add_argument("--n-splits", type=int, default=5)
	ap.add_argument("--gap", type=int, default=0)
	ap.add_argument("--min-val-size", type=int, default=0)
	ap.add_argument("--learning-rate", type=float, default=0.05)
	ap.add_argument("--n-estimators", type=int, default=600)
	ap.add_argument("--num-leaves", type=int, default=63)
	ap.add_argument("--min-data-in-leaf", type=int, default=32)
	ap.add_argument("--feature-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-freq", type=int, default=1)
	ap.add_argument("--random-state", type=int, default=42)
	ap.add_argument("--verbosity", type=int, default=-1)
	ap.add_argument(
		"--signal-optimize-for",
		type=str,
		choices=("msr", "msr_down", "vmsr"),
		default="msr",
	)
	ap.add_argument(
		"--signal-mult-grid",
		type=float,
		nargs="+",
		default=(0.5, 0.75, 1.0, 1.25, 1.5),
	)
	ap.add_argument(
		"--signal-lo-grid",
		type=float,
		nargs="+",
		default=(0.8, 0.9, 1.0),
	)
	ap.add_argument(
		"--signal-hi-grid",
		type=float,
		nargs="+",
		default=(1.0, 1.1, 1.2),
	)
	ap.add_argument(
		"--signal-lam-grid",
		type=float,
		nargs="+",
		default=(0.0,),
	)
	ap.add_argument(
		"--signal-eps",
		type=float,
		default=1e-8,
	)
	return ap.parse_args(argv)


def run_cv(
	base_pipeline: Pipeline,
	X: pd.DataFrame,
	y: pd.Series,
	args: argparse.Namespace,
	signal_mult_grid: tuple,
	signal_lo_grid: tuple,
	signal_hi_grid: tuple,
	signal_lam_grid: tuple,
	signal_eps: float,
	signal_optimize_for: str,
	callbacks: List[Any],
) -> tuple[float, float, Dict[str, float]]:
	"""Run cross-validation and return OOF RMSE, MSR, and best metrics."""
	splitter = TimeSeriesSplit(n_splits=args.n_splits)
	oof_pred = np.full(len(X), np.nan, dtype=float)
	y_np_array = y.to_numpy()

	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
		train_idx = np.array(train_idx)
		val_idx = np.array(val_idx)

		if args.gap > 0:
			if len(train_idx) > args.gap:
				train_idx = train_idx[:-args.gap]
			if len(val_idx) > args.gap:
				val_idx = val_idx[args.gap:]
		if len(train_idx) == 0 or len(val_idx) == 0:
			continue
		if args.min_val_size and len(val_idx) < args.min_val_size:
			continue

		X_train = X.iloc[train_idx]
		y_train = y.iloc[train_idx]
		X_valid = X.iloc[val_idx]
		y_valid = y.iloc[val_idx]

		pipe = cast(Pipeline, clone(base_pipeline))
		
		# SU5のfold_indicesを更新（reset_each_fold=Trueの場合に必要）
		su5_step = pipe.named_steps.get("su5")
		if su5_step is not None and hasattr(su5_step, "fold_indices"):
			su5_step.fold_indices = train_idx
		
		# SU4はraw_data.loc[X.index]で自動的に行を揃えるため、fold_indices更新は不要
		
		# Note: eval_setはパイプライン内で使用できないため、コールバックを使わない
		# 代わりにn_estimatorsで学習を制御
		pipe.fit(X_train, y_train)
		pred = pipe.predict(X_valid)
		pred = _to_1d(pred)
		oof_pred[val_idx] = pred

	valid_mask = ~np.isnan(oof_pred)
	if valid_mask.any():
		oof_rmse = float(math.sqrt(mean_squared_error(y_np_array[valid_mask], oof_pred[valid_mask])))
		
		best_params_global, _ = grid_search_msr(
			y_pred=oof_pred[valid_mask],
			y_true=y_np_array[valid_mask],
			mult_grid=signal_mult_grid,
			lo_grid=signal_lo_grid,
			hi_grid=signal_hi_grid,
			eps=signal_eps,
			optimize_for=signal_optimize_for,
			lam_grid=signal_lam_grid if signal_optimize_for == "vmsr" else (0.0,),
		)
		
		best_metrics = evaluate_msr_proxy(
			oof_pred[valid_mask],
			y_np_array[valid_mask],
			best_params_global,
			eps=signal_eps,
			lam=0.0,
		)
		oof_msr = float(best_metrics["msr"])
		return oof_rmse, oof_msr, best_metrics
	else:
		return float("nan"), float("nan"), {}


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Load configs
	su1_config = load_su1_config(args.config_path)
	su5_config = load_su5_config(args.config_path)
	base_su4_config = load_su4_config(args.config_path)
	preprocess_settings = load_preprocess_policies(args.preprocess_config)

	# Load data
	train_path = infer_train_file(data_dir, args.train_file)
	test_path = infer_test_file(data_dir, args.test_file)
	print(f"[info] train file: {train_path}")
	print(f"[info] test file : {test_path}")

	train_df = load_table(train_path)
	test_df = load_table(test_path)

	X, y, feature_cols = _prepare_features(
		train_df, test_df,
		target_col=args.target_col,
		id_col=args.id_col,
		exclude_lagged=True,
	)
	raw_data = X.copy()

	print(f"[info] feature columns: {len(feature_cols)}")
	print(f"[info] train samples: {len(X)}")

	# Model kwargs
	model_kwargs = {
		"n_estimators": args.n_estimators,
		"learning_rate": args.learning_rate,
		"num_leaves": args.num_leaves,
		"min_data_in_leaf": args.min_data_in_leaf,
		"feature_fraction": args.feature_fraction,
		"bagging_fraction": args.bagging_fraction,
		"bagging_freq": args.bagging_freq,
		"random_state": args.random_state,
		"verbosity": args.verbosity,
		"force_col_wise": True,
	}

	# Signal grids
	signal_mult_grid = tuple(args.signal_mult_grid)
	signal_lo_grid = tuple(args.signal_lo_grid)
	signal_hi_grid = tuple(args.signal_hi_grid)
	signal_lam_grid = tuple(args.signal_lam_grid)
	signal_eps = args.signal_eps
	signal_optimize_for = args.signal_optimize_for

	# Define parameter grid
	param_grid = {
		"top_k_imp_delta": [20, 25, 30],
		"top_k_holiday_cross": [5, 10, 15],
		"winsor_p": [0.95, 0.99],
	}

	# Generate all combinations
	param_combinations = list(product(
		param_grid["top_k_imp_delta"],
		param_grid["top_k_holiday_cross"],
		param_grid["winsor_p"]
	))

	print(f"\n[info] Generated {len(param_combinations)} parameter combinations")
	print(f"[info] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("="*80)

	results: List[Dict[str, Any]] = []
	config_times: List[float] = []
	
	for config_idx, (top_k_delta, top_k_cross, winsor) in enumerate(param_combinations, 1):
		config_start_time = time.time()
		print(f"\n[sweep][{config_idx}/{len(param_combinations)}] " + "="*60)
		print(f"[sweep][{config_idx}/{len(param_combinations)}] Parameters: top_k_imp_delta={top_k_delta}, top_k_holiday_cross={top_k_cross}, winsor_p={winsor}")
		
		# ETA計算
		if config_times:
			avg_config_time = sum(config_times) / len(config_times)
			remaining_configs = len(param_combinations) - config_idx + 1
			eta_seconds = avg_config_time * remaining_configs
			eta = timedelta(seconds=int(eta_seconds))
			print(f"[sweep][{config_idx}/{len(param_combinations)}] ETA: {eta} (avg {avg_config_time:.1f}s/config)")
		print(f"[sweep][{config_idx}/{len(param_combinations)}] Progress: {100*config_idx/len(param_combinations):.1f}%")
		
		# Create SU4Config with custom parameters
		su4_config = SU4Config(
			id_column=base_su4_config.id_column,
			output_prefix=base_su4_config.output_prefix,
			top_k_imp_delta=top_k_delta,
			top_k_holiday_cross=top_k_cross,
			winsor_p=winsor,
			imp_methods=base_su4_config.imp_methods,
			reset_each_fold=base_su4_config.reset_each_fold,
			dtype_flag=base_su4_config.dtype_flag,
			dtype_int=base_su4_config.dtype_int,
			dtype_float=base_su4_config.dtype_float,
		)

		# Build pipeline
		try:
			pipeline = build_pipeline(
				su1_config,
				su5_config,
				su4_config,
				preprocess_settings,
				numeric_fill_value=args.numeric_fill_value,
				model_kwargs=model_kwargs,
				random_state=args.random_state,
				raw_data=raw_data,
				fold_indices=None,  # run_cv内で更新
			)
			
			callbacks = _initialise_callbacks(pipeline.named_steps["model"])
			
			# Run CV
			oof_rmse, oof_msr, best_metrics = run_cv(
				pipeline,
				X,
				y,
				args,
				signal_mult_grid,
				signal_lo_grid,
				signal_hi_grid,
				signal_lam_grid,
				signal_eps,
				signal_optimize_for,
				callbacks,
			)
			
			config_elapsed = time.time() - config_start_time
			config_times.append(config_elapsed)
			
			print(f"[sweep][{config_idx}/{len(param_combinations)}] Result: oof_rmse={oof_rmse:.6f}, oof_msr={oof_msr:.6f}")
			print(f"[sweep][{config_idx}/{len(param_combinations)}] Config completed in {config_elapsed:.1f}s")
			
			results.append({
				"config_id": config_idx,
				"top_k_imp_delta": top_k_delta,
				"top_k_holiday_cross": top_k_cross,
				"winsor_p": winsor,
				"oof_rmse": oof_rmse,
				"oof_msr": oof_msr,
				"elapsed_time": config_elapsed,
				"oof_msr_down": best_metrics.get("msr_down", float("nan")),
				"oof_vmsr": best_metrics.get("vmsr", float("nan")),
				"status": "ok",
			})
		except Exception as e:
			config_elapsed = time.time() - config_start_time
			config_times.append(config_elapsed)
			print(f"[error][{config_idx}/{len(param_combinations)}] Config {config_idx} failed: {e}")
			results.append({
				"config_id": config_idx,
				"top_k_imp_delta": top_k_delta,
				"top_k_holiday_cross": top_k_cross,
				"winsor_p": winsor,
				"oof_rmse": float("nan"),
				"oof_msr": float("nan"),
				"elapsed_time": config_elapsed,
				"oof_msr_down": float("nan"),
				"oof_vmsr": float("nan"),
				"status": "error",
				"error": str(e),
			})

	# Save results
	results_df = pd.DataFrame(results)
	results_df = results_df.sort_values('oof_rmse')
	
	csv_path = output_dir / "sweep_summary.csv"
	results_df.to_csv(csv_path, index=False)
	
	# Calculate total time
	total_sweep_time = sum(config_times) if config_times else 0.0
	avg_config_time = total_sweep_time / len(config_times) if config_times else 0.0
	
	print("\n" + "="*80)
	print("[sweep] Hyperparameter sweep completed")
	print("="*80)
	print(f"[sweep] Total configurations: {len(param_combinations)}")
	print(f"[sweep] Successful: {sum(1 for r in results if r.get('status') == 'ok')}")
	print(f"[sweep] Failed: {sum(1 for r in results if r.get('status') != 'ok')}")
	print(f"[sweep] Total time: {total_sweep_time:.1f}s ({timedelta(seconds=int(total_sweep_time))})")
	print(f"[sweep] Average time per config: {avg_config_time:.1f}s")
	print(f"[sweep] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[ok] Saved results to {csv_path}")

	# Display best configurations
	print("\n" + "="*80)
	print("Top 5 Configurations by OOF RMSE")
	print("="*80)
	top_configs = results_df.head(5)
	for rank, (_, row) in enumerate(top_configs.iterrows(), start=1):
		print(f"Rank {rank}: config_id={row['config_id']}")
		print(f"  top_k_imp_delta={row['top_k_imp_delta']}, top_k_holiday_cross={row['top_k_holiday_cross']}, winsor_p={row['winsor_p']}")
		print(f"  oof_rmse={row['oof_rmse']:.6f}, oof_msr={row['oof_msr']:.6f}")
		print()

	if len(top_configs) > 0:
		best_row = top_configs.iloc[0]
		print("="*80)
		print("Best Configuration:")
		print("="*80)
		print(f"config_id: {best_row['config_id']}")
		print(f"top_k_imp_delta: {best_row['top_k_imp_delta']}")
		print(f"top_k_holiday_cross: {best_row['top_k_holiday_cross']}")
		print(f"winsor_p: {best_row['winsor_p']}")
		print(f"oof_rmse: {best_row['oof_rmse']:.6f}")
		print(f"oof_msr: {best_row['oof_msr']:.6f}")
		print("="*80)

	return 0


if __name__ == "__main__":
	sys.exit(main())
