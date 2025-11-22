#!/usr/bin/env python
"""SU5 OOF sweep script for hyperparameter tuning.

configs/feature_generation.yaml の su5 セクションに記載されたパラメータ候補を組み合わせ、
OOF (TimeSeriesSplit) で RMSE / MSR を評価する。結果は results/ablation/SU5 配下に出力する。
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
from typing import Any, Dict, Iterable, List, Sequence, cast

import numpy as np
import pandas as pd
import yaml
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

from scripts.utils_msr import (  # noqa: E402
	PostProcessParams,
	evaluate_msr_proxy,
	grid_search_msr,
)
from src.feature_generation.su1.feature_su1 import SU1Config, load_su1_config  # noqa: E402
from src.feature_generation.su5.feature_su5 import SU5Config  # noqa: E402
from src.feature_generation.su5.train_su5 import (  # noqa: E402
	SU5FeatureAugmenter,
	_initialise_callbacks,
	_prepare_features,
	_to_1d,
	build_pipeline,
	infer_test_file,
	infer_train_file,
	load_preprocess_policies,
	load_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Sweep SU5 hyperparameters using OOF validation.")
	ap.add_argument("--config-path", type=str, default="configs/feature_generation.yaml", help="Path to feature generation YAML")
	ap.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml", help="Path to preprocess policy YAML")
	ap.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw train/test files")
	ap.add_argument("--train-file", type=str, default=None, help="Optional explicit train file path")
	ap.add_argument("--test-file", type=str, default=None, help="Optional explicit test file path")
	ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
	ap.add_argument("--id-col", type=str, default="date_id")
	ap.add_argument("--out-dir", type=str, default="results/ablation/SU5", help="Directory to store sweep outputs")
	ap.add_argument("--n-splits", type=int, default=5, help="Number of TimeSeriesSplit folds")
	ap.add_argument("--gap", type=int, default=0, help="Gap between train and validation indices")
	ap.add_argument("--min-val-size", type=int, default=0, help="Skip folds with validation size smaller than this after gap trimming")
	ap.add_argument("--numeric-fill-value", type=float, default=0.0, help="Fill value applied after feature generation")
	ap.add_argument("--learning-rate", type=float, default=0.05)
	ap.add_argument("--n-estimators", type=int, default=600)
	ap.add_argument("--num-leaves", type=int, default=63)
	ap.add_argument("--min-data-in-leaf", type=int, default=32)
	ap.add_argument("--feature-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-freq", type=int, default=1)
	ap.add_argument("--random-state", type=int, default=42)
	ap.add_argument("--verbosity", type=int, default=-1)
	ap.add_argument("--signal-optimize-for", type=str, choices=("msr", "msr_down", "vmsr"), default="msr")
	ap.add_argument("--signal-mult-grid", type=float, nargs="+", default=(0.5, 0.75, 1.0, 1.25, 1.5))
	ap.add_argument("--signal-lo-grid", type=float, nargs="+", default=(0.8, 0.9, 1.0))
	ap.add_argument("--signal-hi-grid", type=float, nargs="+", default=(1.0, 1.1, 1.2))
	ap.add_argument("--signal-lam-grid", type=float, nargs="+", default=(0.0,))
	ap.add_argument("--signal-eps", type=float, default=1e-8)
	# SU5-specific grid parameters
	ap.add_argument("--top-k-pairs-grid", type=int, nargs="+", default=None, help="Grid for top_k_pairs (defaults to [5, 10, 20])")
	ap.add_argument("--windows-grid", type=str, nargs="+", default=None, help="Grid for windows as JSON strings (defaults to ['[5]', '[5,20]'])")
	ap.add_argument("--reset-each-fold-grid", type=str, nargs="+", default=None, help="Grid for reset_each_fold (defaults to ['true', 'false'])")
	return ap.parse_args(argv)


def _parse_windows_str(s: str) -> List[int]:
	"""Parse a window configuration string like '[5,20]' into a list."""
	try:
		return json.loads(s)
	except Exception:
		# Try simple comma-separated format
		return [int(x.strip()) for x in s.strip("[]").split(",") if x.strip()]


def build_param_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
	"""Build parameter grid for SU5 sweep."""
	# top_k_pairs
	if args.top_k_pairs_grid:
		top_k_pairs_values = list(args.top_k_pairs_grid)
	else:
		top_k_pairs_values = [5, 10, 20]
	
	# windows
	if args.windows_grid:
		windows_values = [_parse_windows_str(s) for s in args.windows_grid]
	else:
		windows_values = [[5], [5, 20]]
	
	# reset_each_fold
	if args.reset_each_fold_grid:
		reset_values = [s.lower() == "true" for s in args.reset_each_fold_grid]
	else:
		reset_values = [True, False]
	
	# Build Cartesian product
	configs: List[Dict[str, Any]] = []
	for top_k, windows, reset in itertools.product(top_k_pairs_values, windows_values, reset_values):
		configs.append({
			"top_k_pairs": int(top_k),
			"windows": tuple(windows),
			"reset_each_fold": bool(reset),
		})
	
	return configs


def run_single_config(
	config_params: Dict[str, Any],
	su1_config: SU1Config,
	args: argparse.Namespace,
	X: pd.DataFrame,
	y: pd.Series,
	preprocess_settings: Dict[str, Dict[str, Any]],
	model_kwargs: Dict[str, Any],
	signal_mult_grid: tuple,
	signal_lo_grid: tuple,
	signal_hi_grid: tuple,
	signal_lam_grid: tuple,
	signal_eps: float,
	signal_optimize_for: str,
) -> Dict[str, Any]:
	"""Run OOF evaluation for a single SU5 configuration."""
	start_time = time.time()
	
	# Build SU5 config from params
	import numpy as np
	su5_config = SU5Config(
		id_column=su1_config.id_column,
		output_prefix="su5",
		top_k_pairs=int(config_params["top_k_pairs"]),
		top_k_pairs_per_group=None,
		windows=tuple(config_params["windows"]),
		reset_each_fold=bool(config_params["reset_each_fold"]),
		dtype_flag=np.dtype("uint8"),
		dtype_int=np.dtype("int16"),
		dtype_float=np.dtype("float32"),
	)
	
	# Build pipeline
	base_pipeline = build_pipeline(
		su1_config,
		su5_config,
		preprocess_settings,
		numeric_fill_value=args.numeric_fill_value,
		model_kwargs=model_kwargs,
		random_state=args.random_state,
	)
	callbacks = _initialise_callbacks(base_pipeline.named_steps["model"])
	
	# CV setup
	splitter = TimeSeriesSplit(n_splits=args.n_splits)
	X_np = X.reset_index(drop=True)
	y_np = y.reset_index(drop=True)
	y_np_array = y_np.to_numpy()
	
	# Pre-fit augmenter
	su5_prefit = SU5FeatureAugmenter(su1_config, su5_config, fill_value=args.numeric_fill_value)
	su5_prefit.fit(X_np)
	
	# Build fold_indices
	fold_indices_full = np.full(len(X_np), -1, dtype=int)
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np)):
		fold_indices_full[train_idx] = fold_idx
		fold_indices_full[val_idx] = fold_idx
	
	# Transform with fold_indices
	X_augmented_all = su5_prefit.transform(X_np, fold_indices=fold_indices_full)
	core_pipeline_template = cast(Pipeline, Pipeline(base_pipeline.steps[1:]))
	
	# Track feature count
	feature_count_su1 = len(getattr(su5_prefit, "su1_feature_names_", []))
	feature_count_su5 = len(getattr(su5_prefit, "su5_feature_names_", []))
	feature_count_total = X_augmented_all.shape[1]
	
	oof_pred = np.full(len(X_np), np.nan, dtype=float)
	fold_count = 0
	
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np), start=1):
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
		oof_pred[val_idx] = pred
		fold_count += 1
	
	# Calculate metrics
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
		best_metrics_global = evaluate_msr_proxy(
			oof_pred[valid_mask],
			y_np_array[valid_mask],
			best_params_global,
			eps=signal_eps,
			lam=0.0,
		)
		oof_msr = float(best_metrics_global["msr"])
		oof_msr_down = float(best_metrics_global["msr_down"])
	else:
		oof_rmse = float("nan")
		oof_msr = float("nan")
		oof_msr_down = float("nan")
	
	elapsed_time = time.time() - start_time
	
	result = {
		"top_k_pairs": config_params["top_k_pairs"],
		"windows": str(list(config_params["windows"])),
		"reset_each_fold": config_params["reset_each_fold"],
		"oof_rmse": oof_rmse,
		"oof_msr": oof_msr,
		"oof_msr_down": oof_msr_down,
		"feature_count_su1": feature_count_su1,
		"feature_count_su5": feature_count_su5,
		"feature_count_total": feature_count_total,
		"training_time_sec": elapsed_time,
		"folds_used": fold_count,
	}
	
	return result


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	
	# Load configs
	su1_config = load_su1_config(args.config_path)
	preprocess_settings = load_preprocess_policies(args.preprocess_config)
	
	# Load data
	data_dir = Path(args.data_dir)
	train_path = infer_train_file(data_dir, args.train_file)
	test_path = infer_test_file(data_dir, args.test_file)
	print(f"[info] train file: {train_path}")
	print(f"[info] test file: {test_path}")
	
	train_df = load_table(train_path)
	test_df = load_table(test_path)
	
	if args.id_col in train_df.columns:
		train_df = train_df.sort_values(args.id_col).reset_index(drop=True)
	if args.id_col in test_df.columns:
		test_df = test_df.sort_values(args.id_col).reset_index(drop=True)
	
	if args.target_col not in train_df.columns:
		raise KeyError(f"Target column '{args.target_col}' was not found in train data.")
	
	X, y, feature_cols = _prepare_features(train_df, test_df, target_col=args.target_col, id_col=args.id_col)
	
	# Model kwargs
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
	
	signal_mult_grid = tuple(float(x) for x in args.signal_mult_grid)
	signal_lo_grid = tuple(float(x) for x in args.signal_lo_grid)
	signal_hi_grid = tuple(float(x) for x in args.signal_hi_grid)
	signal_lam_grid = tuple(float(x) for x in args.signal_lam_grid)
	signal_eps = args.signal_eps
	signal_optimize_for = args.signal_optimize_for
	
	# Build parameter grid
	param_grid = build_param_grid(args)
	print(f"[info] evaluating {len(param_grid)} parameter configurations")
	
	# Run sweep
	all_results: List[Dict[str, Any]] = []
	for idx, config_params in enumerate(param_grid, start=1):
		print(f"\n[sweep {idx}/{len(param_grid)}] {config_params}")
		try:
			result = run_single_config(
				config_params,
				su1_config,
				args,
				X,
				y,
				preprocess_settings,
				model_kwargs,
				signal_mult_grid,
				signal_lo_grid,
				signal_hi_grid,
				signal_lam_grid,
				signal_eps,
				signal_optimize_for,
			)
			all_results.append(result)
			print(f"  [result] rmse={result['oof_rmse']:.6f} msr={result['oof_msr']:.6f} features={result['feature_count_total']} time={result['training_time_sec']:.1f}s")
		except Exception as exc:
			print(f"  [error] {exc}")
			import traceback
			traceback.print_exc()
	
	if not all_results:
		print("[error] no successful configurations")
		return 1
	
	# Save results
	timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
	json_path = out_dir / f"sweep_{timestamp}.json"
	with json_path.open("w", encoding="utf-8") as fh:
		json.dump(all_results, fh, indent=2, ensure_ascii=False)
	print(f"\n[ok] wrote detailed results: {json_path}")
	
	csv_path = out_dir / "sweep_summary.csv"
	fieldnames = [
		"top_k_pairs",
		"windows",
		"reset_each_fold",
		"oof_rmse",
		"oof_msr",
		"oof_msr_down",
		"feature_count_su1",
		"feature_count_su5",
		"feature_count_total",
		"training_time_sec",
		"folds_used",
	]
	with csv_path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames)
		writer.writeheader()
		for row in all_results:
			writer.writerow(row)
	print(f"[ok] wrote summary: {csv_path}")
	
	# Print best result
	valid_results = [r for r in all_results if not math.isnan(r["oof_msr"])]
	if valid_results:
		best = max(valid_results, key=lambda r: r["oof_msr"])
		print("\n[best] configuration:")
		print(f"  top_k_pairs={best['top_k_pairs']}, windows={best['windows']}, reset_each_fold={best['reset_each_fold']}")
		print(f"  oof_rmse={best['oof_rmse']:.6f}, oof_msr={best['oof_msr']:.6f}")
		print(f"  features: su1={best['feature_count_su1']}, su5={best['feature_count_su5']}, total={best['feature_count_total']}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())
