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
		raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
	if not preprocess_config_path.exists():
		raise FileNotFoundError(f"前処理設定ファイルが見つかりません: {preprocess_config_path}")

	# SU1とSU3の設定を読み込む
	su1_config = load_su1_config(config_path)
	su3_config = load_su3_config(config_path)

	# 前処理設定を読み込む
	preprocess_settings = load_preprocess_policies(preprocess_config_path)

	return su1_config, su3_config, preprocess_settings


def build_param_combinations(
	param_grid: Dict[str, List[Any]],
	include_imputation_trace: bool = False
) -> List[Dict[str, Any]]:
	"""パラメータグリッドから全組み合わせを生成。"""
	combinations = []

	# パラメータリストを抽出
	reappear_values = param_grid.get('reappear_top_k', [20])
	temporal_values = param_grid.get('temporal_top_k', [20])
	holiday_values = param_grid.get('holiday_top_k', [20])

	# itertools.productで全組み合わせを生成
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
		raise ValueError(f"サポートされていないファイル形式: {file_path.suffix}")


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

	# パラメータ値でSU3設定を更新
	# include_imputation_trace は Stage1 では sweep しないため、fallback を使用
	su3_config_dict = {
		'id_column': su3_base_config.id_column,
		'output_prefix': su3_base_config.output_prefix,
		'include_transitions': su3_base_config.include_transitions,
		'transition_group_agg': su3_base_config.transition_group_agg,
		'include_reappearance': su3_base_config.include_reappearance,
		'reappear_clip': su3_base_config.reappear_clip,
		'reappear_top_k': param_config['reappear_top_k'],
		'include_imputation_trace': bool(param_config.get('include_imputation_trace', su3_base_config.include_imputation_trace)),
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

	# パイプラインを構築（SU3 augmenter を含む）
	base_pipeline = build_pipeline(
		su1_config,
		su3_config,
		preprocess_settings,
		numeric_fill_value=numeric_fill_value,
		model_kwargs=model_kwargs,
		random_state=random_state,
	)

	# CV用の準備
	splitter = TimeSeriesSplit(n_splits=n_splits)
	X_reset = train_data.reset_index(drop=True)
	y_reset = pd.Series(target).reset_index(drop=True)
	
	# Pre-fit SU3 augmenter to get full augmented dataset
	# This follows the same pattern as SU2 to ensure fold boundary awareness
	from src.feature_generation.su3.feature_su3 import SU3FeatureAugmenter
	su3_prefit = SU3FeatureAugmenter(su1_config, su3_config)
	su3_prefit.fit(X_reset)
	
	# Build fold_indices array for proper SU3 state reset
	# Use validation indices to assign fold boundaries (avoids TimeSeriesSplit overlap issues)
	# Each validation region gets its fold_idx; earlier training-only regions get fold 0
	fold_indices_all = np.full(len(X_reset), -1, dtype=int)
	for fold_idx_iter, (_, val_idx_iter) in enumerate(splitter.split(X_reset)):
		fold_indices_all[val_idx_iter] = fold_idx_iter
	
	# Assign fold 0 to rows that were never in validation (early training-only data)
	first_assigned = np.where(fold_indices_all >= 0)[0]
	if first_assigned.size == 0:
		raise RuntimeError("No validation indices assigned in TimeSeriesSplit.")
	fold_indices_all[:first_assigned[0]] = 0
	
	# Generate full augmented dataset with fold boundaries
	X_augmented_all = su3_prefit.transform(X_reset, fold_indices=fold_indices_all)
	
	# Create core pipeline without SU3 augmenter (already applied)
	core_pipeline_template = Pipeline(base_pipeline.steps[1:])

	oof_pred = np.full(len(X_reset), np.nan, dtype=float)
	fold_scores: List[Dict[str, Any]] = []

	# クロスバリデーションを実行
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_reset), start=1):
		train_idx = np.array(train_idx)
		val_idx = np.array(val_idx)

		# ギャップを適用
		if gap > 0:
			if len(train_idx) > gap:
				train_idx = train_idx[:-gap]
			if len(val_idx) > gap:
				val_idx = val_idx[gap:]

		if len(train_idx) == 0 or len(val_idx) == 0:
			continue
		if min_val_size > 0 and len(val_idx) < min_val_size:
			continue

		# Use pre-augmented data (SU3 already applied with fold_indices)
		X_train = X_augmented_all.iloc[train_idx]
		y_train = y_reset.iloc[train_idx]
		X_val = X_augmented_all.iloc[val_idx]
		y_val = y_reset.iloc[val_idx]

		# このfold用にパイプラインをクローン（SU3以降のステップのみ）
		pipe = cast(Pipeline, clone(core_pipeline_template))

		# 早期停止なしで学習（スイープでは簡略化）
		# 注: eval_setは特徴変換の複雑さを避けるためスキップ
		fit_kwargs: Dict[str, Any] = {}

		pipe.fit(X_train, y_train, **fit_kwargs)

		# 予測
		pred = pipe.predict(X_val)
		pred = _to_1d(pred)
		oof_pred[val_idx] = pred

		# fold RMSEを計算
		rmse = float(math.sqrt(mean_squared_error(y_val, pred)))

		# fold MSRを計算（デフォルトのシグナルパラメータを使用）
		# NOTE: lo=0.0, hi=2.0 を使用（lo=hi=1.0だとシグナルが定数になりMSR=0）
		signal_params = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)
		metrics = evaluate_msr_proxy(pred, y_val.to_numpy(), signal_params, eps=1e-8, lam=0.0)
		msr = float(metrics['msr'])

		fold_scores.append({
			'fold': fold_idx,
			'rmse': rmse,
			'msr': msr,
		})

	# 全体のOOF指標を計算
	valid_mask = ~np.isnan(oof_pred)
	if valid_mask.any():
		oof_rmse = float(math.sqrt(mean_squared_error(target[valid_mask], oof_pred[valid_mask])))
		# NOTE: lo=0.0, hi=2.0 を使用（lo=hi=1.0だとシグナルが定数になりMSR=0）
		signal_params = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)
		oof_metrics = evaluate_msr_proxy(oof_pred[valid_mask], target[valid_mask], signal_params, eps=1e-8, lam=0.0)
		oof_msr = float(oof_metrics['msr'])
	else:
		oof_rmse = float('nan')
		oof_msr = float('nan')

	# 特徴量数を取得（拡張特徴量から）
	# Pre-augmented dataから直接取得
	n_features = X_augmented_all.shape[1]

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

	# JSON保存（fold別スコアを含む詳細結果）
	json_file = output_dir / f'sweep_{timestamp}.json'
	json_data = {
		'metadata': metadata,
		'results': results,
	}
	with json_file.open('w', encoding='utf-8') as f:
		json.dump(json_data, f, indent=2, ensure_ascii=False)
	print(f"[保存] 詳細結果を保存しました: {json_file}")

	# CSV保存（サマリー）
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

	print(f"[保存] サマリー結果を追記しました: {csv_file}")


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
	print("SU3 ハイパーパラメータスイープ")
	print("=" * 80)
	print(f"データディレクトリ: {args.data_dir}")
	print(f"設定ファイルパス: {args.config_path}")
	print(f"前処理設定: {args.preprocess_config}")
	print(f"出力ディレクトリ: {args.output_dir}")
	print(f"分割数: {args.n_splits}")
	print(f"ギャップ: {args.gap}")
	print(f"代入影響トレース: {args.include_imputation_trace}")
	print()

	# 設定を読み込む
	print("[読込] 設定を読み込んでいます...")
	su1_config, su3_config, preprocess_settings = load_configs(
		args.config_path,
		args.preprocess_config
	)
	print(f"[読込] SU1設定を読み込みました: {len(su1_config.target_groups)}グループ")
	print(f"[読込] SU3設定を読み込みました: prefix={su3_config.output_prefix}")

	# 学習データを読み込む
	print("[読込] 学習データを読み込んでいます...")
	if args.train_file:
		train_file = Path(args.train_file)
	else:
		train_file = infer_train_file(args.data_dir)
	
	print(f"[読込] 読み込み中: {train_file}")
	train_df = load_table(train_file)
	print(f"[読込] 学習データの形状: {train_df.shape}")

	# 特徴量とターゲットを準備
	if args.target_col not in train_df.columns:
		raise ValueError(f"ターゲット列 '{args.target_col}' が学習データに見つかりません")

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

	print(f"[準備] 特徴量列数: {len(feature_cols)}")
	print(f"[準備] ターゲットの形状: {y.shape}")

	# パラメータ組み合わせを生成
	print("[スイープ] パラメータ組み合わせを生成しています...")
	param_combinations = build_param_combinations(
		PARAM_GRID,
		include_imputation_trace=args.include_imputation_trace
	)
	print(f"[スイープ] 全構成数: {len(param_combinations)}")
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

	# 各構成を評価
	results: List[Dict[str, Any]] = []
	config_start_time = time.time()
	for idx, param_config in enumerate(param_combinations, start=1):
		iter_start = time.time()
		
		# 推定残り時間を計算
		if idx > 1:
			elapsed = time.time() - config_start_time
			avg_time_per_config = elapsed / (idx - 1)
			remaining_configs = len(param_combinations) - idx + 1
			estimated_remaining = avg_time_per_config * remaining_configs
			remaining_str = f" (推定残り時間: {estimated_remaining/60:.1f}分)"
		else:
			remaining_str = ""
		
		print(f"[スイープ] 構成 {idx}/{len(param_combinations)} を評価中{remaining_str}")
		print(f"[スイープ] パラメータ: {param_config}")

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

			iter_time = time.time() - iter_start
			print(f"[結果] OOF RMSE: {result['oof_rmse']:.6f}")
			print(f"[結果] OOF MSR: {result['oof_msr']:.6f}")
			print(f"[結果] 特徴量数: {result['n_features']}")
			print(f"[結果] 時間: {result['train_time_sec']:.1f}秒 (この構成: {iter_time:.1f}秒)")
			print()

		except Exception as e:
			print(f"[エラー] 構成 {idx} が失敗しました: {e}")
			print("[エラー] 次の構成に進みます...")
			print()
			continue

	# OOF MSRの降順でソート（大きいほど良い - Sharpe-like指標）
	# タイブレークとしてRMSEの昇順（小さいほど良い）
	results.sort(key=lambda x: (-x['oof_msr'] if not math.isnan(x['oof_msr']) else float('-inf'), x['oof_rmse']))

	# 結果を保存
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

	# ベスト構成を表示
	print()
	print("=" * 80)
	print("ベスト構成")
	print("=" * 80)
	if results:
		best = results[0]
		print(f"構成: {best['config']}")
		print(f"OOF RMSE: {best['oof_rmse']:.6f}")
		print(f"OOF MSR: {best['oof_msr']:.6f}")
		print(f"特徴量数: {best['n_features']}")
		print(f"時間: {best['train_time_sec']:.1f}秒")
	else:
		print("成功した構成がありません")
	print("=" * 80)
	return 0


if __name__ == "__main__":
	sys.exit(main())
