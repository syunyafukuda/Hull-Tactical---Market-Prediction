"""SU3スイープ機能の単体テスト。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_generation.su1.feature_su1 import SU1Config
from src.feature_generation.su3.feature_su3 import SU3Config
from src.feature_generation.su3.sweep_oof import (
	build_param_combinations,
	evaluate_single_config,
	save_results,
)


def _create_test_su1_config() -> SU1Config:
	"""テスト用のSU1Configを構築する。"""
	mapping = {
		"id_column": "date_id",
		"exclude_columns": ["forward_returns", "risk_free_rate", "market_forward_excess_returns"],
		"groups": {
			"include": ["M", "E"],
			"exclude": []
		},
		"gap_clip": 60,
		"run_clip": 60,
		"dtype": {
			"flag": "uint8",
			"run": "int16"
		},
		"include_group_means": {
			"gap_ffill": True,
			"run_na": True,
			"exclude_all_nan": True
		},
		"data": {
			"raw_dir": "data/raw",
			"train_filename": "train.csv",
			"test_filename": "test.csv"
		}
	}
	return SU1Config.from_mapping(mapping, base_dir=Path.cwd())


def _create_test_su3_config() -> SU3Config:
	"""テスト用のSU3Configを構築する。"""
	mapping = {
		"id_column": "date_id",
		"output_prefix": "su3",
		"include_transitions": True,
		"transition_group_agg": True,
		"include_reappearance": True,
		"reappear_clip": 60,
		"reappear_top_k": 3,
		"include_imputation_trace": False,
		"imp_delta_winsorize_p": 0.99,
		"imp_delta_top_k": 20,
		"imp_policy_group_level": True,
		"include_temporal_bias": True,
		"temporal_burn_in": 3,
		"temporal_top_k": 3,
		"include_holiday_interaction": True,
		"holiday_top_k": 3,
		"dtype": {
			"flag": "uint8",
			"int": "int16",
			"float": "float32"
		},
		"reset_each_fold": True,
	}
	return SU3Config.from_mapping(mapping)


def _create_test_preprocess_settings() -> dict:
	"""テスト用の前処理設定を構築する。"""
	return {
		"m_group": {
			"policy": "ffill_bfill",
			"rolling_window": 5,
			"ema_alpha": 0.3,
			"calendar_column": None,
			"policy_params": {},
		},
		"e_group": {
			"policy": "ffill_bfill",
			"rolling_window": 5,
			"ema_alpha": 0.3,
			"calendar_column": None,
			"policy_params": {},
			"all_nan_strategy": "keep_nan",
			"all_nan_fill": 0.0,
		},
		"i_group": {
			"policy": "ffill_bfill",
			"rolling_window": 5,
			"ema_alpha": 0.3,
			"calendar_column": None,
			"policy_params": {},
			"clip_quantile_low": 0.001,
			"clip_quantile_high": 0.999,
			"enable_quantile_clip": True,
		},
		"p_group": {
			"policy": "ffill_bfill",
			"rolling_window": 5,
			"ema_alpha": 0.3,
			"calendar_column": None,
			"policy_params": {},
			"mad_clip_scale": 4.0,
			"mad_clip_min_samples": 25,
			"enable_mad_clip": True,
			"fallback_quantile_low": 0.005,
			"fallback_quantile_high": 0.995,
		},
		"s_group": {
			"policy": "ffill_bfill",
			"rolling_window": 5,
			"ema_alpha": 0.3,
			"calendar_column": None,
			"policy_params": {},
			"mad_clip_scale": 4.0,
			"mad_clip_min_samples": 25,
			"enable_mad_clip": True,
			"fallback_quantile_low": 0.005,
			"fallback_quantile_high": 0.995,
		},
	}


def _create_small_test_data(n_rows: int = 100, n_cols: int = 10) -> pd.DataFrame:
	"""小規模テストデータを作成する（生データを模擬）。"""
	np.random.seed(42)
	
	data = {}
	
	# 生データを模擬（市場データのような形式）
	# M群とE群の列を作成
	for i in range(n_cols):
		# ランダムな市場データ（欠損あり）
		col_data = np.random.randn(n_rows) * 0.01
		# ランダムに欠損を挿入
		mask = np.random.rand(n_rows) < 0.3
		col_data[mask] = np.nan
		data[f"M{i}"] = col_data
	
	for i in range(n_cols):
		col_data = np.random.randn(n_rows) * 0.02
		mask = np.random.rand(n_rows) < 0.25
		col_data[mask] = np.nan
		data[f"E{i}"] = col_data
	
	df = pd.DataFrame(data)
	return df


def test_build_param_combinations() -> None:
	"""パラメータ組み合わせ生成の確認。"""
	param_grid = {
		'reappear_top_k': [10, 20],
		'temporal_top_k': [10, 20],
		'holiday_top_k': [10, 20]
	}
	combinations = build_param_combinations(param_grid, include_imputation_trace=False)
	
	# 2 × 2 × 2 = 8通り
	assert len(combinations) == 8
	
	# 各組み合わせが辞書であること
	for config in combinations:
		assert 'reappear_top_k' in config
		assert 'temporal_top_k' in config
		assert 'holiday_top_k' in config
		assert 'include_imputation_trace' in config
		assert config['include_imputation_trace'] is False
	
	# 全組み合わせが一意であること
	config_tuples = [
		(c['reappear_top_k'], c['temporal_top_k'], c['holiday_top_k'])
		for c in combinations
	]
	assert len(set(config_tuples)) == 8


def test_evaluate_single_config_small_data() -> None:
	"""小規模データでのOOF評価動作確認。"""
	# 小規模データ作成（100行×10列）
	data = _create_small_test_data(n_rows=100, n_cols=5)
	target = np.random.randn(100) * 0.01  # 小さい値にスケール
	
	# 設定作成
	su1_config = _create_test_su1_config()
	su3_config = _create_test_su3_config()
	preprocess_settings = _create_test_preprocess_settings()
	
	param_config = {
		'reappear_top_k': 3,
		'temporal_top_k': 3,
		'holiday_top_k': 3,
		'include_imputation_trace': False
	}
	
	# 評価実行
	result = evaluate_single_config(
		su1_config=su1_config,
		su3_base_config=su3_config,
		param_config=param_config,
		preprocess_settings=preprocess_settings,
		train_data=data,
		target=target,
		id_col='date_id',
		n_splits=2,
		gap=0,
		min_val_size=0,
		model_kwargs={
			'n_estimators': 10,
			'learning_rate': 0.1,
			'random_state': 42,
			'verbosity': -1
		},
		numeric_fill_value=0.0,
		random_state=42
	)
	
	# 結果形式の確認
	assert 'config' in result
	assert 'oof_rmse' in result
	assert 'oof_msr' in result
	assert 'n_features' in result
	assert 'train_time_sec' in result
	assert 'fold_scores' in result
	
	# スコアが数値であること
	assert isinstance(result['oof_rmse'], float)
	assert isinstance(result['oof_msr'], float)
	assert result['oof_rmse'] > 0 or np.isnan(result['oof_rmse'])
	assert result['oof_msr'] >= 0 or np.isnan(result['oof_msr'])  # MSR can be 0.0
	
	# fold数が正しいこと（最大2、gap等で減る可能性あり）
	assert len(result['fold_scores']) <= 2
	assert len(result['fold_scores']) >= 0


def test_save_results() -> None:
	"""結果保存機能の確認。"""
	results = [
		{
			'config': {
				'reappear_top_k': 20,
				'temporal_top_k': 20,
				'holiday_top_k': 20,
				'include_imputation_trace': False
			},
			'oof_rmse': 0.012,
			'oof_msr': 0.0185,
			'n_features': 474,
			'train_time_sec': 45.2,
			'fold_scores': [
				{'fold': 1, 'rmse': 0.0121, 'msr': 0.0186},
				{'fold': 2, 'rmse': 0.0119, 'msr': 0.0184},
			]
		}
	]
	
	metadata = {
		'timestamp': '2025-11-22T15:30:45',
		'n_configs': 1,
		'n_splits': 2,
		'gap': 0,
		'model_params': {
			'learning_rate': 0.05,
			'n_estimators': 600,
		}
	}
	
	with tempfile.TemporaryDirectory() as tmpdir:
		output_dir = Path(tmpdir)
		timestamp = '2025-11-22_153045'
		
		save_results(results, output_dir, timestamp, metadata)
		
		# JSONファイルが作成されていること
		json_file = output_dir / f'sweep_{timestamp}.json'
		assert json_file.exists()
		
		# CSVファイルが作成されていること
		csv_file = output_dir / 'sweep_summary.csv'
		assert csv_file.exists()
		
		# JSONが正しく読めること
		with json_file.open('r') as f:
			data = json.load(f)
			assert 'metadata' in data
			assert 'results' in data
			assert data['metadata']['n_configs'] == 1
			assert len(data['results']) == 1
		
		# CSVが正しく読めること
		import csv
		with csv_file.open('r') as f:
			reader = csv.DictReader(f)
			rows = list(reader)
			assert len(rows) == 1
			assert rows[0]['reappear_top_k'] == '20'
			assert rows[0]['temporal_top_k'] == '20'
			assert rows[0]['holiday_top_k'] == '20'


def test_build_param_combinations_with_imputation() -> None:
	"""代入影響フラグを含むパラメータ組み合わせ生成の確認。"""
	param_grid = {
		'reappear_top_k': [10, 20],
		'temporal_top_k': [10],
		'holiday_top_k': [10]
	}
	combinations = build_param_combinations(param_grid, include_imputation_trace=True)
	
	# 2 × 1 × 1 = 2通り
	assert len(combinations) == 2
	
	# 全てのconfigでinclude_imputation_trace=Trueであること
	for config in combinations:
		assert config['include_imputation_trace'] is True
