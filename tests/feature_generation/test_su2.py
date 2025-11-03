"""SU2 特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su2.feature_su2 import SU2Config, SU2FeatureGenerator


def _build_config(tmp_path: Path) -> SU2Config:
	mapping = {
		"id_column": "date_id",
		"input_sources": ["m", "gap_ffill", "run_na", "run_obs"],
		"target_groups": {"include": ["M", "E", "V"], "exclude": []},
		"rolling_windows": [3, 5],
		"ewma_alpha": [0.3, 0.5],
		"recovery_clip": 10,
		"clip_max": 10,
		"dtype": {"flag": "uint8", "run": "int16", "float": "float32"},
		"output_prefix": "su2",
		"drop_constant_columns": True,
		"features": {
			"rolling": {
				"include_metrics": ["mean", "std", "max_run_na", "max_run_obs", "zscore"],
				"include_current": False,
				"fill_missing_with_zero": True,
			},
			"ewma": {
				"signals": ["m", "gap_ffill"],
				"include_std": True,
				"reset_each_fold": True,
			},
			"transitions": {
				"flip_rate_windows": [3, 5],
				"burst_score_windows": [3, 5],
				"recovery_feature": True,
			},
			"normalization": {
				"minmax_windows": [3, 5],
				"rank_windows": [3, 5],
				"epsilon": 1.0e-6,
			},
		},
	}
	return SU2Config.from_mapping(mapping, base_dir=tmp_path)


def test_su2_all_nan(tmp_path: Path) -> None:
	"""全NaNケース: 全て欠損している列に対する挙動を確認。"""
	config = _build_config(tmp_path)
	
	# SU1 output with all NaN column
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [1, 1, 1, 1, 1],  # All missing
			"gap_ffill/M1": [5, 5, 5, 5, 5],  # Constant gap
			"run_na/M1": [1, 2, 3, 4, 5],  # Increasing run
			"run_obs/M1": [0, 0, 0, 0, 0],  # No observation
		}
	).set_index("date_id")
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df)
	
	# Should generate features without errors
	assert features is not None
	assert len(features) == len(su1_df)
	
	# Check that rolling mean exists
	assert any("roll_mean" in col for col in features.columns)


def test_su2_alternating_nan(tmp_path: Path) -> None:
	"""交互NaNケース: 欠損と観測が交互に現れるパターン。"""
	config = _build_config(tmp_path)
	
	# SU1 output with alternating pattern
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [1, 0, 1, 0, 1, 0],  # Alternating missing/observed
			"gap_ffill/M1": [0, 1, 0, 1, 0, 1],
			"run_na/M1": [1, 0, 1, 0, 1, 0],
			"run_obs/M1": [0, 1, 0, 1, 0, 1],
		}
	).set_index("date_id")
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df)
	
	# Check flip rate is high (many transitions)
	flip_rate_cols = [col for col in features.columns if "flip_rate" in col]
	assert len(flip_rate_cols) > 0
	
	# Flip rate should be non-zero for alternating pattern
	for col in flip_rate_cols:
		assert features[col].iloc[-1] > 0


def test_su2_island_nan(tmp_path: Path) -> None:
	"""島状NaNケース: 欠損が島状に現れるパターン。"""
	config = _build_config(tmp_path)
	
	# SU1 output with island pattern: NaN-OBS-NaN-OBS
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5, 6, 7],
			"m/M1": [1, 1, 0, 0, 1, 1, 0, 0],
			"gap_ffill/M1": [2, 2, 0, 0, 2, 2, 0, 0],
			"run_na/M1": [1, 2, 0, 0, 1, 2, 0, 0],
			"run_obs/M1": [0, 0, 1, 2, 0, 0, 1, 2],
		}
	).set_index("date_id")
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df)
	
	# Check recovery lag feature
	recovery_cols = [col for col in features.columns if "recovery_lag" in col]
	assert len(recovery_cols) > 0
	
	# Recovery lag should reset when transitioning from NA to OBS
	for col in recovery_cols:
		# After island recovery, lag should be low
		assert features[col].iloc[2] >= 0  # First recovery point


def test_su2_fold_boundary_reset(tmp_path: Path) -> None:
	"""折境界リセットケース: foldごとに状態がリセットされることを確認。"""
	config = _build_config(tmp_path)
	
	# SU1 output spanning two folds
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 0, 0, 1, 1, 1],
			"gap_ffill/M1": [0, 0, 0, 1, 2, 3],
			"run_na/M1": [0, 0, 0, 1, 2, 3],
			"run_obs/M1": [1, 2, 3, 0, 0, 0],
		}
	).set_index("date_id")
	
	# Define fold boundaries: fold 0 = [0,1,2], fold 1 = [3,4,5]
	fold_indices = np.array([0, 0, 0, 1, 1, 1])
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df, fold_indices=fold_indices)
	
	# Check EWMA feature resets at fold boundary
	ewma_cols = [col for col in features.columns if "ewma" in col and "m/M1" in col]
	assert len(ewma_cols) > 0
	
	# At fold boundary (index 3), EWMA should start fresh
	for col in ewma_cols:
		# Value at index 3 should be the raw value (fresh start)
		assert features[col].iloc[3] == pytest.approx(su1_df["m/M1"].iloc[3], abs=0.01)


def test_su2_rolling_statistics(tmp_path: Path) -> None:
	"""ローリング統計の基本動作を確認。"""
	config = _build_config(tmp_path)
	
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [0.0, 1.0, 0.0, 1.0, 0.0],
			"gap_ffill/M1": [0, 1, 0, 1, 0],
			"run_na/M1": [0, 1, 0, 1, 0],
			"run_obs/M1": [1, 0, 1, 0, 1],
		}
	).set_index("date_id")
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df)
	
	# Check rolling mean exists and is computed correctly
	mean_cols = [col for col in features.columns if "roll_mean[3]" in col and "m/M1" in col]
	assert len(mean_cols) > 0
	
	# At index 3, rolling mean of past 3 values [0, 1, 0] should be ~0.333
	for col in mean_cols:
		assert features[col].iloc[3] == pytest.approx(1.0/3.0, abs=0.01)


def test_su2_ewma_computation(tmp_path: Path) -> None:
	"""EWMA計算の基本動作を確認。"""
	config = _build_config(tmp_path)
	
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3],
			"m/M1": [1.0, 1.0, 1.0, 0.0],
			"gap_ffill/M1": [0, 0, 0, 0],
			"run_na/M1": [0, 0, 0, 0],
			"run_obs/M1": [0, 0, 0, 0],
		}
	).set_index("date_id")
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df)
	
	# Check EWMA with alpha=0.5 exists
	ewma_cols = [col for col in features.columns if "ewma[0.5]" in col and "m/M1" in col]
	assert len(ewma_cols) > 0
	
	# EWMA should converge towards values
	for col in ewma_cols:
		# After 3 steps of 1.0, EWMA should be close to 1.0
		assert features[col].iloc[2] > 0.9


def test_su2_normalization(tmp_path: Path) -> None:
	"""正規化特徴の基本動作を確認。"""
	# Use config with drop_constant_columns=False for this test
	mapping = {
		"id_column": "date_id",
		"input_sources": ["m", "gap_ffill", "run_na", "run_obs"],
		"target_groups": {"include": ["M", "E", "V"], "exclude": []},
		"rolling_windows": [3, 5],
		"ewma_alpha": [0.3, 0.5],
		"recovery_clip": 10,
		"clip_max": 10,
		"dtype": {"flag": "uint8", "run": "int16", "float": "float32"},
		"output_prefix": "su2",
		"drop_constant_columns": False,  # Don't drop constants for this test
		"features": {
			"rolling": {
				"include_metrics": ["mean", "std", "max_run_na", "max_run_obs", "zscore"],
				"include_current": False,
				"fill_missing_with_zero": True,
			},
			"ewma": {
				"signals": ["m", "gap_ffill"],
				"include_std": True,
				"reset_each_fold": True,
			},
			"transitions": {
				"flip_rate_windows": [3, 5],
				"burst_score_windows": [3, 5],
				"recovery_feature": True,
			},
			"normalization": {
				"minmax_windows": [3, 5],
				"rank_windows": [3, 5],
				"epsilon": 1.0e-6,
			},
		},
	}
	config = SU2Config.from_mapping(mapping, base_dir=tmp_path)
	
	su1_df = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [0, 0, 0, 0, 0],
			"gap_ffill/M1": [0.0, 1.0, 2.0, 3.0, 4.0],  # Linear increase
			"run_na/M1": [0, 0, 0, 0, 0],
			"run_obs/M1": [0, 0, 0, 0, 0],
		}
	).set_index("date_id")
	
	transformer = SU2FeatureGenerator(config)
	transformer.fit(su1_df)
	features = transformer.transform(su1_df)
	
	# Check minmax normalization exists
	minmax_cols = [col for col in features.columns if "minmax" in col and "gap/M1" in col]
	assert len(minmax_cols) > 0
	
	# Minmax normalizes against past window, so values can be > 1.0 if current exceeds past max
	for col in minmax_cols:
		# Check that the normalization is computed (not all zeros for varying data)
		assert features[col].max() > 0.0
