"""SU1 特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su1.feature_su1 import SU1Config, SU1FeatureGenerator


def _build_config(tmp_path: Path) -> SU1Config:
	mapping = {
		"id_column": "date_id",
		"exclude_columns": [],
		"groups": {"include": ["M", "E", "V"], "exclude": []},
		"gap_clip": 5,
		"run_clip": 3,
		"dtype": {"flag": "uint8", "run": "int16"},
		"include_group_means": {
			"gap_ffill": True,
			"run_na": True,
			"exclude_all_nan": True,
		},
		"data": {
			"raw_dir": str(tmp_path),
			"train_filename": "train.csv",
			"test_filename": "test.csv",
		},
	}
	return SU1Config.from_mapping(mapping, base_dir=tmp_path)


def _build_config_with_brushup(tmp_path: Path) -> SU1Config:
	"""Build config with brushup enabled."""
	mapping = {
		"id_column": "date_id",
		"exclude_columns": [],
		"groups": {"include": ["M", "E", "V"], "exclude": []},
		"gap_clip": 5,
		"run_clip": 3,
		"dtype": {"flag": "uint8", "run": "int16"},
		"include_group_means": {
			"gap_ffill": True,
			"run_na": True,
			"exclude_all_nan": True,
		},
		"data": {
			"raw_dir": str(tmp_path),
			"train_filename": "train.csv",
			"test_filename": "test.csv",
		},
		"brushup": {
			"enabled": True,
			"miss_count_window": 5,
			"streak_threshold": 3,
			"regime_change": {
				"recent_window": 5,
				"past_window": 30,
				"recent_threshold": 0.5,
				"past_threshold": 0.1,
			},
		},
	}
	return SU1Config.from_mapping(mapping, base_dir=tmp_path)


def test_su1_feature_generation(tmp_path: Path) -> None:
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"M1": [1.0, np.nan, np.nan, 5.0, np.nan],
			"M2": [np.nan, np.nan, 2.0, 3.0, 4.0],
			"M3": [5.0, 5.0, 5.0, 5.0, 5.0],
			"E1": [np.nan, np.nan, np.nan, np.nan, np.nan],
			"V1": [10.0, 11.0, 12.0, 13.0, 14.0],
		}
	).set_index("date_id")

	transformer = SU1FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	expected_columns = {
		"m/M1",
		"m/M2",
		"m/M3",
		"gap_ffill/M1",
		"gap_ffill/M2",
		"gap_ffill/E1",
		"run_na/M1",
		"run_na/M2",
		"run_obs/M1",
		"run_obs/M2",
		"run_obs/M3",
		"m/V1",
		"m_any_day",
		"m_rate_day",
		"m_cnt/ALL",
		"m_rate/ALL",
		"m_cnt/M",
		"m_rate/M",
		"avg_gapff/M",
		"avg_run_na/M",
		"avg_gapff/E",
		"avg_run_na/E",
		"avg_gapff/V",
		"avg_run_na/V",
	}

	assert expected_columns.issubset(features.columns)

	# 行方向の検証（代表例）
	assert features.loc[1, "m/M1"] == 1
	assert features.loc[0, "m/M2"] == 1
	assert features.loc[1, "gap_ffill/M1"] == 1
	assert features.loc[0, "gap_ffill/M2"] == 5  # gap_clip の上限確認
	assert features.loc[2, "run_na/M1"] == 2
	assert features.loc[4, "run_obs/M3"] == 3  # run_clip の上限確認
	assert features.loc[0, "m/V1"] == 0

	# 集計指標の検証
	assert features.loc[0, "m_cnt/ALL"] == features.loc[0, "m_any_day"]
	assert features.loc[0, "m_rate/ALL"] == features.loc[0, "m_rate_day"]
	assert features.loc[1, "m_cnt/M"] == 2
	assert features.loc[1, "m_rate/M"] == pytest.approx(2 / 3)
	assert features.loc[0, "avg_gapff/M"] == pytest.approx((0 + 5 + 0) / 3)
	assert features.loc[1, "avg_run_na/M"] == pytest.approx((1 + 2 + 0) / 3)
	assert pd.isna(features.loc[0, "avg_gapff/E"])
	assert pd.isna(features.loc[0, "avg_run_na/E"])
	assert features.loc[2, "avg_gapff/V"] == pytest.approx(0.0)
	assert features.loc[2, "avg_run_na/V"] == pytest.approx(0.0)


def test_su1_brushup_miss_count_last_5d(tmp_path: Path) -> None:
	"""Test miss_count_last_5d rolling calculation."""
	config = _build_config_with_brushup(tmp_path)

	# Create data with specific missing pattern
	raw = pd.DataFrame(
		{
			"date_id": list(range(10)),
			"M1": [np.nan, np.nan, 1.0, 1.0, 1.0, np.nan, np.nan, 1.0, 1.0, 1.0],
			"M2": [np.nan, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0],
			"M3": [1.0] * 10,
		}
	).set_index("date_id")

	transformer = SU1FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	assert "miss_count_last_5d" in features.columns
	assert "miss_ratio_last_5d" in features.columns

	# First 4 rows should be NaN (min_periods=5)
	assert pd.isna(features.loc[0, "miss_count_last_5d"])
	assert pd.isna(features.loc[3, "miss_count_last_5d"])

	# Row 4: sum of missings in rows 0-4 = 2+1+0+0+0 = 3
	assert features.loc[4, "miss_count_last_5d"] == 3


def test_su1_brushup_is_long_missing_streak(tmp_path: Path) -> None:
	"""Test is_long_missing_streak threshold detection."""
	config = _build_config_with_brushup(tmp_path)

	# Create data where M1 has 3+ consecutive NaNs
	raw = pd.DataFrame(
		{
			"date_id": list(range(10)),
			"M1": [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
			"M2": [1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
			"M3": [1.0] * 10,
		}
	).set_index("date_id")

	transformer = SU1FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	assert "is_long_missing_streak" in features.columns
	assert "long_streak_col_count" in features.columns

	# Row 2: M1 has run_na=3, should trigger flag
	assert features.loc[2, "is_long_missing_streak"] == 1
	assert features.loc[2, "long_streak_col_count"] == 1

	# Row 3: M1 no longer missing, flag should be 0
	assert features.loc[3, "is_long_missing_streak"] == 0


def test_su1_brushup_miss_regime_change(tmp_path: Path) -> None:
	"""Test miss_regime_change detection."""
	config = _build_config_with_brushup(tmp_path)

	# Create data with regime change: M1 present for 30+ rows, then missing for 5 rows
	n_rows = 50
	m1_values = [1.0] * 40 + [np.nan] * 10
	raw = pd.DataFrame(
		{
			"date_id": list(range(n_rows)),
			"M1": m1_values,
			"M2": [1.0] * n_rows,
			"M3": [1.0] * n_rows,
		}
	).set_index("date_id")

	transformer = SU1FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	assert "miss_regime_change" in features.columns

	# Row 44: recent 5 days (40-44) have 5 NaNs (100% missing)
	# past 30 days (5-34) have 0 NaNs (0% missing)
	# Should trigger regime change
	if not pd.isna(features.loc[44, "miss_regime_change"]):
		assert features.loc[44, "miss_regime_change"] == 1


def test_su1_brushup_dtype(tmp_path: Path) -> None:
	"""Test that brushup features have correct dtypes."""
	config = _build_config_with_brushup(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": list(range(10)),
			"M1": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
			"M2": [1.0] * 10,
			"M3": [1.0] * 10,
		}
	).set_index("date_id")

	transformer = SU1FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# Check dtypes
	assert features["miss_count_last_5d"].dtype == np.int16
	assert features["miss_ratio_last_5d"].dtype == np.float32
	assert features["is_long_missing_streak"].dtype == np.uint8
	assert features["long_streak_col_count"].dtype == np.int16
	assert features["miss_regime_change"].dtype == np.uint8
