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
