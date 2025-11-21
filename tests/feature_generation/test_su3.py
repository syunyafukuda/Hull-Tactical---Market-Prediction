"""SU3 特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su3.feature_su3 import SU3Config, SU3FeatureGenerator


def _build_config(tmp_path: Path) -> SU3Config:
	"""テスト用のSU3Configを生成する。"""
	mapping = {
		"id_column": "date_id",
		"output_prefix": "su3",
		"include_transitions": True,
		"transition_group_agg": True,
		"include_reappearance": True,
		"reappear_clip": 10,
		"reappear_top_k": 3,
		"include_imputation_trace": False,
		"imp_delta_winsorize_p": 0.99,
		"imp_delta_top_k": 20,
		"imp_policy_group_level": True,
		"include_temporal_bias": True,
		"temporal_burn_in": 2,
		"temporal_top_k": 3,
		"include_holiday_interaction": True,
		"holiday_top_k": 3,
		"dtype": {"flag": "uint8", "int": "int16", "float": "float32"},
		"reset_each_fold": True,
	}
	return SU3Config.from_mapping(mapping, base_dir=tmp_path)


def test_su3_config_loading(tmp_path: Path) -> None:
	"""SU3Config の読み込みテスト。"""
	config = _build_config(tmp_path)
	assert config.id_column == "date_id"
	assert config.output_prefix == "su3"
	assert config.include_transitions is True
	assert config.transition_group_agg is True
	assert config.reappear_clip == 10
	assert config.reappear_top_k == 3
	assert config.temporal_burn_in == 2


def test_su3_transition_flags(tmp_path: Path) -> None:
	"""遷移フラグの基本動作確認。"""
	config = _build_config(tmp_path)

	# m/<col>が [0, 0, 1, 1, 0, 0] のパターン
	# 遷移: index 2で 0→1 (obs_to_na), index 4で 1→0 (na_to_obs)
	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 0, 1, 1, 0, 0],  # 観測, 観測, NaN, NaN, 観測, 観測
			"m/M2": [1, 1, 1, 0, 0, 0],  # NaN, NaN, NaN, 観測, 観測, 観測
			"run_na/M1": [0, 0, 1, 2, 0, 0],
			"run_na/M2": [1, 2, 3, 0, 0, 0],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# 群集約遷移率
	assert "su3/trans_rate/M" in features.columns

	# index 0: 初回なので遷移なし
	assert features.loc[0, "su3/trans_rate/M"] == 0.0

	# index 2: M1が 0→1, M2は 1→1 で遷移1/2 = 0.5
	assert features.loc[2, "su3/trans_rate/M"] == pytest.approx(0.5)

	# index 4: M1が 1→0, M2が 0→0 で遷移1/2 = 0.5
	assert features.loc[4, "su3/trans_rate/M"] == pytest.approx(0.5)


def test_su3_reappear_gap(tmp_path: Path) -> None:
	"""再出現間隔の計算テスト。"""
	config = _build_config(tmp_path)

	# m/<col>が [0, 1, 1, 1, 0, 0] のパターン
	# run_na が [0, 1, 2, 3, 0, 0]
	# index 4で再出現（1→0）、その時のrun_na[3]=3
	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 1, 1, 0, 0],
			"run_na/M1": [0, 1, 2, 3, 0, 0],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	assert "su3/reappear_gap/M1" in features.columns
	assert "su3/pos_since_reappear/M1" in features.columns

	# index 0-3: 再出現なし
	assert features.loc[0, "su3/reappear_gap/M1"] == 0
	assert features.loc[1, "su3/reappear_gap/M1"] == 0
	assert features.loc[2, "su3/reappear_gap/M1"] == 0
	assert features.loc[3, "su3/reappear_gap/M1"] == 0

	# index 4: 再出現（run_na[3]=3）
	assert features.loc[4, "su3/reappear_gap/M1"] == 3

	# index 5: 観測継続中
	assert features.loc[5, "su3/reappear_gap/M1"] == 0


def test_su3_pos_since_reappear(tmp_path: Path) -> None:
	"""再出現位置の正規化テスト。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [1, 1, 0, 0, 0, 0],  # NaN, NaN, 観測, 観測, 観測, 観測
			"run_na/M1": [1, 2, 0, 0, 0, 0],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# index 2: 再出現点（pos=0）
	assert features.loc[2, "su3/pos_since_reappear/M1"] == pytest.approx(0.0)

	# index 3: 1日経過（pos=1/60）
	assert features.loc[3, "su3/pos_since_reappear/M1"] == pytest.approx(1.0 / 60.0)

	# index 4: 2日経過（pos=2/60）
	assert features.loc[4, "su3/pos_since_reappear/M1"] == pytest.approx(2.0 / 60.0)


def test_su3_temporal_bias(tmp_path: Path) -> None:
	"""曜日・月次パターン生成テスト。"""
	config = _build_config(tmp_path)

	# date_id % 7 で曜日を模擬
	raw = pd.DataFrame(
		{
			"date_id": [0, 7, 14, 21, 28, 35],  # 全て同じ曜日（0）
			"m/M1": [1, 1, 0, 1, 0, 0],  # NaN, NaN, 観測, NaN, 観測, 観測
			"run_na/M1": [1, 2, 0, 1, 0, 0],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	assert "su3/dow_m_rate/M1" in features.columns

	# burn-in=2なので、3回目以降で計算開始
	# index 2 (date_id=14): 同じ曜日が3回目、NaN率=2/3
	expected_rate = 2.0 / 3.0
	assert features.iloc[2]["su3/dow_m_rate/M1"] == pytest.approx(expected_rate)


def test_su3_holiday_interaction(tmp_path: Path) -> None:
	"""祝日交差テスト。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 1, 0, 1, 0],
			"run_na/M1": [0, 1, 2, 0, 1, 0],
			"holiday_bridge": [0, 1, 0, 1, 1, 0],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	assert "su3/holiday_bridge_m/M1" in features.columns

	# index 1: holiday=1, m=1 → 1
	assert features.loc[1, "su3/holiday_bridge_m/M1"] == 1

	# index 2: holiday=0, m=1 → 0
	assert features.loc[2, "su3/holiday_bridge_m/M1"] == 0

	# index 3: holiday=1, m=0 → 0
	assert features.loc[3, "su3/holiday_bridge_m/M1"] == 0

	# index 4: holiday=1, m=1 → 1
	assert features.loc[4, "su3/holiday_bridge_m/M1"] == 1


def test_su3_fold_reset(tmp_path: Path) -> None:
	"""fold境界でのリセット確認。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 1, 0, 1, 0],  # fold0=[0,1,2], fold1=[3,4,5]
			"run_na/M1": [0, 1, 2, 0, 1, 0],
		}
	)

	fold_indices = np.array([0, 0, 0, 1, 1, 1])

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw, fold_indices=fold_indices)

	# fold境界（index 3）で状態がリセットされるべき
	# index 2→3での遷移はカウントされない
	assert "su3/trans_rate/M" in features.columns

	# index 3は新しいfoldの初回なので遷移なし
	assert features.loc[3, "su3/trans_rate/M"] == 0.0


def test_su3_all_nan_column(tmp_path: Path) -> None:
	"""全NaN列の扱い。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [1, 1, 1, 1, 1, 1],  # 全NaN
			"run_na/M1": [1, 2, 3, 4, 5, 6],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# 遷移なし（すべて0）
	assert features["su3/trans_rate/M"].sum() == 0.0

	# 再出現なし（すべて0）
	if "su3/reappear_gap/M1" in features.columns:
		assert features["su3/reappear_gap/M1"].sum() == 0


def test_su3_alternating_nan(tmp_path: Path) -> None:
	"""交互NaNのテスト。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 0, 1, 0, 1],  # 交互
			"run_na/M1": [0, 1, 0, 1, 0, 1],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# すべての遷移で変化が起きる（初回除く）
	# index 1: 0→1 (遷移1)
	# index 2: 1→0 (遷移1)
	# index 3: 0→1 (遷移1)
	# index 4: 1→0 (遷移1)
	# index 5: 0→1 (遷移1)
	trans_rate = features["su3/trans_rate/M"].values
	assert trans_rate[0] == 0.0  # 初回
	assert trans_rate[1] == pytest.approx(1.0)
	assert trans_rate[2] == pytest.approx(1.0)
	assert trans_rate[3] == pytest.approx(1.0)


def test_su3_island_nan(tmp_path: Path) -> None:
	"""島状NaNのテスト。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5, 6, 7],
			"m/M1": [0, 0, 1, 1, 0, 0, 1, 0],  # 島: [2,3], [6]
			"run_na/M1": [0, 0, 1, 2, 0, 0, 1, 0],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# index 4: 再出現（run_na[3]=2）
	assert features.loc[4, "su3/reappear_gap/M1"] == 2

	# index 7: 再出現（run_na[6]=1）
	assert features.loc[7, "su3/reappear_gap/M1"] == 1


def test_su3_output_shape(tmp_path: Path) -> None:
	"""出力特徴量数の確認。"""
	config = _build_config(tmp_path)

	# 10列のデータを作成
	data: Dict[str, Any] = {"date_id": list(range(20))}
	for i in range(10):
		data[f"m/M{i}"] = np.random.randint(0, 2, 20).tolist()
		data[f"run_na/M{i}"] = np.random.randint(0, 5, 20).tolist()

	raw = pd.DataFrame(data)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# 予想される列数:
	# - trans_rate: 1 (M群のみ)
	# - reappear_gap, pos_since_reappear: 3列 × 2 = 6 (top_k=3)
	# - dow_m_rate, month_m_rate: 3列 × 2 = 6 (top_k=3)
	# - holiday_bridge_m: 0 (holiday列なし)
	# 合計: 1 + 6 + 6 = 13列

	expected_cols = 1 + 6 + 6
	assert len(features.columns) == expected_cols

	# 祝日特徴がないことを確認
	holiday_features = [c for c in features.columns if "holiday" in c]
	assert len(holiday_features) == 0, "No holiday features should be generated without holiday column"


def test_su3_dtype(tmp_path: Path) -> None:
	"""データ型の確認。"""
	config = _build_config(tmp_path)

	raw = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 0, 1, 0, 1],
			"run_na/M1": [0, 1, 0, 1, 0, 1],
			"holiday_bridge": [0, 1, 0, 1, 0, 1],
		}
	)

	transformer = SU3FeatureGenerator(config)
	transformer.fit(raw)
	features = transformer.transform(raw)

	# trans_rate: float32
	assert features["su3/trans_rate/M"].dtype == np.dtype("float32")

	# reappear_gap: int16
	if "su3/reappear_gap/M1" in features.columns:
		assert features["su3/reappear_gap/M1"].dtype == np.dtype("int16")

	# pos_since_reappear: float32
	if "su3/pos_since_reappear/M1" in features.columns:
		assert features["su3/pos_since_reappear/M1"].dtype == np.dtype("float32")

	# holiday_bridge_m: uint8
	if "su3/holiday_bridge_m/M1" in features.columns:
		assert features["su3/holiday_bridge_m/M1"].dtype == np.dtype("uint8")


def test_su3_config_validation(tmp_path: Path) -> None:
	"""設定値のバリデーションテスト。"""
	# reappear_clipが負の値でエラーになることを確認
	mapping = {
		"id_column": "date_id",
		"output_prefix": "su3",
		"include_transitions": True,
		"transition_group_agg": True,
		"include_reappearance": True,
		"reappear_clip": -10,  # 不正な値
		"reappear_top_k": 3,
		"include_imputation_trace": False,
		"imp_delta_winsorize_p": 0.99,
		"imp_delta_top_k": 20,
		"imp_policy_group_level": True,
		"include_temporal_bias": True,
		"temporal_burn_in": 2,
		"temporal_top_k": 3,
		"include_holiday_interaction": True,
		"holiday_top_k": 3,
		"dtype": {"flag": "uint8", "int": "int16", "float": "float32"},
		"reset_each_fold": True,
	}

	with pytest.raises(ValueError, match="reappear_clip must be positive"):
		SU3Config.from_mapping(mapping, base_dir=tmp_path)
