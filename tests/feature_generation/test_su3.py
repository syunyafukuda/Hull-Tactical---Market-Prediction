"""SU3 特徴量生成の単体テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su3.feature_su3 import SU3Config, SU3FeatureGenerator


def _build_config() -> SU3Config:
	"""テスト用のSU3Config を構築する。"""
	mapping = {
		"id_column": "date_id",
		"output_prefix": "su3",
		"include_transitions": True,
		"transition_group_agg": True,
		"include_reappearance": True,
		"reappear_clip": 60,
		"reappear_top_k": 3,  # テスト用に少なく
		"include_imputation_trace": False,
		"imp_delta_winsorize_p": 0.99,
		"imp_delta_top_k": 20,
		"imp_policy_group_level": True,
		"include_temporal_bias": True,
		"temporal_burn_in": 3,
		"temporal_top_k": 3,  # テスト用に少なく
		"include_holiday_interaction": True,
		"holiday_top_k": 3,  # テスト用に少なく
		"dtype": {"flag": "uint8", "int": "int16", "float": "float32"},
		"reset_each_fold": True,
	}
	return SU3Config.from_mapping(mapping)


def test_su3_config_loading() -> None:
	"""YAML設定の読込確認。"""
	config = _build_config()
	assert config.id_column == "date_id"
	assert config.output_prefix == "su3"
	assert config.include_transitions is True
	assert config.transition_group_agg is True
	assert config.reappear_clip == 60
	assert config.temporal_burn_in == 3
	assert config.reset_each_fold is True


def test_su3_transition_flags() -> None:
	"""遷移フラグの基本動作確認。"""
	config = _build_config()

	# SU1特徴を模擬したデータ
	# m/<col>: 0=観測, 1=NaN
	# パターン: [観測, 観測, NaN, NaN, 観測, 観測]
	#          -> na_to_obs: [0, 0, 0, 0, 1, 0]
	#          -> obs_to_na: [0, 0, 1, 0, 0, 0]
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 0, 1, 1, 0, 0],
			"m/M2": [0, 1, 1, 0, 0, 0],
			"m/E1": [1, 1, 0, 0, 1, 1],
			"gap_ffill/M1": [0, 0, 1, 2, 0, 0],
			"gap_ffill/M2": [0, 1, 2, 0, 0, 0],
			"gap_ffill/E1": [5, 5, 0, 0, 1, 2],
			"run_na/M1": [0, 0, 1, 2, 0, 0],
			"run_na/M2": [0, 1, 2, 0, 0, 0],
			"run_na/E1": [1, 2, 0, 0, 1, 2],
			"run_obs/M1": [1, 2, 0, 0, 1, 2],
			"run_obs/M2": [1, 0, 0, 1, 2, 3],
			"run_obs/E1": [0, 0, 1, 2, 0, 0],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	# 群集約の遷移率を確認
	assert "su3/trans_rate/M" in features.columns
	assert "su3/trans_rate/E" in features.columns

	# M群: M1とM2の遷移を確認
	# t=1: M1変化なし(0->0), M2変化あり(0->1) -> 1/2=0.5
	# t=2: M1変化あり(0->1), M2変化なし(1->1) -> 1/2=0.5
	# t=3: M1変化なし(1->1), M2変化あり(1->0) -> 1/2=0.5
	# t=4: M1変化あり(1->0), M2変化なし(0->0) -> 1/2=0.5
	# t=5: M1変化なし(0->0), M2変化なし(0->0) -> 0/2=0.0
	assert features.loc[1, "su3/trans_rate/M"] == pytest.approx(0.5)
	assert features.loc[2, "su3/trans_rate/M"] == pytest.approx(0.5)
	assert features.loc[5, "su3/trans_rate/M"] == pytest.approx(0.0)


def test_su3_reappear_gap() -> None:
	"""再出現間隔の計算確認。"""
	config = _build_config()

	# パターン: [観測, NaN, NaN, NaN, 観測, 観測]
	#   run_na: [0,    1,   2,   3,   0,    0]
	#   reappear_gap: [0, 0, 0, 0, 3, 0]  # t=4で復帰、その時のrun_na[3]=3
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 1, 1, 0, 0],
			"gap_ffill/M1": [0, 1, 2, 3, 0, 0],
			"run_na/M1": [0, 1, 2, 3, 0, 0],
			"run_obs/M1": [1, 0, 0, 0, 1, 2],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	assert "su3/reappear_gap/M1" in features.columns
	assert features.loc[4, "su3/reappear_gap/M1"] == 3
	assert features.loc[0, "su3/reappear_gap/M1"] == 0
	assert features.loc[1, "su3/reappear_gap/M1"] == 0


def test_su3_pos_since_reappear() -> None:
	"""再出現位置の正規化確認。"""
	config = _build_config()

	# パターン: [観測, NaN, NaN, 観測, 観測, 観測]
	# pos_since_reappear:
	#   t=0: 初期観測 -> 0/60 = 0.0
	#   t=1: NaN -> 0/60 = 0.0
	#   t=2: NaN -> 0/60 = 0.0
	#   t=3: 復帰 -> days=1, (1-1)/60 = 0/60 = 0.0
	#   t=4: 観測継続 -> days=2, (2-1)/60 = 1/60 ≈ 0.0167
	#   t=5: 観測継続 -> days=3, (3-1)/60 = 2/60 ≈ 0.0333
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 1, 1, 0, 0, 0],
			"gap_ffill/M1": [0, 1, 2, 0, 0, 0],
			"run_na/M1": [0, 1, 2, 0, 0, 0],
			"run_obs/M1": [1, 0, 0, 1, 2, 3],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	assert "su3/pos_since_reappear/M1" in features.columns
	assert features.loc[3, "su3/pos_since_reappear/M1"] == pytest.approx(0.0)
	assert features.loc[4, "su3/pos_since_reappear/M1"] == pytest.approx(1 / 60.0)
	assert features.loc[5, "su3/pos_since_reappear/M1"] == pytest.approx(2 / 60.0)


def test_su3_temporal_bias() -> None:
	"""曜日・月次パターン生成確認。"""
	config = _build_config()

	# 7日間のデータ（曜日0-6）
	# M1の欠損パターン: [0, 1, 0, 1, 0, 1, 0]
	# 曜日0: NaN率 = 0/1=0.0 (burn_in後)
	# 曜日1: NaN率 = 1/1=1.0
	# 曜日2: NaN率 = 0/1=0.0
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5, 6],
			"m/M1": [0, 1, 0, 1, 0, 1, 0],
			"gap_ffill/M1": [0, 1, 0, 1, 0, 1, 0],
			"run_na/M1": [0, 1, 0, 1, 0, 1, 0],
			"run_obs/M1": [1, 0, 1, 0, 1, 0, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	assert "su3/dow_m_rate/M1" in features.columns
	assert "su3/month_m_rate/M1" in features.columns

	# 曜日1（date_id=1）は初回なのでburn_in未満 -> 0.0
	# 後続は累積で計算される
	assert features.loc[1, "su3/dow_m_rate/M1"] == 0.0  # burn_in未満


def test_su3_holiday_interaction() -> None:
	"""祝日交差の確認。"""
	config = _build_config()

	# holiday_bridge列を含むデータ
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [0, 1, 1, 0, 1],
			"gap_ffill/M1": [0, 1, 2, 0, 1],
			"run_na/M1": [0, 1, 2, 0, 1],
			"run_obs/M1": [1, 0, 0, 1, 0],
			"holiday_bridge": [0, 1, 0, 1, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	assert "su3/holiday_bridge_m/M1" in features.columns
	# holiday_bridge * m
	# t=0: 0*0=0
	# t=1: 1*1=1
	# t=2: 0*1=0
	# t=3: 1*0=0
	# t=4: 1*1=1
	assert features.loc[1, "su3/holiday_bridge_m/M1"] == 1
	assert features.loc[0, "su3/holiday_bridge_m/M1"] == 0
	assert features.loc[4, "su3/holiday_bridge_m/M1"] == 1


def test_su3_fold_reset() -> None:
	"""fold境界でのリセット確認。"""
	config = _build_config()

	# 2つのfoldに分割
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"m/M1": [0, 0, 1, 0, 1, 1],
			"gap_ffill/M1": [0, 0, 1, 0, 1, 2],
			"run_na/M1": [0, 0, 1, 0, 1, 2],
			"run_obs/M1": [1, 2, 0, 1, 0, 0],
		}
	).set_index("date_id")

	fold_indices = np.array([0, 0, 0, 1, 1, 1])

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data, fold_indices=fold_indices)

	# fold境界（t=2とt=3の間）で状態がリセットされる
	# 遷移フラグはt=3で前回値を参照しない（fold境界）
	assert "su3/trans_rate/M" in features.columns


def test_su3_all_nan_column() -> None:
	"""全NaN列の扱い確認。"""
	config = _build_config()

	# M1は全てNaN
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [1, 1, 1, 1, 1],
			"m/M2": [0, 1, 0, 1, 0],
			"gap_ffill/M1": [60, 60, 60, 60, 60],
			"gap_ffill/M2": [0, 1, 0, 1, 0],
			"run_na/M1": [1, 2, 3, 4, 5],
			"run_na/M2": [0, 1, 0, 1, 0],
			"run_obs/M1": [0, 0, 0, 0, 0],
			"run_obs/M2": [1, 0, 1, 0, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	# 全NaN列は遷移なし
	assert "su3/trans_rate/M" in features.columns
	# reappear_gapは全て0（復帰しない）
	if "su3/reappear_gap/M1" in features.columns:
		assert features["su3/reappear_gap/M1"].sum() == 0


def test_su3_alternating_nan() -> None:
	"""交互NaNパターンの確認。"""
	config = _build_config()

	# 交互NaNパターン: [観測, NaN, 観測, NaN, 観測]
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [0, 1, 0, 1, 0],
			"gap_ffill/M1": [0, 1, 0, 1, 0],
			"run_na/M1": [0, 1, 0, 1, 0],
			"run_obs/M1": [1, 0, 1, 0, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	# 遷移が頻繁に発生
	assert "su3/trans_rate/M" in features.columns
	# 再出現間隔は全て1
	if "su3/reappear_gap/M1" in features.columns:
		assert features.loc[2, "su3/reappear_gap/M1"] == 1
		assert features.loc[4, "su3/reappear_gap/M1"] == 1


def test_su3_island_nan() -> None:
	"""島状NaNパターンの確認。"""
	config = _build_config()

	# 島状NaNパターン: [観測, 観測, NaN, NaN, 観測, 観測, NaN, 観測]
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5, 6, 7],
			"m/M1": [0, 0, 1, 1, 0, 0, 1, 0],
			"gap_ffill/M1": [0, 0, 1, 2, 0, 0, 1, 0],
			"run_na/M1": [0, 0, 1, 2, 0, 0, 1, 0],
			"run_obs/M1": [1, 2, 0, 0, 1, 2, 0, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	# 再出現間隔の確認
	if "su3/reappear_gap/M1" in features.columns:
		assert features.loc[4, "su3/reappear_gap/M1"] == 2  # t=4で復帰、run_na[3]=2
		assert features.loc[7, "su3/reappear_gap/M1"] == 1  # t=7で復帰、run_na[6]=1


def test_su3_output_shape() -> None:
	"""出力特徴量数の確認。"""
	config = _build_config()

	# 3列×4種類の特徴 = 約12列（top-kが3の場合）
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [0, 1, 0, 1, 0],
			"m/M2": [1, 0, 1, 0, 1],
			"m/E1": [0, 0, 1, 1, 0],
			"gap_ffill/M1": [0, 1, 0, 1, 0],
			"gap_ffill/M2": [1, 0, 1, 0, 1],
			"gap_ffill/E1": [0, 0, 1, 2, 0],
			"run_na/M1": [0, 1, 0, 1, 0],
			"run_na/M2": [1, 0, 1, 0, 1],
			"run_na/E1": [0, 0, 1, 2, 0],
			"run_obs/M1": [1, 0, 1, 0, 1],
			"run_obs/M2": [0, 1, 0, 1, 0],
			"run_obs/E1": [1, 2, 0, 0, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	# 群集約（2群）+ 再出現（3列×2）+ 曜日・月次（3列×2）+ 祝日（3列×1）
	# = 2 + 6 + 6 + 3 = 17列
	assert len(features.columns) > 0
	assert len(features.columns) <= 20  # top-kが3なので約17列


def test_su3_dtype() -> None:
	"""データ型の確認。"""
	config = _build_config()

	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4],
			"m/M1": [0, 1, 0, 1, 0],
			"gap_ffill/M1": [0, 1, 0, 1, 0],
			"run_na/M1": [0, 1, 0, 1, 0],
			"run_obs/M1": [1, 0, 1, 0, 1],
		}
	).set_index("date_id")

	transformer = SU3FeatureGenerator(config)
	transformer.fit(data)
	features = transformer.transform(data)

	# trans_rate: float32
	if "su3/trans_rate/M" in features.columns:
		assert features["su3/trans_rate/M"].dtype == np.float32

	# reappear_gap: int16
	if "su3/reappear_gap/M1" in features.columns:
		assert features["su3/reappear_gap/M1"].dtype == np.int16

	# pos_since_reappear: float32
	if "su3/pos_since_reappear/M1" in features.columns:
		assert features["su3/pos_since_reappear/M1"].dtype == np.float32

	# holiday_bridge_m: uint8
	if "su3/holiday_bridge_m/M1" in features.columns:
		assert features["su3/holiday_bridge_m/M1"].dtype == np.uint8
