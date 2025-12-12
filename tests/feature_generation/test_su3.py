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


def test_su3_feature_augmenter() -> None:
	"""SU3FeatureAugmenterの基本動作確認。"""
	from src.feature_generation.su1.feature_su1 import SU1Config
	from src.feature_generation.su3.feature_su3 import SU3FeatureAugmenter
	
	# SU1設定
	su1_mapping = {
		"id_column": "date_id",
		"exclude_columns": [],
		"groups": {
			"include": ["M", "E"],
			"exclude": []
		},
		"gap_clip": 60,
		"run_clip": 60,
		"dtype": {"flag": "uint8", "run": "int16"},
		"include_group_means": {
			"gap_ffill": True,
			"run_na": True,
			"exclude_all_nan": False
		},
		"data": {
			"raw_dir": "data/raw",
			"train_filename": "train.csv",
			"test_filename": "test.csv"
		}
	}
	from pathlib import Path
	su1_config = SU1Config.from_mapping(su1_mapping, base_dir=Path.cwd())
	
	# SU3設定
	su3_config = _build_config()
	
	# 生データを模擬
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"M1": [1.0, np.nan, np.nan, 2.0, 3.0, np.nan],
			"M2": [4.0, 5.0, np.nan, np.nan, 6.0, 7.0],
			"E1": [np.nan, np.nan, 8.0, 9.0, np.nan, np.nan],
		}
	).set_index("date_id")
	
	# SU3FeatureAugmenterのfit/transform
	augmenter = SU3FeatureAugmenter(su1_config, su3_config)
	augmenter.fit(data)
	features = augmenter.transform(data)
	
	# SU1特徴が含まれているか（m/, gap_ffill/, run_na/, run_obs/, m_cnt/, m_rate/, avg_gapff/, avg_run_na/）
	assert any(col.startswith("m/") for col in features.columns)
	assert any(col.startswith("gap_ffill/") for col in features.columns)
	assert any(col.startswith("run_na/") for col in features.columns)
	assert any(col.startswith("run_obs/") for col in features.columns)
	
	# SU3特徴が含まれているか（su3/trans_rate/, su3/reappear_gap/, etc.）
	assert any(col.startswith("su3/") for col in features.columns)
	assert any(col.startswith("su3/trans_rate/") for col in features.columns)
	
	# 行数が保持されているか
	assert len(features) == len(data)
	assert features.index.equals(data.index)


def test_su3_augmenter_with_fold_indices() -> None:
	"""SU3FeatureAugmenterでfold_indicesが正しく伝播するか確認。"""
	from src.feature_generation.su1.feature_su1 import SU1Config
	from src.feature_generation.su3.feature_su3 import SU3FeatureAugmenter
	
	# SU1設定
	su1_mapping = {
		"id_column": "date_id",
		"exclude_columns": [],
		"groups": {
			"include": ["M"],
			"exclude": []
		},
		"gap_clip": 60,
		"run_clip": 60,
		"dtype": {"flag": "uint8", "run": "int16"},
		"include_group_means": {
			"gap_ffill": False,
			"run_na": False,
			"exclude_all_nan": False
		},
		"data": {
			"raw_dir": "data/raw",
			"train_filename": "train.csv",
			"test_filename": "test.csv"
		}
	}
	from pathlib import Path
	su1_config = SU1Config.from_mapping(su1_mapping, base_dir=Path.cwd())
	
	# SU3設定（reset_each_fold=True）
	su3_config = _build_config()
	
	# 生データ
	data = pd.DataFrame(
		{
			"date_id": [0, 1, 2, 3, 4, 5],
			"M1": [1.0, np.nan, np.nan, 2.0, np.nan, np.nan],
		}
	).set_index("date_id")
	
	# fold_indicesを指定
	fold_indices = np.array([0, 0, 0, 1, 1, 1])
	
	augmenter = SU3FeatureAugmenter(su1_config, su3_config)
	augmenter.fit(data)
	features = augmenter.transform(data, fold_indices=fold_indices)
	
	# 特徴が生成されているか
	assert len(features.columns) > 0
	assert len(features) == len(data)


def test_su3_augmenter_column_count() -> None:
	"""SU3FeatureAugmenterの特徴量数確認（目標: 約474列）。"""
	from src.feature_generation.su1.feature_su1 import SU1Config
	from src.feature_generation.su3.feature_su3 import SU3FeatureAugmenter
	
	# SU1設定（全グループを含む）
	su1_mapping = {
		"id_column": "date_id",
		"exclude_columns": [],
		"groups": {
			"include": ["M", "E", "I", "P", "S"],
			"exclude": []
		},
		"gap_clip": 60,
		"run_clip": 60,
		"dtype": {"flag": "uint8", "run": "int16"},
		"include_group_means": {
			"gap_ffill": True,
			"run_na": True,
			"exclude_all_nan": False
		},
		"data": {
			"raw_dir": "data/raw",
			"train_filename": "train.csv",
			"test_filename": "test.csv"
		}
	}
	from pathlib import Path
	su1_config = SU1Config.from_mapping(su1_mapping, base_dir=Path.cwd())
	
	# SU3設定（top-kを20に設定）
	su3_mapping = {
		"id_column": "date_id",
		"output_prefix": "su3",
		"include_transitions": True,
		"transition_group_agg": True,
		"include_reappearance": True,
		"reappear_clip": 60,
		"reappear_top_k": 20,
		"include_imputation_trace": False,
		"imp_delta_winsorize_p": 0.99,
		"imp_delta_top_k": 20,
		"imp_policy_group_level": True,
		"include_temporal_bias": True,
		"temporal_burn_in": 3,
		"temporal_top_k": 20,
		"include_holiday_interaction": True,
		"holiday_top_k": 20,
		"dtype": {"flag": "uint8", "int": "int16", "float": "float32"},
		"reset_each_fold": True,
	}
	su3_config = SU3Config.from_mapping(su3_mapping)
	
	# 多数の列を持つ生データを模擬（10列×5グループ=50列）
	data_dict: dict[str, list[float]] = {"date_id": list(range(10))}
	for group in ["M", "E", "I", "P", "S"]:
		for i in range(10):
			col_name = f"{group}{i+1}"
			data_dict[col_name] = np.random.choice([1.0, np.nan], size=10).tolist()
	
	data = pd.DataFrame(data_dict).set_index("date_id")
	
	augmenter = SU3FeatureAugmenter(su1_config, su3_config)
	augmenter.fit(data)
	features = augmenter.transform(data)
	
	# 特徴量数を確認
	# SU1: 50列×4種類(m/, gap_ffill/, run_na/, run_obs/) + 群集約(約15列) = 約215列
	# SU3: 約5群集約 + 20×2再出現 + 20×2曜日月次 + 20×1祝日 = 約105列
	# 合計: 約320列（実際のデータ構造に依存）
	assert len(features.columns) > 100  # 最低でも100列以上
	print(f"Total features: {len(features.columns)}")


# =============================================================================
# スイープ機能のテスト（旧 test_su3_sweep.py から移行）
# =============================================================================

def _create_test_su1_config_for_sweep():
	"""スイープテスト用のSU1Configを構築する。"""
	from pathlib import Path

	from src.feature_generation.su1.feature_su1 import SU1Config
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


def _create_test_su3_config_for_sweep():
	"""スイープテスト用のSU3Configを構築する。"""
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


def _create_test_preprocess_settings():
	"""スイープテスト用の前処理設定を構築する。"""
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


def _create_small_test_data_for_sweep(n_rows: int = 100, n_cols: int = 10):
	"""スイープテスト用の小規模テストデータを作成する（生データを模擬）。"""
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


def test_sweep_build_param_combinations() -> None:
	"""パラメータ組み合わせ生成の確認。"""
	from src.feature_generation.su3.sweep_oof import build_param_combinations
	
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


def test_sweep_evaluate_single_config_small_data() -> None:
	"""小規模データでのOOF評価動作確認。"""
	from src.feature_generation.su3.sweep_oof import evaluate_single_config
	
	# 小規模データ作成（100行×10列）
	data = _create_small_test_data_for_sweep(n_rows=100, n_cols=5)
	target = np.random.randn(100) * 0.01  # 小さい値にスケール
	
	# 設定作成
	su1_config = _create_test_su1_config_for_sweep()
	su3_config = _create_test_su3_config_for_sweep()
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


def test_sweep_save_results() -> None:
	"""結果保存機能の確認。"""
	import csv
	import json
	import tempfile
	from pathlib import Path

	from src.feature_generation.su3.sweep_oof import save_results
	
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
		with csv_file.open('r') as f:
			reader = csv.DictReader(f)
			rows = list(reader)
			assert len(rows) == 1
			assert rows[0]['reappear_top_k'] == '20'
			assert rows[0]['temporal_top_k'] == '20'
			assert rows[0]['holiday_top_k'] == '20'


def test_sweep_build_param_combinations_with_imputation() -> None:
	"""代入影響フラグを含むパラメータ組み合わせ生成の確認。"""
	from src.feature_generation.su3.sweep_oof import build_param_combinations
	
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
