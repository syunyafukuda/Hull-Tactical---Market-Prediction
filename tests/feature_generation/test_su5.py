"""SU5 特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_generation.su5.feature_su5 import SU5Config, SU5FeatureGenerator


def _build_config() -> SU5Config:
	"""テスト用のSU5Config を生成する。"""
	mapping = {
		"id_column": "date_id",
		"output_prefix": "su5",
		"top_k_pairs": 3,
		"top_k_pairs_per_group": None,
		"windows": [5, 20],
		"reset_each_fold": True,
		"dtype": {"flag": "uint8", "int": "int16", "float": "float32"},
	}
	return SU5Config.from_mapping(mapping)


def test_su5_config_loading(tmp_path: Path) -> None:
	"""YAML設定の読込確認。"""
	config_path = tmp_path / "test_config.yaml"
	config_content = """
su5:
  id_column: date_id
  output_prefix: su5
  top_k_pairs: 10
  top_k_pairs_per_group: null
  windows: [5, 20]
  reset_each_fold: true
  dtype:
    flag: uint8
    int: int16
    float: float32
"""
	config_path.write_text(config_content)

	from src.feature_generation.su5.feature_su5 import load_su5_config

	config = load_su5_config(config_path)
	assert config.id_column == "date_id"
	assert config.top_k_pairs == 10
	assert config.windows == (5, 20)
	assert config.reset_each_fold is True


def test_su5_all_observed_columns() -> None:
	"""全て m/<col> == 0 の場合、co_miss_now およびローリングが 0 or NaN になること。"""
	config = _build_config()

	# SU1特徴を模したDataFrame（全て観測済み = m == 0）
	su1_features = pd.DataFrame(
		{
			"date_id": range(10),
			"m/M1": [0] * 10,
			"m/M2": [0] * 10,
			"m/M3": [0] * 10,
		}
	).set_index("date_id")

	transformer = SU5FeatureGenerator(config)
	transformer.fit(su1_features)
	features = transformer.transform(su1_features)

	# co_miss_now は全て 0 のはず
	co_miss_now_cols = [c for c in features.columns if c.startswith("co_miss_now/")]
	for col in co_miss_now_cols:
		assert features[col].sum() == 0, f"{col} should be all zeros"

	# rollrate も全て 0 のはず（窓が十分な場合）
	rollrate_cols = [c for c in features.columns if c.startswith("co_miss_rollrate_")]
	for col in rollrate_cols:
		non_nan_values = features[col].dropna()
		if len(non_nan_values) > 0:
			assert non_nan_values.sum() == 0, f"{col} should be all zeros where not NaN"


def test_su5_all_nan_columns() -> None:
	"""全て m/<col> == 1 の場合、共欠損スコアが最大になり top-k に選ばれること。"""
	config = _build_config()

	# 全列が全行でNaN（m == 1）
	su1_features = pd.DataFrame(
		{
			"date_id": range(10),
			"m/M1": [1] * 10,
			"m/M2": [1] * 10,
			"m/M3": [1] * 10,
		}
	).set_index("date_id")

	transformer = SU5FeatureGenerator(config)
	transformer.fit(su1_features)

	# top_k_pairs に全てのペアが含まれるはず（スコアが同じなので順序は不定だが3ペア選ばれる）
	assert transformer.top_pairs_ is not None
	assert len(transformer.top_pairs_) == 3

	features = transformer.transform(su1_features)

	# co_miss_now は全て 1 のはず
	co_miss_now_cols = [c for c in features.columns if c.startswith("co_miss_now/")]
	for col in co_miss_now_cols:
		assert features[col].sum() == 10, f"{col} should be all ones"


def test_su5_single_co_miss_pair() -> None:
	"""特定 2 列だけが同じ NaN パターンを持つ場合、そのペアが top-1 で選ばれること。"""
	config = _build_config()
	config = SU5Config(
		id_column="date_id",
		output_prefix="su5",
		top_k_pairs=1,  # top-1 のみ選択
		top_k_pairs_per_group=None,
		windows=(5,),
		reset_each_fold=True,
		dtype_flag=np.dtype("uint8"),
		dtype_int=np.dtype("int16"),
		dtype_float=np.dtype("float32"),
	)

	# M1 と M2 のみが同じパターンで欠損
	su1_features = pd.DataFrame(
		{
			"date_id": range(10),
			"m/M1": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
			"m/M2": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
			"m/M3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		}
	).set_index("date_id")

	transformer = SU5FeatureGenerator(config)
	transformer.fit(su1_features)

	# top-1 ペアは M1-M2 のはず
	assert transformer.top_pairs_ is not None
	assert len(transformer.top_pairs_) == 1
	assert transformer.top_pairs_ == [("M1", "M2")] or transformer.top_pairs_ == [("M2", "M1")]

	features = transformer.transform(su1_features)

	# co_miss_now/M1__M2 または co_miss_now/M2__M1 が存在するはず
	co_miss_now_cols = [c for c in features.columns if c.startswith("co_miss_now/")]
	assert len(co_miss_now_cols) == 1

	# パターンが一致しているので、該当行で 1 になっているはず
	col = co_miss_now_cols[0]
	expected_pattern = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
	np.testing.assert_array_equal(np.asarray(features[col].values), expected_pattern)


def test_su5_fold_reset() -> None:
	"""reset_each_fold=True 時に fold 境界を跨いでローリングが続かないこと。"""
	config = _build_config()

	su1_features = pd.DataFrame(
		{
			"date_id": range(10),
			"m/M1": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
			"m/M2": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
		}
	).set_index("date_id")

	# fold_indices: 最初の5行がfold 0、次の5行がfold 1
	fold_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

	transformer = SU5FeatureGenerator(config)
	transformer.fit(su1_features)
	features = transformer.transform(su1_features, fold_indices=fold_indices)

	# rollrate_5 を確認
	rollrate_cols = [c for c in features.columns if "co_miss_rollrate_5" in c]
	assert len(rollrate_cols) > 0

	# fold 1 の最初の行（index=5）では、fold 0 のデータを参照しないはず
	# 窓が5なので、index=5では NaN になるはず
	for col in rollrate_cols:
		# index=5 は新しいfoldの開始なので、窓が足りずNaNになるはず
		assert pd.isna(features.loc[5, col]), f"{col} at index 5 should be NaN due to fold reset"


def test_su5_output_shape() -> None:
	"""top_k_pairs・windows 設定に応じた出力列数になること。"""
	config = SU5Config(
		id_column="date_id",
		output_prefix="su5",
		top_k_pairs=2,
		top_k_pairs_per_group=None,
		windows=(5, 20),
		reset_each_fold=True,
		dtype_flag=np.dtype("uint8"),
		dtype_int=np.dtype("int16"),
		dtype_float=np.dtype("float32"),
	)

	su1_features = pd.DataFrame(
		{
			"date_id": range(10),
			"m/M1": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
			"m/M2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
			"m/M3": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		}
	).set_index("date_id")

	transformer = SU5FeatureGenerator(config)
	transformer.fit(su1_features)
	features = transformer.transform(su1_features)

	# 期待される列数:
	# - co_miss_now: top_k_pairs = 2
	# - co_miss_rollrate: top_k_pairs * len(windows) = 2 * 2 = 4
	# - co_miss_deg: len(m_columns) = 3
	# 合計: 2 + 4 + 3 = 9
	expected_col_count = 2 + 4 + 3
	assert len(features.columns) == expected_col_count


def test_su5_dtype() -> None:
	"""uint8 / int16 / float32 の dtype が正しく適用されていること。"""
	config = _build_config()

	su1_features = pd.DataFrame(
		{
			"date_id": range(10),
			"m/M1": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
			"m/M2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
			"m/M3": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
		}
	).set_index("date_id")

	transformer = SU5FeatureGenerator(config)
	transformer.fit(su1_features)
	features = transformer.transform(su1_features)

	# co_miss_now は uint8
	co_miss_now_cols = [c for c in features.columns if c.startswith("co_miss_now/")]
	for col in co_miss_now_cols:
		assert features[col].dtype == np.uint8, f"{col} should be uint8"

	# co_miss_deg は int16
	co_miss_deg_cols = [c for c in features.columns if c.startswith("co_miss_deg/")]
	for col in co_miss_deg_cols:
		assert features[col].dtype == np.int16, f"{col} should be int16"

	# co_miss_rollrate は float32
	rollrate_cols = [c for c in features.columns if c.startswith("co_miss_rollrate_")]
	for col in rollrate_cols:
		assert features[col].dtype == np.float32, f"{col} should be float32"
