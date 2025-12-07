"""SU11 スタッキングモジュールの単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su11.feature_su11 import (
	SU11Config,
	SU11MetaFeatureBuilder,
	load_su11_config,
)


# ---------------------------------------------------------------------------
# SU11Config Tests
# ---------------------------------------------------------------------------
class TestSU11Config:
	"""SU11Config のテスト。"""

	def test_default_values(self) -> None:
		"""デフォルト値が正しく設定されること。"""
		config = SU11Config()
		assert config.level1_artifacts_dir == "artifacts/SU5"
		assert config.level2_model_type == "ridge"
		assert config.use_extra_features is False
		assert config.extra_feature_names == []
		assert config.ridge_alpha == 0.001
		assert config.lgbm_n_estimators == 50
		assert config.lgbm_max_depth == 3
		assert config.n_splits == 5
		assert config.random_state == 42

	def test_custom_values(self) -> None:
		"""カスタム値が正しく設定されること。"""
		config = SU11Config(
			level1_artifacts_dir="custom/path",
			level2_model_type="lgbm",
			ridge_alpha=0.5,
			lgbm_n_estimators=100,
			n_splits=10,
		)
		assert config.level1_artifacts_dir == "custom/path"
		assert config.level2_model_type == "lgbm"
		assert config.ridge_alpha == 0.5
		assert config.lgbm_n_estimators == 100
		assert config.n_splits == 10

	def test_invalid_model_type(self) -> None:
		"""無効なモデルタイプでエラーになること。"""
		with pytest.raises(ValueError, match="level2_model_type must be one of"):
			SU11Config(level2_model_type="invalid")


class TestLoadSU11Config:
	"""load_su11_config のテスト。"""

	def test_load_from_yaml(self, tmp_path: Path) -> None:
		"""YAML から設定を読み込めること。"""
		config_path = tmp_path / "test_config.yaml"
		config_content = """
su11:
  level1_artifacts_dir: artifacts/SU5
  level2_model_type: ridge
  ridge_alpha: 0.5
  n_splits: 3
  random_state: 123
"""
		config_path.write_text(config_content)

		config = load_su11_config(config_path)
		assert config.level1_artifacts_dir == "artifacts/SU5"
		assert config.level2_model_type == "ridge"
		assert config.ridge_alpha == 0.5
		assert config.n_splits == 3
		assert config.random_state == 123

	def test_load_missing_section(self, tmp_path: Path) -> None:
		"""セクションが存在しない場合はデフォルト設定を返すこと。"""
		config_path = tmp_path / "test_config.yaml"
		config_content = """
other_section:
  key: value
"""
		config_path.write_text(config_content)

		config = load_su11_config(config_path)
		assert config.level2_model_type == "ridge"  # default


# ---------------------------------------------------------------------------
# SU11MetaFeatureBuilder Tests
# ---------------------------------------------------------------------------
class TestSU11MetaFeatureBuilder:
	"""SU11MetaFeatureBuilder のテスト。"""

	@pytest.fixture
	def mock_artifacts(self, tmp_path: Path) -> Path:
		"""テスト用のモックアーティファクトを作成。"""
		artifacts_dir = tmp_path / "artifacts"
		artifacts_dir.mkdir()

		# OOF predictions
		oof_df = pd.DataFrame({
			"row_index": [0, 1, 2, 3, 4],
			"y_true": [0.01, -0.02, 0.015, -0.01, 0.005],
			"y_pred": [0.008, -0.015, 0.012, -0.008, 0.003],
			"fold": [1, 1, 2, 2, 3],
		})
		oof_df.to_csv(artifacts_dir / "oof_predictions.csv", index=False)

		# Submission (test predictions)
		sub_df = pd.DataFrame({
			"prediction": [0.005, -0.003, 0.01],
		})
		sub_df.to_csv(artifacts_dir / "submission.csv", index=False)

		return artifacts_dir

	def test_load_level1_artifacts(self, mock_artifacts: Path) -> None:
		"""Level-1 アーティファクトを読み込めること。"""
		config = SU11Config(level1_artifacts_dir=str(mock_artifacts))
		builder = SU11MetaFeatureBuilder(config)
		builder.load_level1_artifacts()

		assert builder._level1_oof is not None
		assert len(builder._level1_oof) == 5
		assert builder._level1_test_pred is not None
		assert len(builder._level1_test_pred) == 3

	def test_build_level2_train(self, mock_artifacts: Path) -> None:
		"""Level-2 学習用データセットを構築できること。"""
		config = SU11Config(level1_artifacts_dir=str(mock_artifacts))
		builder = SU11MetaFeatureBuilder(config)

		X_L2, y_L2 = builder.build_level2_train()

		assert len(X_L2) == 5
		assert "y_pred_L1" in X_L2.columns
		assert len(y_L2) == 5
		assert y_L2.name == "y_true"

	def test_build_level2_test(self, mock_artifacts: Path) -> None:
		"""Level-2 推論用データセットを構築できること。"""
		config = SU11Config(level1_artifacts_dir=str(mock_artifacts))
		builder = SU11MetaFeatureBuilder(config)
		builder.load_level1_artifacts()

		X_L2_test = builder.build_level2_test()

		assert len(X_L2_test) == 3
		assert "y_pred_L1" in X_L2_test.columns

	def test_build_level2_test_with_array(self, mock_artifacts: Path) -> None:
		"""numpy 配列から Level-2 テストデータを構築できること。"""
		config = SU11Config(level1_artifacts_dir=str(mock_artifacts))
		builder = SU11MetaFeatureBuilder(config)

		test_pred = np.array([0.01, -0.005, 0.002, 0.008])
		X_L2_test = builder.build_level2_test(test_pred)

		assert len(X_L2_test) == 4
		assert "y_pred_L1" in X_L2_test.columns

	def test_get_fold_mapping(self, mock_artifacts: Path) -> None:
		"""fold ごとのインデックスマッピングを取得できること。"""
		config = SU11Config(level1_artifacts_dir=str(mock_artifacts))
		builder = SU11MetaFeatureBuilder(config)

		mapping = builder.get_fold_mapping()

		assert 1 in mapping
		assert 2 in mapping
		assert 3 in mapping
		assert len(mapping[1]) == 2
		assert len(mapping[2]) == 2
		assert len(mapping[3]) == 1

	def test_artifacts_not_found(self, tmp_path: Path) -> None:
		"""アーティファクトが見つからない場合にエラーになること。"""
		config = SU11Config(level1_artifacts_dir=str(tmp_path / "nonexistent"))
		builder = SU11MetaFeatureBuilder(config)

		with pytest.raises(FileNotFoundError, match="OOF predictions not found"):
			builder.load_level1_artifacts()

	def test_with_extra_features(self, mock_artifacts: Path) -> None:
		"""追加特徴を含めて Level-2 データセットを構築できること。"""
		config = SU11Config(
			level1_artifacts_dir=str(mock_artifacts),
			use_extra_features=True,
			extra_feature_names=["extra_col"],
		)
		builder = SU11MetaFeatureBuilder(config)

		# 追加特徴用のデータフレーム
		X_extra = pd.DataFrame({
			"extra_col": [0.1, 0.2, 0.3, 0.4, 0.5],
			"unused_col": [1, 2, 3, 4, 5],
		})

		X_L2, y_L2 = builder.build_level2_train(X_extra=X_extra)

		assert "y_pred_L1" in X_L2.columns
		assert "extra_col" in X_L2.columns
		assert "unused_col" not in X_L2.columns
		assert len(X_L2.columns) == 2


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestSU11Integration:
	"""SU11 の統合テスト。"""

	@pytest.fixture
	def mock_su5_artifacts(self, tmp_path: Path) -> Path:
		"""テスト用の SU5 アーティファクトを作成。"""
		artifacts_dir = tmp_path / "SU5"
		artifacts_dir.mkdir()

		# より多くのデータで OOF predictions を作成
		n_samples = 100
		np.random.seed(42)
		oof_df = pd.DataFrame({
			"row_index": list(range(n_samples)),
			"y_true": np.random.randn(n_samples) * 0.01,
			"y_pred": np.random.randn(n_samples) * 0.01,
			"fold": [i % 5 + 1 for i in range(n_samples)],
		})
		oof_df.to_csv(artifacts_dir / "oof_predictions.csv", index=False)

		return artifacts_dir

	def test_end_to_end_ridge(self, mock_su5_artifacts: Path) -> None:
		"""Ridge モデルでエンドツーエンドの学習ができること。"""
		from sklearn.linear_model import Ridge
		from sklearn.model_selection import TimeSeriesSplit

		config = SU11Config(
			level1_artifacts_dir=str(mock_su5_artifacts),
			level2_model_type="ridge",
			ridge_alpha=1.0,
			n_splits=3,
		)
		builder = SU11MetaFeatureBuilder(config)
		X_L2, y_L2 = builder.build_level2_train()

		# CV で学習
		splitter = TimeSeriesSplit(n_splits=config.n_splits)
		oof_pred = np.full(len(X_L2), np.nan)

		for train_idx, val_idx in splitter.split(X_L2):
			model = Ridge(alpha=config.ridge_alpha)
			model.fit(X_L2.iloc[train_idx], y_L2.iloc[train_idx])
			oof_pred[val_idx] = model.predict(X_L2.iloc[val_idx])

		# OOF 予測が生成されていること
		valid_mask = ~np.isnan(oof_pred)
		assert valid_mask.sum() > 0

		# RMSE が計算できること
		from sklearn.metrics import mean_squared_error
		rmse = np.sqrt(mean_squared_error(y_L2[valid_mask], oof_pred[valid_mask]))
		assert rmse > 0
		assert np.isfinite(rmse)
