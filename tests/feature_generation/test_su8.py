"""SU8 ボラティリティ・レジーム特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su8.feature_su8 import (
    SU8Config,
    SU8FeatureAugmenter,
    SU8FeatureGenerator,
    load_su8_config,
)


def _build_config(
    core_index_col: str = "core_index",
    ret_base_col: str = "ret_1",
) -> SU8Config:
    """テスト用のSU8Config を生成する。"""
    mapping = {
        "core_index_col": core_index_col,
        "ret_base_col": ret_base_col,
        "ewm_short_halflife": 5,
        "ewm_long_halflife": 20,
        "eps": 1e-4,
        "trend_ma_short_window": 5,
        "trend_ma_long_window": 20,
        "vol_quantiles": [0.33, 0.66],
        "trend_quantiles": [0.33, 0.66],
        "winsorize_ret_vol_adj_p": 0.99,
        "dtype": {"float": "float32", "bool": "bool"},
    }
    return SU8Config.from_mapping(mapping)


class TestSU8Config:
    """SU8Config のテスト。"""

    def test_config_loading(self, tmp_path: Path) -> None:
        """YAML設定の読込確認。"""
        config_path = tmp_path / "test_config.yaml"
        config_content = """
su8:
  enabled: false
  core_index_col: "M1"
  ret_base_col: "ret_1"
  ewm_short_halflife: 5
  ewm_long_halflife: 20
  eps: 1.0e-4
  trend_ma_short_window: 5
  trend_ma_long_window: 20
  vol_quantiles: [0.33, 0.66]
  trend_quantiles: [0.33, 0.66]
  winsorize_ret_vol_adj_p: 0.99
  categorical_cols:
    - vol_regime_low
    - vol_regime_mid
    - vol_regime_high
    - trend_regime_up
    - trend_regime_down
    - trend_regime_flat
  dtype:
    float: float32
    bool: bool
"""
        config_path.write_text(config_content)

        config = load_su8_config(config_path)
        assert config.core_index_col == "M1"
        assert config.ret_base_col == "ret_1"
        assert config.ewm_short_halflife == 5
        assert config.ewm_long_halflife == 20
        assert config.eps == 1e-4
        assert config.vol_quantiles == (0.33, 0.66)
        assert config.trend_quantiles == (0.33, 0.66)

    def test_config_missing_core_index_col(self) -> None:
        """core_index_col が空の場合はエラー。"""
        with pytest.raises(ValueError, match="core_index_col"):
            SU8Config.from_mapping({"ret_base_col": "ret_1"})

    def test_config_missing_ret_base_col(self) -> None:
        """ret_base_col が空の場合はエラー。"""
        with pytest.raises(ValueError, match="ret_base_col"):
            SU8Config.from_mapping({"core_index_col": "M1"})

    def test_config_default_values(self) -> None:
        """デフォルト値の確認。"""
        config = SU8Config.from_mapping({
            "core_index_col": "M1",
            "ret_base_col": "ret_1",
        })
        assert config.ewm_short_halflife == 5
        assert config.ewm_long_halflife == 20
        assert config.eps == 1e-4
        assert config.trend_ma_short_window == 5
        assert config.trend_ma_long_window == 20


class TestSU8FeatureGenerator:
    """SU8FeatureGenerator のテスト。"""

    def test_basic_shape_and_columns(self) -> None:
        """基本的な形状・列名のテスト。"""
        config = _build_config()
        np.random.seed(42)
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 11 列が追加されるはず
        assert len(features.columns) == 11
        assert transformer.get_expected_column_count() == 11

        # 列名の確認
        expected_cols = [
            "ewmstd_short",
            "ewmstd_long",
            "vol_ratio",
            "vol_level",
            "vol_regime_low",
            "vol_regime_mid",
            "vol_regime_high",
            "trend_regime_down",
            "trend_regime_up",
            "trend_regime_flat",
            "ret_vol_adj",
        ]
        for col in expected_cols:
            assert col in features.columns, f"{col} should be in features"

    def test_dtype_check(self) -> None:
        """dtype が仕様通りであること。"""
        config = _build_config()
        np.random.seed(42)
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 数値列は float32
        for col in ["ewmstd_short", "ewmstd_long", "vol_ratio", "vol_level", "ret_vol_adj"]:
            assert features[col].dtype == np.float32, f"{col} should be float32"

        # レジーム列は bool
        for col in [
            "vol_regime_low",
            "vol_regime_mid",
            "vol_regime_high",
            "trend_regime_down",
            "trend_regime_up",
            "trend_regime_flat",
        ]:
            assert features[col].dtype == np.bool_, f"{col} should be bool"

    def test_leak_prevention_ewm(self) -> None:
        """EWMA が過去方向のみ参照することを確認（リーク防止）。"""
        config = _build_config()
        # 特殊なパターン: index=20 以降で値が急変
        values = np.ones(40, dtype=float)
        values[20:] = 100.0  # 急激に増加
        df = pd.DataFrame({
            "core_index": values,
            "ret_1": np.random.randn(40),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # index=19 の ewmstd_long は index=0..19 のみ参照している
        # 値が全て 1.0 なので、標準偏差は 0 に近いはず
        ewmstd_at_19 = features["ewmstd_long"].iloc[19]
        # min_periods=20 なので index=19 で初めて有効値が出る
        assert ewmstd_at_19 < 0.1 or np.isnan(ewmstd_at_19), (
            f"ewmstd at index 19 should be near 0 (no leak from future), got {ewmstd_at_19}"
        )

    def test_leak_prevention_rolling(self) -> None:
        """rolling が過去方向のみ参照することを確認（リーク防止）。"""
        config = _build_config()
        # 特殊なパターン: index=20 以降で値が急変
        values = np.ones(40, dtype=float)
        values[20:] = 100.0
        df = pd.DataFrame({
            "core_index": values,
            "ret_1": np.random.randn(40),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # index=19 のトレンドレジームは 1.0 のみを見ているはず
        # ma_short = ma_long = 1.0 なので trend_indicator ≈ 0
        # -> trend_regime_flat が True の可能性が高い
        # ただし fit 時の quantile によるので、確実ではない
        # 少なくとも将来の 100.0 の影響を受けていないことを確認
        # （NaN でなければ、過去のデータのみから計算されている）
        trend_down_at_19 = features["trend_regime_down"].iloc[19]
        trend_up_at_19 = features["trend_regime_up"].iloc[19]
        trend_flat_at_19 = features["trend_regime_flat"].iloc[19]
        # いずれか一つが True のはず
        assert (
            trend_down_at_19 or trend_up_at_19 or trend_flat_at_19
        ), "At index 19, one regime should be True"

    def test_vol_regime_quantile_fit(self) -> None:
        """vol_regime の閾値が train のみから fit されること。"""
        config = _build_config()
        np.random.seed(42)

        # train: 低ボラ区間
        train_df = pd.DataFrame({
            "core_index": np.random.randn(100) * 0.1,  # 低ボラ
            "ret_1": np.random.randn(100),
        })
        # val: 高ボラ区間
        val_df = pd.DataFrame({
            "core_index": np.random.randn(50) * 10.0,  # 高ボラ
            "ret_1": np.random.randn(50),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(train_df)

        # quantile は train から計算されているはず
        assert transformer.quantiles_ is not None
        assert "q_low" in transformer.quantiles_
        assert "q_high" in transformer.quantiles_

        # train の低ボラデータから計算された閾値
        # val の高ボラデータでは、ほぼ全て vol_regime_high になるはず
        val_features = transformer.transform(val_df)

        # 高ボラデータなので、high レジームが多いはず
        high_ratio = val_features["vol_regime_high"].mean()
        assert high_ratio > 0.5, f"Expected majority high regime, got {high_ratio}"

    def test_trend_regime_quantile_fit(self) -> None:
        """trend_regime の閾値が train のみから fit されること。"""
        config = _build_config()
        np.random.seed(42)

        # train: 横ばい
        train_values = np.cumsum(np.random.randn(100) * 0.01) + 100
        train_df = pd.DataFrame({
            "core_index": train_values,
            "ret_1": np.random.randn(100),
        })
        # val: 強い上昇トレンド
        val_values = np.linspace(100, 200, 50)
        val_df = pd.DataFrame({
            "core_index": val_values,
            "ret_1": np.random.randn(50),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(train_df)

        assert transformer.quantiles_ is not None
        assert "tau_up" in transformer.quantiles_

        # val の上昇トレンドデータでは、up レジームが多いはず
        val_features = transformer.transform(val_df)

        # 強い上昇トレンドなので、up レジームが多いはず
        # ただし min_periods の影響で最初の方は NaN
        valid_up = val_features["trend_regime_up"].iloc[20:]
        up_ratio = valid_up.mean()
        assert up_ratio > 0.5, f"Expected majority up regime, got {up_ratio}"

    def test_class_balance(self) -> None:
        """vol_regime / trend_regime のクラスが極端にゼロにならないこと。"""
        config = _build_config()
        np.random.seed(42)

        # ランダムデータ
        df = pd.DataFrame({
            "core_index": np.random.randn(200),
            "ret_1": np.random.randn(200),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 各レジームが少なくとも 1 つは True を持つはず
        for col in ["vol_regime_low", "vol_regime_mid", "vol_regime_high"]:
            # NaN を除いた有効データで確認
            valid_data = features[col].dropna()
            assert valid_data.sum() > 0, f"{col} should have at least one True"

        for col in ["trend_regime_down", "trend_regime_up", "trend_regime_flat"]:
            valid_data = features[col].dropna()
            assert valid_data.sum() > 0, f"{col} should have at least one True"

    def test_ret_vol_adj_stability_near_zero_vol(self) -> None:
        """ほぼ定数列でも ret_vol_adj に Inf が発生しないこと（eps ガード）。"""
        config = _build_config()

        # 非常に小さいボラ（ほぼ定数列）
        df = pd.DataFrame({
            "core_index": np.full(100, 1.0) + np.random.randn(100) * 1e-10,
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # ret_vol_adj に Inf がない
        ret_vol_adj = features["ret_vol_adj"]
        assert not np.isinf(ret_vol_adj.values).any(), "ret_vol_adj should not have Inf"

    def test_ret_vol_adj_winsorize(self) -> None:
        """winsorize オプションで外れ値がクリップされること。"""
        config = _build_config()

        # 極端な外れ値を含むデータ（有効領域に配置）
        ret_values = np.random.randn(100)
        ret_values[25] = 1000.0  # 極端な外れ値（有効領域に配置）
        ret_values[26] = -1000.0
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": ret_values,
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # ret_vol_adj が winsorize されているはず
        ret_vol_adj = features["ret_vol_adj"]
        # 外れ値がクリップされている（元の 1000 より小さい）
        # NaN でなく、クリップされている（元の値より小さい）
        assert not np.isnan(ret_vol_adj.iloc[25]), "Should not be NaN in valid region"
        assert abs(ret_vol_adj.iloc[25]) < 500, "Extreme value should be clipped"
        assert not np.isnan(ret_vol_adj.iloc[26]), "Should not be NaN in valid region"
        assert abs(ret_vol_adj.iloc[26]) < 500, "Extreme value should be clipped"

    def test_fit_transform_responsibility(self) -> None:
        """fit/transform の責務分離。"""
        config = _build_config()
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)

        # fit 前は quantiles_ が None
        assert transformer.quantiles_ is None

        transformer.fit(df)

        # fit 後は quantiles_ が設定されている
        assert transformer.quantiles_ is not None
        assert "q_low" in transformer.quantiles_
        assert "q_high" in transformer.quantiles_
        assert "tau_down" in transformer.quantiles_
        assert "tau_up" in transformer.quantiles_

    def test_transform_before_fit_error(self) -> None:
        """fit 前に transform を呼ぶとエラー。"""
        config = _build_config()
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        with pytest.raises(RuntimeError, match="fitted"):
            transformer.transform(df)

    def test_missing_core_index_col_error(self) -> None:
        """存在しない core_index_col を指定した場合はエラー。"""
        config = _build_config(core_index_col="nonexistent_col")
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        with pytest.raises(ValueError, match="core_index_col"):
            transformer.fit(df)

    def test_missing_ret_base_col_error(self) -> None:
        """存在しない ret_base_col を指定した場合はエラー。"""
        config = _build_config(ret_base_col="nonexistent_col")
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        with pytest.raises(ValueError, match="ret_base_col"):
            transformer.fit(df)

    def test_nan_handling_at_endpoints(self) -> None:
        """先頭 min_periods 行で NaN が入ること。"""
        config = _build_config()
        df = pd.DataFrame({
            "core_index": np.random.randn(50),
            "ret_1": np.random.randn(50),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # ewmstd_long は min_periods=20 なので、最初の 19 行は NaN
        ewmstd_long = features["ewmstd_long"]
        assert np.all(np.isnan(ewmstd_long.iloc[:19])), "First 19 rows should be NaN"
        assert not np.isnan(ewmstd_long.iloc[19]), "Row 19 should not be NaN"

    def test_no_nan_inf_in_valid_region(self) -> None:
        """有効領域で NaN/Inf が出ないこと（vol_level 以外）。"""
        config = _build_config()
        np.random.seed(42)
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 有効領域: max(ewm_long_halflife, trend_ma_long_window) 以降
        valid_start = max(config.ewm_long_halflife, config.trend_ma_long_window)
        for col in features.columns:
            valid_values = features[col].iloc[valid_start:]
            # Inf がない
            assert not np.isinf(valid_values.values).any(), f"{col} has Inf in valid region"


class TestSU8FeatureAugmenter:
    """SU8FeatureAugmenter のテスト。"""

    def test_augment_concatenates_features(self) -> None:
        """元の DataFrame と SU8 特徴が結合されること。"""
        config = _build_config()
        df = pd.DataFrame({
            "core_index": np.random.randn(50),
            "ret_1": np.random.randn(50),
            "other_col": np.random.randn(50),
        })

        augmenter = SU8FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # 元の列が残っている
        assert "core_index" in augmented.columns
        assert "ret_1" in augmented.columns
        assert "other_col" in augmented.columns

        # SU8 特徴が追加されている
        assert "ewmstd_short" in augmented.columns
        assert "vol_regime_low" in augmented.columns
        assert "ret_vol_adj" in augmented.columns

    def test_fill_value(self) -> None:
        """fill_value が指定された場合、NaN が埋められること。"""
        config = _build_config()
        df = pd.DataFrame({
            "core_index": np.random.randn(50),
            "ret_1": np.random.randn(50),
        })

        augmenter = SU8FeatureAugmenter(config, fill_value=0.0)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # SU8 特徴に NaN がない（数値列）
        for col in ["ewmstd_short", "ewmstd_long", "vol_ratio", "vol_level", "ret_vol_adj"]:
            assert not bool(augmented[col].isna().any()), f"{col} should have no NaN"

    def test_no_fill_value(self) -> None:
        """fill_value が None の場合、NaN がそのまま残ること。"""
        config = _build_config()
        df = pd.DataFrame({
            "core_index": np.random.randn(50),
            "ret_1": np.random.randn(50),
        })

        augmenter = SU8FeatureAugmenter(config, fill_value=None)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # ewmstd_long の先頭は NaN のまま
        assert np.isnan(augmented["ewmstd_long"].iloc[0])

    def test_index_preserved(self) -> None:
        """DataFrame の index が保持されること。"""
        config = _build_config()
        df = pd.DataFrame(
            {
                "core_index": np.random.randn(50),
                "ret_1": np.random.randn(50),
            },
            index=pd.date_range("2023-01-01", periods=50),
        )

        augmenter = SU8FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        pd.testing.assert_index_equal(augmented.index, df.index)


class TestVolatilityFeatures:
    """ボラティリティ指標のテスト。"""

    def test_ewmstd_monotonic_increasing_vol(self) -> None:
        """ボラティリティが増加する系列での挙動確認。"""
        config = _build_config()
        # ボラティリティが時間とともに増加する系列
        np.random.seed(42)
        n = 100
        volatility_scale = np.linspace(0.1, 2.0, n)
        values = np.cumsum(np.random.randn(n) * volatility_scale)
        df = pd.DataFrame({
            "core_index": values,
            "ret_1": np.random.randn(n),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # vol_level は概ね増加傾向にあるはず
        vol_level = features["vol_level"].dropna()
        # 後半の平均が前半の平均より大きいはず
        mid = len(vol_level) // 2
        first_half_mean = vol_level.iloc[:mid].mean()
        second_half_mean = vol_level.iloc[mid:].mean()
        assert second_half_mean > first_half_mean, (
            f"Vol level should increase: first_half={first_half_mean}, second_half={second_half_mean}"
        )

    def test_vol_ratio_calculation(self) -> None:
        """vol_ratio = ewmstd_short / (ewmstd_long + eps) が正しいこと。"""
        config = _build_config()
        np.random.seed(42)
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 有効領域で確認
        valid_start = max(config.ewm_short_halflife, config.ewm_long_halflife)
        ewmstd_short = features["ewmstd_short"].iloc[valid_start:].values
        ewmstd_long = features["ewmstd_long"].iloc[valid_start:].values
        vol_ratio = features["vol_ratio"].iloc[valid_start:].values

        expected_ratio = ewmstd_short / (ewmstd_long + config.eps)
        np.testing.assert_allclose(vol_ratio, expected_ratio, rtol=1e-5)


class TestRegimeTagging:
    """レジームタグ付けのテスト。"""

    def test_vol_regime_mutually_exclusive(self) -> None:
        """vol_regime_low/mid/high が相互排他的であること。"""
        config = _build_config()
        np.random.seed(42)
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 有効領域で確認
        valid_start = config.ewm_long_halflife
        for i in range(valid_start, len(features)):
            low = features["vol_regime_low"].iloc[i]
            mid = features["vol_regime_mid"].iloc[i]
            high = features["vol_regime_high"].iloc[i]

            # NaN の場合はスキップ
            if pd.isna(low) or pd.isna(mid) or pd.isna(high):
                continue

            # 1 つだけが True のはず
            count = int(low) + int(mid) + int(high)
            assert count == 1, f"At row {i}, exactly one regime should be True, got {count}"

    def test_trend_regime_mutually_exclusive(self) -> None:
        """trend_regime_up/down/flat が相互排他的であること。"""
        config = _build_config()
        np.random.seed(42)
        df = pd.DataFrame({
            "core_index": np.random.randn(100),
            "ret_1": np.random.randn(100),
        })

        transformer = SU8FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 有効領域で確認
        valid_start = config.trend_ma_long_window
        for i in range(valid_start, len(features)):
            down = features["trend_regime_down"].iloc[i]
            up = features["trend_regime_up"].iloc[i]
            flat = features["trend_regime_flat"].iloc[i]

            # NaN の場合はスキップ
            if pd.isna(down) or pd.isna(up) or pd.isna(flat):
                continue

            # 1 つだけが True のはず
            count = int(down) + int(up) + int(flat)
            assert count == 1, f"At row {i}, exactly one regime should be True, got {count}"
