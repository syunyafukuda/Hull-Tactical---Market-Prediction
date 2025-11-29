"""SU7 モメンタム・リバーサル特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su7.feature_su7 import (
    SU7Config,
    SU7FeatureAugmenter,
    SU7FeatureGenerator,
    load_su7_config,
)


def _build_config(base_cols: list[str] | None = None) -> SU7Config:
    """テスト用のSU7Config を生成する。"""
    if base_cols is None:
        base_cols = ["ret_1", "ret_2"]
    mapping = {
        "su7_base_cols": base_cols,
        "lags": [1, 5, 20],
        "windows": [5, 20],
        "halflife_rsi": 5,
        "eps": 1e-8,
        "rs_max": 100.0,
        "dtype": {"float": "float32", "int": "int8"},
    }
    return SU7Config.from_mapping(mapping)


class TestSU7Config:
    """SU7Config のテスト。"""

    def test_config_loading(self, tmp_path: Path) -> None:
        """YAML設定の読込確認。"""
        config_path = tmp_path / "test_config.yaml"
        config_content = """
su7:
  enabled: true
  su7_base_cols:
    - ret_1d_spx
    - ret_1d_sector1
  lags: [1, 5, 20]
  windows: [5, 20]
  halflife_rsi: 5
  eps: 1.0e-8
  rs_max: 100.0
  dtype:
    float: float32
    int: int8
"""
        config_path.write_text(config_content)

        config = load_su7_config(config_path)
        assert config.su7_base_cols == ("ret_1d_spx", "ret_1d_sector1")
        assert config.lags == (1, 5, 20)
        assert config.windows == (5, 20)
        assert config.halflife_rsi == 5
        assert config.eps == 1e-8
        assert config.rs_max == 100.0

    def test_config_missing_base_cols(self) -> None:
        """su7_base_cols が空の場合はエラー。"""
        with pytest.raises(ValueError, match="su7_base_cols"):
            SU7Config.from_mapping({"su7_base_cols": []})

    def test_config_default_values(self) -> None:
        """デフォルト値の確認。"""
        config = SU7Config.from_mapping({"su7_base_cols": ["col1"]})
        assert config.lags == (1, 5, 20)
        assert config.windows == (5, 20)
        assert config.halflife_rsi == 5
        assert config.eps == 1e-8
        assert config.rs_max == 100.0


class TestSU7FeatureGenerator:
    """SU7FeatureGenerator のテスト。"""

    def test_monotonic_increasing_series(self) -> None:
        """単調増加系列での挙動確認。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.arange(1, 31, dtype=float)})  # 1, 2, ..., 30

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # diff_1 = r_t - r_{t-1} = 1 (単調増加なので差分は常に1)
        diff_1 = features["diff_1/ret_1"]
        assert np.isnan(diff_1[0])  # 先頭は NaN
        np.testing.assert_allclose(diff_1[1:], 1.0, rtol=1e-5)

        # diff_5 = r_t - r_{t-5} = 5
        diff_5 = features["diff_5/ret_1"]
        assert np.all(np.isnan(diff_5[:5]))  # 先頭5行は NaN
        np.testing.assert_allclose(diff_5[5:], 5.0, rtol=1e-5)

        # lag_1 = r_{t-1}
        lag_1 = features["lag_1/ret_1"]
        assert np.isnan(lag_1[0])
        np.testing.assert_allclose(lag_1[1:], np.arange(1, 30, dtype=float), rtol=1e-5)

    def test_monotonic_decreasing_series(self) -> None:
        """単調減少系列での挙動確認。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.arange(30, 0, -1, dtype=float)})  # 30, 29, ..., 1

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # diff_1 = r_t - r_{t-1} = -1 (単調減少なので差分は常に-1)
        diff_1 = features["diff_1/ret_1"]
        assert np.isnan(diff_1[0])
        np.testing.assert_allclose(diff_1[1:], -1.0, rtol=1e-5)

        # sign_r_t は全て正（リターン自体は正）
        sign_r_t = features["sign_r_t/ret_1"]
        np.testing.assert_array_equal(sign_r_t, np.ones(30, dtype=np.int8))

    def test_zero_series(self) -> None:
        """ゼロ系列での挙動確認。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.zeros(30)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # diff/lag は全て 0 or NaN
        diff_1 = features["diff_1/ret_1"]
        assert np.isnan(diff_1[0])
        np.testing.assert_allclose(diff_1[1:], 0.0, rtol=1e-5)

        # roll_ret は全て 0 or NaN
        roll_ret_5 = features["roll_ret_5/ret_1"]
        assert np.all(np.isnan(roll_ret_5[:4]))  # window=5, 先頭4行は NaN
        np.testing.assert_allclose(roll_ret_5[4:], 0.0, rtol=1e-5)

        # sign_r_t は全て 0
        sign_r_t = features["sign_r_t/ret_1"]
        np.testing.assert_array_equal(sign_r_t, np.zeros(30, dtype=np.int8))

    def test_nan_handling_at_endpoints(self) -> None:
        """先頭 max(lags) 日や window 未満区間で NaN が入ること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.random.randn(30)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # diff_1: 先頭1行が NaN
        assert np.isnan(features["diff_1/ret_1"][0])
        assert not np.isnan(features["diff_1/ret_1"][1])

        # diff_5: 先頭5行が NaN
        assert np.all(np.isnan(features["diff_5/ret_1"][:5]))
        assert not np.isnan(features["diff_5/ret_1"][5])

        # diff_20: 先頭20行が NaN
        assert np.all(np.isnan(features["diff_20/ret_1"][:20]))
        assert not np.isnan(features["diff_20/ret_1"][20])

        # roll_ret_5: 先頭4行が NaN (window=5, min_periods=5)
        assert np.all(np.isnan(features["roll_ret_5/ret_1"][:4]))
        assert not np.isnan(features["roll_ret_5/ret_1"][4])

        # roll_ret_20: 先頭19行が NaN
        assert np.all(np.isnan(features["roll_ret_20/ret_1"][:19]))
        assert not np.isnan(features["roll_ret_20/ret_1"][19])

    def test_no_nan_inf_in_valid_region(self) -> None:
        """NaN/Inf が有効領域（十分な履歴がある部分）で出ないこと。"""
        config = _build_config(["ret_1"])
        np.random.seed(42)
        df = pd.DataFrame({"ret_1": np.random.randn(100)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 有効領域: max(max(lags), max(windows) - 1) 以降
        # lag=20 は index >= 20 から有効
        # rolling window=20 は min_periods=20 なので index >= 19 から有効
        valid_start = max(max(config.lags), max(config.windows) - 1)
        for col in features.columns:
            valid_values = features[col].iloc[valid_start:]
            # NaN がない
            assert not valid_values.isna().any(), f"{col} has NaN in valid region"
            # Inf がない
            assert not np.isinf(valid_values.values).any(), f"{col} has Inf in valid region"

    def test_rsi_numerical_stability_all_positive(self) -> None:
        """r_t がほぼ常に正の場合でも rsi_5 に NaN/Inf が発生しないこと。"""
        config = _build_config(["ret_1"])
        # 全て正のリターン
        df = pd.DataFrame({"ret_1": np.abs(np.random.randn(100)) + 0.1})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        rsi = features["rsi_5/ret_1"]
        # NaN/Inf がない
        assert not rsi.isna().any(), "RSI should not have NaN"
        assert not np.isinf(rsi.values).any(), "RSI should not have Inf"
        # RSI は 0〜1 の範囲
        assert (rsi >= 0).all() and (rsi <= 1).all(), "RSI should be in [0, 1]"

    def test_rsi_numerical_stability_all_negative(self) -> None:
        """r_t がほぼ常に負の場合でも rsi_5 に NaN/Inf が発生しないこと。"""
        config = _build_config(["ret_1"])
        # 全て負のリターン
        df = pd.DataFrame({"ret_1": -np.abs(np.random.randn(100)) - 0.1})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        rsi = features["rsi_5/ret_1"]
        # NaN/Inf がない
        assert not rsi.isna().any(), "RSI should not have NaN"
        assert not np.isinf(rsi.values).any(), "RSI should not have Inf"
        # RSI は 0〜1 の範囲
        assert (rsi >= 0).all() and (rsi <= 1).all(), "RSI should be in [0, 1]"

    def test_rsi_numerical_stability_extreme_values(self) -> None:
        """極端な値でも rsi_5 に NaN/Inf が発生しないこと。"""
        config = _build_config(["ret_1"])
        # 極端な値（0に近い値と大きな値の混合）
        values = np.concatenate(
            [
                np.full(10, 1e-10),  # 極小値
                np.full(10, 1e10),  # 極大値
                np.full(10, -1e-10),  # 極小負値
                np.full(10, -1e10),  # 極大負値
                np.zeros(10),  # ゼロ
            ]
        )
        df = pd.DataFrame({"ret_1": values})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        rsi = features["rsi_5/ret_1"]
        # NaN/Inf がない
        assert not rsi.isna().any(), "RSI should not have NaN"
        assert not np.isinf(rsi.values).any(), "RSI should not have Inf"

    def test_leak_prevention_shift(self) -> None:
        """shift が過去方向のみ参照することを確認（リーク防止）。"""
        config = _build_config(["ret_1"])
        # 特殊なパターン: index=10 以降は全て 999
        values = np.arange(30, dtype=float)
        values[10:] = 999.0
        df = pd.DataFrame({"ret_1": values})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # index=9 の lag_1 は index=8 の値を参照 = 8
        assert features["lag_1/ret_1"].iloc[9] == 8.0
        # index=9 の lag_1 は 999 を参照していない（リークがない）
        assert features["lag_1/ret_1"].iloc[9] != 999.0

    def test_leak_prevention_rolling(self) -> None:
        """rolling が過去方向のみ参照することを確認（リーク防止）。"""
        config = _build_config(["ret_1"])
        # 特殊なパターン: index=10 以降は全て 1000
        values = np.ones(30, dtype=float)
        values[10:] = 1000.0
        df = pd.DataFrame({"ret_1": values})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # index=9 の roll_ret_5 は index=5..9 の値を参照 = 5 * 1 = 5
        assert features["roll_ret_5/ret_1"].iloc[9] == 5.0
        # index=9 の roll_ret_5 は 1000 を参照していない（リークがない）
        assert features["roll_ret_5/ret_1"].iloc[9] != 5000.0

    def test_leak_prevention_ewm(self) -> None:
        """ewm が過去方向のみ参照することを確認（リーク防止）。"""
        config = _build_config(["ret_1"])
        # 特殊なパターン: index=20 以降は全て負
        values = np.ones(40, dtype=float)
        values[20:] = -10.0
        df = pd.DataFrame({"ret_1": values})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # index=19 の RSI は正のリターンのみを見ているはず
        # gains は全て 1.0, losses は全て 0.0
        # RS = ema_gain / (ema_loss + eps) → 非常に大きな値（rs_max でクリップ）
        # RSI = rs / (1 + rs) → 1 に近い値
        rsi_at_19 = features["rsi_5/ret_1"].iloc[19]
        assert rsi_at_19 > 0.9, f"RSI at index 19 should be close to 1.0, got {rsi_at_19}"

    def test_column_count(self) -> None:
        """生成される列数が仕様どおりであること。

        B = len(su7_base_cols)
        - diff/lag: 2 * 3 * B = 6B
        - roll_ret/roll_mean_diff: 2 * 2 * B = 4B
        - rsi: B
        - sign: B
        - 合計: 12B
        """
        for n_cols in [1, 2, 6, 8]:
            base_cols = [f"ret_{i}" for i in range(n_cols)]
            config = _build_config(base_cols)

            df = pd.DataFrame({col: np.random.randn(30) for col in base_cols})

            transformer = SU7FeatureGenerator(config)
            transformer.fit(df)
            features = transformer.transform(df)

            expected_count = 12 * n_cols
            actual_count = len(features.columns)
            assert (
                actual_count == expected_count
            ), f"Expected {expected_count} columns for B={n_cols}, got {actual_count}"

    def test_expected_column_count_method(self) -> None:
        """get_expected_column_count() メソッドの確認。"""
        config = _build_config(["ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_6"])
        transformer = SU7FeatureGenerator(config)

        # B=6 の場合、12 * 6 = 72
        assert transformer.get_expected_column_count() == 72

        # 引数で指定した場合
        assert transformer.get_expected_column_count(n_base_cols=8) == 96

    def test_dtype_float32(self) -> None:
        """float32 の dtype が正しく適用されていること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.random.randn(30)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 連続値列は float32
        for col in ["diff_1/ret_1", "lag_1/ret_1", "roll_ret_5/ret_1", "rsi_5/ret_1"]:
            assert features[col].dtype == np.float32, f"{col} should be float32"

    def test_dtype_int8(self) -> None:
        """int8 の dtype が正しく適用されていること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.random.randn(30)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # sign_r_t は int8
        assert features["sign_r_t/ret_1"].dtype == np.int8

    def test_sign_values(self) -> None:
        """sign_r_t が {-1, 0, 1} の値のみを持つこと。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0]})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        sign_r_t = features["sign_r_t/ret_1"]
        expected = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(sign_r_t, expected)

    def test_missing_base_cols_error(self) -> None:
        """存在しない列を指定した場合はエラー。"""
        config = _build_config(["nonexistent_col"])
        df = pd.DataFrame({"ret_1": np.random.randn(30)})

        transformer = SU7FeatureGenerator(config)
        with pytest.raises(ValueError, match="not found in input"):
            transformer.fit(df)


class TestSU7FeatureAugmenter:
    """SU7FeatureAugmenter のテスト。"""

    def test_augment_concatenates_features(self) -> None:
        """元の DataFrame と SU7 特徴が結合されること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame(
            {
                "ret_1": np.random.randn(30),
                "other_col": np.random.randn(30),
            }
        )

        augmenter = SU7FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # 元の列が残っている
        assert "ret_1" in augmented.columns
        assert "other_col" in augmented.columns

        # SU7 特徴が追加されている
        assert "diff_1/ret_1" in augmented.columns
        assert "rsi_5/ret_1" in augmented.columns
        assert "sign_r_t/ret_1" in augmented.columns

    def test_fill_value(self) -> None:
        """fill_value が指定された場合、NaN が埋められること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.random.randn(30)})

        augmenter = SU7FeatureAugmenter(config, fill_value=0.0)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # SU7 特徴に NaN がない
        su7_cols = [c for c in augmented.columns if "/" in c]
        for col in su7_cols:
            assert not augmented[col].isna().any(), f"{col} should have no NaN"

    def test_no_fill_value(self) -> None:
        """fill_value が None の場合、NaN がそのまま残ること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame({"ret_1": np.random.randn(30)})

        augmenter = SU7FeatureAugmenter(config, fill_value=None)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # diff_1 の先頭は NaN のまま
        assert np.isnan(augmented["diff_1/ret_1"].iloc[0])

    def test_index_preserved(self) -> None:
        """DataFrame の index が保持されること。"""
        config = _build_config(["ret_1"])
        df = pd.DataFrame(
            {"ret_1": np.random.randn(30)}, index=pd.date_range("2023-01-01", periods=30)
        )

        augmenter = SU7FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        pd.testing.assert_index_equal(augmented.index, df.index)


class TestRollingSum:
    """roll_ret のテスト（単純和の検証）。"""

    def test_rolling_sum_calculation(self) -> None:
        """roll_ret_W = ∑_{i=t-W+1}^{t} r_i が正しいこと。"""
        config = _build_config(["ret_1"])
        # 1, 2, 3, ..., 30
        df = pd.DataFrame({"ret_1": np.arange(1, 31, dtype=float)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # roll_ret_5 at index=4: sum(1, 2, 3, 4, 5) = 15
        assert features["roll_ret_5/ret_1"].iloc[4] == 15.0

        # roll_ret_5 at index=9: sum(6, 7, 8, 9, 10) = 40
        assert features["roll_ret_5/ret_1"].iloc[9] == 40.0

    def test_roll_mean_diff_calculation(self) -> None:
        """roll_mean_diff_W = mean(diff_1[t-W+1:t]) が正しいこと。"""
        config = _build_config(["ret_1"])
        # 単調増加: diff_1 は常に 1
        df = pd.DataFrame({"ret_1": np.arange(1, 31, dtype=float)})

        transformer = SU7FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # diff_1 = 1 なので、roll_mean_diff_5 = mean([1, 1, 1, 1, 1]) = 1
        # ただし、diff_1 の先頭は NaN なので、index=5 から有効
        roll_mean_diff = features["roll_mean_diff_5/ret_1"]
        # index=5: mean of diff_1[1:6] = mean([1, 1, 1, 1, 1]) = 1.0
        np.testing.assert_allclose(roll_mean_diff.iloc[5], 1.0, rtol=1e-5)
