"""SU9 カレンダー・季節性特徴量生成の単体テスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su9.feature_su9 import (
    SU9Config,
    SU9FeatureAugmenter,
    SU9FeatureGenerator,
    load_su9_config,
)


def _build_config(
    id_column: str = "date_id",
    include_dow: bool = True,
    include_dom: bool = True,
    include_month: bool = True,
    include_month_flags: bool = True,
    include_holiday: bool = True,
    include_year_position: bool = True,
    holiday_calendar_path: str | None = None,
) -> SU9Config:
    """テスト用のSU9Config を生成する。"""
    mapping = {
        "id_column": id_column,
        "include_dow": include_dow,
        "include_dom": include_dom,
        "include_month": include_month,
        "include_month_flags": include_month_flags,
        "include_holiday": include_holiday,
        "include_year_position": include_year_position,
        "holiday_calendar_path": holiday_calendar_path,
        "dtype": {"flag": "uint8", "float": "float32"},
    }
    return SU9Config.from_mapping(mapping)


class TestSU9Config:
    """SU9Config のテスト。"""

    def test_config_loading(self, tmp_path: Path) -> None:
        """YAML設定の読込確認。"""
        config_path = tmp_path / "test_config.yaml"
        config_content = """
su9:
  enabled: false
  id_column: date_id
  include_dow: true
  include_dom: true
  include_month: true
  include_month_flags: true
  include_holiday: true
  include_year_position: true
  holiday_calendar_path: null
  dtype:
    flag: uint8
    float: float32
"""
        config_path.write_text(config_content)

        config = load_su9_config(config_path)
        assert config.id_column == "date_id"
        assert config.include_dow is True
        assert config.include_dom is True
        assert config.include_month is True
        assert config.include_month_flags is True
        assert config.include_holiday is True
        assert config.include_year_position is True
        assert config.holiday_calendar_path is None

    def test_config_missing_id_column(self) -> None:
        """id_column が空の場合はエラー。"""
        with pytest.raises(ValueError, match="id_column"):
            SU9Config.from_mapping({})

    def test_config_default_values(self) -> None:
        """デフォルト値の確認。"""
        config = SU9Config.from_mapping({"id_column": "date_id"})
        assert config.include_dow is True
        assert config.include_dom is True
        assert config.include_month is True
        assert config.include_month_flags is True
        assert config.include_holiday is True
        assert config.include_year_position is True


class TestSU9FeatureGenerator:
    """SU9FeatureGenerator のテスト。"""

    def test_basic_shape_and_columns(self) -> None:
        """基本的な形状・列名のテスト（全特徴量有効時）。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 32 列が追加されるはず (7+3+12+4+4+2)
        assert len(features.columns) == 32
        assert transformer.get_expected_column_count() == 32

        # 列名の確認
        expected_cols = (
            [f"dow_{i}" for i in range(7)]
            + ["dom_early", "dom_mid", "dom_late"]
            + [f"month_{i}" for i in range(1, 13)]
            + ["is_month_start", "is_month_end", "is_qtr_start", "is_qtr_end"]
            + ["is_holiday", "is_holiday_eve", "is_holiday_next", "is_bridge_day"]
            + ["yday_norm", "days_to_year_end"]
        )
        for col in expected_cols:
            assert col in features.columns, f"{col} should be in features"

    def test_dow_one_hot(self) -> None:
        """曜日 one-hot 特徴のテスト。"""
        config = _build_config()
        # date_id = 0, 1, 2, 3, 4 → 月, 火, 水, 木, 金
        df = pd.DataFrame({"date_id": np.arange(5)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 各日で対応する曜日のみが 1
        for i in range(5):
            for j in range(7):
                expected = 1 if i == j else 0
                assert features[f"dow_{j}"].iloc[i] == expected, (
                    f"dow_{j} at row {i} should be {expected}"
                )

        # 土日（dow_5, dow_6）は常に 0（営業日データのため）
        assert (features["dow_5"] == 0).all()
        assert (features["dow_6"] == 0).all()

    def test_dow_one_hot_cyclic(self) -> None:
        """曜日 one-hot が周期的であることを確認。"""
        config = _build_config()
        # date_id = 5, 6, 7, 8, 9 → 月, 火, 水, 木, 金（再び月曜から）
        df = pd.DataFrame({"date_id": np.arange(5, 10)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 各日で対応する曜日のみが 1
        for i in range(5):
            for j in range(5):
                expected = 1 if i == j else 0
                assert features[f"dow_{j}"].iloc[i] == expected, (
                    f"dow_{j} at row {i} should be {expected}"
                )

    def test_dom_bins(self) -> None:
        """月内位置ビン特徴のテスト。"""
        config = _build_config()
        # 1ヶ月 = 22営業日として
        # dom_early: 0-7, dom_mid: 8-14, dom_late: 15-21
        df = pd.DataFrame({"date_id": np.arange(22)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # dom_early: 0-7
        for i in range(8):
            assert features["dom_early"].iloc[i] == 1, f"dom_early at row {i} should be 1"
            assert features["dom_mid"].iloc[i] == 0
            assert features["dom_late"].iloc[i] == 0

        # dom_mid: 8-14
        for i in range(8, 15):
            assert features["dom_early"].iloc[i] == 0
            assert features["dom_mid"].iloc[i] == 1, f"dom_mid at row {i} should be 1"
            assert features["dom_late"].iloc[i] == 0

        # dom_late: 15-21
        for i in range(15, 22):
            assert features["dom_early"].iloc[i] == 0
            assert features["dom_mid"].iloc[i] == 0
            assert features["dom_late"].iloc[i] == 1, f"dom_late at row {i} should be 1"

    def test_month_one_hot(self) -> None:
        """月 one-hot 特徴のテスト。"""
        config = _build_config()
        # 1年 = 252営業日、各月約21営業日
        df = pd.DataFrame({"date_id": np.arange(252)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 各行で1つの月のみが 1
        for i in range(252):
            month_sum = sum(features[f"month_{m}"].iloc[i] for m in range(1, 13))
            assert month_sum == 1, f"Exactly one month should be 1 at row {i}"

    def test_month_flags(self) -> None:
        """月末・期末フラグ特徴のテスト。"""
        config = _build_config()
        # 1ヶ月 = 22営業日として
        df = pd.DataFrame({"date_id": np.arange(44)})  # 2ヶ月分

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # is_month_start: dom == 0
        assert features["is_month_start"].iloc[0] == 1
        assert features["is_month_start"].iloc[22] == 1
        assert features["is_month_start"].iloc[1] == 0

        # is_month_end: dom == 21
        assert features["is_month_end"].iloc[21] == 1
        assert features["is_month_end"].iloc[43] == 1
        assert features["is_month_end"].iloc[20] == 0

    def test_holiday_flags(self) -> None:
        """祝日・ブリッジ特徴のテスト（祝日カレンダーなし）。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(10)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 祝日カレンダーがないので全て 0
        assert (features["is_holiday"] == 0).all()
        assert (features["is_holiday_eve"] == 0).all()
        assert (features["is_holiday_next"] == 0).all()
        assert (features["is_bridge_day"] == 0).all()

    def test_holiday_flags_with_calendar(self, tmp_path: Path) -> None:
        """祝日・ブリッジ特徴のテスト（祝日カレンダーあり）。"""
        # 祝日カレンダーを作成
        calendar_path = tmp_path / "holidays.csv"
        calendar_df = pd.DataFrame({"date_id": [5, 6]})  # date_id=5, 6 が祝日
        calendar_df.to_csv(calendar_path, index=False)

        config = _build_config(holiday_calendar_path=str(calendar_path))
        df = pd.DataFrame({"date_id": np.arange(10)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # is_holiday: date_id=5, 6 が祝日
        assert features["is_holiday"].iloc[5] == 1
        assert features["is_holiday"].iloc[6] == 1
        assert features["is_holiday"].iloc[4] == 0

        # is_holiday_eve: 翌日が祝日（date_id=4, 5 が前日）
        assert features["is_holiday_eve"].iloc[4] == 1
        assert features["is_holiday_eve"].iloc[5] == 1
        assert features["is_holiday_eve"].iloc[6] == 0

        # is_holiday_next: 前日が祝日（date_id=6, 7 が翌日）
        assert features["is_holiday_next"].iloc[6] == 1
        assert features["is_holiday_next"].iloc[7] == 1
        assert features["is_holiday_next"].iloc[5] == 0

    def test_bridge_day_logic(self, tmp_path: Path) -> None:
        """ブリッジデーのテスト。"""
        # 祝日カレンダーを作成（date_id=5 と 7 が祝日、6 がブリッジデー）
        calendar_path = tmp_path / "holidays.csv"
        calendar_df = pd.DataFrame({"date_id": [5, 7]})
        calendar_df.to_csv(calendar_path, index=False)

        config = _build_config(holiday_calendar_path=str(calendar_path))
        df = pd.DataFrame({"date_id": np.arange(10)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # is_bridge_day: date_id=6 は前後が祝日
        assert features["is_bridge_day"].iloc[6] == 1
        # 祝日自体はブリッジデーではない
        assert features["is_bridge_day"].iloc[5] == 0
        assert features["is_bridge_day"].iloc[7] == 0

    def test_year_position(self) -> None:
        """年内ポジション特徴のテスト。"""
        config = _build_config()
        # 1年 = 252営業日
        df = pd.DataFrame({"date_id": np.arange(252)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # yday_norm: 0 → 0.0, 251 → 1.0
        assert np.isclose(features["yday_norm"].iloc[0], 0.0)
        assert np.isclose(features["yday_norm"].iloc[251], 1.0)

        # days_to_year_end: 0 → 1.0, 251 → 0.0
        assert np.isclose(features["days_to_year_end"].iloc[0], 1.0)
        assert np.isclose(features["days_to_year_end"].iloc[251], 0.0)

    def test_no_future_leakage(self) -> None:
        """リークがないことのテスト。

        SU9 は決定可能な日付情報のみを使用するため、
        将来情報を参照しないことを確認。
        """
        config = _build_config()
        # 2つのデータセット（前半と後半）
        df_first = pd.DataFrame({"date_id": np.arange(50)})
        df_full = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)

        # 前半のみで fit/transform
        transformer.fit(df_first)
        features_first = transformer.transform(df_first)

        # 全体で fit/transform
        transformer.fit(df_full)
        features_full = transformer.transform(df_full)

        # 前半の結果は同じはず（後半のデータに依存しない）
        for col in features_first.columns:
            np.testing.assert_array_equal(
                np.asarray(features_first[col].values),
                np.asarray(features_full[col].iloc[:50].values),
                err_msg=f"{col} should not depend on future data",
            )

    def test_fit_transform_responsibility(self) -> None:
        """fit/transform の責務分離。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)

        # fit 前は feature_names_ が None
        assert transformer.feature_names_ is None

        transformer.fit(df)

        # fit 後は feature_names_ が設定されている
        assert transformer.feature_names_ is not None
        assert len(transformer.feature_names_) == 32

    def test_transform_before_fit_error(self) -> None:
        """fit 前に transform を呼ぶとエラー。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)
        with pytest.raises(RuntimeError, match="fitted"):
            transformer.transform(df)

    def test_missing_id_column_error(self) -> None:
        """存在しない id_column を指定した場合はエラー。"""
        config = _build_config(id_column="nonexistent_col")
        df = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)
        with pytest.raises(ValueError, match="id_column"):
            transformer.fit(df)

    def test_partial_features_dow_only(self) -> None:
        """曜日のみ有効の場合。"""
        config = _build_config(
            include_dow=True,
            include_dom=False,
            include_month=False,
            include_month_flags=False,
            include_holiday=False,
            include_year_position=False,
        )
        df = pd.DataFrame({"date_id": np.arange(10)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        assert len(features.columns) == 7
        assert transformer.get_expected_column_count() == 7

    def test_partial_features_holiday_only(self) -> None:
        """祝日のみ有効の場合。"""
        config = _build_config(
            include_dow=False,
            include_dom=False,
            include_month=False,
            include_month_flags=False,
            include_holiday=True,
            include_year_position=False,
        )
        df = pd.DataFrame({"date_id": np.arange(10)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        assert len(features.columns) == 4
        assert transformer.get_expected_column_count() == 4

    def test_dtype_check(self) -> None:
        """dtype が仕様通りであること。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # フラグ列は uint8
        for col in ["dow_0", "dom_early", "month_1", "is_month_start", "is_holiday"]:
            assert features[col].dtype == np.uint8, f"{col} should be uint8"

        # 数値列は float32
        for col in ["yday_norm", "days_to_year_end"]:
            assert features[col].dtype == np.float32, f"{col} should be float32"


class TestSU9FeatureAugmenter:
    """SU9FeatureAugmenter のテスト。"""

    def test_augmentation(self) -> None:
        """元の DataFrame と SU9 特徴が結合されること。"""
        config = _build_config()
        df = pd.DataFrame({
            "date_id": np.arange(50),
            "other_col": np.random.randn(50),
        })

        augmenter = SU9FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # 元の列が残っている
        assert "date_id" in augmented.columns
        assert "other_col" in augmented.columns

        # SU9 特徴が追加されている
        assert "dow_0" in augmented.columns
        assert "dom_early" in augmented.columns
        assert "month_1" in augmented.columns
        assert "is_month_start" in augmented.columns
        assert "is_holiday" in augmented.columns
        assert "yday_norm" in augmented.columns

    def test_column_count_consistency(self) -> None:
        """列数の一貫性テスト。"""
        config = _build_config()
        df = pd.DataFrame({
            "date_id": np.arange(50),
            "other_col": np.random.randn(50),
        })

        augmenter = SU9FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # 元の列数 + SU9 列数 (32)
        assert len(augmented.columns) == 2 + 32

    def test_fill_value(self) -> None:
        """fill_value が指定された場合、NaN が埋められること。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(50)})

        augmenter = SU9FeatureAugmenter(config, fill_value=0.0)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        # SU9 特徴に NaN がない
        for col in augmented.columns:
            if col != "date_id":
                assert not bool(augmented[col].isna().any()), f"{col} should have no NaN"

    def test_index_preserved(self) -> None:
        """DataFrame の index が保持されること。"""
        config = _build_config()
        df = pd.DataFrame(
            {"date_id": np.arange(50)},
            index=pd.date_range("2023-01-01", periods=50),
        )

        augmenter = SU9FeatureAugmenter(config)
        augmenter.fit(df)
        augmented = augmenter.transform(df)

        pd.testing.assert_index_equal(augmented.index, df.index)

    def test_transform_before_fit_error(self) -> None:
        """fit 前に transform を呼ぶとエラー。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(50)})

        augmenter = SU9FeatureAugmenter(config)
        with pytest.raises(RuntimeError):
            augmenter.transform(df)


class TestDomBins:
    """月内位置ビンの詳細テスト。"""

    def test_dom_bins_exclusive(self) -> None:
        """dom_early/mid/late が相互排他的であること。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(100)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # 各行で1つのビンのみが 1
        for i in range(100):
            early = features["dom_early"].iloc[i]
            mid = features["dom_mid"].iloc[i]
            late = features["dom_late"].iloc[i]
            total = early + mid + late
            assert total == 1, f"Exactly one DOM bin should be 1 at row {i}, got {total}"


class TestMonthFlags:
    """月末・期末フラグの詳細テスト。"""

    def test_qtr_start_months(self) -> None:
        """四半期初が正しい月に設定されること。"""
        config = _build_config()
        # 1年分のデータ
        df = pd.DataFrame({"date_id": np.arange(252)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # is_qtr_start が 1 の行を確認
        qtr_starts = features[features["is_qtr_start"] == 1].index.tolist()
        # 少なくとも1つは存在するはず
        assert len(qtr_starts) >= 1, "At least one quarter start should exist"

    def test_qtr_end_months(self) -> None:
        """四半期末が正しい月に設定されること。"""
        config = _build_config()
        # 1年分のデータ
        df = pd.DataFrame({"date_id": np.arange(252)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        # is_qtr_end が 1 の行を確認
        qtr_ends = features[features["is_qtr_end"] == 1].index.tolist()
        # 少なくとも1つは存在するはず
        assert len(qtr_ends) >= 1, "At least one quarter end should exist"


class TestYearPosition:
    """年内ポジションの詳細テスト。"""

    def test_yday_norm_range(self) -> None:
        """yday_norm が [0, 1] の範囲であること。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(1000)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        assert (features["yday_norm"] >= 0).all()
        assert (features["yday_norm"] <= 1).all()

    def test_days_to_year_end_range(self) -> None:
        """days_to_year_end が [0, 1] の範囲であること。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(1000)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        assert (features["days_to_year_end"] >= 0).all()
        assert (features["days_to_year_end"] <= 1).all()

    def test_yday_norm_plus_days_to_year_end(self) -> None:
        """yday_norm + days_to_year_end ≈ 1 であること。"""
        config = _build_config()
        df = pd.DataFrame({"date_id": np.arange(252)})

        transformer = SU9FeatureGenerator(config)
        transformer.fit(df)
        features = transformer.transform(df)

        total = features["yday_norm"] + features["days_to_year_end"]
        np.testing.assert_allclose(total.values, 1.0, rtol=1e-5)
