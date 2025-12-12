from typing import cast

import numpy as np
import pandas as pd
import pytest

from preprocess.M_group.m_group import MGroupImputer


def _df(values):
    return pd.DataFrame(values)


def _fit_transform_df(imputer: MGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.fit_transform(df.copy()))


def _transform_df(imputer: MGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.transform(df.copy()))


def test_ffill_bfill_train_and_transform():
    df = _df({
        "M1": [1.0, np.nan, np.nan, 4.0],
        "M2": [np.nan, 1.0, np.nan, 3.0],
        "E1": [5.0, 6.0, 7.0, 8.0],
    })
    imputer = MGroupImputer(columns=["M1", "M2"], policy="ffill_bfill")
    filled_train = _fit_transform_df(imputer, df)
    expected_train = _df({"M1": [1.0, 1.0, 1.0, 4.0], "M2": [1.0, 1.0, 1.0, 3.0], "E1": [5.0, 6.0, 7.0, 8.0]})
    pd.testing.assert_frame_equal(filled_train, expected_train)
    assert isinstance(imputer._state_.get("warnings"), list)

    val = _df({"M1": [np.nan, 5.0], "M2": [np.nan, np.nan], "E1": [1.0, 1.0]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "M1"] == 4.0
    assert filled_val.loc[0, "M2"] == 3.0
    assert filled_val.loc[1, "M1"] == 5.0
    assert filled_val.loc[1, "M2"] == 3.0


def test_ffill_bfill_alias_matches_primary():
    df = _df({"M1": [np.nan, 1.0, np.nan]})
    base = MGroupImputer(columns=["M1"], policy="ffill_bfill")
    alias = MGroupImputer(columns=["M1"], policy="ffill_train_bfill_in_fit")
    filled_base = _fit_transform_df(base, df)
    filled_alias = _fit_transform_df(alias, df)
    pd.testing.assert_frame_equal(filled_alias, filled_base)

    val = _df({"M1": [np.nan, np.nan]})
    filled_base_val = _transform_df(base, val)
    filled_alias_val = _transform_df(alias, val)
    pd.testing.assert_frame_equal(filled_alias_val, filled_base_val)


def test_rolling_median_policy_uses_window():
    df = _df({"M1": [1.0, np.nan, 3.0, np.nan, 5.0]})
    imputer = MGroupImputer(columns=["M1"], policy="rolling_median_k", rolling_window=2)
    filled = _fit_transform_df(imputer, df)
    # 4 行目は過去 [1.0, 3.0] の中央値 (=2.0) で補完される
    expected = np.array([1.0, 1.0, 3.0, 2.0, 5.0])
    np.testing.assert_allclose(filled["M1"].to_numpy(), expected)

    val = _df({"M1": [np.nan, 10.0, np.nan]})
    filled_val = _transform_df(imputer, val)
    # 学習時のデック終端は [3.0, 5.0] なので中央値は 4.0
    assert filled_val.loc[0, "M1"] == 4.0
    assert filled_val.loc[1, "M1"] == 10.0
    # デックを更新すると [5.0, 10.0] となり中央値は 7.5
    assert filled_val.loc[2, "M1"] == 7.5


def test_ema_policy_smoothed_fill():
    df = _df({"M1": [1.0, np.nan, 3.0, np.nan]})
    imputer = MGroupImputer(columns=["M1"], policy="ema_alpha", ema_alpha=0.5)
    filled = _fit_transform_df(imputer, df)
    expected = [1.0, 1.0, 3.0, 2.0]
    np.testing.assert_allclose(filled["M1"].to_numpy(), expected)

    val = _df({"M1": [np.nan, 6.0, np.nan]})
    filled_val = _transform_df(imputer, val)
    # 学習終了時の EMA 推定値は 2.0
    assert filled_val.loc[0, "M1"] == 2.0
    assert filled_val.loc[1, "M1"] == 6.0
    # 新しい EMA は 0.5*6 + 0.5*2 = 4.0 なので次の NaN を 4.0 で補完
    assert filled_val.loc[2, "M1"] == 4.0


def test_ffill_only_respects_missing_trailing():
    df = _df({"M1": [np.nan, 2.0, np.nan]})
    imputer = MGroupImputer(columns=["M1"], policy="ffill_only")
    filled = _fit_transform_df(imputer, df)
    # 先頭の NaN は中央値 (2.0) で補完される
    np.testing.assert_allclose(np.asarray(filled["M1"], dtype=float), np.array([2.0, 2.0, 2.0]))

    val = _df({"M1": [np.nan, np.nan]})
    filled_val = _transform_df(imputer, val)
    np.testing.assert_allclose(np.asarray(filled_val["M1"], dtype=float), np.array([2.0, 2.0]))


def test_rolling_mean_policy():
    df = _df({"M1": [1.0, np.nan, 4.0]})
    imputer = MGroupImputer(columns=["M1"], policy="rolling_mean_k", rolling_window=2)
    filled = _fit_transform_df(imputer, df)
    np.testing.assert_allclose(filled["M1"].to_numpy(), [1.0, 1.0, 4.0])

    val = _df({"M1": [np.nan, np.nan]})
    filled_val = _transform_df(imputer, val)
    # 学習後のデックは [1.0, 4.0] で平均は 2.5
    assert filled_val.loc[0, "M1"] == 2.5
    # 2.5 を追加するとデックは [4.0, 2.5] となり平均は 3.25
    assert filled_val.loc[1, "M1"] == 3.25


def test_linear_interp_policy_simple_gap():
    df = _df({"M1": [1.0, np.nan, 4.0]})
    imputer = MGroupImputer(columns=["M1"], policy="linear_interp")
    filled = _fit_transform_df(imputer, df)
    np.testing.assert_allclose(filled["M1"].to_numpy(), [1.0, 2.5, 4.0])


def test_mask_plus_mean_adds_flag_column():
    df = _df({"M1": [1.0, np.nan, 3.0]})
    imputer = MGroupImputer(columns=["M1"], policy="mask_plus_mean")
    filled = _fit_transform_df(imputer, df)
    assert "M1_missing_flag" in filled.columns
    np.testing.assert_allclose(filled["M1_missing_flag"].to_numpy(), [0.0, 1.0, 0.0])
    assert not np.isnan(filled["M1"].to_numpy(dtype=float, copy=False)).any()
    assert filled["M1_missing_flag"].dtype.kind in {"i", "u", "f"}

    val = _df({"M1": [np.nan, 2.0]})
    filled_val = _transform_df(imputer, val)
    assert "M1_missing_flag" in filled_val.columns
    assert filled_val.loc[0, "M1_missing_flag"] == 1.0
    assert filled_val.loc[1, "M1_missing_flag"] == 0.0


def test_two_stage_ignores_future_validation_observations():
    df = _df({"M1": [1.0, np.nan, 5.0, np.nan]})
    imputer_high = MGroupImputer(columns=["M1"], policy="two_stage", rolling_window=2)
    _fit_transform_df(imputer_high, df)
    val_future_high = _df({"M1": [np.nan, 999.0]})
    filled_high = _transform_df(imputer_high, val_future_high)

    imputer_low = MGroupImputer(columns=["M1"], policy="two_stage", rolling_window=2)
    _fit_transform_df(imputer_low, df)
    val_future_low = _df({"M1": [np.nan, -999.0]})
    filled_low = _transform_df(imputer_low, val_future_low)
    assert filled_high.loc[0, "M1"] == pytest.approx(filled_low.loc[0, "M1"])


def test_knn_policy_uses_standardization_before_imputing():
    df = _df({"M1": [0.0, np.nan, 2.0], "M2": [1000.0, np.nan, 2000.0]})
    imputer = MGroupImputer(columns=["M1", "M2"], policy="knn_k", policy_params={"knn_neighbors": 1})
    _fit_transform_df(imputer, df)
    means = cast(dict, imputer._state_.get("scaler_means"))
    stds = cast(dict, imputer._state_.get("scaler_stds"))
    assert means is not None and pytest.approx(means["M2"]) == 1500.0
    assert stds is not None and stds["M2"] > 0.0
    val = _df({"M1": [np.nan], "M2": [np.nan]})
    filled_val = _transform_df(imputer, val)
    assert not filled_val.isna().to_numpy(dtype=bool, copy=False).any()


def test_knn_policy_fills_missing_values():
    df = _df({
        "M1": [1.0, np.nan, 2.0],
        "M2": [2.0, 2.5, np.nan],
    })
    imputer = MGroupImputer(columns=["M1", "M2"], policy="knn_k", policy_params={"knn_neighbors": 1})
    filled = _fit_transform_df(imputer, df)
    assert not filled[["M1", "M2"]].isna().to_numpy(dtype=bool, copy=False).any()


def test_dow_median_uses_calendar_column():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09"])
    df = pd.DataFrame({"date": dates, "M1": [1.0, np.nan, 5.0]})
    imputer = MGroupImputer(columns=["M1"], policy="dow_median", calendar_column="date")
    filled = _fit_transform_df(imputer, df)
    # 2024-01-02 (火) の欠損は同曜日の中央値 (1.0 と 5.0 -> 3.0)
    assert filled.loc[1, "M1"] == 3.0


def test_two_stage_policy_handles_leading_gaps():
    df = _df({"M1": [np.nan, np.nan, 2.0, np.nan]})
    imputer = MGroupImputer(columns=["M1"], policy="two_stage", rolling_window=2)
    filled = _fit_transform_df(imputer, df)
    assert not np.isnan(filled["M1"].to_numpy(dtype=float, copy=False)).any()


def test_kalman_transform_does_not_use_future():
    pytest.importorskip("statsmodels")
    df = _df({"M1": [1.0, 1.5, 2.0, np.nan, 2.5, 3.0]})
    imputer = MGroupImputer(columns=["M1"], policy="kalman_local_level")
    _fit_transform_df(imputer, df)
    val_high = _df({"M1": [np.nan, 100.0, np.nan]})
    val_low = _df({"M1": [np.nan, -100.0, np.nan]})
    filled_high = _transform_df(imputer, val_high)
    filled_low = _transform_df(imputer, val_low)
    assert filled_high.loc[0, "M1"] == pytest.approx(filled_low.loc[0, "M1"])


def test_arima_transform_does_not_use_future():
    pytest.importorskip("statsmodels")
    df = _df({"M1": [1.0, 1.2, 1.1, 1.3, np.nan, 1.4, 1.5]})
    imputer = MGroupImputer(columns=["M1"], policy="arima_auto", policy_params={"arima_max_p": 1, "arima_max_q": 1})
    _fit_transform_df(imputer, df)
    val_high = _df({"M1": [np.nan, 50.0, np.nan]})
    val_low = _df({"M1": [np.nan, -50.0, np.nan]})
    filled_high = _transform_df(imputer, val_high)
    filled_low = _transform_df(imputer, val_low)
    assert filled_high.loc[0, "M1"] == pytest.approx(filled_low.loc[0, "M1"])
