import numpy as np
import pandas as pd
import pytest
from typing import cast

from preprocess.E_group.e_group import DGroupImputer


def _df(values):
    return pd.DataFrame(values)


def _fit_transform_df(imputer: DGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.fit_transform(df.copy()))


def _transform_df(imputer: DGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.transform(df.copy()))


def test_auto_column_detection_uses_d_prefix():
    df = _df({"D1": [1.0, np.nan], "D2": [np.nan, 2.0], "M1": [9.0, 9.0]})
    imputer = DGroupImputer(columns=None, policy="ffill_bfill")
    filled = _fit_transform_df(imputer, df)
    assert "M1" in filled.columns
    assert "D1" in imputer.columns_
    assert "D2" in imputer.columns_
    assert filled.loc[0, "D1"] == 1.0
    assert filled.loc[0, "D2"] == 2.0


def test_ffill_bfill_train_and_transform():
    df = _df({
        "D1": [1.0, np.nan, np.nan, 4.0],
        "D2": [np.nan, 1.0, np.nan, 3.0],
        "E1": [5.0, 6.0, 7.0, 8.0],
    })
    imputer = DGroupImputer(columns=["D1", "D2"], policy="ffill_bfill")
    filled_train = _fit_transform_df(imputer, df)
    expected_train = _df({"D1": [1.0, 1.0, 1.0, 4.0], "D2": [1.0, 1.0, 1.0, 3.0], "E1": [5.0, 6.0, 7.0, 8.0]})
    pd.testing.assert_frame_equal(filled_train, expected_train)
    assert isinstance(imputer._state_.get("warnings"), list)

    val = _df({"D1": [np.nan, 5.0], "D2": [np.nan, np.nan], "E1": [1.0, 1.0]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "D1"] == 4.0
    assert filled_val.loc[0, "D2"] == 3.0
    assert filled_val.loc[1, "D1"] == 5.0
    assert filled_val.loc[1, "D2"] == 3.0


def test_ffill_bfill_alias_matches_primary():
    df = _df({"D1": [np.nan, 1.0, np.nan]})
    base = DGroupImputer(columns=["D1"], policy="ffill_bfill")
    alias = DGroupImputer(columns=["D1"], policy="ffill_train_bfill_in_fit")
    filled_base = _fit_transform_df(base, df)
    filled_alias = _fit_transform_df(alias, df)
    pd.testing.assert_frame_equal(filled_alias, filled_base)

    val = _df({"D1": [np.nan, np.nan]})
    filled_base_val = _transform_df(base, val)
    filled_alias_val = _transform_df(alias, val)
    pd.testing.assert_frame_equal(filled_alias_val, filled_base_val)


def test_rolling_median_policy_uses_window():
    df = _df({"D1": [1.0, np.nan, 3.0, np.nan, 5.0]})
    imputer = DGroupImputer(columns=["D1"], policy="rolling_median_k", rolling_window=2)
    filled = _fit_transform_df(imputer, df)
    expected = np.array([1.0, 1.0, 3.0, 2.0, 5.0])
    np.testing.assert_allclose(filled["D1"].to_numpy(), expected)

    val = _df({"D1": [np.nan, 10.0, np.nan]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "D1"] == 4.0
    assert filled_val.loc[1, "D1"] == 10.0
    assert filled_val.loc[2, "D1"] == 7.5


def test_ema_policy_smoothed_fill():
    df = _df({"D1": [1.0, np.nan, 3.0, np.nan]})
    imputer = DGroupImputer(columns=["D1"], policy="ema_alpha", ema_alpha=0.5)
    filled = _fit_transform_df(imputer, df)
    expected = [1.0, 1.0, 3.0, 2.0]
    np.testing.assert_allclose(filled["D1"].to_numpy(), expected)

    val = _df({"D1": [np.nan, 6.0, np.nan]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "D1"] == 2.0
    assert filled_val.loc[1, "D1"] == 6.0
    assert filled_val.loc[2, "D1"] == 4.0


def test_ffill_only_respects_missing_trailing():
    df = _df({"D1": [np.nan, 2.0, np.nan]})
    imputer = DGroupImputer(columns=["D1"], policy="ffill_only")
    filled = _fit_transform_df(imputer, df)
    np.testing.assert_allclose(np.asarray(filled["D1"], dtype=float), np.array([2.0, 2.0, 2.0]))

    val = _df({"D1": [np.nan, np.nan]})
    filled_val = _transform_df(imputer, val)
    np.testing.assert_allclose(np.asarray(filled_val["D1"], dtype=float), np.array([2.0, 2.0]))


def test_rolling_mean_policy():
    df = _df({"D1": [1.0, np.nan, 4.0]})
    imputer = DGroupImputer(columns=["D1"], policy="rolling_mean_k", rolling_window=2)
    filled = _fit_transform_df(imputer, df)
    np.testing.assert_allclose(filled["D1"].to_numpy(), [1.0, 1.0, 4.0])

    val = _df({"D1": [np.nan, np.nan]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "D1"] == 2.5
    assert filled_val.loc[1, "D1"] == 3.25


def test_linear_interp_policy_simple_gap():
    df = _df({"D1": [1.0, np.nan, 4.0]})
    imputer = DGroupImputer(columns=["D1"], policy="linear_interp")
    filled = _fit_transform_df(imputer, df)
    np.testing.assert_allclose(filled["D1"].to_numpy(), [1.0, 2.5, 4.0])


def test_mask_plus_mean_adds_flag_column():
    df = _df({"D1": [1.0, np.nan, 3.0]})
    imputer = DGroupImputer(columns=["D1"], policy="mask_plus_mean")
    filled = _fit_transform_df(imputer, df)
    assert "Dmask__D1" in filled.columns
    np.testing.assert_allclose(filled["Dmask__D1"].to_numpy(), [0.0, 1.0, 0.0])
    assert not np.isnan(filled["D1"].to_numpy(dtype=float, copy=False)).any()
    assert filled["Dmask__D1"].dtype.kind in {"i", "u", "f"}

    val = _df({"D1": [np.nan, 2.0]})
    filled_val = _transform_df(imputer, val)
    assert "Dmask__D1" in filled_val.columns
    assert filled_val.loc[0, "Dmask__D1"] == 1.0
    assert filled_val.loc[1, "Dmask__D1"] == 0.0


def test_two_stage_ignores_future_validation_observations():
    df = _df({"D1": [1.0, np.nan, 5.0, np.nan]})
    imputer_high = DGroupImputer(columns=["D1"], policy="two_stage", rolling_window=2)
    _fit_transform_df(imputer_high, df)
    val_future_high = _df({"D1": [np.nan, 999.0]})
    filled_high = _transform_df(imputer_high, val_future_high)

    imputer_low = DGroupImputer(columns=["D1"], policy="two_stage", rolling_window=2)
    _fit_transform_df(imputer_low, df)
    val_future_low = _df({"D1": [np.nan, -999.0]})
    filled_low = _transform_df(imputer_low, val_future_low)
    assert filled_high.loc[0, "D1"] == pytest.approx(filled_low.loc[0, "D1"])


def test_knn_policy_uses_standardization_before_imputing():
    df = _df({"D1": [0.0, np.nan, 2.0], "D2": [1000.0, np.nan, 2000.0]})
    imputer = DGroupImputer(columns=["D1", "D2"], policy="knn_k", policy_params={"knn_neighbors": 1})
    _fit_transform_df(imputer, df)
    means = cast(dict, imputer._state_.get("scaler_means"))
    stds = cast(dict, imputer._state_.get("scaler_stds"))
    assert means is not None and pytest.approx(means["D2"]) == 1500.0
    assert stds is not None and stds["D2"] > 0.0
    val = _df({"D1": [np.nan], "D2": [np.nan]})
    filled_val = _transform_df(imputer, val)
    assert not filled_val.isna().to_numpy(dtype=bool, copy=False).any()


def test_knn_policy_fills_missing_values():
    df = _df({
        "D1": [1.0, np.nan, 2.0],
        "D2": [2.0, 2.5, np.nan],
    })
    imputer = DGroupImputer(columns=["D1", "D2"], policy="knn_k", policy_params={"knn_neighbors": 1})
    filled = _fit_transform_df(imputer, df)
    assert not filled[["D1", "D2"]].isna().to_numpy(dtype=bool, copy=False).any()


def test_dow_median_uses_calendar_column():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09"])
    df = pd.DataFrame({"date": dates, "D1": [1.0, np.nan, 5.0]})
    imputer = DGroupImputer(columns=["D1"], policy="dow_median", calendar_column="date")
    filled = _fit_transform_df(imputer, df)
    assert filled.loc[1, "D1"] == 3.0


def test_two_stage_policy_handles_leading_gaps():
    df = _df({"D1": [np.nan, np.nan, 2.0, np.nan]})
    imputer = DGroupImputer(columns=["D1"], policy="two_stage", rolling_window=2)
    filled = _fit_transform_df(imputer, df)
    assert not np.isnan(filled["D1"].to_numpy(dtype=float, copy=False)).any()


def test_kalman_transform_does_not_use_future():
    pytest.importorskip("statsmodels")
    df = _df({
        "date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "D1": [1.0, 1.5, 2.0, np.nan, 2.5, 3.0],
    })
    imputer = DGroupImputer(columns=["D1"], policy="kalman_local_level", calendar_column="date")
    _fit_transform_df(imputer, df)
    val_high = _df({
        "date": pd.date_range("2024-01-07", periods=3, freq="D"),
        "D1": [np.nan, 100.0, np.nan],
    })
    val_low = _df({
        "date": pd.date_range("2024-01-07", periods=3, freq="D"),
        "D1": [np.nan, -100.0, np.nan],
    })
    filled_high = _transform_df(imputer, val_high)
    filled_low = _transform_df(imputer, val_low)
    assert filled_high.loc[0, "D1"] == pytest.approx(filled_low.loc[0, "D1"])


def test_arima_transform_does_not_use_future():
    pytest.importorskip("statsmodels")
    df = _df({
        "date": pd.date_range("2024-01-01", periods=7, freq="D"),
        "D1": [1.0, 1.2, 1.1, 1.3, np.nan, 1.4, 1.5],
    })
    imputer = DGroupImputer(
        columns=["D1"],
        policy="arima_auto",
        policy_params={"arima_max_p": 1, "arima_max_q": 1},
        calendar_column="date",
    )
    _fit_transform_df(imputer, df)
    val_high = _df({
        "date": pd.date_range("2024-01-08", periods=3, freq="D"),
        "D1": [np.nan, 50.0, np.nan],
    })
    val_low = _df({
        "date": pd.date_range("2024-01-08", periods=3, freq="D"),
        "D1": [np.nan, -50.0, np.nan],
    })
    filled_high = _transform_df(imputer, val_high)
    filled_low = _transform_df(imputer, val_low)
    assert filled_high.loc[0, "D1"] == pytest.approx(filled_low.loc[0, "D1"])


def test_calendar_policy_requires_explicit_calendar_column():
    with pytest.raises(ValueError):
        DGroupImputer(columns=["D1"], policy="dow_median")


def test_all_nan_strategy_keep_nan():
    df = _df({"D1": [np.nan, np.nan, np.nan]})
    imputer = DGroupImputer(columns=["D1"], policy="ffill_only", all_nan_strategy="keep_nan")
    filled = _fit_transform_df(imputer, df)
    assert bool(filled["D1"].isna().all())
    val = _df({"D1": [np.nan, np.nan]})
    transformed = _transform_df(imputer, val)
    assert bool(transformed["D1"].isna().all())


def test_all_nan_strategy_fill_constant():
    df = _df({"D1": [np.nan, np.nan, np.nan]})
    imputer = DGroupImputer(columns=["D1"], policy="ffill_only", all_nan_strategy="fill_constant", all_nan_fill=7.5)
    filled = _fit_transform_df(imputer, df)
    assert np.allclose(filled["D1"].to_numpy(dtype=float, copy=False), np.full(3, 7.5))
    val = _df({"D1": [np.nan]})
    transformed = _transform_df(imputer, val)
    assert np.allclose(transformed["D1"].to_numpy(dtype=float, copy=False), np.array([7.5]))
