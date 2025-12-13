from typing import cast

import numpy as np
import pandas as pd
import pytest

from preprocess.V_group.v_group import VGroupImputer


def _df(values):
    return pd.DataFrame(values)


def _fit_transform_df(imputer: VGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.fit_transform(df.copy()))


def _transform_df(imputer: VGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.transform(df.copy()))


def test_auto_column_selection_defaults_to_v_group():
    df = _df({
        "V1": [1.0, np.nan, 3.0],
        "V2": [np.nan, 2.0, np.nan],
        "M1": [5.0, 6.0, 7.0],
    })
    imputer = VGroupImputer(policy="ffill_bfill")
    filled = _fit_transform_df(imputer, df)
    assert set(getattr(imputer, "columns_", [])) == {"V1", "V2"}
    assert "M1" in filled.columns


def test_calendar_required_policy_without_calendar_raises():
    df = _df({"V1": [1.0, np.nan, 3.0]})
    imputer = VGroupImputer(columns=["V1"], policy="dow_median")
    with pytest.raises(ValueError):
        imputer.fit(df)


def test_missing_calendar_column_raises_keyerror():
    df = _df({"V1": [1.0, np.nan, 3.0]})
    imputer = VGroupImputer(columns=["V1"], policy="dow_median", calendar_column="date_id")
    with pytest.raises(KeyError):
        imputer.fit(df)


def test_mask_plus_mean_renames_flag_column():
    df = _df({"V1": [1.0, np.nan, 3.0]})
    imputer = VGroupImputer(columns=["V1"], policy="mask_plus_mean")
    filled = _fit_transform_df(imputer, df)
    assert "Vmask__V1" in filled.columns
    assert "V1_missing_flag" not in filled.columns
    assert "Vmask__V1" in getattr(imputer, "extra_columns_", [])

    val = _df({"V1": [np.nan, 2.0]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "Vmask__V1"] == 1.0
    assert filled_val.loc[1, "Vmask__V1"] == 0.0


def test_quantile_clip_and_log_transform_apply_during_transform():
    df = _df({"V1": [0.0, 1.0, 2.0, 3.0, 4.0]})
    imputer = VGroupImputer(
        columns=["V1"],
        policy="ffill_bfill",
        clip_quantile_low=0.25,
        clip_quantile_high=0.75,
        enable_quantile_clip=True,
        log_transform=True,
    )
    _fit_transform_df(imputer, df)

    bounds = getattr(imputer, "_clip_bounds_", {})
    low, high = bounds.get("V1", (None, None))
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(3.0)
    offsets = getattr(imputer, "_log_offsets_", {})
    assert offsets.get("V1", 0.0) == pytest.approx(0.0, abs=1e-9)

    val = _df({"V1": [-10.0, 10.0]})
    transformed = _transform_df(imputer, val)
    expected = np.log1p(np.array([1.0, 3.0], dtype=float))
    np.testing.assert_allclose(transformed["V1"].to_numpy(dtype=float, copy=False), expected, rtol=1e-5)


def test_disable_quantile_clip_and_log_transform_preserves_values():
    df = _df({"V1": [0.0, 1.0, 2.0]})
    imputer = VGroupImputer(
        columns=["V1"],
        policy="ffill_bfill",
        enable_quantile_clip=False,
        log_transform=False,
    )
    _fit_transform_df(imputer, df)
    val = _df({"V1": [-5.0, 5.0]})
    transformed = _transform_df(imputer, val)
    np.testing.assert_allclose(transformed["V1"].to_numpy(dtype=float, copy=False), [-5.0, 5.0])
    assert getattr(imputer, "_clip_bounds_", {}) == {}
    assert getattr(imputer, "_log_offsets_", {}) == {}


def test_negative_values_generate_positive_log_offsets():
    df = _df({"V1": [-2.0, -1.0, 0.0]})
    imputer = VGroupImputer(
        columns=["V1"],
        policy="ffill_bfill",
        enable_quantile_clip=False,
        log_transform=True,
        log_offset_epsilon=1e-6,
    )
    _fit_transform_df(imputer, df)

    offsets = getattr(imputer, "_log_offsets_", {})
    assert offsets.get("V1") == pytest.approx(2.000001, rel=1e-6)

    val = _df({"V1": [-3.0, 1.0]})
    transformed = _transform_df(imputer, val)
    expected = np.array([0.0, np.log1p(1.0 + offsets["V1"])], dtype=float)
    np.testing.assert_allclose(transformed["V1"].to_numpy(dtype=float, copy=False), expected, rtol=1e-5)
