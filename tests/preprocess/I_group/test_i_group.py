from typing import cast

import numpy as np
import pandas as pd
import pytest

from preprocess.I_group.i_group import IGroupImputer


def _df(values):
    return pd.DataFrame(values)


def _fit_transform_df(imputer: IGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.fit_transform(df.copy()))


def _transform_df(imputer: IGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.transform(df.copy()))


def test_auto_column_selection_defaults_to_i_group():
    df = _df({
        "I1": [1.0, np.nan, 3.0],
        "I2": [np.nan, 2.0, np.nan],
        "M1": [5.0, 6.0, 7.0],
    })
    imputer = IGroupImputer(policy="ffill_bfill")
    _fit_transform_df(imputer, df)
    assert set(getattr(imputer, "columns_", [])) == {"I1", "I2"}


def test_calendar_required_policy_without_calendar_raises():
    df = _df({
        "I1": [1.0, np.nan, 3.0],
        "date_id": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    imputer = IGroupImputer(policy="dow_median", calendar_column=None)
    with pytest.raises(ValueError):
        imputer.fit(df)


def test_missing_calendar_column_raises_keyerror():
    df = _df({"I1": [1.0, np.nan, 3.0]})
    imputer = IGroupImputer(policy="dow_median", calendar_column="date_id")
    with pytest.raises(KeyError):
        imputer.fit(df)


def test_quantile_clip_applies_on_transform():
    df = _df({"I1": [0.0, 1.0, 100.0, 2.0, 3.0]})
    imputer = IGroupImputer(columns=["I1"], policy="ffill_bfill", enable_quantile_clip=True)
    _fit_transform_df(imputer, df)
    bounds = getattr(imputer, "_clip_bounds_", {})
    assert "I1" in bounds
    low, high = bounds["I1"]
    assert low <= high

    val = _df({"I1": [999.0, -999.0]})
    clipped = _transform_df(imputer, val)
    assert clipped.loc[0, "I1"] == pytest.approx(high)
    assert clipped.loc[1, "I1"] == pytest.approx(low)


def test_mask_plus_mean_renames_flag_column():
    df = _df({"I1": [1.0, np.nan, 3.0]})
    imputer = IGroupImputer(columns=["I1"], policy="mask_plus_mean")
    filled = _fit_transform_df(imputer, df)
    assert "Imask__I1" in filled.columns
    assert "I1_missing_flag" not in filled.columns
    assert "Imask__I1" in getattr(imputer, "extra_columns_", [])


def test_calendar_warnings_propagate_into_state():
    df = _df({
        "I1": [1.0, 2.0, 3.0],
        "date_id": ["2024-01-01", "not-a-date", "2024-01-01"],
    })
    imputer = IGroupImputer(columns=["I1"], policy="ffill_bfill", calendar_column="date_id")
    _fit_transform_df(imputer, df)
    warnings = getattr(imputer, "_state_", {}).get("warnings", [])
    assert any("calendar_column" in str(msg) for msg in warnings)
