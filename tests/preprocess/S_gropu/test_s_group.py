import numpy as np
import pandas as pd
import pytest
from typing import cast

from preprocess.S_group.s_group import SGroupImputer


def _df(values):
    return pd.DataFrame(values)


def _fit_transform_df(imputer: SGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.fit_transform(df.copy()))


def _transform_df(imputer: SGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.transform(df.copy()))


def test_auto_column_selection_defaults_to_s_group():
    df = _df({
        "S1": [1.0, np.nan, 3.0],
        "S2": [np.nan, 2.0, np.nan],
        "M1": [5.0, 6.0, 7.0],
    })
    imputer = SGroupImputer(policy="ffill_bfill")
    filled = _fit_transform_df(imputer, df)
    assert set(getattr(imputer, "columns_", [])) == {"S1", "S2"}
    assert "M1" in filled.columns


def test_calendar_required_policy_without_calendar_raises():
    df = _df({"S1": [1.0, np.nan, 3.0]})
    imputer = SGroupImputer(columns=["S1"], policy="dow_median")
    with pytest.raises(ValueError):
        imputer.fit(df)


def test_missing_calendar_column_raises_keyerror():
    df = _df({"S1": [1.0, np.nan, 3.0]})
    imputer = SGroupImputer(columns=["S1"], policy="dow_median", calendar_column="date_id")
    with pytest.raises(KeyError):
        imputer.fit(df)


def test_mask_plus_mean_renames_flag_column():
    df = _df({"S1": [1.0, np.nan, 3.0]})
    imputer = SGroupImputer(columns=["S1"], policy="mask_plus_mean")
    filled = _fit_transform_df(imputer, df)
    assert "Smask__S1" in filled.columns
    assert "S1_missing_flag" not in filled.columns
    assert "Smask__S1" in getattr(imputer, "extra_columns_", [])

    val = _df({"S1": [np.nan, 2.0]})
    filled_val = _transform_df(imputer, val)
    assert filled_val.loc[0, "Smask__S1"] == 1.0
    assert filled_val.loc[1, "Smask__S1"] == 0.0


def test_mad_clip_applies_during_transform():
    df = _df({"S1": [0.0, 1.0, 2.0, 3.0]})
    imputer = SGroupImputer(
        columns=["S1"],
        policy="ffill_bfill",
        mad_clip_scale=1.0,
        mad_clip_min_samples=1,
        enable_mad_clip=True,
    )
    _fit_transform_df(imputer, df)

    val = _df({"S1": [-10.0, 100.0]})
    clipped = _transform_df(imputer, val)
    assert clipped.loc[0, "S1"] == pytest.approx(0.5)
    assert clipped.loc[1, "S1"] == pytest.approx(2.5)

    bounds = getattr(imputer, "_state_", {}).get("mad_clip_bounds", {})
    low, high = bounds.get("S1", (None, None))
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(2.5)


def test_mad_clip_can_be_disabled():
    df = _df({"S1": [0.0, 1.0, 2.0, 3.0]})
    imputer = SGroupImputer(columns=["S1"], policy="ffill_bfill", enable_mad_clip=False)
    _fit_transform_df(imputer, df)

    val = _df({"S1": [-10.0, 100.0]})
    transformed = _transform_df(imputer, val)
    np.testing.assert_allclose(transformed["S1"].to_numpy(dtype=float, copy=False), [-10.0, 100.0])

    bounds = getattr(imputer, "_state_", {}).get("mad_clip_bounds", {})
    assert "S1" not in bounds
