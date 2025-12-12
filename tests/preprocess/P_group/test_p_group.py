from typing import cast

import numpy as np
import pandas as pd
import pytest

from preprocess.P_group.p_group import PGroupImputer


def _df(values):
    return pd.DataFrame(values)


def _fit_transform_df(imputer: PGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.fit_transform(df.copy()))


def _transform_df(imputer: PGroupImputer, df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, imputer.transform(df.copy()))


def test_auto_column_selection_defaults_to_p_prefix():
    df = _df({
        "P1": [1.0, np.nan, 3.0],
        "P2": [np.nan, 2.0, np.nan],
        "M1": [5.0, 6.0, 7.0],
    })
    imputer = PGroupImputer(policy="ffill_bfill")
    _fit_transform_df(imputer, df)
    assert set(getattr(imputer, "columns_", [])) == {"P1", "P2"}


def test_calendar_required_policy_without_calendar_raises():
    df = _df({
        "P1": [1.0, np.nan, 3.0],
        "date_id": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    imputer = PGroupImputer(policy="dow_median", calendar_column=None)
    with pytest.raises(ValueError):
        imputer.fit(df)


def test_missing_calendar_column_raises_keyerror():
    df = _df({"P1": [1.0, np.nan, 3.0]})
    imputer = PGroupImputer(policy="dow_median", calendar_column="date_id")
    with pytest.raises(KeyError):
        imputer.fit(df)


def test_mad_clip_applies_after_transform():
    df = _df({"P1": [10.0, 11.0, 12.0, 13.0, 14.0]})
    imputer = PGroupImputer(
        columns=["P1"],
        policy="ffill_bfill",
        mad_clip_scale=1.0,
        mad_clip_min_samples=3,
    )
    _fit_transform_df(imputer, df)
    bounds = getattr(imputer, "_clip_bounds_", {})
    assert "P1" in bounds
    low, high = bounds["P1"]
    assert low < high

    val = _df({"P1": [-999.0, 999.0]})
    clipped = _transform_df(imputer, val)
    assert clipped.loc[0, "P1"] == pytest.approx(low)
    assert clipped.loc[1, "P1"] == pytest.approx(high)


def test_mad_clip_falls_back_to_quantiles_when_mad_invalid():
    values = [1.0, 1.0, 1.0, 200.0, 1.0]
    df = _df({"P1": values})
    imputer = PGroupImputer(
        columns=["P1"],
        policy="ffill_bfill",
        mad_clip_scale=2.0,
        mad_clip_min_samples=10,
        fallback_quantile_low=0.1,
        fallback_quantile_high=0.9,
    )
    _fit_transform_df(imputer, df)
    bounds = getattr(imputer, "_clip_bounds_", {})
    assert "P1" in bounds
    low, high = bounds["P1"]
    expected_low = pd.Series(values).quantile(0.1)
    expected_high = pd.Series(values).quantile(0.9)
    assert low == pytest.approx(float(expected_low))
    assert high == pytest.approx(float(expected_high))

    val = _df({"P1": [-1000.0, 1000.0]})
    clipped = _transform_df(imputer, val)
    assert clipped.loc[0, "P1"] == pytest.approx(low)
    assert clipped.loc[1, "P1"] == pytest.approx(high)


def test_mask_plus_mean_generates_prefixed_columns():
    df = _df({"P1": [1.0, np.nan, 3.0]})
    imputer = PGroupImputer(columns=["P1"], policy="mask_plus_mean")
    filled = _fit_transform_df(imputer, df)
    assert "Pmask__P1" in filled.columns
    assert "P1_missing_flag" not in filled.columns
    assert "Pmask__P1" in getattr(imputer, "extra_columns_", [])


def test_calendar_warnings_propagate_into_state():
    df = _df({
        "P1": [1.0, 2.0, 3.0],
        "date_id": ["2024-01-01", "invalid", "2024-01-01"],
    })
    imputer = PGroupImputer(columns=["P1"], policy="ffill_bfill", calendar_column="date_id")
    _fit_transform_df(imputer, df)
    warnings = getattr(imputer, "_state_", {}).get("warnings", [])
    assert any("calendar_column" in str(msg) for msg in warnings)
    mad_state = getattr(imputer, "_state_", {}).get("mad_clip_bounds")
    assert isinstance(mad_state, dict) and "P1" in mad_state