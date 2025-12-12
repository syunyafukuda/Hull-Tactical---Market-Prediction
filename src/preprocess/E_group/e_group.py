from __future__ import annotations

import math
from typing import Any, Hashable, Iterable, List, Mapping, cast

import numpy as np
import pandas as pd

from preprocess.M_group.m_group import MGroupImputer as _BaseImputer


class EGroupImputer(_BaseImputer):
    """Imputer tailored for E-group features leveraging the M-group policies."""

    CALENDAR_REQUIRED_POLICIES = {
        "dow_median",
        "dom_median",
        "month_median",
        "holiday_bridge",
        "time_interp",
        "kalman_local_level",
        "state_space_custom",
        "arima_auto",
    }

    def __init__(
        self,
        columns: Iterable[Hashable] | None = None,
        policy: str = "ffill_bfill",
        rolling_window: int = 5,
        ema_alpha: float = 0.3,
        calendar_column: str | None = None,
        policy_params: Mapping[str, Any] | None = None,
        random_state: int = 42,
        all_nan_strategy: str = "keep_nan",
        all_nan_fill: float = 0.0,
    ) -> None:
        self._user_calendar_column = calendar_column
        self.all_nan_strategy = all_nan_strategy
        strategy_choices = {"keep_nan", "fill_zero", "fill_constant"}
        if self.all_nan_strategy not in strategy_choices:
            raise ValueError(f"all_nan_strategy must be one of {sorted(strategy_choices)}")
        self.all_nan_fill = float(all_nan_fill)
        self.all_nan_fill_value_ = float(all_nan_fill)
        self.all_nan_columns_: List[str] = []
        self._prefit_warnings: List[str] = []
        super().__init__(
            columns=columns,
            policy=policy,
            rolling_window=rolling_window,
            ema_alpha=ema_alpha,
            calendar_column=calendar_column,
            policy_params=policy_params,
            random_state=random_state,
        )
        if self.policy in self.CALENDAR_REQUIRED_POLICIES and self._user_calendar_column is None:
            raise ValueError(
                f"Policy '{self.policy_requested}' requires calendar_column to be provided explicitly."
            )

    def fit(self, X: pd.DataFrame, y: Any = None):  # type: ignore[override]
        frame = self._ensure_dataframe(X).copy()
        if self.columns is None:
            selected = [c for c in frame.columns if isinstance(c, str) and c.startswith("E")]
            self.columns = selected

        if self.columns is None:
            self.columns = []

        numeric_cols = []
        for col in list(self.columns):
            if col not in frame.columns:
                continue
            frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce")
            numeric_cols.append(col)
        self.columns = numeric_cols

        all_nan_columns: List[str] = []
        for col in self.columns:
            series = cast(pd.Series, frame[col])
            if series.isna().all():
                all_nan_columns.append(str(col))
        self.all_nan_columns_ = all_nan_columns
        self._prefit_warnings = []
        if self.policy in self.CALENDAR_REQUIRED_POLICIES:
            calendar_col = self._user_calendar_column or self.calendar_column
            if calendar_col is None:
                raise ValueError(
                    f"Policy '{self.policy_requested}' requires calendar_column but none was supplied."
                )
            if calendar_col not in frame.columns:
                raise ValueError(f"Calendar column '{calendar_col}' not found in training frame.")
            calendar_series = pd.to_datetime(frame[calendar_col], errors="coerce")
            if calendar_series.isna().any():
                self._prefit_warnings.append("calendar_column_contains_non_parseable_values")
            if calendar_series.duplicated().any():
                self._prefit_warnings.append("calendar_column_contains_duplicates")

        fitted = super().fit(frame, y)

        if self._prefit_warnings:
            for msg in self._prefit_warnings:
                self._record_warning(msg)

        extra_columns = getattr(self, "extra_columns_", None)
        if isinstance(extra_columns, list) and extra_columns:
            rename_map: dict[str, str] = {}
            for col in extra_columns:
                rename_map[col] = self._rename_generated_column(col)
            if rename_map:
                self._train_filled_ = self._train_filled_.rename(columns=rename_map)
                renamed_extra: List[str] = [str(rename_map.get(col, col)) for col in extra_columns]
                self.extra_columns_ = renamed_extra
                output_cols = list(getattr(self, "_output_columns_", []))
                self._output_columns_ = [str(rename_map.get(col, col)) for col in output_cols]
                state = getattr(self, "_state_", {})
                if isinstance(state, dict):
                    for key, value in list(state.items()):
                        if isinstance(value, pd.DataFrame):
                            state[key] = value.rename(columns=rename_map)
                    mask_map = state.get("mask_map")
                    if isinstance(mask_map, dict):
                        state["mask_map"] = {k: rename_map.get(v, v) for k, v in mask_map.items()}

        if self.all_nan_columns_:
            fill_value: float | None
            if self.all_nan_strategy == "keep_nan":
                fill_value = np.nan
            elif self.all_nan_strategy == "fill_zero":
                fill_value = 0.0
            else:  # fill_constant
                fill_value = self.all_nan_fill_value_

            for col in self.all_nan_columns_:
                if col in self._train_filled_.columns:
                    if fill_value is None:
                        self._train_filled_.loc[:, col] = np.nan
                    elif isinstance(fill_value, float) and math.isnan(fill_value):
                        self._train_filled_.loc[:, col] = np.nan
                    else:
                        self._train_filled_.loc[:, col] = float(fill_value)
            state = getattr(self, "_state_", {})
            if isinstance(state, dict):
                state["all_nan_columns"] = list(self.all_nan_columns_)
                state["all_nan_strategy"] = self.all_nan_strategy
                state["all_nan_fill_value"] = float(self.all_nan_fill_value_)
        return fitted

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        transformed = super().transform(X)
        if self.all_nan_columns_:
            if self.all_nan_strategy == "keep_nan":
                fill_value = np.nan
            elif self.all_nan_strategy == "fill_zero":
                fill_value = 0.0
            else:
                fill_value = self.all_nan_fill_value_
            for col in self.all_nan_columns_:
                if col in transformed.columns:
                    if fill_value is None:
                        transformed.loc[:, col] = np.nan
                    elif isinstance(fill_value, float) and math.isnan(fill_value):
                        transformed.loc[:, col] = np.nan
                    else:
                        transformed.loc[:, col] = float(fill_value)
        return transformed

    def _rename_generated_column(self, name: str) -> str:
        if name.endswith("_missing_flag"):
            base = name[: -len("_missing_flag")]
            return f"Emask__{base}"
        if name.startswith("E__") or name.startswith("Emask__"):
            return name
        return f"E__{name}"
