from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import (
    Any,
    Deque,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import Ridge

try:  # optional dependency
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.structural import UnobservedComponents
except ImportError:  # pragma: no cover - statsmodels optional at runtime
    ARIMA = None  # type: ignore[assignment]
    UnobservedComponents = None  # type: ignore[assignment]


class MGroupImputer(TransformerMixin, BaseEstimator):
    """時間情報を考慮して M 系特徴量の欠損を補完する推定器。

    Parameters
    ----------
    columns:
        対象とする列。指定しない場合は ``fit`` 時に ``"M"`` で始まる列を自動選択。
    policy:
        採用する補完ポリシー。 ``SUPPORTED_POLICIES`` を参照。
    rolling_window:
        ローリング系ポリシーで用いる窓幅。
    ema_alpha:
        指数移動平均ポリシーで用いる平滑化係数。
    calendar_column:
        曜日や月次など季節性ポリシーが参照する日時列名。指定しない場合はポリシー内で自動探索。
    policy_params:
        ポリシー固有のハイパーパラメータを指定する辞書。キーは文字列、値は数値または文字列。
    random_state:
        多変量モデルを利用するポリシーの乱数シード。

    Notes
    -----
        - **重要:** ``ffill_bfill`` は学習時のみ後方補完を併用し、 ``transform`` では前方方向の値だけで補完します。
            エイリアス ``ffill_train_bfill_in_fit`` でも指定可能で、推論時の未来参照を防ぎつつ学習時には末尾値をウォームスタートできます。
        - ``kalman_*`` と ``arima_auto`` ポリシーは statsmodels の ``fittedvalues``（フィルタによる一歩先推定）だけを使用し、将来の平滑値を参照しません。
    """

    _BASE_POLICIES: Tuple[str, ...] = (
        "ffill_bfill",
        "ffill_only",
        "rolling_median_k",
        "rolling_mean_k",
        "ema_alpha",
        "linear_interp",
        "spline_interp_deg",
        "time_interp",
        "backfill_robust",
        "winsorized_median_k",
        "quantile_fill",
        "dow_median",
        "dom_median",
        "month_median",
        "holiday_bridge",
        "knn_k",
        "pca_reconstruct_r",
        "mice",
        "missforest",
        "ridge_stack",
        "kalman_local_level",
        "arima_auto",
        "state_space_custom",
        "mask_plus_mean",
        "two_stage",
    )
    POLICY_ALIASES: Dict[str, str] = {
        "ffill_train_bfill_in_fit": "ffill_bfill",
    }
    SUPPORTED_POLICIES: Tuple[str, ...] = _BASE_POLICIES + tuple(POLICY_ALIASES.keys())

    def __init__(
        self,
        columns: Iterable[Hashable] | None = None,
        policy: str = "ffill_bfill",
        rolling_window: int = 5,
        ema_alpha: float = 0.3,
        calendar_column: str | None = None,
        policy_params: Mapping[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.columns = columns
        resolved_policy = self.POLICY_ALIASES.get(policy, policy)
        self.policy_requested = policy
        self.policy = resolved_policy
        self.rolling_window = int(rolling_window)
        self.ema_alpha = float(ema_alpha)
        self.calendar_column = calendar_column
        self.policy_params = policy_params
        self._policy_params: Dict[str, Any] = dict(policy_params) if policy_params is not None else {}
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        X_df = self._ensure_dataframe(X).copy()
        if self.columns is None:
            cols: List[str] = [c for c in X_df.columns if isinstance(c, str) and c.startswith("M")]
        else:
            cols = [c for c in self.columns if isinstance(c, str) and c in X_df.columns]
        self.columns_ = cols
        self.extra_columns_: List[str] = []
        if not self.columns_:
            self._train_index = X_df.index.copy()
            self._train_filled_ = pd.DataFrame(index=X_df.index.copy())
            self._state_ = {}
            self._calendar_fit_values_ = None
            self._output_columns_ = []
            self._medians_dict_ = {}
            return self

        if self.policy not in self.SUPPORTED_POLICIES:
            raise ValueError(f"Unsupported policy '{self.policy}'. Supported: {list(self.SUPPORTED_POLICIES)}")
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")

        calendar_series = self._extract_calendar_series(X_df)
        self._calendar_column_name_ = calendar_series.name if calendar_series is not None else None
        self._calendar_fit_values_ = calendar_series.copy() if calendar_series is not None else None

        data = cast(pd.DataFrame, X_df.loc[:, self.columns_].copy())
        medians_series = data.median(numeric_only=True)
        if not isinstance(medians_series, pd.Series):  # pragma: no cover - defensive
            raise TypeError("Expected pandas Series from DataFrame.median().")
        medians_series = medians_series.reindex(self.columns_)
        medians_series = medians_series.fillna(0.0)
        medians_dict: Dict[Hashable, float] = {}
        for idx_label in medians_series.index:
            idx_position = cast(int, medians_series.index.get_loc(idx_label))
            medians_dict[idx_label] = float(cast(float, medians_series.iloc[idx_position]))
        self._medians_dict_ = medians_dict

        if self.policy == "ffill_bfill":
            filled, state = self._fit_ffill(data, medians_dict, use_bfill=True)
        elif self.policy == "ffill_only":
            filled, state = self._fit_ffill(data, medians_dict, use_bfill=False)
        elif self.policy in {"rolling_median_k", "rolling_mean_k"}:
            filled, state = self._fit_rolling(
                data,
                use_median=self.policy == "rolling_median_k",
                medians_lookup=medians_dict,
            )
            state["medians"] = medians_dict
        elif self.policy == "ema_alpha":
            filled, state = self._fit_ema(data, medians_lookup=medians_dict)
        elif self.policy == "linear_interp":
            filled, state = self._fit_linear_interp(data, medians_dict)
        elif self.policy == "spline_interp_deg":
            filled, state = self._fit_spline_interp(data, medians_dict)
        elif self.policy == "time_interp":
            filled, state = self._fit_time_interp(data, medians_dict, calendar_series)
        elif self.policy == "backfill_robust":
            filled, state = self._fit_backfill_robust(data, medians_dict)
        elif self.policy == "winsorized_median_k":
            filled, state = self._fit_winsorized_median(data, medians_dict)
        elif self.policy == "quantile_fill":
            filled, state = self._fit_quantile_fill(data, medians_dict)
        elif self.policy in {"dow_median", "dom_median", "month_median"}:
            filled, state = self._fit_seasonal_median(data, medians_dict, calendar_series)
        elif self.policy == "holiday_bridge":
            filled, state = self._fit_holiday_bridge(data, medians_dict, calendar_series)
        elif self.policy == "knn_k":
            filled, state = self._fit_knn(data)
        elif self.policy == "pca_reconstruct_r":
            filled, state = self._fit_pca_reconstruct(data, medians_dict)
        elif self.policy == "mice":
            filled, state = self._fit_mice(data)
        elif self.policy == "missforest":
            filled, state = self._fit_missforest(data)
        elif self.policy == "ridge_stack":
            filled, state = self._fit_ridge_stack(data, medians_dict)
        elif self.policy == "kalman_local_level":
            filled, state = self._fit_kalman(data, medians_dict, level_only=True)
        elif self.policy == "arima_auto":
            filled, state = self._fit_arima_auto(data, medians_dict)
        elif self.policy == "state_space_custom":
            filled, state = self._fit_kalman(data, medians_dict, level_only=False)
        elif self.policy == "mask_plus_mean":
            filled, state = self._fit_mask_plus_mean(data, medians_dict)
        elif self.policy == "two_stage":
            filled, state = self._fit_two_stage(data, medians_dict)
        else:  # pragma: no cover - safeguarded above
            raise ValueError(self.policy)

        state = self._prepare_state(state)
        self.extra_columns_ = [c for c in filled.columns if c not in self.columns_]
        self._train_index = X_df.index.copy()
        self._train_filled_ = filled
        self._state_ = state
        self._output_columns_ = list(filled.columns)
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        self._validate_fitted()
        X_df = self._ensure_dataframe(X)
        if not self.columns_:
            return X_df
        df = X_df.copy()
        if df.index.equals(self._train_index):
            for col in self._output_columns_:
                if col in df.columns:
                    df.loc[:, col] = self._train_filled_.loc[:, col].values
                else:
                    df[col] = self._train_filled_.loc[:, col].values
            return df

        subset = cast(pd.DataFrame, df.loc[:, self.columns_])
        expect_calendar = bool(getattr(self, "_calendar_column_name_", None))
        calendar_series = self._extract_calendar_series(df, expect_existing=expect_calendar)

        if self.policy == "ffill_bfill":
            filled = self._transform_ffill(subset, use_bfill=True)
        elif self.policy == "ffill_only":
            filled = self._transform_ffill(subset, use_bfill=False)
        elif self.policy in {"rolling_median_k", "rolling_mean_k"}:
            filled = self._transform_rolling(subset, use_median=self.policy == "rolling_median_k")
        elif self.policy == "ema_alpha":
            filled = self._transform_ema(subset)
        elif self.policy == "linear_interp":
            filled = self._transform_linear_interp(subset)
        elif self.policy == "spline_interp_deg":
            filled = self._transform_spline_interp(subset)
        elif self.policy == "time_interp":
            filled = self._transform_time_interp(subset, calendar_series)
        elif self.policy == "backfill_robust":
            filled = self._transform_backfill_robust(subset)
        elif self.policy == "winsorized_median_k":
            filled = self._transform_winsorized_median(subset)
        elif self.policy == "quantile_fill":
            filled = self._transform_quantile_fill(subset)
        elif self.policy in {"dow_median", "dom_median", "month_median"}:
            filled = self._transform_seasonal_median(subset, calendar_series)
        elif self.policy == "holiday_bridge":
            filled = self._transform_holiday_bridge(subset, calendar_series)
        elif self.policy == "knn_k":
            filled = self._transform_knn(subset)
        elif self.policy == "pca_reconstruct_r":
            filled = self._transform_pca_reconstruct(subset)
        elif self.policy == "mice":
            filled = self._transform_mice(subset)
        elif self.policy == "missforest":
            filled = self._transform_missforest(subset)
        elif self.policy == "ridge_stack":
            filled = self._transform_ridge_stack(subset)
        elif self.policy == "kalman_local_level":
            filled = self._transform_kalman(subset, calendar_series)
        elif self.policy == "arima_auto":
            filled = self._transform_arima_auto(subset)
        elif self.policy == "state_space_custom":
            filled = self._transform_kalman(subset, calendar_series)
        elif self.policy == "mask_plus_mean":
            filled = self._transform_mask_plus_mean(subset)
        elif self.policy == "two_stage":
            filled = self._transform_two_stage(subset)
        else:  # pragma: no cover - safeguarded in fit
            raise ValueError(self.policy)

        for col in self.columns_:
            df.loc[:, col] = filled.loc[:, col].values
        for col in self.extra_columns_:
            df.loc[:, col] = filled.loc[:, col].values
        return df

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        if hasattr(self, "_output_columns_") and self._output_columns_:
            return np.array(self._output_columns_)
        return np.array(self.columns_)

    # ------------------------------------------------------------------
    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("MGroupImputer expects a pandas DataFrame as input.")

    @staticmethod
    def _ensure_warning_list(state: Dict[str, Any]) -> List[str]:
        warnings = state.get("warnings")
        if isinstance(warnings, list):
            return warnings
        warnings_list: List[str] = []
        state["warnings"] = warnings_list
        return warnings_list

    def _prepare_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_warning_list(state)
        return state

    def _record_warning(self, message: str) -> None:
        if not hasattr(self, "_state_"):
            return
        warnings = self._state_.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append(message)
        else:  # pragma: no cover - defensive
            self._state_["warnings"] = [message]

    @staticmethod
    def _effective_history(frame: pd.DataFrame, requested_len: int) -> tuple[int, pd.DataFrame]:
        if requested_len <= 0 or frame.empty:
            return 0, frame.iloc[0:0].copy()
        effective = min(requested_len, len(frame))
        return effective, frame.tail(effective).copy()

    def _empty_tail_frame(self) -> pd.DataFrame:
        if not hasattr(self, "columns_") or not self.columns_:
            return pd.DataFrame()
        return pd.DataFrame({col: pd.Series(dtype=float) for col in self.columns_})

    def _state_tail_frame(self) -> pd.DataFrame:
        stored = self._state_.get("tail") if hasattr(self, "_state_") else None
        if isinstance(stored, pd.DataFrame):
            return stored
        return self._empty_tail_frame()

    @staticmethod
    def _deque_median(values: Sequence[float]) -> float:
        seq = list(values)
        if not seq:
            return float("nan")
        arr = np.asarray(seq, dtype=float)
        return float(np.median(arr))

    def _validate_fitted(self):
        if not hasattr(self, "_state_"):
            raise AttributeError("MGroupImputer is not fitted.")

    # ローリング補完用の内部処理 -------------------------------------------
    def _fit_rolling(self, data: pd.DataFrame, *, use_median: bool, medians_lookup: dict[Hashable, float]):
        filled = data.copy()
        deques: dict[str, Deque[float]] = {col: deque(maxlen=self.rolling_window) for col in self.columns_}
        for index_label in filled.index:
            for col in self.columns_:
                original_val = data.at[index_label, col]
                if pd.isna(original_val):
                    dq = deques[col]
                    if dq:
                        if use_median:
                            fill_value = self._deque_median(dq)
                        else:
                            fill_value = float(np.mean(dq))
                            if np.isfinite(fill_value):
                                dq.append(fill_value)
                    else:
                        lookup = medians_lookup.get(col)
                        fill_value = float(lookup) if lookup is not None else float("nan")
                        if not use_median and np.isfinite(fill_value):
                            dq.append(fill_value)
                    filled.at[index_label, col] = fill_value
                else:
                    fill_value = float(cast(float, original_val))
                    filled.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
        state = {
            "deques": {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_},
            "use_median": use_median,
        }
        return filled, state

    def _transform_rolling(self, data: pd.DataFrame, *, use_median: bool):
        filled = data.copy()
        state = cast(dict[str, Deque[float]], deepcopy(self._state_["deques"]))
        medians_lookup = cast(dict[Hashable, float], self._state_["medians"])
        for index_label in filled.index:
            for col in self.columns_:
                original_val = data.at[index_label, col]
                dq = state[col]
                if pd.isna(original_val):
                    if dq:
                        fill_value = self._deque_median(dq) if use_median else float(np.mean(dq))
                    else:
                        lookup = medians_lookup.get(col)
                        fill_value = float(lookup) if lookup is not None else float("nan")
                    filled.at[index_label, col] = fill_value
                    if not use_median:
                        dq.append(fill_value)
                else:
                    fill_value = float(cast(float, original_val))
                    filled.at[index_label, col] = fill_value
                    dq.append(fill_value)
        return filled

    # EMA 補完用の内部処理 ------------------------------------------------
    def _fit_ema(self, data: pd.DataFrame, *, medians_lookup: dict[Hashable, float]):
        filled = data.copy()
        ema_state: dict[str, float | None] = {col: None for col in self.columns_}
        for index_label in filled.index:
            for col in self.columns_:
                val = filled.at[index_label, col]
                ema_val = ema_state[col]
                if pd.isna(val):
                    if ema_val is not None:
                        fill_value = ema_val
                    else:
                        lookup = medians_lookup.get(col)
                        fill_value = float(lookup) if lookup is not None else float("nan")
                    filled.at[index_label, col] = fill_value
                    ema_val = float(fill_value)
                else:
                    val_float = float(cast(float, val))
                    ema_val = val_float if ema_val is None else self.ema_alpha * val_float + (1 - self.ema_alpha) * ema_val
                    filled.at[index_label, col] = val_float
                ema_state[col] = ema_val
        state = {
            "ema": ema_state,
            "medians": medians_lookup.copy(),
        }
        return filled, state

    def _transform_ema(self, data: pd.DataFrame):
        filled = data.copy()
        ema_state = cast(dict[str, float | None], deepcopy(self._state_["ema"]))
        medians_lookup = cast(dict[Hashable, float], self._state_["medians"])
        for index_label in filled.index:
            for col in self.columns_:
                val = filled.at[index_label, col]
                ema_val = ema_state[col]
                if pd.isna(val):
                    if ema_val is not None:
                        fill_value = ema_val
                    else:
                        lookup = medians_lookup.get(col)
                        fill_value = float(lookup) if lookup is not None else float("nan")
                    filled.at[index_label, col] = fill_value
                    ema_val = float(fill_value)
                else:
                    val_float = float(cast(float, val))
                    ema_val = val_float if ema_val is None else self.ema_alpha * val_float + (1 - self.ema_alpha) * ema_val
                    filled.at[index_label, col] = val_float
                ema_state[col] = ema_val
        return filled

    # 共通ユーティリティ --------------------------------------------------
    def _extract_calendar_series(self, X_df: pd.DataFrame, expect_existing: bool = False) -> pd.Series | None:
        """ポリシーが必要とする場合に日時列を抽出する。"""
        col_name = getattr(self, "_calendar_column_name_", None)
        if col_name is not None and col_name in X_df.columns:
            return cast(pd.Series, X_df[col_name])
        if col_name is not None and col_name not in X_df.columns:
            if expect_existing:
                raise KeyError(
                    f"Calendar column '{col_name}' required by policy '{self.policy}' is missing in transform input."
                )
            return None

        if self.calendar_column and self.calendar_column in X_df.columns:
            self._calendar_column_name_ = self.calendar_column
            return cast(pd.Series, X_df[self.calendar_column])
        if self.calendar_column and self.calendar_column not in X_df.columns:
            if expect_existing:
                raise KeyError(
                    f"Calendar column '{self.calendar_column}' not found in input DataFrame for policy '{self.policy}'."
                )
            return None

        for candidate in ("date", "date_id", "timestamp", "datetime", "evaluation_date"):
            if candidate in X_df.columns:
                self._calendar_column_name_ = candidate
                return cast(pd.Series, X_df[candidate])

        if expect_existing:
            raise KeyError(
                "Calendar column is required for the selected policy. Provide calendar_column explicitly or include a compatible column in the input."
            )
        return None

    def _get_policy_param(self, key: str, default: Any) -> Any:
        return self._policy_params.get(key, default)

    def _history_length(self) -> int:
        raw = int(self._get_policy_param("history_length", max(self.rolling_window, 2)))
        return max(2, raw)

    def _compute_scaler_stats(self, data: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for col in self.columns_:
            series = data[col].astype(float)
            values = series.to_numpy()
            with np.errstate(all="ignore"):
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values))
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
            means[col] = mean
            stds[col] = std
        return means, stds

    def _standardize_with_stats(
        self,
        data: pd.DataFrame,
        means: Mapping[str, float],
        stds: Mapping[str, float],
    ) -> pd.DataFrame:
        scaled = data.astype(float).copy()
        for col in self.columns_:
            mean = float(means.get(col, 0.0))
            std = float(stds.get(col, 1.0))
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
            scaled[col] = (scaled[col] - mean) / std
        return scaled

    def _destandardize_with_stats(
        self,
        data: pd.DataFrame,
        means: Mapping[str, float],
        stds: Mapping[str, float],
    ) -> pd.DataFrame:
        restored = data.astype(float).copy()
        for col in self.columns_:
            mean = float(means.get(col, 0.0))
            std = float(stds.get(col, 1.0))
            restored[col] = restored[col] * std + mean
        return restored

    @staticmethod
    def _series_any(series: pd.Series) -> bool:
        """Return whether a boolean Series contains any True values."""
        return bool(np.any(series.to_numpy(dtype=bool, copy=False)))

    def _calendar_to_datetime(self, series: pd.Series | None) -> pd.Series | None:
        if series is None:
            return None
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        origin = self._get_policy_param("calendar_origin", None)
        unit = self._get_policy_param("calendar_unit", "D")
        try:
            if origin is not None:
                return pd.to_datetime(series, unit=unit, origin=origin)
            return pd.to_datetime(series)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to convert calendar column to datetime. Consider specifying calendar_origin and calendar_unit.") from exc

    # フィル前後処理 -------------------------------------------------------
    def _fit_ffill(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float], *, use_bfill: bool):
        filled = data.ffill()
        if use_bfill:
            filled = filled.bfill()
        filled = filled.fillna(pd.Series(medians_lookup))

        last_values: Dict[str, float] = {}
        for col in self.columns_:
            series = filled[col].dropna()
            if not series.empty:
                last_values[col] = float(series.iloc[-1])
        state = {
            "last": last_values,
            "medians": medians_lookup,
            "use_bfill": use_bfill,
        }
        return filled, state

    def _transform_ffill(self, data: pd.DataFrame, *, use_bfill: bool):
        """Apply forward-only fills during transform (``use_bfill`` kept for compatibility)."""
        filled = data.ffill()
        last_values = cast(Dict[str, float], self._state_.get("last", {}))
        if last_values:
            filled = filled.fillna(last_values)
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        filled = filled.fillna(pd.Series(medians_lookup))
        return filled

    # 線形/スプライン/時間補間 --------------------------------------------
    def _fit_linear_interp(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        requested_history = self._history_length()
        filled = data.interpolate(method="linear", limit_direction="forward")
        filled = filled.ffill()
        filled = filled.fillna(pd.Series(medians_lookup))
        history_len, tail = self._effective_history(filled, requested_history)
        state = {
            "history_len": history_len,
            "tail": tail,
            "medians": medians_lookup,
        }
        return filled, state

    def _transform_linear_interp(self, data: pd.DataFrame):
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        tail = self._state_tail_frame()
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in linear_interp: tail={len(tail)} expected={history_len}")
        combined = pd.concat([tail, data], axis=0)
        combined = combined.interpolate(method="linear", limit_direction="forward")
        combined = combined.ffill()
        combined = combined.fillna(pd.Series(medians_lookup))
        result = combined.iloc[history_len:].copy()
        result.index = data.index
        new_tail = combined.tail(history_len).copy()
        self._state_["tail"] = new_tail
        self._state_["history_len"] = len(new_tail)
        return result

    def _fit_spline_interp(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        order = int(self._get_policy_param("spline_degree", 3))
        requested_history = self._history_length()
        try:
            filled = data.interpolate(method="spline", order=order, limit_direction="forward")
        except ValueError as exc:  # pragma: no cover - SciPy 未導入など
            raise RuntimeError("spline_interp_deg policy requires scipy to be installed.") from exc
        filled = filled.ffill()
        filled = filled.fillna(pd.Series(medians_lookup))
        history_len, tail = self._effective_history(filled, requested_history)
        state = {
            "history_len": history_len,
            "tail": tail,
            "medians": medians_lookup,
            "order": order,
        }
        return filled, state

    def _transform_spline_interp(self, data: pd.DataFrame):
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        tail = self._state_tail_frame()
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        order = int(self._state_.get("order", 3))
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in spline_interp: tail={len(tail)} expected={history_len}")
        combined = pd.concat([tail, data], axis=0)
        try:
            combined = combined.interpolate(method="spline", order=order, limit_direction="forward")
        except ValueError as exc:  # pragma: no cover
            raise RuntimeError("spline_interp_deg policy requires scipy during transform as well.") from exc
        combined = combined.ffill()
        combined = combined.fillna(pd.Series(medians_lookup))
        result = combined.iloc[history_len:].copy()
        result.index = data.index
        new_tail = combined.tail(history_len).copy()
        self._state_["tail"] = new_tail
        self._state_["history_len"] = len(new_tail)
        return result

    def _fit_time_interp(
        self,
        data: pd.DataFrame,
        medians_lookup: Dict[Hashable, float],
        calendar_series: pd.Series | None,
    ):
        calendar_dt = self._calendar_to_datetime(calendar_series)
        if calendar_dt is None:
            raise ValueError("time_interp policy requires a datetime-like calendar column.")
        requested_history = self._history_length()
        working = data.copy()
        working.index = pd.DatetimeIndex(calendar_dt)
        filled = working.interpolate(method="time", limit_direction="forward")
        filled = filled.ffill()
        filled = filled.fillna(pd.Series(medians_lookup))
        history_len, tail = self._effective_history(filled, requested_history)
        tail_calendar = calendar_dt.tail(history_len).copy() if history_len else calendar_dt.iloc[0:0].copy()
        filled_reset = filled.copy()
        filled_reset.index = data.index
        state = {
            "history_len": history_len,
            "tail": tail,
            "tail_calendar": tail_calendar,
            "medians": medians_lookup,
        }
        return filled_reset, state

    def _transform_time_interp(self, data: pd.DataFrame, calendar_series: pd.Series | None):
        calendar_dt = self._calendar_to_datetime(calendar_series)
        if calendar_dt is None:
            raise ValueError("time_interp policy requires a datetime-like calendar column during transform.")
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        tail = self._state_tail_frame()
        tail_calendar = cast(pd.Series, self._state_.get("tail_calendar", pd.Series(dtype="datetime64[ns]")))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in time_interp: tail={len(tail)} expected={history_len}")
        if len(tail_calendar) != history_len:
            raise RuntimeError(
                f"history_len mismatch in time_interp calendar: tail_calendar={len(tail_calendar)} expected={history_len}"
            )

        combined = pd.concat([tail, data], axis=0)
        combined_calendar = pd.concat([tail_calendar, calendar_dt], axis=0)
        combined.index = pd.DatetimeIndex(pd.to_datetime(combined_calendar))
        combined = combined.interpolate(method="time", limit_direction="forward")
        combined = combined.ffill()
        combined = combined.fillna(pd.Series(medians_lookup))
        result = combined.iloc[history_len:].copy()
        result.index = data.index
        new_tail = combined.tail(history_len).copy()
        new_tail_calendar = (
            combined_calendar.tail(history_len).copy() if history_len else combined_calendar.iloc[0:0].copy()
        )
        self._state_["tail"] = new_tail
        self._state_["tail_calendar"] = new_tail_calendar
        self._state_["history_len"] = len(new_tail)
        return result

    # ロバスト/分位系 ----------------------------------------------------
    def _fit_backfill_robust(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        deques: Dict[str, Deque[float]] = {col: deque(maxlen=self.rolling_window) for col in self.columns_}
        last_valid: Dict[str, float | None] = {col: None for col in self.columns_}
        fallback = data.copy()

        for index_label in data.index:
            for col in self.columns_:
                original = data.at[index_label, col]
                if pd.isna(original):
                    dq = deques[col]
                    if dq:
                        fallback_val = self._deque_median(dq)
                    else:
                        fallback_val = float(medians_lookup.get(col, np.nan))
                    fallback.at[index_label, col] = fallback_val
                else:
                    valf = float(cast(float, original))
                    fallback.at[index_label, col] = valf
                    deques[col].append(valf)
                    last_valid[col] = valf

        filled = fallback.copy()

        requested_history = self._history_length()
        history_len, tail = self._effective_history(filled, requested_history)
        state = {
            "medians": medians_lookup,
            "deques": {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_},
            "last_valid": last_valid,
            "history_len": history_len,
            "tail": tail,
        }
        return filled, state

    def _transform_backfill_robust(self, data: pd.DataFrame):
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        stored_deques = cast(Dict[str, Deque[float]], self._state_.get("deques", {}))
        base_deques = {col: deque(stored_deques.get(col, deque()), maxlen=self.rolling_window) for col in self.columns_}
        last_valid = cast(Dict[str, float | None], self._state_.get("last_valid", {}))
        fallback = data.copy()
        for col in self.columns_:
            last_val = last_valid.get(col)
            if last_val is not None:
                base_deques[col].append(last_val)

        for index_label in data.index:
            for col in self.columns_:
                original = data.at[index_label, col]
                if pd.isna(original):
                    dq = base_deques[col]
                    if dq:
                        fallback_val = self._deque_median(dq)
                    else:
                        fallback_val = float(medians_lookup.get(col, np.nan))
                    fallback.at[index_label, col] = fallback_val
                else:
                    valf = float(cast(float, original))
                    fallback.at[index_label, col] = valf
                    base_deques[col].append(valf)

        filled = fallback.copy()

        for col in self.columns_:
            series = filled[col]
            if not series.empty:
                last_valid[col] = float(series.iloc[-1])
        self._state_["last_valid"] = last_valid
        self._state_["deques"] = {col: deque(base_deques[col], maxlen=self.rolling_window) for col in self.columns_}
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        history_len = min(history_len, len(filled)) if len(filled) else 0
        tail_updated = filled.tail(history_len).copy() if history_len else filled.iloc[0:0].copy()
        self._state_["tail"] = tail_updated
        self._state_["history_len"] = len(tail_updated)
        return filled

    def _winsorized_stat(self, values: Sequence[float], clip: float) -> float:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return float("nan")
        lower = float(np.quantile(arr, clip))
        upper = float(np.quantile(arr, 1.0 - clip))
        clipped = np.clip(arr, lower, upper)
        return float(np.median(clipped))

    def _fit_winsorized_median(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        clip = float(self._get_policy_param("winsor_clip", 0.1))
        clip = min(max(0.0, clip), 0.49)
        deques: Dict[str, Deque[float]] = {col: deque(maxlen=self.rolling_window) for col in self.columns_}
        filled = data.copy()
        for index_label in data.index:
            for col in self.columns_:
                original = data.at[index_label, col]
                if pd.isna(original):
                    dq = deques[col]
                    if dq:
                        fill_value = self._winsorized_stat(dq, clip)
                    else:
                        fill_value = float(medians_lookup.get(col, np.nan))
                    filled.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
                else:
                    valf = float(cast(float, original))
                    filled.at[index_label, col] = valf
                    deques[col].append(valf)
        state = {
            "medians": medians_lookup,
            "deques": {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_},
            "clip": clip,
        }
        return filled, state

    def _transform_winsorized_median(self, data: pd.DataFrame):
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        clip = float(self._state_.get("clip", 0.1))
        stored_deques = cast(Dict[str, Deque[float]], self._state_.get("deques", {}))
        deques = {col: deque(stored_deques.get(col, deque()), maxlen=self.rolling_window) for col in self.columns_}
        filled = data.copy()
        for index_label in data.index:
            for col in self.columns_:
                original = data.at[index_label, col]
                if pd.isna(original):
                    dq = deques[col]
                    if dq:
                        fill_value = self._winsorized_stat(dq, clip)
                    else:
                        fill_value = float(medians_lookup.get(col, np.nan))
                    filled.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
                else:
                    valf = float(cast(float, original))
                    filled.at[index_label, col] = valf
                    deques[col].append(valf)
        self._state_["deques"] = {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_}
        return filled

    def _fit_seasonal_median(
        self,
        data: pd.DataFrame,
        medians_lookup: Dict[Hashable, float],
        calendar_series: pd.Series | None,
    ):
        calendar_dt = self._calendar_to_datetime(calendar_series)
        if calendar_dt is None:
            raise ValueError(f"{self.policy} policy requires a datetime-like calendar column.")
        if self.policy == "dow_median":
            keys = calendar_dt.dt.dayofweek
        elif self.policy == "dom_median":
            keys = calendar_dt.dt.day
        else:  # month_median
            keys = calendar_dt.dt.month

        stats: Dict[str, Dict[int, float]] = {}
        filled = data.copy()
        for col in self.columns_:
            frame = pd.DataFrame({"key": keys, "value": data[col]})
            grouped = frame.groupby("key")["value"].agg(["median", "count"])
            stats_for_col: Dict[int, float] = {}
            fallback = float(medians_lookup.get(col, np.nan))
            for idx, row in grouped.iterrows():
                idx_key = int(cast(int, idx))
                count = int(row["count"])
                median_raw = row["median"]
                if isinstance(median_raw, (int, float, np.floating)):
                    median_val = float(median_raw)
                else:
                    median_val = fallback
                if count >= 2 and not np.isnan(median_val):
                    stats_for_col[idx_key] = median_val
                else:
                    stats_for_col[idx_key] = fallback
            stats[col] = stats_for_col
            fill_map = keys.map(stats_for_col)
            col_series = data[col].copy()
            col_series = col_series.fillna(fill_map)
            col_series = col_series.fillna(medians_lookup.get(col, np.nan))
            filled[col] = col_series.astype(float)

        state = {
            "medians": medians_lookup,
            "seasonal_stats": stats,
        }
        return filled, state

    def _transform_seasonal_median(self, data: pd.DataFrame, calendar_series: pd.Series | None):
        calendar_dt = self._calendar_to_datetime(calendar_series)
        if calendar_dt is None:
            raise ValueError(f"{self.policy} policy requires a datetime-like calendar column during transform.")
        if self.policy == "dow_median":
            keys = calendar_dt.dt.dayofweek
        elif self.policy == "dom_median":
            keys = calendar_dt.dt.day
        else:
            keys = calendar_dt.dt.month
        stats = cast(Dict[str, Dict[int, float]], self._state_.get("seasonal_stats", {}))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        filled = data.copy()
        for col in self.columns_:
            col_series = data[col].copy()
            mapping = stats.get(col, {})
            fill_map = keys.map(mapping)
            col_series = col_series.fillna(fill_map)
            col_series = col_series.fillna(medians_lookup.get(col, np.nan))
            filled[col] = col_series.astype(float)
        return filled

    def _apply_holiday_bridge(
        self,
        frame: pd.DataFrame,
        medians_lookup: Dict[Hashable, float],
        window: int,
        *,
        online_start: int | None = None,
    ):
        bridged = frame.copy()
        for col in self.columns_:
            series = bridged[col].copy()
            values = series.values
            n = len(values)
            i = 0
            while i < n:
                if not np.isnan(values[i]):
                    i += 1
                    continue
                start = i
                while i < n and np.isnan(values[i]):
                    i += 1
                end = i
                length = end - start
                prev_idx = start - 1
                next_idx = end
                prev_val = values[prev_idx] if prev_idx >= 0 and not np.isnan(values[prev_idx]) else None
                next_val = None
                if next_idx < n and not np.isnan(values[next_idx]):
                    if online_start is None or next_idx < online_start:
                        next_val = values[next_idx]
                if prev_val is not None and next_val is not None and length <= window:
                    fill_val = float((prev_val + next_val) / 2.0)
                    values[start:end] = fill_val
                else:
                    segment = series.iloc[max(0, start - self.rolling_window) : start].dropna()
                    if not segment.empty:
                        fill_val = float(segment.median())
                    elif prev_val is not None:
                        fill_val = float(prev_val)
                    elif next_val is not None:
                        fill_val = float(next_val)
                    else:
                        fill_val = float(medians_lookup.get(col, np.nan))
                    values[start:end] = fill_val
            bridged[col] = values
        return bridged

    def _fit_holiday_bridge(
        self,
        data: pd.DataFrame,
        medians_lookup: Dict[Hashable, float],
        calendar_series: pd.Series | None,
    ):
        if self._calendar_to_datetime(calendar_series) is None:
            raise ValueError("holiday_bridge policy requires a datetime-like calendar column.")
        window = int(self._get_policy_param("holiday_window", 2))
        requested_history = self._history_length()
        bridged = self._apply_holiday_bridge(data, medians_lookup, window)
        history_len, tail = self._effective_history(bridged, requested_history)
        state = {
            "medians": medians_lookup,
            "window": window,
            "history_len": history_len,
            "tail": tail,
        }
        return bridged, state

    def _transform_holiday_bridge(self, data: pd.DataFrame, calendar_series: pd.Series | None):
        if self._calendar_to_datetime(calendar_series) is None:
            raise ValueError("holiday_bridge policy requires a datetime-like calendar column during transform.")
        window = int(self._state_.get("window", 2))
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        tail = self._state_tail_frame()
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in holiday_bridge: tail={len(tail)} expected={history_len}")
        combined = pd.concat([tail, data], axis=0)
        bridged = self._apply_holiday_bridge(combined, medians_lookup, window, online_start=len(tail))
        result = bridged.iloc[history_len:].copy()
        result.index = data.index
        new_tail = bridged.tail(history_len).copy()
        self._state_["tail"] = new_tail
        self._state_["history_len"] = len(new_tail)
        return result

    # 多変量補完 ----------------------------------------------------------
    def _fit_knn(self, data: pd.DataFrame):
        n_neighbors = int(self._get_policy_param("knn_neighbors", min(5, len(data))))
        n_neighbors = max(1, n_neighbors)
        means, stds = self._compute_scaler_stats(data)
        scaled = self._standardize_with_stats(data, means, stds)
        available_cols: List[str] = []
        for col in self.columns_:
            notna_series = cast(pd.Series, data[col].notna())
            if self._series_any(notna_series):
                available_cols.append(col)
        missing_cols = [col for col in self.columns_ if col not in available_cols]

        if not available_cols:
            # No informative columns => fall back to medians.
            medians_series = pd.Series({col: self._medians_dict_.get(col, 0.0) for col in self.columns_})
            filled = data.fillna(medians_series)
            state = {
                "imputer": None,
                "scaler_means": means,
                "scaler_stds": stds,
                "available_cols": available_cols,
                "missing_cols": missing_cols,
                "medians": self._medians_dict_,
            }
            return filled, state

        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        scaled_subset = scaled.loc[:, available_cols]
        filled_array = imputer.fit_transform(scaled_subset)
        filled_scaled_subset = pd.DataFrame(
            filled_array,
            columns=pd.Index(available_cols),
            index=data.index,
        )

        filled_scaled = scaled.copy()
        filled_scaled.loc[:, available_cols] = filled_scaled_subset
        filled = self._destandardize_with_stats(filled_scaled, means, stds)

        medians_lookup = self._medians_dict_
        for col in missing_cols:
            median_val = float(medians_lookup.get(col, 0.0))
            mask_missing = cast(pd.Series, filled[col].isna())
            if self._series_any(mask_missing):
                filled.loc[mask_missing, col] = median_val

        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if self._series_any(~mask_series):
                filled.loc[~mask_series, col] = data.loc[~mask_series, col].astype(float)

        state = {
            "imputer": imputer,
            "scaler_means": means,
            "scaler_stds": stds,
            "available_cols": available_cols,
            "missing_cols": missing_cols,
            "medians": medians_lookup,
        }
        return filled, state

    def _transform_knn(self, data: pd.DataFrame):
        imputer = cast(KNNImputer, self._state_.get("imputer"))
        means = cast(Dict[str, float], self._state_.get("scaler_means", {}))
        stds = cast(Dict[str, float], self._state_.get("scaler_stds", {}))
        available_cols = cast(List[str], self._state_.get("available_cols", []))
        missing_cols = cast(List[str], self._state_.get("missing_cols", []))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))

        scaled = self._standardize_with_stats(data, means, stds)

        if imputer is not None and available_cols:
            scaled_subset = scaled.loc[:, available_cols]
            filled_array = imputer.transform(scaled_subset)
            filled_scaled_subset = pd.DataFrame(
                filled_array,
                columns=pd.Index(available_cols),
                index=data.index,
            )
            scaled.loc[:, available_cols] = filled_scaled_subset
        filled = self._destandardize_with_stats(scaled, means, stds)

        for col in missing_cols:
            median_val = float(medians_lookup.get(col, 0.0))
            mask_missing = cast(pd.Series, filled[col].isna())
            if self._series_any(mask_missing):
                filled.loc[mask_missing, col] = median_val

        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if self._series_any(~mask_series):
                filled.loc[~mask_series, col] = data.loc[~mask_series, col].astype(float)
        return filled

    def _fit_pca_reconstruct(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        n_features = max(1, len(self.columns_))
        default_components = max(1, min(n_features - 1, n_features // 2)) if n_features > 1 else 1
        components = int(self._get_policy_param("pca_components", default_components))
        components = max(1, min(components, n_features))
        medians_series = pd.Series(medians_lookup)
        filled_reference = data.fillna(medians_series)
        rng = np.random.RandomState(self.random_state)
        pca = PCA(n_components=components, random_state=rng)
        pca.fit(filled_reference.values)
        reconstructed = pca.inverse_transform(pca.transform(filled_reference.values))
        recon_df = pd.DataFrame(reconstructed, columns=data.columns, index=data.index)
        filled = data.copy()
        for col in self.columns_:
            mask = data[col].isna()
            filled.loc[mask, col] = recon_df.loc[mask, col]
            filled.loc[~mask, col] = data.loc[~mask, col].astype(float)
        state = {
            "pca": pca,
            "medians": medians_lookup,
        }
        return filled, state

    def _transform_pca_reconstruct(self, data: pd.DataFrame):
        pca = cast(PCA, self._state_.get("pca"))
        if pca is None:
            raise RuntimeError("PCA model missing; ensure fit was called.")
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        medians_series = pd.Series(medians_lookup)
        filled_reference = data.fillna(medians_series)
        reconstructed = pca.inverse_transform(pca.transform(filled_reference.values))
        recon_df = pd.DataFrame(reconstructed, columns=data.columns, index=data.index)
        filled = data.copy()
        for col in self.columns_:
            mask = data[col].isna()
            filled.loc[mask, col] = recon_df.loc[mask, col]
            filled.loc[~mask, col] = data.loc[~mask, col].astype(float)
        return filled

    def _fit_mice(self, data: pd.DataFrame):
        max_iter = int(self._get_policy_param("mice_max_iter", 10))
        means, stds = self._compute_scaler_stats(data)
        scaled = self._standardize_with_stats(data, means, stds)
        available_cols: List[str] = []
        for col in self.columns_:
            notna_series = cast(pd.Series, data[col].notna())
            if self._series_any(notna_series):
                available_cols.append(col)
        missing_cols = [col for col in self.columns_ if col not in available_cols]

        if not available_cols:
            medians_series = pd.Series({col: self._medians_dict_.get(col, 0.0) for col in self.columns_})
            filled = data.fillna(medians_series)
            state = {
                "imputer": None,
                "scaler_means": means,
                "scaler_stds": stds,
                "available_cols": available_cols,
                "missing_cols": missing_cols,
                "medians": self._medians_dict_,
            }
            return filled, state

        scaled_subset = scaled.loc[:, available_cols]
        rng = np.random.RandomState(self.random_state)
        imputer = IterativeImputer(random_state=rng, max_iter=max_iter, sample_posterior=False)
        filled_array = imputer.fit_transform(scaled_subset)
        filled_scaled_subset = pd.DataFrame(
            filled_array,
            columns=pd.Index(available_cols),
            index=data.index,
        )

        filled_scaled = scaled.copy()
        filled_scaled.loc[:, available_cols] = filled_scaled_subset
        filled = self._destandardize_with_stats(filled_scaled, means, stds)

        medians_lookup = self._medians_dict_
        for col in missing_cols:
            median_val = float(medians_lookup.get(col, 0.0))
            mask_missing = cast(pd.Series, filled[col].isna())
            if self._series_any(mask_missing):
                filled.loc[mask_missing, col] = median_val

        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if self._series_any(~mask_series):
                filled.loc[~mask_series, col] = data.loc[~mask_series, col].astype(float)
        state = {
            "imputer": imputer,
            "scaler_means": means,
            "scaler_stds": stds,
            "available_cols": available_cols,
            "missing_cols": missing_cols,
            "medians": medians_lookup,
        }
        return filled, state

    def _transform_mice(self, data: pd.DataFrame):
        imputer = cast(IterativeImputer, self._state_.get("imputer"))
        means = cast(Dict[str, float], self._state_.get("scaler_means", {}))
        stds = cast(Dict[str, float], self._state_.get("scaler_stds", {}))
        available_cols = cast(List[str], self._state_.get("available_cols", []))
        missing_cols = cast(List[str], self._state_.get("missing_cols", []))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))

        scaled = self._standardize_with_stats(data, means, stds)
        if imputer is not None and available_cols:
            scaled_subset = scaled.loc[:, available_cols]
            filled_array = imputer.transform(scaled_subset)
            filled_scaled_subset = pd.DataFrame(
                filled_array,
                columns=pd.Index(available_cols),
                index=data.index,
            )
            scaled.loc[:, available_cols] = filled_scaled_subset
        filled = self._destandardize_with_stats(scaled, means, stds)

        for col in missing_cols:
            median_val = float(medians_lookup.get(col, 0.0))
            mask_missing = cast(pd.Series, filled[col].isna())
            if self._series_any(mask_missing):
                filled.loc[mask_missing, col] = median_val

        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if self._series_any(~mask_series):
                filled.loc[~mask_series, col] = data.loc[~mask_series, col].astype(float)
        return filled

    def _fit_missforest(self, data: pd.DataFrame):
        max_iter = int(self._get_policy_param("missforest_max_iter", 5))
        n_estimators = int(self._get_policy_param("missforest_estimators", 200))
        means, stds = self._compute_scaler_stats(data)
        scaled = self._standardize_with_stats(data, means, stds)
        available_cols: List[str] = []
        for col in self.columns_:
            notna_series = cast(pd.Series, data[col].notna())
            if self._series_any(notna_series):
                available_cols.append(col)
        missing_cols = [col for col in self.columns_ if col not in available_cols]

        if not available_cols:
            medians_series = pd.Series({col: self._medians_dict_.get(col, 0.0) for col in self.columns_})
            filled = data.fillna(medians_series)
            state = {
                "imputer": None,
                "scaler_means": means,
                "scaler_stds": stds,
                "available_cols": available_cols,
                "missing_cols": missing_cols,
                "medians": self._medians_dict_,
            }
            return filled, state

        scaled_subset = scaled.loc[:, available_cols]
        rng_estim = np.random.RandomState(self.random_state)
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=rng_estim,
            n_jobs=-1,
            max_depth=self._get_policy_param("missforest_max_depth", None),
        )
        rng_imputer = np.random.RandomState(self.random_state)
        imputer = IterativeImputer(
            estimator=estimator,
            random_state=rng_imputer,
            max_iter=max_iter,
            sample_posterior=False,
            initial_strategy="median",
        )
        filled_array = imputer.fit_transform(scaled_subset)
        filled_scaled_subset = pd.DataFrame(
            filled_array,
            columns=pd.Index(available_cols),
            index=data.index,
        )

        filled_scaled = scaled.copy()
        filled_scaled.loc[:, available_cols] = filled_scaled_subset
        filled = self._destandardize_with_stats(filled_scaled, means, stds)

        medians_lookup = self._medians_dict_
        for col in missing_cols:
            median_val = float(medians_lookup.get(col, 0.0))
            mask_missing = cast(pd.Series, filled[col].isna())
            if self._series_any(mask_missing):
                filled.loc[mask_missing, col] = median_val

        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if self._series_any(~mask_series):
                filled.loc[~mask_series, col] = data.loc[~mask_series, col].astype(float)
        state = {
            "imputer": imputer,
            "scaler_means": means,
            "scaler_stds": stds,
            "available_cols": available_cols,
            "missing_cols": missing_cols,
            "medians": medians_lookup,
        }
        return filled, state

    def _transform_missforest(self, data: pd.DataFrame):
        imputer = cast(IterativeImputer, self._state_.get("imputer"))
        means = cast(Dict[str, float], self._state_.get("scaler_means", {}))
        stds = cast(Dict[str, float], self._state_.get("scaler_stds", {}))
        available_cols = cast(List[str], self._state_.get("available_cols", []))
        missing_cols = cast(List[str], self._state_.get("missing_cols", []))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))

        scaled = self._standardize_with_stats(data, means, stds)
        if imputer is not None and available_cols:
            scaled_subset = scaled.loc[:, available_cols]
            filled_array = imputer.transform(scaled_subset)
            filled_scaled_subset = pd.DataFrame(
                filled_array,
                columns=pd.Index(available_cols),
                index=data.index,
            )
            scaled.loc[:, available_cols] = filled_scaled_subset
        filled = self._destandardize_with_stats(scaled, means, stds)

        for col in missing_cols:
            median_val = float(medians_lookup.get(col, 0.0))
            mask_missing = cast(pd.Series, filled[col].isna())
            if self._series_any(mask_missing):
                filled.loc[mask_missing, col] = median_val

        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if self._series_any(~mask_series):
                filled.loc[~mask_series, col] = data.loc[~mask_series, col].astype(float)
        return filled

    def _fit_ridge_stack(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        alpha = float(self._get_policy_param("ridge_alpha", 1.0))
        medians_series = pd.Series(medians_lookup)
        filled_reference = data.fillna(medians_series)
        models: Dict[str, Ridge] = {}
        for col in self.columns_:
            mask = cast(pd.Series, data[col].notna())
            if int(mask.sum()) < 2:
                continue
            target = data.loc[mask, col].astype(float)
            features = filled_reference.loc[mask, [c for c in self.columns_ if c != col]]
            model = Ridge(alpha=alpha, random_state=np.random.RandomState(self.random_state))
            model.fit(features, target)
            models[col] = model

        filled = filled_reference.copy()
        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if not self._series_any(mask_series):
                filled[col] = data[col].astype(float)
                continue
            model = models.get(col)
            if model is None:
                filled.loc[mask_series, col] = medians_lookup.get(col, np.nan)
                continue
            feature_cols = [c for c in self.columns_ if c != col]
            preds = model.predict(filled.loc[mask_series, feature_cols])
            filled.loc[mask_series, col] = preds

        state = {
            "models": models,
            "medians": medians_lookup,
        }
        return filled, state

    def _transform_ridge_stack(self, data: pd.DataFrame):
        models = cast(Dict[str, Ridge], self._state_.get("models", {}))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        medians_series = pd.Series(medians_lookup)
        filled = data.fillna(medians_series)
        for col in self.columns_:
            mask_series = cast(pd.Series, data[col].isna()).astype(bool)
            if not self._series_any(mask_series):
                filled[col] = data[col].astype(float)
                continue
            model = models.get(col)
            if model is None:
                filled.loc[mask_series, col] = medians_lookup.get(col, np.nan)
                continue
            feature_cols = [c for c in self.columns_ if c != col]
            preds = model.predict(filled.loc[mask_series, feature_cols])
            filled.loc[mask_series, col] = preds
        return filled

    def _fit_quantile_fill(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        quantile = float(self._get_policy_param("quantile", 0.5))
        quantile = min(max(0.0, quantile), 1.0)
        deques: Dict[str, Deque[float]] = {col: deque(maxlen=self.rolling_window) for col in self.columns_}
        filled = data.copy()
        for index_label in data.index:
            for col in self.columns_:
                original = data.at[index_label, col]
                if pd.isna(original):
                    dq = deques[col]
                    if dq:
                        fill_value = float(np.quantile(list(dq), quantile))
                    else:
                        fill_value = float(medians_lookup.get(col, np.nan))
                    filled.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
                else:
                    valf = float(cast(float, original))
                    filled.at[index_label, col] = valf
                    deques[col].append(valf)
        state = {
            "medians": medians_lookup,
            "deques": {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_},
            "quantile": quantile,
        }
        return filled, state

    def _transform_quantile_fill(self, data: pd.DataFrame):
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        quantile = float(self._state_.get("quantile", 0.5))
        stored_deques = cast(Dict[str, Deque[float]], self._state_.get("deques", {}))
        deques = {col: deque(stored_deques.get(col, deque()), maxlen=self.rolling_window) for col in self.columns_}
        filled = data.copy()
        for index_label in data.index:
            for col in self.columns_:
                original = data.at[index_label, col]
                if pd.isna(original):
                    dq = deques[col]
                    if dq:
                        fill_value = float(np.quantile(list(dq), quantile))
                    else:
                        fill_value = float(medians_lookup.get(col, np.nan))
                    filled.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
                else:
                    valf = float(cast(float, original))
                    filled.at[index_label, col] = valf
                    deques[col].append(valf)
        self._state_["deques"] = {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_}
        return filled

    def _fit_kalman(
        self,
        data: pd.DataFrame,
        medians_lookup: Dict[Hashable, float],
        *,
        level_only: bool,
    ):
        """Fit UnobservedComponents models and backfill using filter (one-step-ahead) estimates."""
        if UnobservedComponents is None:
            raise RuntimeError("kalman_* policies require the 'statsmodels' package.")
        models: Dict[str, Any] = {}
        filled = data.copy()
        requested_history = self._history_length()
        warnings: List[str] = []
        for col in self.columns_:
            series = data[col].astype(float)
            if series.notna().sum() < 3:
                filled[col] = series.fillna(medians_lookup.get(col, np.nan))
                continue
            model = UnobservedComponents(series, level="local level", trend=not level_only)  # type: ignore[arg-type]
            try:
                res = cast(Any, model.fit(disp=False))
                fitted_series = pd.Series(cast(Any, res.fittedvalues), index=series.index)
                if hasattr(res, "remove_data"):
                    # Drop cached training arrays to keep serialized artifacts small while
                    # retaining parameters required for forward filtering.
                    res.remove_data()
            except Exception as exc:  # pragma: no cover
                res = None
                fitted_series = series.fillna(method="ffill")
                warnings.append(f"kalman_fit_fallback[{col}]: {type(exc).__name__}")
            col_filled = series.copy()
            mask = series.isna()
            col_filled.loc[mask] = fitted_series.loc[mask]
            col_filled = col_filled.fillna(medians_lookup.get(col, np.nan))
            filled[col] = col_filled
            if res is not None:
                models[col] = res
        history_len, tail = self._effective_history(filled, requested_history)
        state = {
            "models": models,
            "medians": medians_lookup,
            "history_len": history_len,
            "tail": tail,
            "warnings": warnings,
        }
        return filled, state

    def _transform_kalman(self, data: pd.DataFrame, calendar_series: pd.Series | None):
        models = cast(Dict[str, Any], self._state_.get("models", {}))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        tail = self._state_tail_frame()
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in kalman transform: tail={len(tail)} expected={history_len}")
        combined = pd.concat([tail, data], axis=0)
        result = combined.copy()
        for col in self.columns_:
            series = combined[col].astype(float)
            res = models.get(col)
            if res is None:
                filled_col = series.fillna(method="ffill").fillna(medians_lookup.get(col, np.nan))
                self._record_warning(f"kalman_transform_fallback[{col}]: model_missing")
            else:
                try:
                    series_for_filter = series.copy()
                    for idx in data.index:
                        if idx in series_for_filter.index:
                            series_for_filter.at[idx] = np.nan
                    applied = res.apply(series_for_filter)
                    forecasts_obj = getattr(applied, "forecasts", None)
                    if forecasts_obj is None:
                        filter_results = getattr(applied, "filter_results", None)
                        forecasts_obj = getattr(filter_results, "forecasts", None)
                    if forecasts_obj is not None:
                        forecasts_arr = np.asarray(forecasts_obj)
                        if forecasts_arr.ndim == 2:
                            forecasts_arr = forecasts_arr.reshape(forecasts_arr.shape[0], -1)
                            if forecasts_arr.shape[0] == 1:
                                forecasts_arr = forecasts_arr[0]
                            else:
                                forecasts_arr = forecasts_arr.mean(axis=0)
                        fitted_values = forecasts_arr.reshape(-1)
                        if len(fitted_values) < len(series):
                            fitted_values = np.pad(fitted_values, (len(series) - len(fitted_values), 0), mode="edge")
                    else:
                        raise AttributeError("kalman_forecast_unavailable")
                except Exception as exc:  # pragma: no cover
                    fitted_values = series.fillna(method="ffill")
                    self._record_warning(f"kalman_transform_fallback[{col}]: {type(exc).__name__}")
                fitted_series = pd.Series(fitted_values, index=combined.index)
                filled_col = series.copy()
                mask = series.isna()
                filled_col.loc[mask] = fitted_series.loc[mask]
                filled_col = filled_col.fillna(medians_lookup.get(col, np.nan))
            result[col] = filled_col
        tail_updated = result.tail(history_len).copy()
        self._state_["tail"] = tail_updated
        self._state_["history_len"] = len(tail_updated)
        return result.iloc[history_len:].copy()

    def _fit_arima_auto(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        """Auto-select ARIMA orders and use filter outputs to keep the transform forward-only."""
        if ARIMA is None:
            raise RuntimeError("arima_auto policy requires the 'statsmodels' package.")
        max_p = int(self._get_policy_param("arima_max_p", 2))
        max_d = int(self._get_policy_param("arima_max_d", 1))
        max_q = int(self._get_policy_param("arima_max_q", 2))
        models: Dict[str, Any] = {}
        filled = data.copy()
        requested_history = self._history_length()
        warnings: List[str] = []
        for col in self.columns_:
            series = data[col].astype(float)
            if series.notna().sum() < 5:
                filled[col] = series.fillna(medians_lookup.get(col, np.nan))
                continue
            best_res = None
            best_aic = np.inf
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        if p == d == q == 0:
                            continue
                        try:
                            res = ARIMA(
                                series,
                                order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(method_kwargs={"warn_convergence": False})
                            if res.aic < best_aic:
                                best_aic = float(res.aic)
                                best_res = res
                        except Exception:
                            continue
            if best_res is None:
                filled[col] = series.fillna(method="ffill").fillna(medians_lookup.get(col, np.nan))
                warnings.append(f"arima_fit_fallback[{col}]: no_model")
            else:
                fitted_series = pd.Series(best_res.fittedvalues, index=series.index)  # filter (one-step) predictions
                col_filled = series.copy()
                mask = series.isna()
                col_filled.loc[mask] = fitted_series.loc[mask]
                col_filled = col_filled.fillna(medians_lookup.get(col, np.nan))
                filled[col] = col_filled
                models[col] = best_res
        history_len, tail = self._effective_history(filled, requested_history)
        state = {
            "models": models,
            "medians": medians_lookup,
            "history_len": history_len,
            "tail": tail,
            "warnings": warnings,
        }
        return filled, state

    def _transform_arima_auto(self, data: pd.DataFrame):
        models = cast(Dict[str, Any], self._state_.get("models", {}))
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        tail = self._state_tail_frame()
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in arima transform: tail={len(tail)} expected={history_len}")
        combined = pd.concat([tail, data], axis=0)
        result = combined.copy()
        for col in self.columns_:
            series = combined[col].astype(float)
            res = models.get(col)
            if res is None:
                filled_col = series.fillna(method="ffill").fillna(medians_lookup.get(col, np.nan))
                self._record_warning(f"arima_transform_fallback[{col}]: model_missing")
            else:
                try:
                    forecast_res = res.get_forecast(steps=len(data))
                    forecast_mean = np.asarray(forecast_res.predicted_mean)
                    forecast_series = pd.Series(forecast_mean, index=data.index)
                except Exception as exc:  # pragma: no cover
                    fitted_values = series.fillna(method="ffill")
                    self._record_warning(f"arima_transform_fallback[{col}]: {type(exc).__name__}")
                    filled_col = series.copy()
                    mask = series.isna()
                    filled_col.loc[mask] = fitted_values.loc[mask] if isinstance(fitted_values, pd.Series) else fitted_values
                    filled_col = filled_col.fillna(medians_lookup.get(col, np.nan))
                else:
                    filled_col = series.copy()
                    mask_combined = series.isna()
                    mask_new = mask_combined.loc[data.index]
                    missing_indices = mask_new.index[mask_new]
                    if not missing_indices.empty:
                        filled_col.loc[missing_indices] = forecast_series.loc[missing_indices]
                    filled_col = filled_col.fillna(medians_lookup.get(col, np.nan))
            result[col] = filled_col
        tail_updated = result.tail(history_len).copy()
        self._state_["tail"] = tail_updated
        self._state_["history_len"] = len(tail_updated)
        return result.iloc[history_len:].copy()

    # マスク系・二段補完 --------------------------------------------------
    def _fit_mask_plus_mean(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        medians_series = pd.Series(medians_lookup)
        filled = data.fillna(medians_series)
        mask_map: Dict[str, str] = {}
        for col in self.columns_:
            mask_col = f"{col}_missing_flag"
            mask_map[col] = mask_col
            filled[mask_col] = data[col].isna().astype(float)
        state = {
            "means": medians_lookup,
            "mask_map": mask_map,
        }
        return filled, state

    def _transform_mask_plus_mean(self, data: pd.DataFrame):
        means = cast(Dict[Hashable, float], self._state_.get("means", {}))
        mask_map = cast(Dict[str, str], self._state_.get("mask_map", {}))
        filled = data.fillna(pd.Series(means))
        for col, mask_col in mask_map.items():
            filled[mask_col] = data[col].isna().astype(float)
        return filled

    def _fit_two_stage(self, data: pd.DataFrame, medians_lookup: Dict[Hashable, float]):
        requested_history = self._history_length()
        first_pass = data.interpolate(method="linear", limit_direction="forward")
        first_pass = first_pass.ffill()
        first_pass = first_pass.fillna(pd.Series(medians_lookup))

        deques: Dict[str, Deque[float]] = {col: deque(maxlen=self.rolling_window) for col in self.columns_}
        second_pass = first_pass.copy()
        for index_label in first_pass.index:
            for col in self.columns_:
                value = second_pass.at[index_label, col]
                if pd.isna(data.at[index_label, col]):
                    dq = deques[col]
                    if dq:
                        fill_value = self._deque_median(dq)
                    else:
                        fill_value = float(medians_lookup.get(col, np.nan))
                    second_pass.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
                else:
                    valf = float(cast(float, value))
                    second_pass.at[index_label, col] = valf
                    deques[col].append(valf)

        history_len, tail = self._effective_history(second_pass, requested_history)
        state = {
            "medians": medians_lookup,
            "history_len": history_len,
            "tail": tail,
            "deques": {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_},
        }
        return second_pass, state

    def _transform_two_stage(self, data: pd.DataFrame):
        medians_lookup = cast(Dict[Hashable, float], self._state_.get("medians", {}))
        history_len = cast(int, self._state_.get("history_len", self._history_length()))
        tail = self._state_tail_frame()
        deques_state = cast(Dict[str, Deque[float]], self._state_.get("deques", {}))
        if len(tail) != history_len:
            raise RuntimeError(f"history_len mismatch in two_stage: tail={len(tail)} expected={history_len}")
        tail_base = tail.reindex(columns=self.columns_, fill_value=np.nan)
        combined = pd.concat([tail_base, data], axis=0)
        first_pass = combined.interpolate(method="linear", limit_direction="forward")
        first_pass = first_pass.ffill()
        first_pass = first_pass.fillna(pd.Series(medians_lookup))

        deques = {col: deque(deques_state.get(col, deque()), maxlen=self.rolling_window) for col in self.columns_}
        second_pass = first_pass.copy()
        combined_original = pd.concat([tail_base, data], axis=0)
        for index_label in second_pass.index:
            for col in self.columns_:
                original_val = combined_original.at[index_label, col]
                if pd.isna(original_val):
                    dq = deques[col]
                    if dq:
                        fill_value = self._deque_median(dq)
                    else:
                        fill_value = float(medians_lookup.get(col, np.nan))
                    second_pass.at[index_label, col] = fill_value
                    deques[col].append(fill_value)
                else:
                    valf = float(cast(float, second_pass.at[index_label, col]))
                    second_pass.at[index_label, col] = valf
                    deques[col].append(valf)

        self._state_["deques"] = {col: deque(deques[col], maxlen=self.rolling_window) for col in self.columns_}
        new_tail = second_pass.tail(history_len).copy()
        self._state_["tail"] = new_tail
        self._state_["history_len"] = len(new_tail)
        result = second_pass.iloc[history_len:].copy()
        result.index = data.index
        return result
