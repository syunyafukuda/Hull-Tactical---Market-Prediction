from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, cast

import numpy as np
import pandas as pd

from preprocess.M_group.m_group import MGroupImputer as _BaseMGroupImputer


class VGroupImputer(_BaseMGroupImputer):
	"""Imputer tailored for volatility (V-group) features.

	The implementation extends :class:`MGroupImputer` with V-specific behaviour:

	- Automatically scopes to columns starting with ``"V"`` when no explicit column
	  list is supplied.
	- Helper columns created by policies such as ``mask_plus_mean`` are renamed so
	  they stay V-namespaced (``Vmask__`` / ``Vextra__``).
	- After imputation, values can be quantile-clipped and log-transformed to
	  stabilise skewed volatility distributions.
	"""

	CALENDAR_REQUIRED_POLICIES = {
		"dow_median",
		"dom_median",
		"month_median",
		"holiday_bridge",
		"time_interp",
	}

	def __init__(
		self,
		columns: Iterable[str] | None = None,
		policy: str = "ffill_bfill",
		rolling_window: int = 5,
		ema_alpha: float = 0.3,
		calendar_column: str | None = None,
		policy_params: Mapping[str, Any] | None = None,
		random_state: int = 42,
		*,
		clip_quantile_low: float = 0.01,
		clip_quantile_high: float = 0.99,
		enable_quantile_clip: bool = True,
		log_transform: bool = True,
		log_offset_epsilon: float = 1e-6,
	) -> None:
		self._user_calendar_column = calendar_column
		self.clip_quantile_low = float(clip_quantile_low)
		self.clip_quantile_high = float(clip_quantile_high)
		self.enable_quantile_clip = bool(enable_quantile_clip)
		self.log_transform = bool(log_transform)
		self.log_offset_epsilon = float(log_offset_epsilon)
		self._clip_bounds_: Dict[str, tuple[float, float]] = {}
		self._log_offsets_: Dict[str, float] = {}
		self._prefit_warnings: list[str] = []

		super().__init__(
			columns=columns,
			policy=policy,
			rolling_window=rolling_window,
			ema_alpha=ema_alpha,
			calendar_column=calendar_column,
			policy_params=policy_params,
			random_state=random_state,
		)

	# ------------------------------------------------------------------
	def fit(self, X: pd.DataFrame, y: Any = None):  # type: ignore[override]
		frame = self._ensure_dataframe(X).copy()

		selected_columns = self._resolve_columns(frame)
		numeric_columns: list[str] = []
		for col in selected_columns:
			if col not in frame.columns:
				continue
			frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce")
			numeric_columns.append(col)
		self.columns = numeric_columns

		self._prefit_warnings = []
		calendar_column = self._resolve_calendar_column(frame)
		if calendar_column is not None and calendar_column in frame.columns:
			calendar_series = pd.to_datetime(frame[calendar_column], errors="coerce")
			if calendar_series.isna().any():
				self._prefit_warnings.append("calendar_column_contains_non_parseable_values")
			if calendar_series.duplicated().any():
				self._prefit_warnings.append("calendar_column_contains_duplicates")

		fitted = super().fit(frame, y)

		self._relabel_generated_columns()
		self._postprocess_training_frame()

		if hasattr(self, "_state_") and isinstance(self._state_, dict):
			state = self._state_
			if self._prefit_warnings:
				warnings = state.setdefault("warnings", [])
				if isinstance(warnings, list):
					warnings.extend(self._prefit_warnings)
			state["clip_bounds"] = dict(self._clip_bounds_)
			state["clip_quantile_low"] = self.clip_quantile_low
			state["clip_quantile_high"] = self.clip_quantile_high
			state["enable_quantile_clip"] = self.enable_quantile_clip
			state["log_offsets"] = dict(self._log_offsets_)
			state["log_transform"] = self.log_transform
			state["log_offset_epsilon"] = self.log_offset_epsilon

		return fitted

	# ------------------------------------------------------------------
	def transform(self, X: pd.DataFrame):  # type: ignore[override]
		frame = self._ensure_dataframe(X).copy()
		for col in getattr(self, "columns_", []):
			if col in frame.columns:
				frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce")

		transformed = super().transform(frame)
		return self._postprocess_dataframe(transformed)

	# ------------------------------------------------------------------
	def _postprocess_training_frame(self) -> None:
		train_filled = getattr(self, "_train_filled_", None)
		if not isinstance(train_filled, pd.DataFrame):
			self._clip_bounds_ = {}
			self._log_offsets_ = {}
			return

		processed = train_filled.copy()
		for col in getattr(self, "columns_", []):
			if col in processed.columns:
				processed.loc[:, col] = pd.to_numeric(processed[col], errors="coerce")

		if self.enable_quantile_clip:
			self._clip_bounds_ = self._compute_clip_bounds(processed)
			if self._clip_bounds_:
				processed = self._apply_clip(processed)
		else:
			self._clip_bounds_ = {}

		if self.log_transform:
			self._log_offsets_ = self._compute_log_offsets(processed)
			if self._log_offsets_:
				processed = self._apply_log_transform(processed)
		else:
			self._log_offsets_ = {}

		self._train_filled_ = processed

	def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
		if not isinstance(df, pd.DataFrame):
			return df
		processed = df.copy()
		for col in getattr(self, "columns_", []):
			if col in processed.columns:
				processed.loc[:, col] = pd.to_numeric(processed[col], errors="coerce")

		if self.enable_quantile_clip and self._clip_bounds_:
			processed = self._apply_clip(processed)
		if self.log_transform and self._log_offsets_:
			processed = self._apply_log_transform(processed)
		return processed

	# ------------------------------------------------------------------
	def _resolve_columns(self, frame: pd.DataFrame) -> list[str]:
		if self.columns is None:
			return [c for c in frame.columns if isinstance(c, str) and c.startswith("V")]
		return [c for c in self.columns if isinstance(c, str)]

	def _resolve_calendar_column(self, frame: pd.DataFrame) -> str | None:
		calendar_column = self._user_calendar_column or self.calendar_column
		if self.policy in self.CALENDAR_REQUIRED_POLICIES and calendar_column is None:
			raise ValueError(
				f"Policy '{self.policy}' requires a calendar column but none was provided."
			)
		if calendar_column is not None and calendar_column not in frame.columns:
			raise KeyError(
				f"Calendar column '{calendar_column}' not found in input DataFrame."
			)
		return calendar_column

	def _relabel_generated_columns(self) -> None:
		rename_map: Dict[str, str] = {}
		extra_columns = getattr(self, "extra_columns_", [])
		if not extra_columns:
			return
		for col in extra_columns:
			rename_map[col] = self._rename_generated_column(col)

		if not rename_map:
			return

		train_filled = getattr(self, "_train_filled_", None)
		if isinstance(train_filled, pd.DataFrame):
			self._train_filled_ = train_filled.rename(columns=rename_map)

		self.extra_columns_ = [str(rename_map.get(col, col)) for col in extra_columns]
		if hasattr(self, "_output_columns_"):
			self._output_columns_ = [
				str(rename_map.get(col, col)) for col in getattr(self, "_output_columns_", [])
			]

		state = getattr(self, "_state_", None)
		if isinstance(state, dict):
			for key, value in list(state.items()):
				if isinstance(value, pd.DataFrame):
					state[key] = value.rename(columns=rename_map)
			mask_map = state.get("mask_map")
			if isinstance(mask_map, dict):
				state["mask_map"] = {k: rename_map.get(v, v) for k, v in mask_map.items()}

	def _rename_generated_column(self, name: str) -> str:
		if name.endswith("_missing_flag"):
			base_name = name[: -len("_missing_flag")]
			return f"Vmask__{base_name}"
		if name.startswith("V"):
			return name
		return f"Vextra__{name}"

	def _compute_clip_bounds(self, frame: pd.DataFrame) -> Dict[str, tuple[float, float]]:
		if not self.enable_quantile_clip:
			return {}
		q_low = float(self.clip_quantile_low)
		q_high = float(self.clip_quantile_high)
		if not (0.0 <= q_low < q_high <= 1.0):
			return {}

		bounds: Dict[str, tuple[float, float]] = {}
		for col in getattr(self, "columns_", []):
			if col not in frame.columns:
				continue
			series_numeric = cast(pd.Series, pd.to_numeric(frame[col], errors="coerce"))
			series = series_numeric.dropna()
			if series.empty:
				continue
			low = float(series.quantile(q_low))
			high = float(series.quantile(q_high))
			if not np.isfinite(low) or not np.isfinite(high) or high <= low:
				continue
			bounds[col] = (low, high)
		return bounds

	def _apply_clip(self, frame: pd.DataFrame) -> pd.DataFrame:
		result = frame.copy()
		for col, (low, high) in self._clip_bounds_.items():
			if col not in result.columns:
				continue
			numeric = cast(pd.Series, pd.to_numeric(result[col], errors="coerce"))
			result.loc[:, col] = numeric.clip(lower=low, upper=high)
		return result

	def _compute_log_offsets(self, frame: pd.DataFrame) -> Dict[str, float]:
		if not self.log_transform:
			return {}
		offsets: Dict[str, float] = {}
		epsilon = max(self.log_offset_epsilon, 0.0)
		for col in getattr(self, "columns_", []):
			if col not in frame.columns:
				continue
			series_numeric = cast(pd.Series, pd.to_numeric(frame[col], errors="coerce"))
			series = series_numeric.dropna()
			if series.empty:
				offsets[col] = 0.0
				continue
			min_val = float(series.min())
			offsets[col] = -min_val + epsilon if min_val < 0.0 else 0.0
		return offsets

	def _apply_log_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
		result = frame.copy()
		for col in getattr(self, "columns_", []):
			if col not in result.columns:
				continue
			series = cast(pd.Series, pd.to_numeric(result[col], errors="coerce"))
			offset = self._log_offsets_.get(col, 0.0)
			adjusted = series.add(offset)
			clipped = cast(pd.Series, adjusted.clip(lower=0.0))
			result.loc[:, col] = np.log1p(clipped)
		return result



