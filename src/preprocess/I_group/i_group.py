from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from preprocess.M_group.m_group import MGroupImputer as _BaseMGroupImputer


class IGroupImputer(_BaseMGroupImputer):
	"""Specialised imputer for I-group (inventory) features.

	Extends :class:`MGroupImputer` with I-specific defaults:
	- Automatically scopes to columns starting with ``"I"`` when no column list is provided.
	- Renames generated helper columns so they remain I-namespaced (e.g. ``Imask__``).
	- Optionally applies quantile clipping after imputation to cap extreme values.
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
		clip_quantile_low: float = 0.001,
		clip_quantile_high: float = 0.999,
		enable_quantile_clip: bool = True,
	) -> None:
		self._user_calendar_column = calendar_column
		self.clip_quantile_low = float(clip_quantile_low)
		self.clip_quantile_high = float(clip_quantile_high)
		self.enable_quantile_clip = bool(enable_quantile_clip)
		self._clip_bounds_: Dict[str, tuple[float, float]] = {}
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
	def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
		frame = self._ensure_dataframe(X).copy()

		selected_columns = self._resolve_columns(frame)
		numeric_columns: list[str] = []
		for col in selected_columns:
			if col not in frame.columns:
				continue
			frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce")
			numeric_columns.append(col)
		self.columns = numeric_columns

		calendar_column = self._resolve_calendar_column(frame)
		if calendar_column is not None and calendar_column in frame.columns:
			calendar_series = pd.to_datetime(frame[calendar_column], errors="coerce")
			if calendar_series.isna().any():
				self._prefit_warnings.append("calendar_column_contains_non_parseable_values")
			if calendar_series.duplicated().any():
				self._prefit_warnings.append("calendar_column_contains_duplicates")

		fitted = super().fit(frame, y)

		self._relabel_generated_columns()
		self._clip_bounds_ = self._compute_clip_bounds()

		if hasattr(self, "_state_") and isinstance(self._state_, dict):
			if self._prefit_warnings:
				warnings = self._state_.setdefault("warnings", [])
				if isinstance(warnings, list):
					warnings.extend(self._prefit_warnings)
			self._state_["clip_bounds"] = dict(self._clip_bounds_)
			self._state_["clip_quantile_low"] = self.clip_quantile_low
			self._state_["clip_quantile_high"] = self.clip_quantile_high
			self._state_["enable_quantile_clip"] = self.enable_quantile_clip

		return fitted

	# ------------------------------------------------------------------
	def transform(self, X: pd.DataFrame):  # type: ignore[override]
		frame = self._ensure_dataframe(X).copy()
		for col in getattr(self, "columns_", []):
			if col in frame.columns:
				frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce")

		transformed = super().transform(frame)

		if self.enable_quantile_clip and self._clip_bounds_:
			for col, (low, high) in self._clip_bounds_.items():
				if col in transformed.columns:
					transformed.loc[:, col] = transformed[col].clip(lower=low, upper=high)

		return transformed

	# ------------------------------------------------------------------
	def _resolve_columns(self, frame: pd.DataFrame) -> list[str]:
		if self.columns is None:
			return [c for c in frame.columns if isinstance(c, str) and c.startswith("I")]
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

		if isinstance(getattr(self, "_train_filled_", None), pd.DataFrame):
			self._train_filled_ = self._train_filled_.rename(columns=rename_map)

		self.extra_columns_ = [str(rename_map.get(col, col)) for col in extra_columns]
		if hasattr(self, "_output_columns_"):
			self._output_columns_ = [str(rename_map.get(col, col)) for col in getattr(self, "_output_columns_", [])]

		if hasattr(self, "_state_") and isinstance(self._state_, dict):
			for key, value in list(self._state_.items()):
				if isinstance(value, pd.DataFrame):
					self._state_[key] = value.rename(columns=rename_map)
			mask_map = self._state_.get("mask_map")
			if isinstance(mask_map, dict):
				self._state_["mask_map"] = {k: rename_map.get(v, v) for k, v in mask_map.items()}

	def _rename_generated_column(self, name: str) -> str:
		if name.endswith("_missing_flag"):
			base_name = name[: -len("_missing_flag")]
			return f"Imask__{base_name}"
		if name.startswith("I"):
			return name
		return f"Iextra__{name}"

	def _compute_clip_bounds(self) -> Dict[str, tuple[float, float]]:
		if not self.enable_quantile_clip:
			return {}
		q_low = float(self.clip_quantile_low)
		q_high = float(self.clip_quantile_high)
		if q_low < 0.0 or q_high > 1.0 or q_low >= q_high:
			return {}
		filled = getattr(self, "_train_filled_", None)
		if not isinstance(filled, pd.DataFrame):
			return {}

		bounds: Dict[str, tuple[float, float]] = {}
		for col in getattr(self, "columns_", []):
			if col not in filled.columns:
				continue
			numeric_values = pd.to_numeric(filled[col], errors="coerce")
			series = pd.Series(numeric_values).dropna()
			if series.empty:
				continue
			low_value = float(series.quantile(q_low))
			high_value = float(series.quantile(q_high))
			if not np.isfinite(low_value) or not np.isfinite(high_value):
				continue
			if high_value <= low_value:
				continue
			bounds[col] = (low_value, high_value)
		return bounds

