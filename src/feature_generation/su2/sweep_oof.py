#!/usr/bin/env python
"""SU2 OOF sweep script for hyperparameter tuning.

configs/feature_generation.yaml の su2 セクションに記載されたパラメータ候補を組み合わせ、
OOF (TimeSeriesSplit) で RMSE / MSR / vMSR を評価する。結果は
results/ablation/SU2 配下に CSV として書き出す。
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import itertools
import json
import shutil
import sys
from pathlib import Path
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
	if str(path) not in sys.path:
		sys.path.append(str(path))

from scripts.utils_msr import (  # noqa: E402
	PostProcessParams,
	evaluate_msr_proxy,
	grid_search_msr,
)
from src.feature_generation.su1.feature_su1 import SU1Config, load_su1_config  # noqa: E402
from src.feature_generation.su2.feature_su2 import SU2Config  # noqa: E402
from src.feature_generation.su2.train_su2 import (  # noqa: E402
	SU2FeatureAugmenter,
	_initialise_callbacks,
	_prepare_features,
	_to_1d,
	build_pipeline,
	infer_test_file,
	infer_train_file,
	load_preprocess_policies,
	load_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Sweep SU2 hyperparameters using OOF validation.")
	ap.add_argument("--config-path", type=str, default="configs/feature_generation.yaml", help="Path to feature generation YAML")
	ap.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml", help="Path to preprocess policy YAML")
	ap.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw train/test files")
	ap.add_argument("--train-file", type=str, default=None, help="Optional explicit train file path")
	ap.add_argument("--test-file", type=str, default=None, help="Optional explicit test file path")
	ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
	ap.add_argument("--id-col", type=str, default="date_id")
	ap.add_argument("--out-dir", type=str, default="results/ablation/SU2", help="Directory to store sweep outputs")
	ap.add_argument("--n-splits", type=int, default=5, help="Number of TimeSeriesSplit folds")
	ap.add_argument("--gap", type=int, default=0, help="Gap between train and validation indices")
	ap.add_argument("--min-val-size", type=int, default=0, help="Skip folds with validation size smaller than this after gap trimming")
	ap.add_argument("--numeric-fill-value", type=float, default=0.0, help="Fill value applied after SU1/SU2 feature generation")
	ap.add_argument("--rolling-subset-sizes", type=int, nargs="*", default=None, help="Subset sizes (prefix) to evaluate for rolling windows. Defaults to all prefix sizes.")
	ap.add_argument("--ewma-subset-sizes", type=int, nargs="*", default=None, help="Subset sizes (prefix) to evaluate for EWMA alphas. Defaults to all prefix sizes.")
	ap.add_argument("--learning-rate", type=float, default=0.05)
	ap.add_argument("--n-estimators", type=int, default=600)
	ap.add_argument("--num-leaves", type=int, default=63)
	ap.add_argument("--min-data-in-leaf", type=int, default=32)
	ap.add_argument("--feature-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-fraction", type=float, default=0.9)
	ap.add_argument("--bagging-freq", type=int, default=1)
	ap.add_argument("--random-state", type=int, default=42)
	ap.add_argument("--verbosity", type=int, default=-1)
	ap.add_argument("--signal-optimize-for", type=str, choices=("msr", "msr_down", "vmsr"), default="msr")
	ap.add_argument("--signal-mult-grid", type=float, nargs="+", default=(0.5, 0.75, 1.0, 1.25, 1.5))
	ap.add_argument("--signal-lo-grid", type=float, nargs="+", default=(0.8, 0.9, 1.0))
	ap.add_argument("--signal-hi-grid", type=float, nargs="+", default=(1.0, 1.1, 1.2))
	ap.add_argument("--signal-lam-grid", type=float, nargs="+", default=(0.0,))
	ap.add_argument("--signal-eps", type=float, default=1e-8)
	ap.add_argument("--max-combos", type=int, default=None, help="Optional cap on number of candidate combinations to evaluate")
	# execution controls
	ap.add_argument("--num-threads", type=int, default=-1, help="Number of threads for model training (LightGBM n_jobs)")
	# staged search controls
	ap.add_argument("--from-best", type=str, default=None, help="Path to an existing sweep_summary.csv to refine from (Stage 2)")
	ap.add_argument("--topk", type=int, default=12, help="Top-K candidates to refine when using --from-best")
	# alias for topk (user may pass --top-k)
	ap.add_argument("--top-k", dest="top_k_alias", type=int, default=None, help="Alias for --topk")
	# ranking strategy for Stage 2 selection
	ap.add_argument("--rank-by", type=str, choices=("msr", "rmse", "joint"), default="msr", help="Ranking metric for selecting top-k from summary (Stage 2)")
	# diversity toggle: pick best per structural group before ranking remainder
	ap.add_argument("--diversity", action="store_true", help="Ensure structural diversity (one per (metrics,signals,roll,ewma,trans,normalization)) in Stage 2 pre-selection")
	ap.add_argument("--fix-input-sources", nargs="*", default=None, help="Fix input_sources to this list (e.g., m gap_ffill run_na run_obs)")
	ap.add_argument("--fix-target-all", action="store_true", help="Force target_groups to all groups [D,M,E,I,P,S,V]")
	return ap.parse_args(argv)


def _load_raw_config(path: Path) -> Mapping[str, Any]:
	with path.open("r", encoding="utf-8") as fh:
		return cast(Mapping[str, Any], yaml.safe_load(fh) or {})


def _prefix_subsets(values: Iterable[Any], subset_sizes: Sequence[int] | None) -> List[Tuple[Any, ...]]:
	unique_values = list(dict.fromkeys(values))
	if not unique_values:
		return []
	if subset_sizes:
		sizes = [size for size in subset_sizes if size >= 1]
		if not sizes:
			sizes = [len(unique_values)]
	else:
		sizes = list(range(1, len(unique_values) + 1))
	result: List[Tuple[Any, ...]] = []
	for size in sizes:
		k = min(len(unique_values), size)
		subset = tuple(unique_values[:k])
		if subset not in result:
			result.append(subset)
	return result


def _stringify_sequence(seq: Iterable[Any]) -> str:
	return "|".join(str(item) for item in seq)


def _ensure_list(value: Any, fallback: Sequence[Any]) -> List[Any]:
	result: List[Any]
	if value is None:
		result = list(fallback)
	elif isinstance(value, (list, tuple, set)):
		result = list(value)
		if not result:
			result = list(fallback)
	else:
		result = [value]
	if not result:
		result = list(fallback)
	return result


def _ensure_sequence_candidates(value: Any, fallback: Sequence[Any] | Sequence[Sequence[Any]] | None) -> List[List[Any]]:
	if value is None:
		if fallback is None:
			return [[]]
		fallback_list = list(fallback)
		if not fallback_list:
			return [[]]
		first = fallback_list[0]
		if isinstance(first, (list, tuple, set)):
			return [list(seq) for seq in fallback_list]
		return [list(fallback_list)]

	if isinstance(value, (list, tuple)):
		value_list = list(value)
		if not value_list:
			return _ensure_sequence_candidates(None, fallback)
		first = value_list[0]
		if isinstance(first, (list, tuple, set)):
			return [list(seq) for seq in value_list]
		return [list(value_list)]

	return [[value]]


def _build_candidate_mappings(
	base_mapping: Mapping[str, Any],
	rolling_candidates: Sequence[Tuple[Any, ...]],
	ewma_candidates: Sequence[Tuple[Any, ...]],
) -> List[Dict[str, Any]]:
	return _build_candidate_mappings_adv(
		base_mapping,
		rolling_candidates,
		ewma_candidates,
		max_combos=None,
		allow_transition_windows_variation=True,
		allow_normalization_windows_variation=True,
		fixed_fields=None,
	)


def _build_candidate_mappings_adv(
	base_mapping: Mapping[str, Any],
	rolling_candidates: Sequence[Tuple[Any, ...]],
	ewma_candidates: Sequence[Tuple[Any, ...]],
	max_combos: int | None = None,
	allow_transition_windows_variation: bool = True,
	allow_normalization_windows_variation: bool = True,
	fixed_fields: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
	candidates: List[Dict[str, Any]] = []
	features_base = cast(Dict[str, Any], copy.deepcopy(base_mapping.get("features", {}))) if isinstance(base_mapping.get("features"), MappingABC) else {}
	rolling_base = cast(Dict[str, Any], copy.deepcopy(features_base.get("rolling", {}))) if isinstance(features_base.get("rolling"), MappingABC) else {}
	ewma_base = cast(Dict[str, Any], copy.deepcopy(features_base.get("ewma", {}))) if isinstance(features_base.get("ewma"), MappingABC) else {}
	transitions_base = cast(Dict[str, Any], copy.deepcopy(features_base.get("transitions", {}))) if isinstance(features_base.get("transitions"), MappingABC) else {}
	normalization_base = cast(Dict[str, Any], copy.deepcopy(features_base.get("normalization", {}))) if isinstance(features_base.get("normalization"), MappingABC) else {}

	default_toggle_values = {
		"include_rolling": bool(rolling_base) if isinstance(rolling_base, MappingABC) else True,
		"include_ewma": bool(ewma_base.get("signals")) if isinstance(ewma_base, MappingABC) else True,
		"include_transitions": bool(transitions_base) if isinstance(transitions_base, MappingABC) else True,
		"include_normalization": bool(normalization_base) if isinstance(normalization_base, MappingABC) else False,
	}
	toggle_options: Dict[str, List[bool]] = {}
	for key, default_value in default_toggle_values.items():
		if fixed_fields and key in fixed_fields:
			# hard-fix the toggle to a single value
			raw_list = [bool(fixed_fields[key])]
		else:
			raw_list = _ensure_list(base_mapping.get(key), [default_value])
		if not raw_list:
			raw_list = [default_value]
		toggle_options[key] = [bool(item) for item in raw_list]

	# Additional fine-grained toggles/candidates
	if fixed_fields and "rolling_include_current" in fixed_fields:
		rolling_include_current_candidates = [bool(fixed_fields["rolling_include_current"])]
	else:
		rolling_include_current_candidates = [
			bool(v) for v in _ensure_list(
				rolling_base.get("include_current") if isinstance(rolling_base, MappingABC) else None,
				[bool(rolling_base.get("include_current", False))] if isinstance(rolling_base, MappingABC) else [False],
			)
		]

	if fixed_fields and "ewma_include_std" in fixed_fields:
		ewma_include_std_candidates = [bool(fixed_fields["ewma_include_std"])]
	else:
		ewma_include_std_candidates = [
			bool(v) for v in _ensure_list(
				ewma_base.get("include_std") if isinstance(ewma_base, MappingABC) else None,
				[bool(ewma_base.get("include_std", True))] if isinstance(ewma_base, MappingABC) else [True],
			)
		]
	if fixed_fields and "ewma_reset_each_fold" in fixed_fields:
		ewma_reset_each_fold_candidates = [bool(fixed_fields["ewma_reset_each_fold"])]
	else:
		ewma_reset_each_fold_candidates = [
			bool(v) for v in _ensure_list(
				ewma_base.get("reset_each_fold") if isinstance(ewma_base, MappingABC) else None,
				[bool(ewma_base.get("reset_each_fold", True))] if isinstance(ewma_base, MappingABC) else [True],
			)
		]

	# clip_max may be a scalar or a list of candidates in YAML; avoid eagerly evaluating fallbacks
	clip_raw = fixed_fields.get("clip_max") if fixed_fields and "clip_max" in fixed_fields else base_mapping.get("clip_max")
	if isinstance(clip_raw, (list, tuple, set)):
		clip_max_candidates = [int(v) for v in clip_raw]
	elif clip_raw is None:
		clip_max_candidates = [60]
	else:
		clip_max_candidates = [int(clip_raw)]
	if fixed_fields and "drop_constant_columns" in fixed_fields:
		drop_constant_candidates = [bool(fixed_fields["drop_constant_columns"])]
	else:
		drop_constant_candidates = [
			bool(v) for v in _ensure_list(base_mapping.get("drop_constant_columns"), [bool(base_mapping.get("drop_constant_columns", True))])
		]

	# dtype can be either a mapping or a list of mappings to sweep
	dtype_options: List[Mapping[str, Any]] = []
	dtype_field = fixed_fields.get("dtype") if fixed_fields and "dtype" in fixed_fields else base_mapping.get("dtype")
	if isinstance(dtype_field, list):
		dtype_options = [cast(Mapping[str, Any], d) for d in dtype_field or []]
		if not dtype_options:
			dtype_options = [cast(Mapping[str, Any], base_mapping.get("dtype", {}))]
	elif isinstance(dtype_field, MappingABC):
		dtype_options = [cast(Mapping[str, Any], dtype_field or {})]
	else:
		dtype_options = []

	metrics_fallback = rolling_base.get("include_metrics") if isinstance(rolling_base, MappingABC) else None
	if fixed_fields and "include_metrics" in fixed_fields:
		include_metrics_candidates = [list(fixed_fields["include_metrics"])]
	else:
		include_metrics_candidates = _ensure_sequence_candidates(base_mapping.get("include_metrics"), metrics_fallback)
	if not include_metrics_candidates:
		include_metrics_candidates = _ensure_sequence_candidates(None, [["mean", "std"]])

	signals_fallback = ewma_base.get("signals") if isinstance(ewma_base, MappingABC) else None
	if fixed_fields and "signals" in fixed_fields:
		signals_candidates = [list(fixed_fields["signals"])]
	else:
		signals_candidates = _ensure_sequence_candidates(base_mapping.get("signals"), signals_fallback)
	if not signals_candidates:
		signals_candidates = _ensure_sequence_candidates(None, [["m", "gap_ffill"]])

	trans_windows_raw = transitions_base.get("windows") if isinstance(transitions_base, MappingABC) else None
	if trans_windows_raw is not None and allow_transition_windows_variation:
		trans_windows_candidates = [tuple(int(v) for v in seq) for seq in _ensure_sequence_candidates(trans_windows_raw, None) if seq]
	else:
		fallback_windows: List[List[int]] = []
		if isinstance(transitions_base, MappingABC):
			flip_w = transitions_base.get("flip_rate_windows")
			if isinstance(flip_w, (list, tuple)) and flip_w:
				fallback_windows.append([int(v) for v in flip_w])
			burst_w = transitions_base.get("burst_score_windows")
			if isinstance(burst_w, (list, tuple)) and burst_w and not fallback_windows:
				fallback_windows.append([int(v) for v in burst_w])
		trans_windows_candidates = [tuple(seq) for seq in fallback_windows] if fallback_windows else []

	norm_windows_raw = normalization_base.get("windows") if isinstance(normalization_base, MappingABC) else None
	if norm_windows_raw is not None and allow_normalization_windows_variation:
		norm_windows_candidates = [tuple(int(v) for v in seq) for seq in _ensure_sequence_candidates(norm_windows_raw, None) if seq]
	else:
		fallback_norm_windows: List[List[int]] = []
		if isinstance(normalization_base, MappingABC):
			minmax_w = normalization_base.get("minmax_windows")
			if isinstance(minmax_w, (list, tuple)) and minmax_w:
				fallback_norm_windows.append([int(v) for v in minmax_w])
			rank_w = normalization_base.get("rank_windows")
			if isinstance(rank_w, (list, tuple)) and rank_w and not fallback_norm_windows:
				fallback_norm_windows.append([int(v) for v in rank_w])
		norm_windows_candidates = [tuple(seq) for seq in fallback_norm_windows] if fallback_norm_windows else []

	recovery_clip_fallback = base_mapping.get("recovery_clip", 60)
	if fixed_fields and "recovery_clip" in fixed_fields:
		recovery_clip_candidates = [int(fixed_fields["recovery_clip"])]
	else:
		recovery_clip_candidates = [int(v) for v in _ensure_list(transitions_base.get("recovery_clip"), [recovery_clip_fallback])]

	mode_default: List[str]
	if isinstance(normalization_base, MappingABC) and "mode" in normalization_base:
		mode_default = _ensure_list(normalization_base.get("mode"), ["both"])
	elif isinstance(normalization_base, MappingABC):
		both_present = "minmax_windows" in normalization_base and "rank_windows" in normalization_base
		if both_present:
			mode_default = ["both"]
		elif "minmax_windows" in normalization_base:
			mode_default = ["minmax"]
		elif "rank_windows" in normalization_base:
			mode_default = ["rank"]
		else:
			mode_default = ["both"]
	else:
		mode_default = ["both"]
	if fixed_fields and "normalization_mode" in fixed_fields:
		norm_mode_candidates = [str(fixed_fields["normalization_mode"]).lower()]
	else:
		norm_mode_candidates = [str(v) for v in _ensure_list(base_mapping.get("normalization_mode"), mode_default)]

	epsilon_fallback_value = normalization_base.get("epsilon", base_mapping.get("epsilon", 1.0e-6)) if isinstance(normalization_base, MappingABC) else base_mapping.get("epsilon", 1.0e-6)
	if epsilon_fallback_value is None:
		epsilon_fallback_value = 1.0e-6
	if fixed_fields and "epsilon" in fixed_fields:
		norm_epsilon_candidates = [float(fixed_fields["epsilon"])]
	else:
		norm_epsilon_candidates = [float(v) for v in _ensure_list(normalization_base.get("epsilon") if isinstance(normalization_base, MappingABC) else None, [epsilon_fallback_value])]

	if fixed_fields and "input_sources" in fixed_fields:
		input_sources_candidates = [list(fixed_fields["input_sources"])]
	else:
		input_sources_candidates = _ensure_sequence_candidates(base_mapping.get("input_sources"), base_mapping.get("input_sources", ["m", "gap_ffill", "run_na", "run_obs"]))
	if not input_sources_candidates:
		input_sources_candidates = [["m", "gap_ffill", "run_na", "run_obs"]]

	target_groups_section = base_mapping.get("target_groups", {})
	if fixed_fields and "target_groups" in fixed_fields and isinstance(fixed_fields["target_groups"], MappingABC):
		include_candidates = [list(fixed_fields["target_groups"].get("include", []))]
		exclude_candidates = [list(fixed_fields["target_groups"].get("exclude", []))]
	elif isinstance(target_groups_section, MappingABC):
		include_candidates = _ensure_sequence_candidates(target_groups_section.get("include"), target_groups_section.get("include", ["D", "M", "E", "I", "P", "S", "V"]))
		exclude_candidates = _ensure_sequence_candidates(target_groups_section.get("exclude"), target_groups_section.get("exclude", [[]]))
	else:
		include_candidates = [["D", "M", "E", "I", "P", "S", "V"]]
		exclude_candidates = [[]]

	for rolling, ewma in itertools.product(rolling_candidates, ewma_candidates):
		trans_window_options = trans_windows_candidates if (allow_transition_windows_variation and trans_windows_candidates) else [tuple(int(v) for v in rolling)]
		norm_window_options = norm_windows_candidates if (allow_normalization_windows_variation and norm_windows_candidates) else [tuple(int(v) for v in rolling)]

		for inc_roll, inc_ewma, inc_trans, inc_norm in itertools.product(
			toggle_options["include_rolling"],
			toggle_options["include_ewma"],
			toggle_options["include_transitions"],
			toggle_options["include_normalization"],
		):
			for metrics in include_metrics_candidates:
				for sigs in signals_candidates:
					roll_current_iter = (rolling_include_current_candidates if inc_roll else [False])
					for roll_inc_current in roll_current_iter:
						ewma_std_iter = (ewma_include_std_candidates if inc_ewma else [False])
						for ewma_inc_std in ewma_std_iter:
							reset_iter = (ewma_reset_each_fold_candidates if inc_ewma else [True])
							for ewma_reset in reset_iter:
								for clip_max_val in (clip_max_candidates or [60]):
									for drop_const in (drop_constant_candidates or [True]):
										for dtype_map in (dtype_options or [None]):
											trans_windows_iter = trans_window_options if inc_trans else [tuple()]
											for tw in trans_windows_iter:
												recovery_iter = recovery_clip_candidates if inc_trans else [None]
												for rc in recovery_iter:
													norm_mode_iter = norm_mode_candidates if inc_norm else [None]
													for nm in norm_mode_iter:
														norm_windows_iter = norm_window_options if inc_norm else [tuple()]
														for nw in norm_windows_iter:
															eps_iter = norm_epsilon_candidates if inc_norm else [None]
															for eps in eps_iter:
																for srcs in input_sources_candidates:
																	for include_groups, exclude_groups in itertools.product(include_candidates, exclude_candidates):
																		candidate = cast(Dict[str, Any], copy.deepcopy(base_mapping))
																		candidate["rolling_windows"] = list(rolling)
																		candidate["ewma_alpha"] = list(ewma)
																		candidate["include_rolling"] = bool(inc_roll)
																		candidate["include_ewma"] = bool(inc_ewma)
																		candidate["include_transitions"] = bool(inc_trans)
																		candidate["include_normalization"] = bool(inc_norm)
																		candidate["include_metrics"] = list(metrics) if inc_roll else []
																		candidate["signals"] = list(sigs) if inc_ewma else []
																		candidate["input_sources"] = list(srcs)
																		candidate["target_groups"] = {"include": list(include_groups), "exclude": list(exclude_groups)}
																		candidate["clip_max"] = int(clip_max_val)
																		candidate["drop_constant_columns"] = bool(drop_const)
																		if dtype_map is not None:
																			candidate["dtype"] = cast(Dict[str, Any], copy.deepcopy(dict(dtype_map)))
																		if inc_trans and rc is not None:
																			candidate["recovery_clip"] = int(rc)
																		if inc_norm and eps is not None:
																			candidate["epsilon"] = float(eps)

																		features = cast(Dict[str, Any], candidate.setdefault("features", {}))

																		if inc_roll:
																			rolling_section = cast(Dict[str, Any], features.setdefault("rolling", {}))
																			rolling_section["include_metrics"] = list(metrics)
																			rolling_section["include_current"] = bool(roll_inc_current)
																		else:
																			features.pop("rolling", None)

																		if inc_ewma:
																			ewma_section = cast(Dict[str, Any], features.setdefault("ewma", {}))
																			ewma_section["signals"] = list(sigs)
																			ewma_section["include_std"] = bool(ewma_inc_std)
																			ewma_section["reset_each_fold"] = bool(ewma_reset)
																		else:
																			features.pop("ewma", None)

																		if inc_trans:
																			transitions_section = cast(Dict[str, Any], features.setdefault("transitions", {}))
																			windows_list = list(tw) if tw else list(rolling)
																			transitions_section["flip_rate_windows"] = windows_list
																			transitions_section["burst_score_windows"] = windows_list
																			if rc is not None:
																				transitions_section["recovery_clip"] = int(rc)
																			candidate["transition_windows"] = windows_list
																		else:
																			features.pop("transitions", None)
																			candidate["transition_windows"] = []

																		if inc_norm:
																			norm_section = cast(Dict[str, Any], features.setdefault("normalization", {}))
																			windows_list = list(nw) if nw else list(rolling)
																			mode_value = (nm or "both").lower()
																			norm_section["mode"] = mode_value
																			if mode_value == "minmax":
																				norm_section["minmax_windows"] = windows_list
																				norm_section.pop("rank_windows", None)
																			elif mode_value == "rank":
																				norm_section["rank_windows"] = windows_list
																				norm_section.pop("minmax_windows", None)
																			else:
																				norm_section["minmax_windows"] = windows_list
																				norm_section["rank_windows"] = windows_list
																			if eps is not None:
																				norm_section["epsilon"] = float(eps)
																			candidate["normalization_mode"] = mode_value
																			candidate["normalization_windows"] = windows_list
																			candidate["normalization_epsilon"] = float(eps) if eps is not None else float(norm_section.get("epsilon", epsilon_fallback_value))
																		else:
																			features.pop("normalization", None)
																			candidate["normalization_mode"] = None
																			candidate["normalization_windows"] = []
																			candidate["normalization_epsilon"] = None

																		if not features:
																			candidate.pop("features", None)

																		candidates.append(candidate)
																		if max_combos is not None and len(candidates) >= max_combos:
																			return candidates
	return candidates


def _json_safe(value: Any) -> Any:
	"""Recursively convert objects into JSON serializable structures."""
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, (np.integer, np.floating)):
		return value.item()
	if isinstance(value, (dt.datetime, dt.date)):
		return value.isoformat()
	if isinstance(value, pd.Timestamp):
		return value.isoformat()
	if isinstance(value, np.ndarray):
		return [_json_safe(item) for item in value.tolist()]
	if isinstance(value, (list, tuple, set)):
		return [_json_safe(item) for item in value]
	if isinstance(value, MappingABC):
		return {str(key): _json_safe(val) for key, val in value.items()}
	return str(value)


def _evaluate_candidate(
	X: pd.DataFrame,
	y: pd.Series,
	su1_config: SU1Config,
	su2_config: SU2Config,
	preprocess_settings: Mapping[str, Dict[str, Any]],
	split_indices: Sequence[Tuple[np.ndarray, np.ndarray]],
	args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
	model_kwargs = {
		"learning_rate": args.learning_rate,
		"n_estimators": args.n_estimators,
		"num_leaves": args.num_leaves,
		"min_data_in_leaf": args.min_data_in_leaf,
		"feature_fraction": args.feature_fraction,
		"bagging_fraction": args.bagging_fraction,
		"bagging_freq": args.bagging_freq,
		"random_state": args.random_state,
		"n_jobs": int(args.num_threads) if getattr(args, "num_threads", -1) is not None else -1,
		"verbosity": args.verbosity,
	}
	base_pipeline = build_pipeline(
		su1_config,
		su2_config,
		preprocess_settings,
		numeric_fill_value=args.numeric_fill_value,
		model_kwargs=model_kwargs,
		random_state=args.random_state,
	)
	callbacks = _initialise_callbacks(base_pipeline.named_steps["model"])

	X_np = X.reset_index(drop=True)
	y_np = y.reset_index(drop=True)
	y_np_array = y_np.to_numpy()

	fold_indices_all = np.full(len(X_np), -1, dtype=int)
	for fold_id, (train_idx, val_idx) in enumerate(split_indices):
		fold_indices_all[train_idx] = fold_id
		fold_indices_all[val_idx] = fold_id
	if np.any(fold_indices_all < 0):
		raise RuntimeError("Fold index assignment failed; found unassigned rows.")

	# fit performs structural preparation only; SU2FeatureAugmenter computes any statistics with past-only data inside transform.
	su2_prefit = SU2FeatureAugmenter(su1_config, su2_config, fill_value=args.numeric_fill_value)
	su2_prefit.fit(X_np)
	X_augmented_all = su2_prefit.transform(X_np, fold_indices=fold_indices_all)
	core_pipeline_template = Pipeline(base_pipeline.steps[1:])

	oof_pred = np.full(len(X_np), np.nan, dtype=float)
	fold_logs: List[Dict[str, Any]] = []

	signal_mult_grid = tuple(float(x) for x in args.signal_mult_grid)
	signal_lo_grid = tuple(float(x) for x in args.signal_lo_grid)
	signal_hi_grid = tuple(float(x) for x in args.signal_hi_grid)
	signal_lam_grid = tuple(float(x) for x in args.signal_lam_grid)

	for fold_idx, (train_idx_raw, val_idx_raw) in enumerate(split_indices, start=1):
		train_idx = train_idx_raw.copy()
		val_idx = val_idx_raw.copy()
		if args.gap > 0:
			if len(train_idx) > args.gap:
				train_idx = train_idx[:-args.gap]
			else:
				train_idx = np.array([], dtype=int)
			if len(val_idx) > args.gap:
				val_idx = val_idx[args.gap:]
			else:
				val_idx = np.array([], dtype=int)
		if len(train_idx) == 0 or len(val_idx) == 0:
			fold_logs.append(
				{
					"fold": fold_idx,
					"status": "skipped",
					"reason": "empty after gap",
					"train_size": int(len(train_idx)),
					"val_size": int(len(val_idx)),
				}
			)
			continue
		if args.min_val_size and len(val_idx) < args.min_val_size:
			fold_logs.append(
				{
					"fold": fold_idx,
					"status": "skipped",
					"reason": "val<min_val_size",
					"train_size": int(len(train_idx)),
					"val_size": int(len(val_idx)),
				}
			)
			continue

		X_train = X_augmented_all.iloc[train_idx]
		y_train = y_np.iloc[train_idx]
		X_valid = X_augmented_all.iloc[val_idx]
		y_valid = y_np.iloc[val_idx]

		fold_pipeline = cast(Pipeline, clone(core_pipeline_template))
		fit_kwargs: Dict[str, Any] = {}
		if callbacks:
			fit_kwargs["model__callbacks"] = callbacks
			fit_kwargs["model__eval_set"] = [(X_valid, y_valid)]
			fit_kwargs["model__eval_metric"] = "rmse"
		fit_kwargs.setdefault("model__feature_name", None)
		fold_pipeline.fit(X_train, y_train, **fit_kwargs)
		pred = fold_pipeline.predict(X_valid)
		pred = _to_1d(pred)
		oof_pred[val_idx] = pred

		if len(pred):
			residuals = pred - y_valid.to_numpy()
			rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
		else:
			rmse_val = float("nan")

		best_params, fold_grid = grid_search_msr(
			y_pred=pred,
			y_true=y_valid.to_numpy(),
			mult_grid=signal_mult_grid,
			lo_grid=signal_lo_grid,
			hi_grid=signal_hi_grid,
			eps=float(args.signal_eps),
			optimize_for=args.signal_optimize_for,
			lam_grid=signal_lam_grid if args.signal_optimize_for == "vmsr" else (0.0,),
		)
		if args.signal_optimize_for == "vmsr":
			matching = [row for row in fold_grid if row["mult"] == best_params.mult and row["lo"] == best_params.lo and row["hi"] == best_params.hi]
			lam_val = float(max(matching, key=lambda r: r.get("vmsr", float("-inf"))).get("vmsr_lam", 0.0)) if matching else float(signal_lam_grid[0]) if signal_lam_grid else 0.0
		else:
			lam_val = 0.0
		fold_metrics = evaluate_msr_proxy(pred, y_valid.to_numpy(), best_params, eps=float(args.signal_eps), lam=lam_val)
		fold_logs.append(
			{
				"fold": fold_idx,
				"status": "ok",
				"train_size": int(len(train_idx)),
				"val_size": int(len(val_idx)),
				"rmse": rmse_val,
				"best_mult": float(best_params.mult),
				"best_lo": float(best_params.lo),
				"best_hi": float(best_params.hi),
				"best_lam": float(lam_val),
				"msr": float(fold_metrics["msr"]),
				"msr_down": float(fold_metrics["msr_down"]),
				"vmsr": float(fold_metrics["vmsr"]),
				"mean": float(fold_metrics["mean"]),
				"std": float(fold_metrics["std"]),
				"std_down": float(fold_metrics["std_down"]),
			}
		)

	valid_mask = ~np.isnan(oof_pred)
	coverage = float(np.mean(valid_mask)) if valid_mask.size else 0.0
	if valid_mask.any():
		overall_rmse = float(np.sqrt(np.mean((oof_pred[valid_mask] - y_np_array[valid_mask]) ** 2)))
		best_params_global, grid_all = grid_search_msr(
			y_pred=oof_pred[valid_mask],
			y_true=y_np_array[valid_mask],
			mult_grid=signal_mult_grid,
			lo_grid=signal_lo_grid,
			hi_grid=signal_hi_grid,
			eps=float(args.signal_eps),
			optimize_for=args.signal_optimize_for,
			lam_grid=signal_lam_grid if args.signal_optimize_for == "vmsr" else (0.0,),
		)
		if args.signal_optimize_for == "vmsr":
			matching = [row for row in grid_all if row["mult"] == best_params_global.mult and row["lo"] == best_params_global.lo and row["hi"] == best_params_global.hi]
			lam_global = float(max(matching, key=lambda r: r.get("vmsr", float("-inf"))).get("vmsr_lam", 0.0)) if matching else float(signal_lam_grid[0]) if signal_lam_grid else 0.0
		else:
			lam_global = 0.0
		best_metrics_global = evaluate_msr_proxy(oof_pred[valid_mask], y_np_array[valid_mask], best_params_global, eps=float(args.signal_eps), lam=lam_global)
	else:
		overall_rmse = float("nan")
		best_params_global = PostProcessParams()
		lam_global = 0.0
		best_metrics_global = {
			"rmse": float("nan"),
			"msr": float("nan"),
			"msr_down": float("nan"),
			"vmsr": float("nan"),
			"vmsr_lam": float("nan"),
			"mean": float("nan"),
			"std": float("nan"),
			"std_down": float("nan"),
		}

	metrics = {
		"oof_rmse": overall_rmse,
		"coverage": coverage,
		"best_mult": float(best_params_global.mult),
		"best_lo": float(best_params_global.lo),
		"best_hi": float(best_params_global.hi),
		"best_lam": float(lam_global),
		"oof_msr": float(best_metrics_global["msr"]),
		"oof_msr_down": float(best_metrics_global["msr_down"]),
		"oof_vmsr": float(best_metrics_global["vmsr"]),
		"oof_mean": float(best_metrics_global["mean"]),
		"oof_std": float(best_metrics_global["std"]),
		"oof_std_down": float(best_metrics_global["std_down"]),
	}
	return metrics, fold_logs


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)

	config_path = Path(args.config_path).resolve()
	if not config_path.exists():
		raise FileNotFoundError(f"Config not found: {config_path}")
	preprocess_config_path = Path(args.preprocess_config).resolve()
	if not preprocess_config_path.exists():
		raise FileNotFoundError(f"Preprocess config not found: {preprocess_config_path}")

	data_dir = Path(args.data_dir).resolve()
	train_path = infer_train_file(data_dir, args.train_file)
	test_path = infer_test_file(data_dir, args.test_file)
	print(f"[info] train file: {train_path}")
	print(f"[info] test file : {test_path}")

	train_df = load_table(train_path)
	test_df = load_table(test_path)

	su1_config = load_su1_config(config_path)
	preprocess_settings = load_preprocess_policies(preprocess_config_path)
	sanitised_preprocess: Dict[str, Dict[str, Any]] = {}
	for key, section in preprocess_settings.items():
		if isinstance(section, MappingABC):
			sanitised_section = dict(section)
			sanitised_section["calendar_column"] = None
			sanitised_preprocess[key] = sanitised_section
		else:
			sanitised_preprocess[key] = section
	preprocess_settings = sanitised_preprocess

	X, y, _ = _prepare_features(train_df, test_df, target_col=args.target_col, id_col=args.id_col)
	splitter = TimeSeriesSplit(n_splits=args.n_splits)
	split_indices = list(splitter.split(X))

	raw_config = _load_raw_config(config_path)
	su2_mapping = cast(Mapping[str, Any], raw_config.get("su2", {}))
	if not su2_mapping:
		raise ValueError("'su2' section is missing in feature_generation.yaml")

	rolling_candidates = _prefix_subsets(su2_mapping.get("rolling_windows", []), args.rolling_subset_sizes)
	ewma_candidates = _prefix_subsets(su2_mapping.get("ewma_alpha", []), args.ewma_subset_sizes)
	# Fallbacks: avoid calling SU2Config loader because YAML may contain list-valued sweep candidates
	if not rolling_candidates:
		fallback_roll = _ensure_list(su2_mapping.get("rolling_windows"), [5, 10])
		rolling_candidates = [tuple(int(v) for v in fallback_roll)]
	if not ewma_candidates:
		fallback_ewma = _ensure_list(su2_mapping.get("ewma_alpha"), [0.1, 0.3])
		ewma_candidates = [tuple(float(v) for v in fallback_ewma)]

	# Stage 1 (coarse) default: restrict windows variation and fix auxiliary fields if requested
	fixed_fields_stage1: Dict[str, Any] | None = None
	if args.fix_input_sources or args.fix_target_all:
		fixed_fields_stage1 = {}
		if args.fix_input_sources:
			fixed_fields_stage1["input_sources"] = list(args.fix_input_sources)
		if args.fix_target_all:
			fixed_fields_stage1["target_groups"] = {"include": ["D", "M", "E", "I", "P", "S", "V"], "exclude": []}

	# Freeze fine-grained params in coarse stage to defaults in config mapping
	def _defaults_from_mapping(m: Mapping[str, Any]) -> Dict[str, Any]:
		features = cast(Mapping[str, Any], m.get("features", {})) if isinstance(m.get("features"), MappingABC) else {}
		rolling_m = cast(Mapping[str, Any], features.get("rolling", {})) if isinstance(features.get("rolling"), MappingABC) else {}
		ewma_m = cast(Mapping[str, Any], features.get("ewma", {})) if isinstance(features.get("ewma"), MappingABC) else {}
		norm_m = cast(Mapping[str, Any], features.get("normalization", {})) if isinstance(features.get("normalization"), MappingABC) else {}
		clip_val_raw = m.get("clip_max", 60)
		if isinstance(clipp := m.get("clip_max"), (list, tuple)) and clipp:
			clip_fixed = int(list(clipp)[0])
		else:
			clip_fixed = int(clip_val_raw if not isinstance(clip_val_raw, (list, tuple)) else 60)
		norm_mode_value = (
			norm_m.get("mode") if isinstance(norm_m, MappingABC) and "mode" in norm_m else None
		)
		if not norm_mode_value:
			if isinstance(norm_m, MappingABC) and ("minmax_windows" in norm_m and "rank_windows" in norm_m):
				norm_mode_value = "both"
			elif isinstance(norm_m, MappingABC) and "minmax_windows" in norm_m:
				norm_mode_value = "minmax"
			elif isinstance(norm_m, MappingABC) and "rank_windows" in norm_m:
				norm_mode_value = "rank"
			else:
				norm_mode_value = "both"
		epsilon_fixed = float(norm_m.get("epsilon", m.get("epsilon", 1.0e-6)))
		defaults: Dict[str, Any] = {
			"rolling_include_current": bool(rolling_m.get("include_current", False)),
			"ewma_include_std": bool(ewma_m.get("include_std", True)),
			"ewma_reset_each_fold": bool(ewma_m.get("reset_each_fold", True)),
			"clip_max": clip_fixed,
			"drop_constant_columns": bool(m.get("drop_constant_columns", True)),
			"dtype": m.get("dtype", {}),
			"normalization_mode": norm_mode_value,
			"epsilon": epsilon_fixed,
			"recovery_clip": int(m.get("recovery_clip", 60)),
		}
		return defaults

	defaults_stage1 = _defaults_from_mapping(su2_mapping)
	if fixed_fields_stage1 is None:
		fixed_fields_stage1 = {}
	fixed_fields_stage1.update(defaults_stage1)

	# If Stage 2 (refinement) is requested via --from-best, rebuild candidate_mappings accordingly
	if args.from_best:
		from_path = Path(args.from_best).resolve()
		if not from_path.exists():
			raise FileNotFoundError(f"from-best summary not found: {from_path}")
		base_dir_for_best = from_path.parent
		summary_df = pd.read_csv(from_path)
		if summary_df.empty:
			raise ValueError("from-best summary is empty")
		# apply alias if provided
		if getattr(args, "top_k_alias", None) is not None:
			args.topk = int(args.top_k_alias)
		# ranking strategy
		if args.rank_by == "msr":
			summary_df = summary_df.sort_values(by=["oof_msr", "oof_rmse"], ascending=[False, True])
		elif args.rank_by == "rmse":
			summary_df = summary_df.sort_values(by="oof_rmse", ascending=True)
		else:  # joint
			# normalise metrics then combine: joint_score = norm_msr - norm_rmse
			msr_vals = summary_df["oof_msr"].to_numpy()
			rmse_vals = summary_df["oof_rmse"].to_numpy()
			if msr_vals.size and rmse_vals.size:
				msr_norm = (msr_vals - msr_vals.min()) / (msr_vals.ptp() + 1e-12)
				rmse_norm = (rmse_vals - rmse_vals.min()) / (rmse_vals.ptp() + 1e-12)
				joint_score = msr_norm - rmse_norm
				summary_df = summary_df.assign(joint_score=joint_score).sort_values(by="joint_score", ascending=False)
		topk = int(args.topk)
		if topk < 1:
			topk = 1
		# diversity pre-selection if requested
		if args.diversity:
			group_cols = ["include_metrics", "signals", "rolling_windows", "ewma_alpha", "include_transitions", "include_normalization"]
			unique_groups = {}
			ordered_rows = []
			for _, r in summary_df.iterrows():
				key = tuple(r.get(c) for c in group_cols)
				if key not in unique_groups:
					unique_groups[key] = True
					ordered_rows.append(r)
					if len(ordered_rows) >= topk:
						break
			diverse_df = pd.DataFrame(ordered_rows)
			if len(diverse_df) < topk:
				# fill remaining from original order excluding already taken keys
				remaining = []
				for _, r in summary_df.iterrows():
					key = tuple(r.get(c) for c in group_cols)
					if key in unique_groups:
						continue
					remaining.append(r)
					if len(ordered_rows) + len(remaining) >= topk:
						break
				full_rows = ordered_rows + remaining
				selected_df = pd.DataFrame(full_rows)
			else:
				selected_df = diverse_df
		else:
			selected_df = summary_df.head(topk)
		candidate_mappings: List[Dict[str, Any]] = []
		for _, row in selected_df.iterrows():
			cid = int(row["config_id"]) if "config_id" in row else None
			if cid is None:
				continue
			cand_yaml = base_dir_for_best / f"su2_config_{cid:02d}.yaml"
			if not cand_yaml.exists():
				continue
			with cand_yaml.open("r", encoding="utf-8") as fh:
				cand_map = cast(Dict[str, Any], yaml.safe_load(fh) or {})
			# robust epsilon handling: allow list/None and fallback to 1e-6
			def _coerce_eps(v: Any) -> float:
				if isinstance(v, (list, tuple)):
					v = v[0] if len(v) > 0 else 1.0e-6
				if v is None:
					v = 1.0e-6
				return float(v)
			_eps_value = _coerce_eps(cand_map.get("normalization_epsilon", su2_mapping.get("epsilon", 1.0e-6)))
			roll_vals = tuple(int(v) for v in cand_map.get("rolling_windows", su2_mapping.get("rolling_windows", [])))
			ewma_vals = tuple(float(v) for v in cand_map.get("ewma_alpha", su2_mapping.get("ewma_alpha", [])))
			fixed_fields_fine: Dict[str, Any] = {
				"include_rolling": bool(cand_map.get("include_rolling", True)),
				"include_ewma": bool(cand_map.get("include_ewma", True)),
				"include_transitions": bool(cand_map.get("include_transitions", True)),
				"include_normalization": bool(cand_map.get("include_normalization", False)),
				"include_metrics": list(cand_map.get("include_metrics", [])),
				"signals": list(cand_map.get("signals", [])),
				"rolling_include_current": bool(cand_map.get("features", {}).get("rolling", {}).get("include_current", False)) if isinstance(cand_map.get("features"), MappingABC) else False,
				"ewma_include_std": bool(cand_map.get("features", {}).get("ewma", {}).get("include_std", True)) if isinstance(cand_map.get("features"), MappingABC) else True,
				"ewma_reset_each_fold": bool(cand_map.get("features", {}).get("ewma", {}).get("reset_each_fold", True)) if isinstance(cand_map.get("features"), MappingABC) else True,
				"clip_max": (lambda _v: int(_v if not isinstance(_v, (list, tuple)) else (_v[0] if _v else 60)))(cand_map.get("clip_max", su2_mapping.get("clip_max", 60))),
				"drop_constant_columns": bool(cand_map.get("drop_constant_columns", su2_mapping.get("drop_constant_columns", True))),
				"dtype": cand_map.get("dtype", su2_mapping.get("dtype", {})),
				"normalization_mode": cand_map.get("normalization_mode", su2_mapping.get("normalization_mode", "both")),
				"epsilon": float(_eps_value),
			}
			cand_candidates = _build_candidate_mappings_adv(
				su2_mapping,
				[roll_vals],
				[ewma_vals],
				max_combos=None,
				allow_transition_windows_variation=True,
				allow_normalization_windows_variation=True,
				fixed_fields=fixed_fields_fine,
			)
			candidate_mappings.extend(cand_candidates)
		if args.max_combos is not None:
			candidate_mappings = candidate_mappings[: args.max_combos]
	else:
		candidate_mappings = _build_candidate_mappings_adv(
			su2_mapping,
			rolling_candidates,
			ewma_candidates,
			max_combos=args.max_combos,
			allow_transition_windows_variation=False,
			allow_normalization_windows_variation=False,
			fixed_fields=fixed_fields_stage1,
		)
	if not candidate_mappings:
		candidate_mappings = [dict(su2_mapping)]
	elif args.max_combos is not None:
		print(f"[info] evaluating {len(candidate_mappings)} candidate(s) (max-combos={args.max_combos})")

	out_dir = Path(args.out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	summary_records: List[Dict[str, Any]] = []
	for idx, mapping in enumerate(candidate_mappings, start=1):
		candidate_config = SU2Config.from_mapping(mapping, base_dir=config_path.parent)
		print(
			f"[info] evaluating candidate {idx}/{len(candidate_mappings)} | rolling={_stringify_sequence(mapping['rolling_windows'])}"
			f" | ewma={_stringify_sequence(mapping['ewma_alpha'])}"
		)
		metrics, fold_logs = _evaluate_candidate(
			X,
			y,
			su1_config,
			candidate_config,
			preprocess_settings,
			split_indices,
			args,
		)
		target_groups_mapping = mapping.get("target_groups", {})
		if not isinstance(target_groups_mapping, MappingABC):
			target_groups_mapping = {}
		normalization_epsilon_value = mapping.get("normalization_epsilon")
		if normalization_epsilon_value is not None:
			normalization_epsilon_value = float(normalization_epsilon_value)
		summary_record = {
			"config_id": idx,
			"rolling_windows": _stringify_sequence(mapping["rolling_windows"]),
			"ewma_alpha": _stringify_sequence(mapping["ewma_alpha"]),
			"include_rolling": bool(mapping.get("include_rolling", True)),
			"include_ewma": bool(mapping.get("include_ewma", True)),
			"include_transitions": bool(mapping.get("include_transitions", True)),
			"include_normalization": bool(mapping.get("include_normalization", False)),
			"include_metrics": _stringify_sequence(mapping.get("include_metrics", [])),
			"signals": _stringify_sequence(mapping.get("signals", [])),
			"rolling_include_current": bool(mapping.get("features", {}).get("rolling", {}).get("include_current", False)) if isinstance(mapping.get("features"), MappingABC) else False,
			"ewma_include_std": bool(mapping.get("features", {}).get("ewma", {}).get("include_std", True)) if isinstance(mapping.get("features"), MappingABC) else True,
			"ewma_reset_each_fold": bool(mapping.get("features", {}).get("ewma", {}).get("reset_each_fold", True)) if isinstance(mapping.get("features"), MappingABC) else True,
			"transition_windows": _stringify_sequence(mapping.get("transition_windows", [])),
			"recovery_clip": mapping.get("recovery_clip"),
			"normalization_mode": str(mapping.get("normalization_mode", "")) if mapping.get("normalization_mode") else "",
			"normalization_windows": _stringify_sequence(mapping.get("normalization_windows", [])),
			"normalization_epsilon": normalization_epsilon_value,
			"input_sources": _stringify_sequence(mapping.get("input_sources", [])),
			"target_include": _stringify_sequence(target_groups_mapping.get("include", [])),
			"target_exclude": _stringify_sequence(target_groups_mapping.get("exclude", [])),
			"clip_max": mapping.get("clip_max"),
			"drop_constant_columns": mapping.get("drop_constant_columns"),
			"dtype": json.dumps(mapping.get("dtype"), ensure_ascii=False) if isinstance(mapping.get("dtype"), MappingABC) else "",
			"oof_rmse": metrics["oof_rmse"],
			"oof_msr": metrics["oof_msr"],
			"oof_msr_down": metrics["oof_msr_down"],
			"oof_vmsr": metrics["oof_vmsr"],
			"coverage": metrics["coverage"],
			"best_mult": metrics["best_mult"],
			"best_lo": metrics["best_lo"],
			"best_hi": metrics["best_hi"],
			"best_lam": metrics["best_lam"],
			"oof_mean": metrics["oof_mean"],
			"oof_std": metrics["oof_std"],
			"oof_std_down": metrics["oof_std_down"],
		}
		summary_records.append(summary_record)

		fold_df = pd.DataFrame(fold_logs)
		fold_path = out_dir / f"su2_config_{idx:02d}_folds.csv"
		fold_df.to_csv(fold_path, index=False)
		candidate_yaml_path = out_dir / f"su2_config_{idx:02d}.yaml"
		with candidate_yaml_path.open("w", encoding="utf-8") as fh:
			yaml.safe_dump(mapping, fh, sort_keys=False, allow_unicode=True)
		candidate_json_path = out_dir / f"su2_config_{idx:02d}.json"
		candidate_json = {
			"config": mapping,
			"metrics": metrics,
			"folds": fold_logs,
		}
		with candidate_json_path.open("w", encoding="utf-8") as fh:
			json.dump(_json_safe(candidate_json), fh, ensure_ascii=False, indent=2)

	summary_df = pd.DataFrame(summary_records)
	if not summary_df.empty:
		summary_df = summary_df.sort_values(by=["oof_msr", "oof_rmse"], ascending=[False, True])
	summary_path = out_dir / "sweep_summary.csv"
	summary_df.to_csv(summary_path, index=False)
	print(f"[ok] wrote summary: {summary_path}")

	if not summary_df.empty:
		best_row = summary_df.iloc[0]
		best_idx = int(best_row["config_id"])
		best_yaml_src = out_dir / f"su2_config_{best_idx:02d}.yaml"
		best_yaml_dst = out_dir / "best_config.yaml"
		if best_yaml_src.exists():
			shutil.copyfile(best_yaml_src, best_yaml_dst)
		print(
			"[best] config_id={config_id} oof_msr={oof_msr:.6f} oof_rmse={oof_rmse:.6f} "
			"rolling={rolling_windows} ewma={ewma_alpha}".format(**best_row.to_dict())
		)

	return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
	sys.exit(main())
