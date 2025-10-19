#!/usr/bin/env python
"""I 系欠損補完ポリシーを横断的に実行し、CV 指標を集計するスクリプト。"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List
import statistics


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from preprocess.I_group.i_group import IGroupImputer  # noqa: E402
from preprocess.E_group.e_group import EGroupImputer  # noqa: E402
from preprocess.M_group.m_group import MGroupImputer  # noqa: E402

TRAIN_SCRIPT = THIS_DIR / "train_pre_i.py"

POLICY_SUITES = {
    "A": [
        "ffill_bfill",
        "ffill_only",
        "rolling_median_k",
        "rolling_mean_k",
        "ema_alpha",
    ],
    "B": [
        "linear_interp",
        "spline_interp_deg",
        "time_interp",
    ],
    "C": [
        "backfill_robust",
        "winsorized_median_k",
        "quantile_fill",
    ],
    "D": [
        "dow_median",
        "dom_median",
        "month_median",
        "holiday_bridge",
    ],
    "E": [
        "knn_k",
        "pca_reconstruct_r",
        "mice",
        "missforest",
        "ridge_stack",
    ],
    "F": [
        "kalman_local_level",
        "arima_auto",
        "state_space_custom",
    ],
    "G": [
        "mask_plus_mean",
        "two_stage",
    ],
}

SUITE_POLICIES = list(chain.from_iterable(POLICY_SUITES.values()))
SUPPORTED_POLICIES = list(IGroupImputer.SUPPORTED_POLICIES)
ALL_POLICIES = list(dict.fromkeys(SUITE_POLICIES + SUPPORTED_POLICIES))
POLICY_SUITES["full"] = list(ALL_POLICIES)
METRIC_OBJECTIVES = {
    "oof_rmse": "min",
    "msr": "max",
    "msr_down": "max",
    "vmsr": "max",
    "coverage": "max",
    "duration_sec": "min",
}
DEFAULT_SORT_KEY = "oof_rmse"


def load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_policy_param_options(raw_items: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for raw in raw_items:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid policy param '{raw}'. Use key=value format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid policy param '{raw}' (empty key).")
        value_str = value.strip()
        lowered = value_str.lower()
        if lowered in {"true", "false"}:
            params[key] = lowered == "true"
            continue
        try:
            params[key] = int(value_str)
            continue
        except ValueError:
            pass
        try:
            params[key] = float(value_str)
            continue
        except ValueError:
            pass
        params[key] = value_str
    return params


def serialize_policy_params(params: Dict[str, Any]) -> List[str]:
    serialized: List[str] = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        else:
            value_str = str(value)
        serialized.append(f"{key}={value_str}")
    return serialized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/Preprocessing_I")
    parser.add_argument("--results-dir", type=str, default="results/ablation/I_group")
    parser.add_argument(
        "--suite",
        dest="suites",
        action="append",
        choices=sorted(POLICY_SUITES.keys()),
        default=[],
        help="Named policy suite(s) to evaluate (e.g. A, B, full). Can be repeated.",
    )
    parser.add_argument("--policies", nargs="*", default=None, help="Explicit list of policies to sweep.")
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument(
        "--calendar-col",
        type=str,
        default="date_id",
        help="Fallback calendar column forwarded to train_pre_i.py when explicit group overrides are not provided (empty string to disable).",
    )
    parser.add_argument(
        "--i-calendar-col",
        type=str,
        default=None,
        help="Optional override for the I policy calendar column (default: reuse --calendar-col)",
    )
    parser.add_argument(
        "--e-calendar-col",
        type=str,
        default=None,
        help="Optional override for the E policy calendar column (default: reuse --calendar-col)",
    )
    parser.add_argument(
        "--m-calendar-col",
        type=str,
        default=None,
        help="Optional override for the M policy calendar column (default: reuse --e-calendar-col or --calendar-col).",
    )
    parser.add_argument(
        "--i-policy-param",
        action="append",
        default=[],
        help="Additional policy parameters passed as key=value (repeat option).",
    )
    parser.add_argument(
        "--e-policy-param",
        action="append",
        default=[],
        help="Additional E policy parameters forwarded to train_pre_i.py (repeatable).",
    )
    parser.add_argument(
        "--m-policy-param",
        action="append",
        default=[],
        help="Additional M policy parameters forwarded to train_pre_i.py (repeatable).",
    )
    parser.add_argument("--extra-args", type=str, default="", help="additional args passed to train_pre_i.py")
    parser.add_argument(
        "--pp-aggregate",
        type=str,
        default="refit",
        choices=["refit", "median", "vote"],
        help="post-process aggregation strategy passed to train_pre_i.py",
    )
    parser.add_argument(
        "--i-clip-low",
        type=float,
        default=0.001,
        help="Lower quantile for I-group post-impute clipping.",
    )
    parser.add_argument(
        "--i-clip-high",
        type=float,
        default=0.999,
        help="Upper quantile for I-group post-impute clipping.",
    )
    parser.add_argument("--disable-i-clip", action="store_true", help="Disable I-group quantile clipping during sweep runs.")
    parser.add_argument("--skip-on-error", action="store_true", help="continue sweep when a policy run fails")
    parser.add_argument("--fail-fast", action="store_true", help="stop immediately on the first failure even if skip-on-error is enabled")
    parser.add_argument("--tag", type=str, default=None, help="custom tag for result files (default: timestamp)")
    parser.add_argument(
        "--sort-by",
        type=str,
        default=DEFAULT_SORT_KEY,
        choices=sorted(METRIC_OBJECTIVES.keys()),
        help="Metric used to sort summary output.",
    )
    parser.add_argument(
        "--m-policy",
        type=str,
        default=None,
        choices=sorted(MGroupImputer.SUPPORTED_POLICIES),
        help="Optional override for the M policy executed before each I run.",
    )
    parser.add_argument("--m-rolling-window", type=int, default=None, help="Override for --m-rolling-window passed to train_pre_i.py.")
    parser.add_argument("--m-ema-alpha", type=float, default=None, help="Override for --m-ema-alpha passed to train_pre_i.py.")
    parser.add_argument(
        "--e-policy",
        type=str,
        default=None,
        choices=sorted(EGroupImputer.SUPPORTED_POLICIES),
        help="Optional override for the E policy executed before each I run.",
    )
    parser.add_argument("--e-rolling-window", type=int, default=None, help="Override for --e-rolling-window passed to train_pre_i.py.")
    parser.add_argument("--e-ema-alpha", type=float, default=None, help="Override for --e-ema-alpha passed to train_pre_i.py.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of concurrent policy runs (default: 1).")
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=None,
        help="Override LightGBM n_jobs passed to train_pre_i.py (default: -1, forced to 1 when --max-workers>1 unless set)",
    )
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for each train_pre_i.py invocation.")
    parser.add_argument("--retries", type=int, default=0, help="Number of retries per policy on failure or timeout.")
    parser.add_argument("--resume", action="store_true", help="Skip policies whose metrics already exist for the run tag.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed forwarded to train_pre_i.py")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.model_n_jobs is None and args.max_workers > 1:
        print("[info] max-workers>1 detected; LightGBM n_jobs will be forced to 1 per run.")

    timestamp = args.tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    summary_rows: list[dict[str, Any]] = []
    extra_args = shlex.split(args.extra_args) if args.extra_args else []
    i_policy_param_dict = parse_policy_param_options(args.i_policy_param or [])
    i_policy_param_cli = serialize_policy_params(i_policy_param_dict)
    e_policy_param_dict = parse_policy_param_options(args.e_policy_param or [])
    e_policy_param_cli = serialize_policy_params(e_policy_param_dict)
    m_policy_param_dict = parse_policy_param_options(args.m_policy_param or [])
    m_policy_param_cli = serialize_policy_params(m_policy_param_dict)

    selected_policies: list[str] = []
    if args.policies:
        selected_policies.extend(args.policies)
    suites_requested = args.suites or (["full"] if not args.policies else [])
    for suite in suites_requested:
        selected_policies.extend(POLICY_SUITES.get(suite, []))
    if not selected_policies:
        selected_policies = list(ALL_POLICIES)
    # preserve order while deduplicating
    selected_policies = list(dict.fromkeys(selected_policies))
    supported_policies = set(IGroupImputer.SUPPORTED_POLICIES)
    unsupported = [policy for policy in selected_policies if policy not in supported_policies]
    if unsupported:
        print(f"[warn] unsupported policies requested: {unsupported} -> they will be skipped")
        selected_policies = [policy for policy in selected_policies if policy in supported_policies]
    if not selected_policies:
        raise RuntimeError("No policies resolved for sweep. Check --suite/--policies options.")

    sort_metric = args.sort_by
    sort_direction = METRIC_OBJECTIVES.get(sort_metric, "min")

    if i_policy_param_dict:
        print(f"[info] I policy param overrides: {i_policy_param_dict}")
    if e_policy_param_dict:
        print(f"[info] E policy param overrides: {e_policy_param_dict}")
    if m_policy_param_dict:
        print(f"[info] M policy param overrides: {m_policy_param_dict}")

    def summarize_result(
        policy: str,
        metrics: dict[str, Any],
        duration: float | None,
        status: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "policy": metrics.get("i_policy", policy),
            "rolling_window": metrics.get("i_rolling_window", fallback.get("i_rolling_window")),
            "ema_alpha": metrics.get("i_ema_alpha", fallback.get("i_ema_alpha")),
            "calendar_col": metrics.get("i_calendar_col", fallback.get("i_calendar_col")),
            "m_policy": metrics.get("m_policy"),
            "m_calendar_col": metrics.get("m_calendar_col", fallback.get("m_calendar_col")),
            "e_policy": metrics.get("e_policy"),
            "e_calendar_col": metrics.get("e_calendar_col", fallback.get("e_calendar_col")),
            "status": status,
            "pp_aggregate": metrics.get("pp_aggregate", fallback.get("pp_aggregate")),
            "n_splits": metrics.get("n_splits"),
            "gap": metrics.get("gap"),
            "min_val_size": metrics.get("min_val_size"),
            "optimize_for": metrics.get("optimize_for"),
            "i_post_impute_nan_ratio": metrics.get("i_post_impute_nan_ratio"),
            "oof_rmse": metrics.get("oof_rmse"),
            "coverage": metrics.get("coverage"),
            "duration_sec": duration,
            "m_imputer_warning_count": metrics.get("m_imputer_warning_count"),
            "m_imputer_warnings": metrics.get("m_imputer_warnings"),
            "e_imputer_warning_count": metrics.get("e_imputer_warning_count"),
            "e_imputer_warnings": metrics.get("e_imputer_warnings"),
            "i_imputer_warning_count": metrics.get("i_imputer_warning_count"),
            "i_imputer_warnings": metrics.get("i_imputer_warnings"),
            "i_clip_low": metrics.get("i_clip_low", fallback.get("i_clip_low")),
            "i_clip_high": metrics.get("i_clip_high", fallback.get("i_clip_high")),
            "i_quantile_clip_enabled": metrics.get(
                "i_quantile_clip_enabled", fallback.get("i_quantile_clip_enabled")
            ),
            "train_path": metrics.get("train_path"),
            "test_path": metrics.get("test_path"),
            "random_seed": metrics.get("random_seed", fallback.get("random_seed")),
            "model_n_jobs": metrics.get("model_n_jobs", fallback.get("model_n_jobs")),
        }
        i_columns = metrics.get("i_columns")
        i_column_count = metrics.get("i_column_count")
        if i_column_count is None and isinstance(i_columns, list):
            i_column_count = len(i_columns)
        row["i_column_count"] = i_column_count
        m_columns = metrics.get("m_columns")
        if row.get("m_policy") and isinstance(m_columns, list):
            row.setdefault("m_column_count", len(m_columns))
        e_columns = metrics.get("e_columns")
        if row.get("e_policy") and isinstance(e_columns, list):
            row.setdefault("e_column_count", metrics.get("e_column_count", len(e_columns)))
        oof_metrics = metrics.get("oof_metrics", {})
        for key in ("msr", "msr_down", "vmsr"):
            row[key] = oof_metrics.get(key)
        for k, v in (metrics.get("i_policy_params") or {}).items():
            row[f"i_param_{k}"] = v
        for k, v in (metrics.get("e_policy_params") or {}).items():
            row[f"e_param_{k}"] = v
        for k, v in (metrics.get("m_policy_params") or {}).items():
            row[f"m_param_{k}"] = v
        i_clip_bounds = metrics.get("i_clip_bounds")
        if isinstance(i_clip_bounds, dict):
            row["i_clip_bounds"] = json.dumps(i_clip_bounds, ensure_ascii=False)

        fold_metrics = metrics.get("fold_metrics") or []

        def _fold_std(key: str) -> float | None:
            values: list[float] = []
            for entry in fold_metrics:
                val = entry.get(key)
                if isinstance(val, (int, float)):
                    values.append(float(val))
            if not values:
                return None
            if len(values) == 1:
                return 0.0
            return float(statistics.pstdev(values))

        row["pp_best_mult_std"] = _fold_std("best_mult")
        row["pp_best_lo_std"] = _fold_std("best_lo")
        row["pp_best_hi_std"] = _fold_std("best_hi")
        row["pp_best_lam_std"] = _fold_std("best_lam")

        clip_bounds = metrics.get("i_clip_bounds") or {}
        ratios: list[float] = []
        if isinstance(clip_bounds, dict):
            for bounds in clip_bounds.values():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    try:
                        low_v = float(bounds[0])
                        high_v = float(bounds[1])
                    except (TypeError, ValueError):
                        continue
                    width = high_v - low_v
                    denom = max(abs(high_v), abs(low_v), 1e-9)
                    ratio = width / denom if denom > 0 else math.nan
                    if not math.isnan(ratio) and not math.isinf(ratio):
                        ratios.append(ratio)
        if ratios:
            row["i_clip_rel_width_mean"] = float(sum(ratios) / len(ratios))
            row["i_clip_rel_width_min"] = float(min(ratios))
            row["i_clip_rel_width_max"] = float(max(ratios))
        return row

    def execute_policy(policy: str) -> dict[str, Any]:
        metrics_path = results_dir / f"{timestamp}_i_group_{policy}.json"
        computed_model_n_jobs = (
            args.model_n_jobs if args.model_n_jobs is not None else (1 if args.max_workers > 1 else -1)
        )
        if metrics_path.exists():
            if args.resume:
                print(f"[info][{policy}] metrics already exist at {metrics_path}; skipping execution")
                metrics = load_metrics(metrics_path)
                fallback = {
                    "i_rolling_window": args.rolling_window,
                    "i_ema_alpha": args.ema_alpha,
                    "i_calendar_col": args.i_calendar_col if args.i_calendar_col is not None else args.calendar_col,
                    "pp_aggregate": args.pp_aggregate,
                    "i_clip_low": args.i_clip_low,
                    "i_clip_high": args.i_clip_high,
                    "i_quantile_clip_enabled": not args.disable_i_clip,
                    "random_seed": args.random_seed,
                    "e_calendar_col": args.e_calendar_col if args.e_calendar_col is not None else args.calendar_col,
                    "m_calendar_col": args.m_calendar_col
                    if args.m_calendar_col is not None
                    else (args.e_calendar_col if args.e_calendar_col is not None else args.calendar_col),
                    "model_n_jobs": computed_model_n_jobs,
                }
                return summarize_result(policy, metrics, duration=None, status="cached", fallback=fallback)
            raise FileExistsError(
                f"Metrics file already exists for policy={policy} at {metrics_path}. Use --resume to skip existing runs."
            )

        effective_i_calendar = args.i_calendar_col if args.i_calendar_col is not None else args.calendar_col
        effective_e_calendar = args.e_calendar_col if args.e_calendar_col is not None else args.calendar_col
        if args.m_calendar_col is not None:
            effective_m_calendar = args.m_calendar_col
        elif effective_e_calendar is not None:
            effective_m_calendar = effective_e_calendar
        else:
            effective_m_calendar = effective_i_calendar

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data-dir",
            args.data_dir,
            "--out-dir",
            args.out_dir,
            "--i-policy",
            policy,
            "--i-rolling-window",
            str(args.rolling_window),
            "--i-ema-alpha",
            str(args.ema_alpha),
            "--no-artifacts",
            "--metrics-path",
            str(metrics_path),
            "--pp-aggregate",
            args.pp_aggregate,
            "--random-seed",
            str(args.random_seed),
            "--i-clip-low",
            str(args.i_clip_low),
            "--i-clip-high",
            str(args.i_clip_high),
        ]
        if args.disable_i_clip:
            cmd.append("--disable-i-clip")
        if effective_i_calendar is not None:
            cmd.extend(["--i-calendar-col", effective_i_calendar])
        if effective_e_calendar is not None:
            cmd.extend(["--e-calendar-col", effective_e_calendar])
        if effective_m_calendar is not None:
            cmd.extend(["--m-calendar-col", effective_m_calendar])
        if args.m_policy:
            cmd.extend(["--m-policy", args.m_policy])
        if args.m_rolling_window is not None:
            cmd.extend(["--m-rolling-window", str(args.m_rolling_window)])
        if args.m_ema_alpha is not None:
            cmd.extend(["--m-ema-alpha", str(args.m_ema_alpha)])
        if args.e_policy:
            cmd.extend(["--e-policy", args.e_policy])
        if args.e_rolling_window is not None:
            cmd.extend(["--e-rolling-window", str(args.e_rolling_window)])
        if args.e_ema_alpha is not None:
            cmd.extend(["--e-ema-alpha", str(args.e_ema_alpha)])
        for param in m_policy_param_cli:
            cmd.extend(["--m-policy-param", param])
        for param in e_policy_param_cli:
            cmd.extend(["--e-policy-param", param])
        for param in i_policy_param_cli:
            cmd.extend(["--i-policy-param", param])
        cmd.extend(["--model-n-jobs", str(computed_model_n_jobs)])
        cmd.extend(extra_args)

        attempts = args.retries + 1
        last_error: str | None = None
        fallback = {
            "i_rolling_window": args.rolling_window,
            "i_ema_alpha": args.ema_alpha,
            "i_calendar_col": effective_i_calendar,
            "pp_aggregate": args.pp_aggregate,
            "i_clip_low": args.i_clip_low,
            "i_clip_high": args.i_clip_high,
            "i_quantile_clip_enabled": not args.disable_i_clip,
            "random_seed": args.random_seed,
            "e_calendar_col": effective_e_calendar,
            "m_calendar_col": effective_m_calendar,
            "model_n_jobs": computed_model_n_jobs,
        }

        for attempt in range(1, attempts + 1):
            print(
                f"[info][{policy}] attempt {attempt}/{attempts} seed={args.random_seed} start | data_dir={args.data_dir}"
            )
            start_ts = time.perf_counter()
            print(f"[cmd][{policy}] {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=False, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                duration_sec = time.perf_counter() - start_ts
                last_error = f"timeout after {args.timeout} seconds"
                print(f"[warn][{policy}] {last_error}")
                if metrics_path.exists():
                    metrics_path.unlink(missing_ok=True)
            else:
                duration_sec = time.perf_counter() - start_ts
                if result.returncode == 0 and metrics_path.exists():
                    metrics = load_metrics(metrics_path)
                    row = summarize_result(policy, metrics, duration_sec, status="ok", fallback=fallback)
                    coverage = metrics.get("coverage")
                    cov_msg = f"coverage={coverage:.3f}" if isinstance(coverage, (float, int)) else "coverage=n/a"
                    print(f"[info][{policy}] completed in {duration_sec:.2f}s | {cov_msg}")
                    return row
                last_error = f"exit code {result.returncode}"
                print(f"[warn][{policy}] {last_error}")
                if metrics_path.exists():
                    metrics_path.unlink(missing_ok=True)
            if attempt < attempts:
                print(f"[info][{policy}] retrying ({attempt + 1}/{attempts}) after error: {last_error}")

        error_message = f"policy={policy} failed after {attempts} attempt(s): {last_error}"
        if args.skip_on_error and not args.fail_fast:
            print(f"[warn] {error_message}")
            return {"policy": policy, "status": "error", "error": error_message}
        raise RuntimeError(error_message)

    def schedule_policies(policies: List[str]) -> None:
        if args.max_workers <= 1:
            for policy in policies:
                try:
                    row = execute_policy(policy)
                except Exception as exc:
                    if args.skip_on_error and not args.fail_fast:
                        print(f"[warn] policy={policy} aborted: {exc}")
                        summary_rows.append({"policy": policy, "status": "error", "error": str(exc)})
                        continue
                    raise
                if row:
                    summary_rows.append(row)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {executor.submit(execute_policy, policy): policy for policy in policies}
                for future in as_completed(future_map):
                    policy = future_map[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        if args.skip_on_error and not args.fail_fast:
                            print(f"[warn] policy={policy} aborted: {exc}")
                            summary_rows.append({"policy": policy, "status": "error", "error": str(exc)})
                            continue
                        for pending in future_map:
                            if pending is not future:
                                pending.cancel()
                        raise
                    if row:
                        summary_rows.append(row)

    schedule_policies(selected_policies)

    summary_path = results_dir / f"{timestamp}_i_group_summary.csv"
    if summary_rows:
        import csv

        base_fieldnames = [
            "policy",
            "status",
            "rolling_window",
            "ema_alpha",
            "calendar_col",
            "pp_aggregate",
            "n_splits",
            "gap",
            "min_val_size",
            "optimize_for",
            "i_column_count",
            "i_post_impute_nan_ratio",
            "i_clip_low",
            "i_clip_high",
            "i_quantile_clip_enabled",
            "oof_rmse",
            "coverage",
            "msr",
            "msr_down",
            "vmsr",
            "duration_sec",
            "m_policy",
            "m_calendar_col",
            "m_column_count",
            "m_imputer_warning_count",
            "m_imputer_warnings",
            "e_policy",
            "e_calendar_col",
            "e_column_count",
            "e_imputer_warning_count",
            "e_imputer_warnings",
            "i_imputer_warning_count",
            "i_imputer_warnings",
            "random_seed",
            "model_n_jobs",
            "pp_best_mult_std",
            "pp_best_lo_std",
            "pp_best_hi_std",
            "pp_best_lam_std",
            "i_clip_rel_width_mean",
            "i_clip_rel_width_min",
            "i_clip_rel_width_max",
            "train_path",
            "test_path",
        ]
        dynamic_keys = sorted({key for row in summary_rows for key in row.keys() if key not in base_fieldnames})
        fieldnames = base_fieldnames + dynamic_keys

        if sort_metric in METRIC_OBJECTIVES:

            def _safe_numeric(row: dict[str, Any], key: str) -> float:
                value = row.get(key)
                numeric: float
                try:
                    numeric = float(value) if value is not None else math.nan
                except (TypeError, ValueError):
                    numeric = math.nan
                if math.isnan(numeric):
                    numeric = math.inf if METRIC_OBJECTIVES.get(key) == "min" else -math.inf
                return numeric

            reverse = sort_direction == "max"
            summary_rows.sort(key=lambda r: _safe_numeric(r, sort_metric), reverse=reverse)

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[ok] wrote summary CSV: {summary_path}")

        def build_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
            ok_rows = [row for row in rows if row.get("status") in {"ok", "cached"}]

            def safe_float(val: Any) -> float | None:
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    return None
                if math.isnan(num) or math.isinf(num):
                    return None
                return num

            metric_details: dict[str, Any] = {}
            for metric, objective in METRIC_OBJECTIVES.items():
                values_raw = [
                    (safe_float(row.get(metric)), row.get("policy", ""))
                    for row in ok_rows
                ]
                values = [(val, pol) for val, pol in values_raw if val is not None]
                if not values:
                    continue
                numbers = [val for val, _ in values]
                if objective == "min":
                    best_val, best_policy = min(values, key=lambda item: item[0])
                    ranked = sorted(values, key=lambda item: item[0])
                else:
                    best_val, best_policy = max(values, key=lambda item: item[0])
                    ranked = sorted(values, key=lambda item: item[0], reverse=True)
                metric_details[metric] = {
                    "best_value": best_val,
                    "best_policy": best_policy,
                    "mean": statistics.fmean(numbers),
                    "median": statistics.median(numbers),
                    "stddev": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
                    "ranked": [
                        {"policy": pol, "value": val}
                        for val, pol in ranked
                    ],
                }
            return {
                "timestamp": timestamp,
                "policies_evaluated": [row.get("policy") for row in rows],
                "suites_requested": suites_requested,
                "metrics": metric_details,
            }

        metric_summary = build_metric_summary(summary_rows)
        stats_path = results_dir / f"{timestamp}_i_group_summary_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(metric_summary, f, indent=2, ensure_ascii=False)
        print(f"[ok] wrote summary stats: {stats_path}")
        for metric, details in metric_summary.get("metrics", {}).items():
            best_policy = details.get("best_policy")
            best_value = details.get("best_value")
            if best_policy is None or best_value is None:
                continue
            try:
                best_value_str = f"{float(best_value):.6f}"
            except (TypeError, ValueError):
                best_value_str = str(best_value)
            print(f"[best] {metric}: {best_policy} ({best_value_str})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
