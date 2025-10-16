#!/usr/bin/env python
"""D 系欠損補完ポリシーを横断的に実行し、CV 指標を集計するスクリプト。"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List
import statistics

from preprocess.D_group.d_group import DGroupImputer
from preprocess.M_group.m_group import MGroupImputer


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_pre_d.py"

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

ALL_POLICIES = sorted(set(chain.from_iterable(POLICY_SUITES.values())))
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
    parser.add_argument("--out-dir", type=str, default="artifacts/Preprocessing_D")
    parser.add_argument("--results-dir", type=str, default="results/ablation/D_group")
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
    help="Calendar column forwarded to train_pre_d.py (empty string to disable).",
    )
    parser.add_argument(
        "--policy-param",
        action="append",
        default=[],
        help="Additional policy parameters passed as key=value (repeat option).",
    )
    parser.add_argument("--extra-args", type=str, default="", help="additional args passed to train_pre_d.py")
    parser.add_argument("--pp-aggregate", type=str, default="refit", choices=["refit", "median", "vote"], help="post-process aggregation strategy passed to train_pre_d.py")
    parser.add_argument(
        "--m-policy",
        type=str,
        default="ridge_stack",
        choices=sorted(MGroupImputer.SUPPORTED_POLICIES),
    help="M group policy executed prior to each D run (default: ridge_stack).",
    )
    parser.add_argument(
        "--m-policy-param",
        action="append",
        default=[],
        help="Additional M policy parameters forwarded as key=value (repeatable).",
    )
    parser.add_argument(
        "--m-calendar-col",
        type=str,
        default=None,
        help="Optional calendar column override for the M policy (default: reuse --calendar-col).",
    )
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
    parser.add_argument("--max-workers", type=int, default=1, help="Number of concurrent policy runs (default: 1).")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for each train_pre_d.py invocation.")
    parser.add_argument("--retries", type=int, default=0, help="Number of retries per policy on failure or timeout.")
    parser.add_argument("--resume", action="store_true", help="Skip policies whose metrics already exist for the run tag.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed forwarded to train_pre_d.py")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = args.tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    summary_rows: list[dict[str, Any]] = []
    extra_args = shlex.split(args.extra_args) if args.extra_args else []
    policy_param_dict = parse_policy_param_options(args.policy_param or [])
    policy_param_cli = serialize_policy_params(policy_param_dict)
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
    supported_policies = set(DGroupImputer.SUPPORTED_POLICIES)
    unsupported = [policy for policy in selected_policies if policy not in supported_policies]
    if unsupported:
        print(f"[warn] unsupported policies requested: {unsupported} -> they will be skipped")
        selected_policies = [policy for policy in selected_policies if policy in supported_policies]
    if not selected_policies:
        raise RuntimeError("No policies resolved for sweep. Check --suite/--policies options.")

    sort_metric = args.sort_by
    sort_direction = METRIC_OBJECTIVES.get(sort_metric, "min")

    if policy_param_dict:
        print(f"[info] policy param overrides: {policy_param_dict}")
    if m_policy_param_dict:
        print(f"[info] M policy param overrides: {m_policy_param_dict}")

    def summarize_result(policy: str, metrics: dict[str, Any], duration: float | None, status: str = "ok") -> dict[str, Any]:
        row = {
            "policy": metrics.get("d_policy", policy),
            "rolling_window": metrics.get("d_rolling_window", args.rolling_window),
            "ema_alpha": metrics.get("d_ema_alpha", args.ema_alpha),
            "calendar_col": metrics.get("d_calendar_col", args.calendar_col),
            "m_policy": metrics.get("m_policy", args.m_policy),
            "m_calendar_col": metrics.get(
                "m_calendar_col",
                args.m_calendar_col if args.m_calendar_col is not None else args.calendar_col,
            ),
            "status": status,
            "pp_aggregate": metrics.get("pp_aggregate", args.pp_aggregate),
            "n_splits": metrics.get("n_splits"),
            "gap": metrics.get("gap"),
            "min_val_size": metrics.get("min_val_size"),
            "optimize_for": metrics.get("optimize_for"),
            "d_column_count": metrics.get("d_column_count"),
            "d_post_impute_nan_ratio": metrics.get("d_post_impute_nan_ratio"),
            "oof_rmse": metrics.get("oof_rmse"),
            "coverage": metrics.get("coverage"),
            "duration_sec": duration,
            "m_imputer_warning_count": metrics.get("m_imputer_warning_count"),
            "m_imputer_warnings": metrics.get("m_imputer_warnings"),
            "d_imputer_warning_count": metrics.get("d_imputer_warning_count"),
            "d_imputer_warnings": metrics.get("d_imputer_warnings"),
            "train_path": metrics.get("train_path"),
            "test_path": metrics.get("test_path"),
        }
        oof_metrics = metrics.get("oof_metrics", {})
        for key in ("msr", "msr_down", "vmsr"):
            row[key] = oof_metrics.get(key)
        params_logged = metrics.get("d_policy_params", {})
        for k, v in params_logged.items():
            row[f"param_{k}"] = v
        params_logged_m = metrics.get("m_policy_params", {})
        for k, v in params_logged_m.items():
            row[f"m_param_{k}"] = v
        row["random_seed"] = metrics.get("random_seed", args.random_seed)
        return row

    def execute_policy(policy: str) -> dict[str, Any]:
        metrics_path = results_dir / f"{timestamp}_d_group_{policy}.json"
        if metrics_path.exists():
            if args.resume:
                print(f"[info][{policy}] metrics already exist at {metrics_path}; skipping execution")
                metrics = load_metrics(metrics_path)
                return summarize_result(policy, metrics, duration=None, status="cached")
            raise FileExistsError(
                f"Metrics file already exists for policy={policy} at {metrics_path}. Use --resume to skip existing runs."
            )

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data-dir",
            args.data_dir,
            "--out-dir",
            args.out_dir,
            "--d-policy",
            policy,
            "--d-rolling-window",
            str(args.rolling_window),
            "--d-ema-alpha",
            str(args.ema_alpha),
            "--d-calendar-col",
            args.calendar_col,
            "--no-artifacts",
            "--metrics-path",
            str(metrics_path),
            "--pp-aggregate",
            args.pp_aggregate,
            "--random-seed",
            str(args.random_seed),
            "--m-policy",
            args.m_policy,
        ]
        if args.m_calendar_col is not None:
            cmd.extend(["--m-calendar-col", args.m_calendar_col])
        for param in m_policy_param_cli:
            cmd.extend(["--m-policy-param", param])
        for param in policy_param_cli:
            cmd.extend(["--d-policy-param", param])
        cmd.extend(extra_args)

        attempts = args.retries + 1
        last_error: str | None = None
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
                    row = summarize_result(policy, metrics, duration_sec, status="ok")
                    coverage = metrics.get("coverage")
                    train_path = metrics.get("train_path")
                    cov_msg = f"coverage={coverage:.3f}" if isinstance(coverage, (float, int)) else "coverage=n/a"
                    train_msg = f"train={train_path}" if train_path else "train=unknown"
                    print(f"[info][{policy}] completed in {duration_sec:.2f}s | {cov_msg} | {train_msg}")
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
                        # cancel remaining futures before re-raising
                        for pending in future_map:
                            if pending is not future:
                                pending.cancel()
                        raise
                    if row:
                        summary_rows.append(row)

    schedule_policies(selected_policies)

    summary_path = results_dir / f"{timestamp}_d_group_summary.csv"
    if summary_rows:
        import csv

        base_fieldnames = [
            "policy",
            "status",
            "rolling_window",
            "ema_alpha",
            "calendar_col",
            "m_policy",
            "m_calendar_col",
            "pp_aggregate",
            "n_splits",
            "gap",
            "min_val_size",
            "optimize_for",
            "d_column_count",
            "d_post_impute_nan_ratio",
            "oof_rmse",
            "coverage",
            "msr",
            "msr_down",
            "vmsr",
            "duration_sec",
            "m_imputer_warning_count",
            "m_imputer_warnings",
            "d_imputer_warning_count",
            "d_imputer_warnings",
            "random_seed",
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
        stats_path = results_dir / f"{timestamp}_d_group_summary_stats.json"
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
