#!/usr/bin/env python
"""M 系欠損補完ポリシーを横断的に実行し、CV 指標を集計するスクリプト。"""

from __future__ import annotations

import argparse
import json
import time
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_pre_m.py"

DEFAULT_POLICIES = [
    "ffill_bfill",
    "ffill_only",
    "rolling_median_k",
    "rolling_mean_k",
    "ema_alpha",
]


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"[cmd] {' '.join(cmd)}")
    return subprocess.run(cmd, check=False)


def load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/Preprocessing_M")
    parser.add_argument("--results-dir", type=str, default="results/ablation")
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument(
        "--calendar-col",
        type=str,
        default="date_id",
        help="Calendar column forwarded to train_pre_m.py (empty string to disable).",
    )
    parser.add_argument(
        "--policy-param",
        action="append",
        default=[],
        help="Additional policy parameters passed as key=value (repeat option).",
    )
    parser.add_argument("--extra-args", type=str, default="", help="additional args passed to train_pre_m.py")
    parser.add_argument("--pp-aggregate", type=str, default="refit", choices=["refit", "median", "vote"], help="post-process aggregation strategy passed to train_pre_m.py")
    parser.add_argument("--skip-on-error", action="store_true", help="continue sweep when a policy run fails")
    parser.add_argument("--fail-fast", action="store_true", help="stop immediately on the first failure even if skip-on-error is enabled")
    parser.add_argument("--tag", type=str, default=None, help="custom tag for result files (default: timestamp)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = args.tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    summary_rows: list[dict[str, Any]] = []
    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    for policy in args.policies:
        metrics_path = results_dir / f"{timestamp}_m_group_{policy}.json"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data-dir",
            args.data_dir,
            "--out-dir",
            args.out_dir,
            "--m-policy",
            policy,
            "--m-rolling-window",
            str(args.rolling_window),
            "--m-ema-alpha",
            str(args.ema_alpha),
            "--m-calendar-col",
            args.calendar_col,
            "--no-artifacts",
            "--metrics-path",
            str(metrics_path),
            "--pp-aggregate",
            args.pp_aggregate,
        ]
        for param in args.policy_param:
            if param:
                cmd.extend(["--m-policy-param", param])
        cmd.extend(extra_args)
        start_ts = time.perf_counter()
        result = run_cmd(cmd)
        duration_sec = time.perf_counter() - start_ts
        if result.returncode != 0:
            err_msg = f"policy={policy} failed with exit code {result.returncode}"
            if args.skip_on_error and not args.fail_fast:
                print(f"[warn] {err_msg}", file=sys.stderr)
                summary_rows.append({"policy": policy, "status": "error", "error": err_msg})
                continue
            raise subprocess.CalledProcessError(result.returncode, cmd)
        if not metrics_path.exists():
            err_msg = f"metrics not found for policy={policy} at {metrics_path}"
            if args.skip_on_error and not args.fail_fast:
                print(f"[warn] {err_msg}", file=sys.stderr)
                summary_rows.append({"policy": policy, "status": "error", "error": err_msg})
                continue
            raise FileNotFoundError(err_msg)

        metrics = load_metrics(metrics_path)
        row = {
            "policy": metrics.get("m_policy", policy),
            "rolling_window": metrics.get("m_rolling_window", args.rolling_window),
            "ema_alpha": metrics.get("m_ema_alpha", args.ema_alpha),
            "calendar_col": metrics.get("m_calendar_col", args.calendar_col),
            "status": "ok",
            "pp_aggregate": metrics.get("pp_aggregate", args.pp_aggregate),
            "n_splits": metrics.get("n_splits"),
            "gap": metrics.get("gap"),
            "min_val_size": metrics.get("min_val_size"),
            "optimize_for": metrics.get("optimize_for"),
            "m_column_count": metrics.get("m_column_count"),
            "m_post_impute_nan_ratio": metrics.get("m_post_impute_nan_ratio"),
            "oof_rmse": metrics.get("oof_rmse"),
            "coverage": metrics.get("coverage"),
            "duration_sec": duration_sec,
            "m_imputer_warning_count": metrics.get("m_imputer_warning_count"),
            "m_imputer_warnings": metrics.get("m_imputer_warnings"),
        }
        oof_metrics = metrics.get("oof_metrics", {})
        for key in ("msr", "msr_down", "vmsr"):
            row[key] = oof_metrics.get(key)
        params_logged = metrics.get("m_policy_params", {})
        for k, v in params_logged.items():
            row[f"param_{k}"] = v
        summary_rows.append(row)

    summary_path = results_dir / f"{timestamp}_m_group_summary.csv"
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
            "m_column_count",
            "m_post_impute_nan_ratio",
            "oof_rmse",
            "coverage",
            "msr",
            "msr_down",
            "vmsr",
        ]
        dynamic_keys = sorted({key for row in summary_rows for key in row.keys() if key not in base_fieldnames})
        fieldnames = base_fieldnames + dynamic_keys

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[ok] wrote summary CSV: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
