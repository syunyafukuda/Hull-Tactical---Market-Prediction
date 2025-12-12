#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSR-proxy 学習スクリプト (V 系特徴量版):
- data/raw/{train.parquet|train.csv} を自動検出（--train-file で明示可）
- 目的変数(--target-col)、ID列(--id-col) を除き、説明変数を構成
- LightGBM を使用（ベースラインを踏襲）
- 時系列CVで RMSE と MSR プロキシを同時評価
- post-process（mult, lo, hi）をfold内予測でグリッド探索し、fold平均で最適化
- 学習済みモデルとメタ情報（MSR関連を含む）を artifacts/Preprocessing_V/ に保存
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb  # type: ignore
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None  # type: ignore
    lgb = None  # type: ignore
    HAS_LGBM = False

import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ensure project/src roots are importable when executed as a script
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from preprocess.E_group.e_group import EGroupImputer  # noqa: E402
from preprocess.I_group.i_group import IGroupImputer  # noqa: E402
from preprocess.M_group.m_group import MGroupImputer  # noqa: E402
from preprocess.P_group.p_group import PGroupImputer  # noqa: E402
from preprocess.S_group.s_group import SGroupImputer  # noqa: E402
from preprocess.V_group.v_group import VGroupImputer  # noqa: E402
from scripts.utils_msr import (  # noqa: E402
    PostProcessParams,
    evaluate_msr_proxy,
    grid_search_msr,
)


def _to_1d_np(pred) -> np.ndarray:
    """Ensure prediction is a 1-D numpy array.
    Some estimators may return a tuple (pred, aux). We only need the first.
    """
    if isinstance(pred, tuple):
        pred = pred[0]
    arr = np.asarray(pred)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr.astype(float, copy=False)


def infer_train_file(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--train-file not found: {p}")
        return p
    # heuristic: prefer parquet
    candidates = [
        data_dir / "train.parquet",
        data_dir / "train.csv",
        data_dir / "Train.parquet",
        data_dir / "Train.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: first parquet/csv under data_dir
    for ext in ("parquet", "csv"):
        found = list(data_dir.glob(f"*.{ext}"))
        if found:
            return found[0]
    raise FileNotFoundError(f"No train file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def parse_policy_params(raw_items: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"Invalid policy param '{raw}'. Use key=value format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid policy param '{raw}': empty key")
        value_str = value.strip()
        if value_str.lower() in {"true", "false"}:
            params[key] = value_str.lower() == "true"
            continue
        try:
            params[key] = int(value_str)
            continue
        except ValueError:
            # Not an int; try float next
            pass
        try:
            params[key] = float(value_str)
            continue
        except ValueError:
            # Not a float; treat as string
            pass
        params[key] = value_str
    return params


def _aggregate_postprocess_from_folds(
    fold_logs: List[dict],
    strategy: str,
    optimize_for: str,
) -> Tuple[PostProcessParams, float]:
    if not fold_logs:
        return PostProcessParams(), 0.0

    mults = [float(log.get("best_mult", 1.0)) for log in fold_logs]
    los = [float(log.get("best_lo", 0.0)) for log in fold_logs]
    his = [float(log.get("best_hi", 2.0)) for log in fold_logs]
    lams = [float(log.get("best_lam", 0.0)) for log in fold_logs]

    def _vote(values: List[float]) -> float:
        if not values:
            return 0.0
        rounded = [round(v, 6) for v in values]
        counts = Counter(rounded)
        best_round, _ = max(counts.items(), key=lambda kv: (kv[1], -abs(kv[0])))
        for original in values:
            if round(original, 6) == best_round:
                return float(original)
        return float(values[0])

    if strategy == "median":
        agg_mult = float(np.median(mults))
        agg_lo = float(np.median(los))
        agg_hi = float(np.median(his))
        agg_lam = float(np.median(lams)) if optimize_for == "vmsr" else 0.0
    elif strategy == "vote":
        agg_mult = _vote(mults)
        agg_lo = _vote(los)
        agg_hi = _vote(his)
        agg_lam = _vote(lams) if optimize_for == "vmsr" else 0.0
    else:
        agg_mult, agg_lo, agg_hi = 1.0, 0.0, 2.0
        agg_lam = 0.0

    if agg_hi < agg_lo:
        agg_lo, agg_hi = agg_hi, agg_lo

    return PostProcessParams(mult=agg_mult, lo=agg_lo, hi=agg_hi), float(agg_lam)


def infer_test_file(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--test-file not found: {p}")
        return p
    candidates = [
        data_dir / "test.parquet",
        data_dir / "test.csv",
        data_dir / "Test.parquet",
        data_dir / "Test.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    for ext in ("parquet", "csv"):
        found = list(data_dir.glob(f"*.{ext}"))
        found_sorted = sorted(found, key=lambda p: ("test" not in p.stem.lower(), p.name))
        if found_sorted:
            return found_sorted[0]
    raise FileNotFoundError(f"No test file found under {data_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/raw", help="directory containing train/test files")
    ap.add_argument("--train-file", type=str, default=None, help="explicit path to train file")
    ap.add_argument("--test-file", type=str, default=None, help="explicit path to test file (for column alignment)")
    ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
    ap.add_argument("--id-col", type=str, default="date_id")
    ap.add_argument("--out-dir", type=str, default="artifacts/Preprocessing_V")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--metrics-path", type=str, default=None, help="optional path to dump CV metrics JSON")
    ap.add_argument("--random-seed", type=int, default=42, help="Random seed shared across preprocessing and model components.")

    ap.add_argument("--m-policy", type=str, default="ridge_stack", choices=list(MGroupImputer.SUPPORTED_POLICIES), help="M group imputer policy applied prior to E/I/V policies (default: ridge_stack).")
    ap.add_argument("--m-rolling-window", type=int, default=5)
    ap.add_argument("--m-ema-alpha", type=float, default=0.3)
    ap.add_argument(
        "--m-policy-param",
        action="append",
        default=[],
        help="Additional M policy parameters as key=value pairs. Repeat option for multiple parameters.",
    )
    ap.add_argument(
        "--m-calendar-col",
        type=str,
        default=None,
    help="Optional calendar column override for the M policy (default: follow V/I/E precedence).",
    )

    ap.add_argument("--e-policy", type=str, default="ffill_bfill", choices=list(EGroupImputer.SUPPORTED_POLICIES))
    ap.add_argument("--e-rolling-window", type=int, default=5)
    ap.add_argument("--e-ema-alpha", type=float, default=0.3)
    ap.add_argument(
        "--e-policy-param",
        action="append",
        default=[],
        help="Additional E policy parameters as key=value pairs. Repeat option for multiple parameters.",
    )
    ap.add_argument(
        "--e-calendar-col",
        type=str,
        default="date_id",
        help="Calendar column consumed by the E policy (empty string to disable).",
    )
    ap.add_argument(
        "--e-all-nan-strategy",
        type=str,
        default="keep_nan",
        choices=["keep_nan", "fill_zero", "fill_constant"],
        help="Fallback handling for E columns fully missing during training.",
    )
    ap.add_argument(
        "--e-all-nan-fill",
        type=float,
        default=0.0,
        help="Constant value used when --e-all-nan-strategy=fill_constant.",
    )

    ap.add_argument("--i-policy", type=str, default="ffill_bfill", choices=list(IGroupImputer.SUPPORTED_POLICIES))
    ap.add_argument("--i-rolling-window", type=int, default=5)
    ap.add_argument("--i-ema-alpha", type=float, default=0.3)
    ap.add_argument(
        "--i-policy-param",
        action="append",
        default=[],
        help="Additional I policy parameters as key=value pairs. Repeat option for multiple parameters.",
    )
    ap.add_argument(
        "--i-calendar-col",
        type=str,
        default="date_id",
        help="Calendar column consumed by the I policy (empty string to disable).",
    )
    ap.add_argument("--i-clip-low", type=float, default=0.001, help="Lower quantile used for post-impute clipping of I columns.")
    ap.add_argument("--i-clip-high", type=float, default=0.999, help="Upper quantile used for post-impute clipping of I columns.")
    ap.add_argument("--disable-i-clip", action="store_true", help="Disable quantile clipping for I columns after imputation.")

    ap.add_argument(
        "--p-policy",
        type=str,
        default="ffill_bfill",
        choices=list(PGroupImputer.SUPPORTED_POLICIES),
    help="P group imputer policy applied between I and V stages.",
    )
    ap.add_argument("--p-rolling-window", type=int, default=5)
    ap.add_argument("--p-ema-alpha", type=float, default=0.3)
    ap.add_argument(
        "--p-policy-param",
        action="append",
        default=[],
        help="Additional P policy parameters as key=value pairs. Repeat option for multiple parameters.",
    )
    ap.add_argument(
        "--p-calendar-col",
        type=str,
        default="date_id",
        help="Calendar column consumed by the P policy (empty string to disable).",
    )
    ap.add_argument("--p-mad-scale", type=float, default=4.0, help="MAD scaling coefficient for P valuation clipping.")
    ap.add_argument("--p-mad-min-samples", type=int, default=25, help="Minimum samples required before applying MAD clipping for P columns.")
    ap.add_argument("--disable-p-mad-clip", action="store_true", help="Disable MAD/quantile clipping for P columns after imputation.")
    ap.add_argument("--p-fallback-quantile-low", type=float, default=0.005, help="Lower quantile used when MAD clip is unavailable for P columns.")
    ap.add_argument("--p-fallback-quantile-high", type=float, default=0.995, help="Upper quantile used when MAD clip is unavailable for P columns.")

    ap.add_argument(
        "--s-policy",
        type=str,
        default="ffill_bfill",
        choices=list(SGroupImputer.SUPPORTED_POLICIES),
        help="S group imputer policy applied before the V stage.",
    )
    ap.add_argument("--s-rolling-window", type=int, default=5)
    ap.add_argument("--s-ema-alpha", type=float, default=0.3)
    ap.add_argument(
        "--s-policy-param",
        action="append",
        default=[],
        help="Additional S policy parameters as key=value pairs. Repeat option for multiple parameters.",
    )
    ap.add_argument(
        "--s-calendar-col",
        type=str,
        default="date_id",
        help="Calendar column consumed by the S policy (empty string to disable).",
    )
    ap.add_argument("--s-mad-scale", type=float, default=4.0, help="MAD scaling coefficient for S sentiment clipping.")
    ap.add_argument("--s-mad-min-samples", type=int, default=25, help="Minimum samples required before applying MAD clipping for S columns.")
    ap.add_argument("--disable-s-mad-clip", action="store_true", help="Disable MAD/quantile clipping for S columns after imputation.")
    ap.add_argument("--s-fallback-quantile-low", type=float, default=0.005, help="Lower quantile used when MAD clip is unavailable for S columns.")
    ap.add_argument("--s-fallback-quantile-high", type=float, default=0.995, help="Upper quantile used when MAD clip is unavailable for S columns.")

    ap.add_argument(
        "--v-policy",
        type=str,
        default="ffill_bfill",
        choices=list(VGroupImputer.SUPPORTED_POLICIES),
        help="V group imputer policy applied last in the preprocessing chain.",
    )
    ap.add_argument("--v-rolling-window", type=int, default=5)
    ap.add_argument("--v-ema-alpha", type=float, default=0.3)
    ap.add_argument(
        "--v-policy-param",
        action="append",
        default=[],
        help="Additional V policy parameters as key=value pairs. Repeat option for multiple parameters.",
    )
    ap.add_argument(
        "--v-calendar-col",
        type=str,
        default="date_id",
        help="Calendar column consumed by the V policy (empty string to disable).",
    )
    ap.add_argument("--v-clip-low", type=float, default=0.01, help="Lower quantile used for post-impute clipping of V columns.")
    ap.add_argument("--v-clip-high", type=float, default=0.99, help="Upper quantile used for post-impute clipping of V columns.")
    ap.add_argument("--disable-v-clip", action="store_true", help="Disable quantile clipping for V columns after imputation.")
    ap.add_argument("--disable-v-log", action="store_true", help="Disable log1p transform for V columns after imputation.")
    ap.add_argument("--v-log-epsilon", type=float, default=1e-6, help="Offset epsilon applied before log1p when V columns contain negatives.")

    # 時系列CV
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=5, help="optional gap between train and val for leakage safety (default: 5)")
    ap.add_argument("--min-val-size", type=int, default=0, help="skip folds whose validation size after gap is below this threshold")
    # post-process grid
    ap.add_argument("--mult-grid", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--lo-grid", type=float, nargs="*", default=[0.8, 0.9, 1.0])
    ap.add_argument("--hi-grid", type=float, nargs="*", default=[1.0, 1.1, 1.2])
    ap.add_argument("--optimize-for", type=str, default="msr", choices=["msr", "msr_down", "vmsr"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--lam-grid", type=float, nargs="*", default=[0.0, 0.25, 0.5], help="penalty lambda grid for vMSR")
    ap.add_argument("--pp-aggregate", type=str, default="refit", choices=["refit", "median", "vote"], help="strategy to combine fold-level post-process parameters for OOF evaluation")
    # LGBM params
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--min-data-in-leaf", type=int, default=20)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument("--model-n-jobs", type=int, default=-1, help="LightGBM worker threads (default: use all cores)")
    args = ap.parse_args()

    seed = int(args.random_seed)
    np.random.seed(seed)
    random.seed(seed)

    raw_m_policy_params = [item for item in (args.m_policy_param or []) if item]
    m_policy_params = parse_policy_params(raw_m_policy_params)
    m_policy_params.setdefault("random_state", seed)
    m_policy_params.setdefault("random_seed", seed)

    raw_e_policy_params = [item for item in (args.e_policy_param or []) if item]
    e_policy_params = parse_policy_params(raw_e_policy_params)
    e_policy_params.setdefault("random_state", seed)
    e_policy_params.setdefault("random_seed", seed)

    raw_i_policy_params = [item for item in (args.i_policy_param or []) if item]
    i_policy_params = parse_policy_params(raw_i_policy_params)
    i_policy_params.setdefault("random_state", seed)
    i_policy_params.setdefault("random_seed", seed)

    raw_p_policy_params = [item for item in (args.p_policy_param or []) if item]
    p_policy_params = parse_policy_params(raw_p_policy_params)
    p_policy_params.setdefault("random_state", seed)
    p_policy_params.setdefault("random_seed", seed)

    raw_s_policy_params = [item for item in (args.s_policy_param or []) if item]
    s_policy_params = parse_policy_params(raw_s_policy_params)
    s_policy_params.setdefault("random_state", seed)
    s_policy_params.setdefault("random_seed", seed)

    raw_v_policy_params = [item for item in (args.v_policy_param or []) if item]
    v_policy_params = parse_policy_params(raw_v_policy_params)
    v_policy_params.setdefault("random_state", seed)
    v_policy_params.setdefault("random_seed", seed)

    v_clip_low = float(args.v_clip_low)
    v_clip_high = float(args.v_clip_high)
    if not (0.0 <= v_clip_low < v_clip_high <= 1.0):
        raise ValueError("--v-clip-low/--v-clip-high must satisfy 0 ≤ low < high ≤ 1.")
    v_log_epsilon = float(args.v_log_epsilon)

    p_calendar_col = args.p_calendar_col.strip() if args.p_calendar_col is not None else "date_id"
    if p_calendar_col == "":
        p_calendar_col = None
    e_calendar_col = args.e_calendar_col.strip() if args.e_calendar_col is not None else "date_id"
    if e_calendar_col == "":
        e_calendar_col = None
    i_calendar_col = args.i_calendar_col.strip() if args.i_calendar_col is not None else "date_id"
    if i_calendar_col == "":
        i_calendar_col = None
    s_calendar_col = args.s_calendar_col.strip() if args.s_calendar_col is not None else "date_id"
    if s_calendar_col == "":
        s_calendar_col = None
    v_calendar_col = args.v_calendar_col.strip() if args.v_calendar_col is not None else "date_id"
    if v_calendar_col == "":
        v_calendar_col = None
    if args.m_calendar_col is None:
        m_calendar_col = None
    else:
        m_calendar_col = args.m_calendar_col.strip()
        if m_calendar_col == "":
            m_calendar_col = None

    # prefer V-centric calendars and fall back through upstream groups
    calendar_priority = (
        v_calendar_col,
        e_calendar_col,
        i_calendar_col,
        p_calendar_col,
        s_calendar_col,
    )
    if m_calendar_col is None:
        for candidate in calendar_priority:
            if candidate is not None:
                m_calendar_col = candidate
                break

    calendar_columns: List[str] = []
    for col in (*calendar_priority, m_calendar_col):
        if col and col not in calendar_columns:
            calendar_columns.append(col)
    calendar_cols_set = {col for col in calendar_columns if col}
    primary_calendar_col = calendar_columns[0] if calendar_columns else None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = infer_train_file(data_dir, args.train_file)
    print(f"[info] train file: {train_path}")

    df = load_table(train_path)
    if "date_id" in df.columns:
        df = df.sort_values("date_id").reset_index(drop=True)
    generated_lagged: list[str] = []
    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] test file (for column alignment): {test_path}")
    df_test_head = load_table(test_path)

    if args.target_col not in df.columns:
        raise KeyError(f"target column '{args.target_col}' not found in {list(df.columns)[:20]}...")
    for cal_col in calendar_columns:
        if cal_col and cal_col not in df.columns:
            raise KeyError(f"calendar column '{cal_col}' not found in train data")
        if cal_col and cal_col not in df_test_head.columns:
            raise KeyError(f"calendar column '{cal_col}' not found in test data")

    y = df[args.target_col].astype(float)
    base_drop = {
        args.target_col,
        args.id_col,
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
        "is_scored",
    }
    for cal_col in calendar_columns:
        if cal_col:
            base_drop.discard(cal_col)
    X_all = df.drop(columns=[c for c in base_drop if c in df.columns])
    test_cols = set(df_test_head.columns.tolist())
    feature_cols: list[str] = []
    for col in X_all.columns:
        if col not in test_cols:
            continue
        if col in base_drop:
            continue
        if col.startswith("lagged_"):
            continue
        if col in calendar_cols_set:
            continue
        feature_cols.append(col)
    if not feature_cols:
        raise RuntimeError("No common feature columns between train and test after drops.")

    calendar_cols_in_data = [col for col in calendar_columns if col and col in X_all.columns and col in test_cols]
    pipeline_input_columns = list(feature_cols) + [col for col in calendar_cols_in_data if col not in feature_cols]

    X = X_all.loc[:, pipeline_input_columns].copy()
    pipeline_input_columns = list(X.columns)

    feature_frame = X.loc[:, feature_cols]
    num_cols = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    e_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith("E")]
    i_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith("I")]
    p_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith("P")]
    s_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith("S")]
    v_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith("V")]

    e_mask_cols: list[str] = []
    if args.e_policy == "mask_plus_mean":
        e_mask_cols = [f"Emask__{col}" for col in e_cols]
    i_mask_cols: list[str] = []
    if args.i_policy == "mask_plus_mean":
        i_mask_cols = [f"Imask__{col}" for col in i_cols]
    p_mask_cols: list[str] = []
    if args.p_policy == "mask_plus_mean":
        p_mask_cols = [f"Pmask__{col}" for col in p_cols]
    s_mask_cols: list[str] = []
    if args.s_policy == "mask_plus_mean":
        s_mask_cols = [f"Smask__{col}" for col in s_cols]
    v_mask_cols: list[str] = []
    if args.v_policy == "mask_plus_mean":
        v_mask_cols = [f"Vmask__{col}" for col in v_cols]
    mask_cols = list(dict.fromkeys(e_mask_cols + i_mask_cols + p_mask_cols + s_mask_cols + v_mask_cols))
    num_cols_with_masks = num_cols + [c for c in mask_cols if c not in num_cols]

    if e_cols:
        X.loc[:, e_cols] = X.loc[:, e_cols].apply(pd.to_numeric, errors="coerce")
    if i_cols:
        X.loc[:, i_cols] = X.loc[:, i_cols].apply(pd.to_numeric, errors="coerce")
    if p_cols:
        X.loc[:, p_cols] = X.loc[:, p_cols].apply(pd.to_numeric, errors="coerce")
    if s_cols:
        X.loc[:, s_cols] = X.loc[:, s_cols].apply(pd.to_numeric, errors="coerce")
    if v_cols:
        X.loc[:, v_cols] = X.loc[:, v_cols].apply(pd.to_numeric, errors="coerce")

    encoder_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False
    cat_encoder = OneHotEncoder(**encoder_kwargs)  # type: ignore[arg-type]
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols_with_masks),
            ("cat", cat_encoder, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    m_imputer = MGroupImputer(
        columns=None,
        policy=args.m_policy,
        rolling_window=args.m_rolling_window,
        ema_alpha=args.m_ema_alpha,
        calendar_column=m_calendar_col,
        policy_params=m_policy_params,
        random_state=seed,
    )
    e_imputer = EGroupImputer(
        columns=None,
        policy=args.e_policy,
        rolling_window=args.e_rolling_window,
        ema_alpha=args.e_ema_alpha,
        calendar_column=e_calendar_col,
        policy_params=e_policy_params,
        random_state=seed,
        all_nan_strategy=args.e_all_nan_strategy,
        all_nan_fill=args.e_all_nan_fill,
    )
    i_imputer = IGroupImputer(
        columns=None,
        policy=args.i_policy,
        rolling_window=args.i_rolling_window,
        ema_alpha=args.i_ema_alpha,
        calendar_column=i_calendar_col,
        policy_params=i_policy_params,
        random_state=seed,
        clip_quantile_low=args.i_clip_low,
        clip_quantile_high=args.i_clip_high,
        enable_quantile_clip=not args.disable_i_clip,
    )
    p_imputer = PGroupImputer(
        columns=None,
        policy=args.p_policy,
        rolling_window=args.p_rolling_window,
        ema_alpha=args.p_ema_alpha,
        calendar_column=p_calendar_col,
        policy_params=p_policy_params,
        random_state=seed,
        mad_clip_scale=args.p_mad_scale,
        mad_clip_min_samples=args.p_mad_min_samples,
        enable_mad_clip=not args.disable_p_mad_clip,
        fallback_quantile_low=args.p_fallback_quantile_low,
        fallback_quantile_high=args.p_fallback_quantile_high,
    )
    s_imputer = SGroupImputer(
        columns=None,
        policy=args.s_policy,
        rolling_window=args.s_rolling_window,
        ema_alpha=args.s_ema_alpha,
        calendar_column=s_calendar_col,
        policy_params=s_policy_params,
        random_state=seed,
        mad_clip_scale=args.s_mad_scale,
        mad_clip_min_samples=args.s_mad_min_samples,
        enable_mad_clip=not args.disable_s_mad_clip,
        fallback_quantile_low=args.s_fallback_quantile_low,
        fallback_quantile_high=args.s_fallback_quantile_high,
    )
    v_imputer = VGroupImputer(
        columns=None,
        policy=args.v_policy,
        rolling_window=args.v_rolling_window,
        ema_alpha=args.v_ema_alpha,
        calendar_column=v_calendar_col,
        policy_params=v_policy_params,
        random_state=seed,
        clip_quantile_low=v_clip_low,
        clip_quantile_high=v_clip_high,
        enable_quantile_clip=not args.disable_v_clip,
        log_transform=not args.disable_v_log,
        log_offset_epsilon=v_log_epsilon,
    )
    if not HAS_LGBM or LGBMRegressor is None:  # type: ignore[truthy-function]
        raise RuntimeError("LightGBM is required but not installed. Please install 'lightgbm'.")
    model = LGBMRegressor(  # type: ignore[operator]
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.bagging_fraction,
        colsample_bytree=args.feature_fraction,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        bagging_freq=args.bagging_freq,
        random_state=seed,
        n_jobs=args.model_n_jobs,
    )
    pipe = Pipeline(
        [
            ("m_imputer", m_imputer),
            ("e_imputer", e_imputer),
            ("i_imputer", i_imputer),
            ("p_imputer", p_imputer),
            ("s_imputer", s_imputer),
            ("v_imputer", v_imputer),
            ("pre", pre),
            ("model", model),
        ]
    )

    n_rounds = int(model.get_params().get("n_estimators", 100))
    log_period = max(1, n_rounds // 10)
    callbacks = []
    if HAS_LGBM and lgb is not None:
        try:
            callbacks.append(lgb.log_evaluation(period=log_period))

            def _progress_cb(env):
                it = getattr(env, "iteration", 0)
                if it % log_period == 0 or it == n_rounds:
                    pct = 100.0 * it / n_rounds if n_rounds else 0.0
                    print(f"[progress] iteration {it}/{n_rounds} ({pct:5.1f}%)")

            callbacks.append(_progress_cb)
        except Exception:
            callbacks = []

    print(f"[info] time-series CV: splits={args.n_splits} gap={args.gap}")
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    oof_pred = np.full(len(X), np.nan, dtype=float)
    fold_logs: list[dict] = []

    X_np = X.reset_index(drop=True)
    y_np = y.reset_index(drop=True)

    optimize_for = args.optimize_for if args.optimize_for in ("msr", "msr_down", "vmsr") else "msr"

    fold_id = 0
    val_sizes: list[int] = []
    for train_idx, val_idx in tscv.split(X_np):
        fold_id += 1
        if args.gap > 0:
            if len(train_idx) > args.gap:
                train_idx = train_idx[:-args.gap]
            if len(val_idx) > args.gap:
                val_idx = val_idx[args.gap:]
        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"[warn][fold {fold_id}] empty train/val after gap; skip fold.")
            continue
        if args.min_val_size and len(val_idx) < args.min_val_size:
            print(
                f"[warn][fold {fold_id}] validation size {len(val_idx)} < min_val_size={args.min_val_size}; skip fold."
            )
            continue
        val_sizes.append(len(val_idx))
        X_tr, X_va = X_np.iloc[train_idx], X_np.iloc[val_idx]
        y_tr, y_va = y_np.iloc[train_idx], y_np.iloc[val_idx]

        model_f = LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            subsample=args.bagging_fraction,
            colsample_bytree=args.feature_fraction,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            bagging_freq=args.bagging_freq,
            random_state=seed,
            n_jobs=args.model_n_jobs,
        )
        pipe_f = Pipeline(
            [
                ("m_imputer", clone(m_imputer)),
                ("e_imputer", clone(e_imputer)),
                ("i_imputer", clone(i_imputer)),
                ("p_imputer", clone(p_imputer)),
                ("s_imputer", clone(s_imputer)),
                ("v_imputer", clone(v_imputer)),
                ("pre", clone(pre)),
                ("model", model_f),
            ]
        )

        fit_kwargs_f: dict = {}
        if callbacks:
            fit_kwargs_f["model__callbacks"] = callbacks

        try:
            pipe_f.fit(X_tr, y_tr, **fit_kwargs_f)
        except TypeError:
            pipe_f.fit(X_tr, y_tr)
        yhat_va = pipe_f.predict(X_va)
        yhat_va = _to_1d_np(yhat_va)
        oof_pred[val_idx] = yhat_va
        rmse_va = float(np.sqrt(mean_squared_error(y_va, yhat_va)))

        best_params, grid_res = grid_search_msr(
            y_pred=yhat_va,
            y_true=y_va.to_numpy(dtype=float, copy=False),
            mult_grid=args.mult_grid,
            lo_grid=args.lo_grid,
            hi_grid=args.hi_grid,
            eps=args.eps,
            optimize_for=optimize_for,
            lam_grid=args.lam_grid if optimize_for == "vmsr" else [0.0],
        )
        if optimize_for == "vmsr":
            candidates = [
                r
                for r in grid_res
                if r["mult"] == best_params.mult and r["lo"] == best_params.lo and r["hi"] == best_params.hi
            ]
            if candidates:
                best_row = max(candidates, key=lambda r: r.get("vmsr", -1e18))
                lam_best = float(best_row.get("vmsr_lam", 0.0))
            else:
                lam_best = float(args.lam_grid[0]) if args.lam_grid else 0.0
        else:
            lam_best = 0.0

        best_metrics = evaluate_msr_proxy(
            yhat_va,
            y_va.to_numpy(dtype=float, copy=False),
            best_params,
            eps=args.eps,
            lam=lam_best,
        )
        e_missing_ratio_fold: Dict[str, float] = {}
        for col in e_cols:
            if col not in X_va.columns:
                continue
            converted = cast(pd.Series, pd.to_numeric(X_va[col], errors="coerce"))
            e_missing_ratio_fold[col] = float(converted.isna().mean())
        i_missing_ratio_fold: Dict[str, float] = {}
        for col in i_cols:
            if col not in X_va.columns:
                continue
            converted = cast(pd.Series, pd.to_numeric(X_va[col], errors="coerce"))
            i_missing_ratio_fold[col] = float(converted.isna().mean())
        p_missing_ratio_fold: Dict[str, float] = {}
        for col in p_cols:
            if col not in X_va.columns:
                continue
            converted = cast(pd.Series, pd.to_numeric(X_va[col], errors="coerce"))
            p_missing_ratio_fold[col] = float(converted.isna().mean())
        v_missing_ratio_fold: Dict[str, float] = {}
        for col in v_cols:
            if col not in X_va.columns:
                continue
            converted = cast(pd.Series, pd.to_numeric(X_va[col], errors="coerce"))
            v_missing_ratio_fold[col] = float(converted.isna().mean())

        fold_logs.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "coverage": float(np.isfinite(yhat_va).mean()),
                "rmse_val": rmse_va,
                "best_mult": best_params.mult,
                "best_lo": best_params.lo,
                "best_hi": best_params.hi,
                "best_lam": lam_best,
                **{f"best_{k}": v for k, v in best_metrics.items()},
                "e_missing_ratio": e_missing_ratio_fold,
                "i_missing_ratio": i_missing_ratio_fold,
                "p_missing_ratio": p_missing_ratio_fold,
                "v_missing_ratio": v_missing_ratio_fold,
            }
        )
        log_msg = (
            f"[cv][fold {fold_id}] rmse={rmse_va:.6f} | best({optimize_for}) mult={best_params.mult} lo={best_params.lo} hi={best_params.hi} "
            f"| msr={best_metrics['msr']:.6f} msr_down={best_metrics['msr_down']:.6f}"
        )
        if optimize_for == "vmsr":
            log_msg += f" vmsr={best_metrics['vmsr']:.6f} lam={lam_best:.2f}"
        log_msg += f" mean={best_metrics['mean']:.6e} std={best_metrics['std']:.6e}"
        print(log_msg)

    if val_sizes:
        vs = np.array(val_sizes)
        vmin, vmed, vmax = int(vs.min()), float(np.median(vs)), int(vs.max())
        print(f"[cv] val_size stats: min={vmin} median={vmed:.1f} max={vmax} n_folds={len(val_sizes)} gap={args.gap}")
        if vmed > 0.0:
            small_folds = [int(v) for v in vs if v < 0.5 * vmed]
            if small_folds:
                print(f"[warn] validation size below 50% of median detected: {small_folds}")

    mask = np.isfinite(oof_pred)
    if not np.any(mask):
        print("[warn] no OOF predictions available; check CV settings.")
        rmse_oof = float("nan")
    else:
        rmse_oof = float(
            np.sqrt(mean_squared_error(y_np.to_numpy(dtype=float, copy=False)[mask], oof_pred[mask]))
        )
    coverage = float(np.mean(mask)) if len(mask) else 0.0
    print(f"[metric][oof] rmse={rmse_oof:.6f} coverage={coverage:.3f}")

    pp_strategy = args.pp_aggregate
    if not np.any(mask):
        print("[warn] OOF empty after gap; skip OOF optimization.")
        best_params_global = PostProcessParams()
        lam_best_global = 0.0
        grid_all: list[dict] = []
        best_global_metrics = {
            "rmse": float("nan"),
            "msr": float("nan"),
            "msr_down": float("nan"),
            "vmsr": float("nan"),
            "vmsr_lam": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "std_down": float("nan"),
        }
    else:
        grid_all = []
        if pp_strategy == "refit":
            best_params_global, grid_all = grid_search_msr(
                y_pred=oof_pred[mask],
                y_true=y_np.to_numpy(dtype=float, copy=False)[mask],
                mult_grid=args.mult_grid,
                lo_grid=args.lo_grid,
                hi_grid=args.hi_grid,
                eps=args.eps,
                optimize_for=optimize_for,
                lam_grid=args.lam_grid if optimize_for == "vmsr" else [0.0],
            )
            if optimize_for == "vmsr":
                candidates = [
                    r
                    for r in grid_all
                    if r["mult"] == best_params_global.mult and r["lo"] == best_params_global.lo and r["hi"] == best_params_global.hi
                ]
                if candidates:
                    best_row = max(candidates, key=lambda r: r.get("vmsr", -1e18))
                    lam_best_global = float(best_row.get("vmsr_lam", 0.0))
                else:
                    lam_best_global = float(args.lam_grid[0]) if args.lam_grid else 0.0
            else:
                lam_best_global = 0.0
        else:
            best_params_global, lam_best_global = _aggregate_postprocess_from_folds(fold_logs, pp_strategy, optimize_for)
        best_global_metrics = evaluate_msr_proxy(
            oof_pred[mask],
            y_np.to_numpy(dtype=float, copy=False)[mask],
            best_params_global,
            eps=args.eps,
            lam=lam_best_global if optimize_for == "vmsr" else 0.0,
        )
        best_global_metrics["vmsr_lam"] = float(lam_best_global if optimize_for == "vmsr" else 0.0)
    oof_msg = (
        f"[cv][oof] best({optimize_for}) mult={best_params_global.mult} lo={best_params_global.lo} hi={best_params_global.hi} "
        f"| msr={best_global_metrics['msr']:.6f} msr_down={best_global_metrics['msr_down']:.6f}"
    )
    if optimize_for == "vmsr":
        oof_msg += f" vmsr={best_global_metrics['vmsr']:.6f} lam={lam_best_global:.2f}"
    oof_msg += f" mean={best_global_metrics['mean']:.6e} std={best_global_metrics['std']:.6e}"
    print(oof_msg)

    try:
        pipe.fit(X, y, **({"model__eval_metric": "rmse"} if callbacks else {}))
    except TypeError:
        pipe.fit(X, y)

    m_imputer_fitted = pipe.named_steps.get("m_imputer")
    e_imputer_fitted = pipe.named_steps.get("e_imputer")
    i_imputer_fitted = pipe.named_steps.get("i_imputer")
    p_imputer_fitted = pipe.named_steps.get("p_imputer")
    s_imputer_fitted = pipe.named_steps.get("s_imputer")
    v_imputer_fitted = pipe.named_steps.get("v_imputer")

    fitted_m_columns: List[str] = []
    fitted_m_extra_columns: List[str] = []
    m_imputer_warning_messages: List[str] = []
    m_imputer_warning_count = 0
    training_m_missing_ratio: Dict[str, float] = {}
    try:
        if m_imputer_fitted is not None:
            fitted_m_columns = list(getattr(m_imputer_fitted, "columns_", []) or [])
            fitted_m_extra_columns = list(getattr(m_imputer_fitted, "extra_columns_", []) or [])
            state_dict_m = getattr(m_imputer_fitted, "_state_", {})
            if isinstance(state_dict_m, dict):
                warnings_list_m = state_dict_m.get("warnings", [])
                if isinstance(warnings_list_m, list):
                    m_imputer_warning_messages = list(
                        dict.fromkeys(str(msg) for msg in warnings_list_m if isinstance(msg, str))
                    )
                    m_imputer_warning_count = len(m_imputer_warning_messages)
            for col in fitted_m_columns:
                if col in df.columns:
                    converted_col = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                    training_m_missing_ratio[col] = float(converted_col.isna().mean())
    except Exception:
        fitted_m_columns = []
        fitted_m_extra_columns = []
        m_imputer_warning_messages = []
        m_imputer_warning_count = 0
        training_m_missing_ratio = {}

    fitted_e_columns: List[str] = []
    fitted_e_extra_columns: List[str] = []
    e_post_impute_nan_ratio = float("nan")
    e_imputer_warning_messages: List[str] = []
    e_imputer_warning_count = 0
    training_e_missing_ratio: Dict[str, float] = {}
    imputer_all_nan_columns: List[str] = []
    try:
        if e_imputer_fitted is not None:
            fitted_e_columns = list(getattr(e_imputer_fitted, "columns_", []) or [])
            fitted_e_extra_columns = list(getattr(e_imputer_fitted, "extra_columns_", []) or [])
            imputer_all_nan_columns = list(getattr(e_imputer_fitted, "all_nan_columns_", []) or [])
            transformed_train_e = cast(pd.DataFrame, e_imputer_fitted.transform(X))
            if fitted_e_columns:
                e_post_impute_nan_ratio = float(
                    np.isnan(transformed_train_e[fitted_e_columns].to_numpy(dtype=float, copy=False)).mean()
                )
            state_dict_e = getattr(e_imputer_fitted, "_state_", {})
            if isinstance(state_dict_e, dict):
                warnings_list = state_dict_e.get("warnings", [])
                if isinstance(warnings_list, list):
                    e_imputer_warning_messages = list(
                        dict.fromkeys(str(msg) for msg in warnings_list if isinstance(msg, str))
                    )
                    e_imputer_warning_count = len(e_imputer_warning_messages)
            for col in fitted_e_columns:
                if col in df.columns:
                    converted_col = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                    training_e_missing_ratio[col] = float(converted_col.isna().mean())
    except Exception:
        e_post_impute_nan_ratio = float("nan")
        e_imputer_warning_messages = []
        e_imputer_warning_count = 0
        training_e_missing_ratio = {}
        imputer_all_nan_columns = []

    e_column_count = len(fitted_e_columns)

    fitted_i_columns: List[str] = []
    fitted_i_extra_columns: List[str] = []
    i_post_impute_nan_ratio = float("nan")
    i_imputer_warning_messages: List[str] = []
    i_imputer_warning_count = 0
    training_i_missing_ratio: Dict[str, float] = {}
    i_clip_bounds: Dict[str, Tuple[float, float]] = {}
    try:
        if i_imputer_fitted is not None:
            fitted_i_columns = list(getattr(i_imputer_fitted, "columns_", []) or [])
            fitted_i_extra_columns = list(getattr(i_imputer_fitted, "extra_columns_", []) or [])
            transformed_train_i = cast(pd.DataFrame, i_imputer_fitted.transform(X))
            if fitted_i_columns:
                i_post_impute_nan_ratio = float(
                    np.isnan(transformed_train_i[fitted_i_columns].to_numpy(dtype=float, copy=False)).mean()
                )
            state_dict_i = getattr(i_imputer_fitted, "_state_", {})
            if isinstance(state_dict_i, dict):
                warnings_list_i = state_dict_i.get("warnings", [])
                if isinstance(warnings_list_i, list):
                    i_imputer_warning_messages = list(
                        dict.fromkeys(str(msg) for msg in warnings_list_i if isinstance(msg, str))
                    )
                    i_imputer_warning_count = len(i_imputer_warning_messages)
                clip_bounds_state = state_dict_i.get("clip_bounds")
                if isinstance(clip_bounds_state, dict):
                    for key, value in clip_bounds_state.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            i_clip_bounds[str(key)] = (low_val, high_val)
            if not i_clip_bounds and hasattr(i_imputer_fitted, "_clip_bounds_"):
                raw_bounds = getattr(i_imputer_fitted, "_clip_bounds_", {})
                if isinstance(raw_bounds, dict):
                    for key, value in raw_bounds.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            i_clip_bounds[str(key)] = (low_val, high_val)
            for col in fitted_i_columns:
                if col in df.columns:
                    converted_col = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                    training_i_missing_ratio[col] = float(converted_col.isna().mean())
    except Exception:
        fitted_i_columns = []
        fitted_i_extra_columns = []
        i_post_impute_nan_ratio = float("nan")
        i_imputer_warning_messages = []
        i_imputer_warning_count = 0
        training_i_missing_ratio = {}
        i_clip_bounds = {}

    i_clip_bounds_serializable = {
        key: [float(bounds[0]), float(bounds[1])] for key, bounds in i_clip_bounds.items()
    }

    fitted_p_columns: List[str] = []
    fitted_p_extra_columns: List[str] = []
    p_post_impute_nan_ratio = float("nan")
    p_imputer_warning_messages: List[str] = []
    p_imputer_warning_count = 0
    training_p_missing_ratio: Dict[str, float] = {}
    p_clip_bounds: Dict[str, Tuple[float, float]] = {}
    try:
        if p_imputer_fitted is not None:
            fitted_p_columns = list(getattr(p_imputer_fitted, "columns_", []) or [])
            fitted_p_extra_columns = list(getattr(p_imputer_fitted, "extra_columns_", []) or [])
            transformed_train_p = cast(pd.DataFrame, p_imputer_fitted.transform(X))
            if fitted_p_columns:
                p_post_impute_nan_ratio = float(
                    np.isnan(transformed_train_p[fitted_p_columns].to_numpy(dtype=float, copy=False)).mean()
                )
            state_dict_p = getattr(p_imputer_fitted, "_state_", {})
            if isinstance(state_dict_p, dict):
                warnings_list_p = state_dict_p.get("warnings", [])
                if isinstance(warnings_list_p, list):
                    p_imputer_warning_messages = list(
                        dict.fromkeys(str(msg) for msg in warnings_list_p if isinstance(msg, str))
                    )
                    p_imputer_warning_count = len(p_imputer_warning_messages)
                clip_bounds_state_p = state_dict_p.get("clip_bounds") or state_dict_p.get("mad_clip_bounds")
                if isinstance(clip_bounds_state_p, dict):
                    for key, value in clip_bounds_state_p.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            p_clip_bounds[str(key)] = (low_val, high_val)
            if not p_clip_bounds and hasattr(p_imputer_fitted, "_clip_bounds_"):
                raw_bounds_p = getattr(p_imputer_fitted, "_clip_bounds_", {})
                if isinstance(raw_bounds_p, dict):
                    for key, value in raw_bounds_p.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            p_clip_bounds[str(key)] = (low_val, high_val)
            for col in fitted_p_columns:
                if col in df.columns:
                    converted_col = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                    training_p_missing_ratio[col] = float(converted_col.isna().mean())
    except Exception:
        fitted_p_columns = []
        fitted_p_extra_columns = []
        p_post_impute_nan_ratio = float("nan")
        p_imputer_warning_messages = []
        p_imputer_warning_count = 0
        training_p_missing_ratio = {}
        p_clip_bounds = {}

    p_clip_bounds_serializable = {
        key: [float(bounds[0]), float(bounds[1])] for key, bounds in p_clip_bounds.items()
    }

    fitted_s_columns: List[str] = []
    fitted_s_extra_columns: List[str] = []
    s_post_impute_nan_ratio = float("nan")
    s_imputer_warning_messages: List[str] = []
    s_imputer_warning_count = 0
    training_s_missing_ratio: Dict[str, float] = {}
    s_clip_bounds: Dict[str, Tuple[float, float]] = {}
    try:
        if s_imputer_fitted is not None:
            fitted_s_columns = list(getattr(s_imputer_fitted, "columns_", []) or [])
            fitted_s_extra_columns = list(getattr(s_imputer_fitted, "extra_columns_", []) or [])
            transformed_train_s = cast(pd.DataFrame, s_imputer_fitted.transform(X))
            if fitted_s_columns:
                s_post_impute_nan_ratio = float(
                    np.isnan(transformed_train_s[fitted_s_columns].to_numpy(dtype=float, copy=False)).mean()
                )
            state_dict_s = getattr(s_imputer_fitted, "_state_", {})
            if isinstance(state_dict_s, dict):
                warnings_list_s = state_dict_s.get("warnings", [])
                if isinstance(warnings_list_s, list):
                    s_imputer_warning_messages = list(
                        dict.fromkeys(str(msg) for msg in warnings_list_s if isinstance(msg, str))
                    )
                    s_imputer_warning_count = len(s_imputer_warning_messages)
                clip_bounds_state_s = state_dict_s.get("mad_clip_bounds") or state_dict_s.get("clip_bounds")
                if isinstance(clip_bounds_state_s, dict):
                    for key, value in clip_bounds_state_s.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            s_clip_bounds[str(key)] = (low_val, high_val)
            if not s_clip_bounds and hasattr(s_imputer_fitted, "_clip_bounds_"):
                raw_bounds_s = getattr(s_imputer_fitted, "_clip_bounds_", {})
                if isinstance(raw_bounds_s, dict):
                    for key, value in raw_bounds_s.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            s_clip_bounds[str(key)] = (low_val, high_val)
            for col in fitted_s_columns:
                if col in df.columns:
                    converted_col = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                    training_s_missing_ratio[col] = float(converted_col.isna().mean())
    except Exception:
        fitted_s_columns = []
        fitted_s_extra_columns = []
        s_post_impute_nan_ratio = float("nan")
        s_imputer_warning_messages = []
        s_imputer_warning_count = 0
        training_s_missing_ratio = {}
        s_clip_bounds = {}

    s_clip_bounds_serializable = {
        key: [float(bounds[0]), float(bounds[1])] for key, bounds in s_clip_bounds.items()
    }

    fitted_v_columns: List[str] = []
    fitted_v_extra_columns: List[str] = []
    v_post_impute_nan_ratio = float("nan")
    v_imputer_warning_messages: List[str] = []
    v_imputer_warning_count = 0
    training_v_missing_ratio: Dict[str, float] = {}
    v_clip_bounds: Dict[str, Tuple[float, float]] = {}
    v_log_offsets: Dict[str, float] = {}
    v_quantile_clip_enabled = bool(not args.disable_v_clip)
    v_log_transform_enabled = bool(not args.disable_v_log)
    try:
        if v_imputer_fitted is not None:
            fitted_v_columns = list(getattr(v_imputer_fitted, "columns_", []) or [])
            fitted_v_extra_columns = list(getattr(v_imputer_fitted, "extra_columns_", []) or [])
            transformed_train_v = cast(pd.DataFrame, v_imputer_fitted.transform(X))
            if fitted_v_columns:
                v_post_impute_nan_ratio = float(
                    np.isnan(transformed_train_v[fitted_v_columns].to_numpy(dtype=float, copy=False)).mean()
                )
            state_dict_v = getattr(v_imputer_fitted, "_state_", {})
            if isinstance(state_dict_v, dict):
                warnings_list_v = state_dict_v.get("warnings", [])
                if isinstance(warnings_list_v, list):
                    v_imputer_warning_messages = list(
                        dict.fromkeys(str(msg) for msg in warnings_list_v if isinstance(msg, str))
                    )
                    v_imputer_warning_count = len(v_imputer_warning_messages)
                clip_bounds_state_v = (
                    state_dict_v.get("mad_clip_bounds")
                    or state_dict_v.get("clip_bounds")
                )
                if isinstance(clip_bounds_state_v, dict):
                    for key, value in clip_bounds_state_v.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            v_clip_bounds[str(key)] = (low_val, high_val)
                log_offsets_state_v = state_dict_v.get("log_offsets")
                if isinstance(log_offsets_state_v, dict):
                    for key, value in log_offsets_state_v.items():
                        try:
                            v_log_offsets[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
                if "enable_quantile_clip" in state_dict_v:
                    v_quantile_clip_enabled = bool(
                        state_dict_v.get("enable_quantile_clip", v_quantile_clip_enabled)
                    )
                if "log_transform" in state_dict_v:
                    v_log_transform_enabled = bool(
                        state_dict_v.get("log_transform", v_log_transform_enabled)
                    )
            if not v_clip_bounds and hasattr(v_imputer_fitted, "_clip_bounds_"):
                raw_bounds_v = getattr(v_imputer_fitted, "_clip_bounds_", {})
                if isinstance(raw_bounds_v, dict):
                    for key, value in raw_bounds_v.items():
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                low_val = float(value[0])
                                high_val = float(value[1])
                            except (TypeError, ValueError):
                                continue
                            v_clip_bounds[str(key)] = (low_val, high_val)
            if not v_log_offsets and hasattr(v_imputer_fitted, "_log_offsets_"):
                raw_offsets_v = getattr(v_imputer_fitted, "_log_offsets_", {})
                if isinstance(raw_offsets_v, dict):
                    for key, value in raw_offsets_v.items():
                        try:
                            v_log_offsets[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
            v_quantile_clip_enabled = bool(
                getattr(v_imputer_fitted, "enable_quantile_clip", v_quantile_clip_enabled)
            )
            v_log_transform_enabled = bool(
                getattr(v_imputer_fitted, "log_transform", v_log_transform_enabled)
            )
            for col in fitted_v_columns:
                if col in df.columns:
                    converted_col = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                    training_v_missing_ratio[col] = float(converted_col.isna().mean())
    except Exception:
        fitted_v_columns = []
        fitted_v_extra_columns = []
        v_post_impute_nan_ratio = float("nan")
        v_imputer_warning_messages = []
        v_imputer_warning_count = 0
        training_v_missing_ratio = {}
        v_clip_bounds = {}
        v_log_offsets = {}
        v_quantile_clip_enabled = bool(not args.disable_v_clip)
        v_log_transform_enabled = bool(not args.disable_v_log)

    v_clip_bounds_serializable = {
        key: [float(bounds[0]), float(bounds[1])] for key, bounds in v_clip_bounds.items()
    }
    v_log_offsets_serializable = {
        key: float(value) for key, value in v_log_offsets.items()
    }

    model_path = out_dir / "model_pre_v.pkl"
    if not args.no_artifacts:
        joblib.dump(pipe, model_path, compress=3)

    try:
        import sklearn as sk  # type: ignore
        sk_ver = getattr(sk, "__version__", "unknown")
    except Exception:
        sk_ver = "unknown"
    try:
        lgb_ver = getattr(lgb, "__version__", "not_installed") if lgb is not None else "not_installed"
    except Exception:
        lgb_ver = "unknown"

    oof_best_metrics_serializable = {
        k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in best_global_metrics.items()
    }
    fold_logs_serializable = [
        {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in log.items()}
        for log in fold_logs
    ]

    feature_list_spec = {
        "pipeline_input_columns": pipeline_input_columns,
        "model_feature_columns": feature_cols,
        "calendar_column": primary_calendar_col,
        "calendar_columns": calendar_columns,
        "m_columns": fitted_m_columns,
        "m_generated_columns": fitted_m_extra_columns,
        "e_columns": fitted_e_columns,
        "e_generated_columns": fitted_e_extra_columns,
        "e_all_nan_columns": imputer_all_nan_columns,
        "i_columns": fitted_i_columns,
        "i_generated_columns": fitted_i_extra_columns,
        "i_clip_bounds": i_clip_bounds_serializable,
        "p_columns": fitted_p_columns,
        "p_generated_columns": fitted_p_extra_columns,
        "p_clip_bounds": p_clip_bounds_serializable,
        "s_columns": fitted_s_columns,
        "s_generated_columns": fitted_s_extra_columns,
        "s_clip_bounds": s_clip_bounds_serializable,
        "v_columns": fitted_v_columns,
        "v_generated_columns": fitted_v_extra_columns,
        "v_clip_bounds": v_clip_bounds_serializable,
        "v_log_offsets": v_log_offsets_serializable,
        "v_quantile_clip_enabled": bool(v_quantile_clip_enabled),
        "v_log_transform_enabled": bool(v_log_transform_enabled),
        "v_log_epsilon": float(v_log_epsilon),
        "numeric_feature_columns": num_cols,
        "numeric_feature_columns_with_masks": num_cols_with_masks,
        "categorical_feature_columns": cat_cols,
        "mask_feature_columns": mask_cols,
    }

    meta = {
        "train_path": str(train_path),
        "target_col": args.target_col,
        "id_col": args.id_col,
        "model_type": type(model).__name__,
        "model_n_jobs": int(args.model_n_jobs),
        "numeric_cols": num_cols_with_masks,
        "categorical_cols": cat_cols,
        "oof_rmse": float(rmse_oof),
        "feature_count": len(num_cols_with_masks) + len(cat_cols),
        "feature_columns": feature_cols,
        "pipeline_input_columns": pipeline_input_columns,
        "test_path_for_alignment": str(test_path),
        "generated_lagged_features": generated_lagged,
        "cv": fold_logs_serializable,
        "oof_best_params": {
            "mult": best_params_global.mult,
            "lo": best_params_global.lo,
            "hi": best_params_global.hi,
            "lam": lam_best_global,
        },
        "oof_best_metrics": oof_best_metrics_serializable,
        "optimize_for": args.optimize_for,
        "mult_grid": list(map(float, args.mult_grid)),
        "lo_grid": list(map(float, args.lo_grid)),
        "hi_grid": list(map(float, args.hi_grid)),
        "eps": float(args.eps),
        "n_splits": int(args.n_splits),
        "gap": int(args.gap),
        "min_val_size": int(args.min_val_size),
        "pp_aggregate": args.pp_aggregate,
        "random_state": seed,
        "postprocess_defaults": {"clip_min": 0.0, "clip_max": 2.0, "use_post_process": True},
        "sklearn_version": sk_ver,
        "lightgbm_version": lgb_ver,
        "calendar_columns": calendar_columns,
        "m_policy": args.m_policy,
        "m_policy_params": m_policy_params,
        "m_calendar_col": m_calendar_col,
        "m_columns": fitted_m_columns,
        "m_generated_columns": fitted_m_extra_columns,
        "m_imputer_warning_count": int(m_imputer_warning_count),
        "m_imputer_warnings": m_imputer_warning_messages,
        "m_training_missing_ratio": training_m_missing_ratio,
        "e_policy": args.e_policy,
        "e_rolling_window": int(args.e_rolling_window),
        "e_ema_alpha": float(args.e_ema_alpha),
        "e_policy_params": e_policy_params,
        "e_calendar_col": e_calendar_col,
        "e_columns": fitted_e_columns,
        "e_generated_columns": fitted_e_extra_columns,
        "e_all_nan_columns": imputer_all_nan_columns,
        "e_training_missing_ratio": training_e_missing_ratio,
        "e_all_nan_strategy": getattr(e_imputer_fitted, "all_nan_strategy", None),
        "e_all_nan_fill_value": getattr(e_imputer_fitted, "all_nan_fill_value_", None),
        "e_column_count": int(e_column_count),
        "e_post_impute_nan_ratio": float(e_post_impute_nan_ratio),
        "e_imputer_warning_count": int(e_imputer_warning_count),
        "e_imputer_warnings": e_imputer_warning_messages,
        "i_policy": args.i_policy,
        "i_rolling_window": int(args.i_rolling_window),
        "i_ema_alpha": float(args.i_ema_alpha),
        "i_policy_params": i_policy_params,
        "i_calendar_col": i_calendar_col,
        "i_columns": fitted_i_columns,
        "i_generated_columns": fitted_i_extra_columns,
        "i_training_missing_ratio": training_i_missing_ratio,
        "i_clip_bounds": i_clip_bounds_serializable,
        "i_clip_low": float(args.i_clip_low),
        "i_clip_high": float(args.i_clip_high),
        "i_quantile_clip_enabled": bool(not args.disable_i_clip),
        "i_post_impute_nan_ratio": float(i_post_impute_nan_ratio),
        "i_imputer_warning_count": int(i_imputer_warning_count),
        "i_imputer_warnings": i_imputer_warning_messages,
        "p_policy": args.p_policy,
        "p_rolling_window": int(args.p_rolling_window),
        "p_ema_alpha": float(args.p_ema_alpha),
        "p_policy_params": p_policy_params,
        "p_calendar_col": p_calendar_col,
        "p_columns": fitted_p_columns,
        "p_column_count": int(len(fitted_p_columns)),
        "p_generated_columns": fitted_p_extra_columns,
        "p_training_missing_ratio": training_p_missing_ratio,
        "p_clip_bounds": p_clip_bounds_serializable,
        "p_mad_scale": float(args.p_mad_scale),
        "p_mad_min_samples": int(args.p_mad_min_samples),
        "p_fallback_quantile_low": float(args.p_fallback_quantile_low),
        "p_fallback_quantile_high": float(args.p_fallback_quantile_high),
        "p_mad_clip_enabled": bool(not args.disable_p_mad_clip),
        "p_post_impute_nan_ratio": float(p_post_impute_nan_ratio),
        "p_imputer_warning_count": int(p_imputer_warning_count),
        "p_imputer_warnings": p_imputer_warning_messages,
    "s_policy": args.s_policy,
    "s_rolling_window": int(args.s_rolling_window),
    "s_ema_alpha": float(args.s_ema_alpha),
    "s_policy_params": s_policy_params,
    "s_calendar_col": s_calendar_col,
    "s_columns": fitted_s_columns,
    "s_column_count": int(len(fitted_s_columns)),
    "s_generated_columns": fitted_s_extra_columns,
    "s_training_missing_ratio": training_s_missing_ratio,
    "s_clip_bounds": s_clip_bounds_serializable,
    "s_mad_scale": float(args.s_mad_scale),
    "s_mad_min_samples": int(args.s_mad_min_samples),
    "s_fallback_quantile_low": float(args.s_fallback_quantile_low),
    "s_fallback_quantile_high": float(args.s_fallback_quantile_high),
    "s_mad_clip_enabled": bool(not args.disable_s_mad_clip),
    "s_post_impute_nan_ratio": float(s_post_impute_nan_ratio),
    "s_imputer_warning_count": int(s_imputer_warning_count),
    "s_imputer_warnings": s_imputer_warning_messages,
        "v_policy": args.v_policy,
        "v_rolling_window": int(args.v_rolling_window),
        "v_ema_alpha": float(args.v_ema_alpha),
        "v_policy_params": v_policy_params,
        "v_calendar_col": v_calendar_col,
        "v_columns": fitted_v_columns,
        "v_generated_columns": fitted_v_extra_columns,
        "v_training_missing_ratio": training_v_missing_ratio,
        "v_clip_bounds": v_clip_bounds_serializable,
        "v_clip_low": float(v_clip_low),
        "v_clip_high": float(v_clip_high),
        "v_quantile_clip_enabled": bool(v_quantile_clip_enabled),
        "v_log_transform_enabled": bool(v_log_transform_enabled),
        "v_log_offsets": v_log_offsets_serializable,
        "v_log_epsilon": float(v_log_epsilon),
        "v_post_impute_nan_ratio": float(v_post_impute_nan_ratio),
        "v_imputer_warning_count": int(v_imputer_warning_count),
        "v_imputer_warnings": v_imputer_warning_messages,
        "random_seed": seed,
    }
    feature_list_path = out_dir / "feature_list.json"

    if not args.no_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(feature_list_path, "w", encoding="utf-8") as fp:
            json.dump(feature_list_spec, fp, indent=2)
        meta["feature_list_path"] = str(feature_list_path)
        with open(out_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ok] saved: {model_path}, model_meta.json")
    else:
        meta["feature_list_path"] = str(feature_list_path)

    try:
        if not args.no_artifacts:
            if grid_all:
                pd.DataFrame(grid_all).to_csv(out_dir / "oof_grid_results.csv", index=False)
            pd.DataFrame(fold_logs).to_csv(out_dir / "cv_fold_logs.csv", index=False)
            print("[ok] saved: oof_grid_results.csv, cv_fold_logs.csv")
    except Exception:
        # Silently ignore errors when saving optional artifact files
        pass

    if args.metrics_path:
        metrics_out = {
            "m_policy": args.m_policy,
            "m_policy_params": m_policy_params,
            "m_calendar_col": m_calendar_col,
            "m_columns": fitted_m_columns,
            "m_generated_columns": fitted_m_extra_columns,
            "m_imputer_warning_count": int(m_imputer_warning_count),
            "m_imputer_warnings": m_imputer_warning_messages,
            "m_training_missing_ratio": training_m_missing_ratio,
            "e_policy": args.e_policy,
            "e_rolling_window": int(args.e_rolling_window),
            "e_ema_alpha": float(args.e_ema_alpha),
            "e_policy_params": e_policy_params,
            "e_calendar_col": e_calendar_col,
            "e_columns": fitted_e_columns,
            "e_generated_columns": fitted_e_extra_columns,
            "e_all_nan_columns": imputer_all_nan_columns,
            "e_training_missing_ratio": training_e_missing_ratio,
            "e_all_nan_strategy": getattr(e_imputer_fitted, "all_nan_strategy", None),
            "e_all_nan_fill_value": getattr(e_imputer_fitted, "all_nan_fill_value_", None),
            "i_policy": args.i_policy,
            "i_rolling_window": int(args.i_rolling_window),
            "i_ema_alpha": float(args.i_ema_alpha),
            "i_policy_params": i_policy_params,
            "i_calendar_col": i_calendar_col,
            "i_columns": fitted_i_columns,
            "i_generated_columns": fitted_i_extra_columns,
            "i_training_missing_ratio": training_i_missing_ratio,
            "i_clip_bounds": i_clip_bounds_serializable,
            "i_clip_low": float(args.i_clip_low),
            "i_clip_high": float(args.i_clip_high),
            "i_quantile_clip_enabled": bool(not args.disable_i_clip),
            "i_post_impute_nan_ratio": float(i_post_impute_nan_ratio),
            "i_imputer_warning_count": int(i_imputer_warning_count),
            "i_imputer_warnings": i_imputer_warning_messages,
            "p_policy": args.p_policy,
            "p_rolling_window": int(args.p_rolling_window),
            "p_ema_alpha": float(args.p_ema_alpha),
            "p_policy_params": p_policy_params,
            "p_calendar_col": p_calendar_col,
            "p_columns": fitted_p_columns,
            "p_column_count": int(len(fitted_p_columns)),
            "p_generated_columns": fitted_p_extra_columns,
            "p_training_missing_ratio": training_p_missing_ratio,
            "p_clip_bounds": p_clip_bounds_serializable,
            "p_mad_scale": float(args.p_mad_scale),
            "p_mad_min_samples": int(args.p_mad_min_samples),
            "p_fallback_quantile_low": float(args.p_fallback_quantile_low),
            "p_fallback_quantile_high": float(args.p_fallback_quantile_high),
            "p_mad_clip_enabled": bool(not args.disable_p_mad_clip),
            "p_post_impute_nan_ratio": float(p_post_impute_nan_ratio),
            "p_imputer_warning_count": int(p_imputer_warning_count),
            "p_imputer_warnings": p_imputer_warning_messages,
            "s_policy": args.s_policy,
            "s_rolling_window": int(args.s_rolling_window),
            "s_ema_alpha": float(args.s_ema_alpha),
            "s_policy_params": s_policy_params,
            "s_calendar_col": s_calendar_col,
            "s_columns": fitted_s_columns,
            "s_column_count": int(len(fitted_s_columns)),
            "s_generated_columns": fitted_s_extra_columns,
            "s_training_missing_ratio": training_s_missing_ratio,
            "s_clip_bounds": s_clip_bounds_serializable,
            "s_mad_scale": float(args.s_mad_scale),
            "s_mad_min_samples": int(args.s_mad_min_samples),
            "s_fallback_quantile_low": float(args.s_fallback_quantile_low),
            "s_fallback_quantile_high": float(args.s_fallback_quantile_high),
            "s_mad_clip_enabled": bool(not args.disable_s_mad_clip),
            "s_post_impute_nan_ratio": float(s_post_impute_nan_ratio),
            "s_imputer_warning_count": int(s_imputer_warning_count),
            "s_imputer_warnings": s_imputer_warning_messages,
            "v_policy": args.v_policy,
            "v_rolling_window": int(args.v_rolling_window),
            "v_ema_alpha": float(args.v_ema_alpha),
            "v_policy_params": v_policy_params,
            "v_calendar_col": v_calendar_col,
            "v_columns": fitted_v_columns,
            "v_generated_columns": fitted_v_extra_columns,
            "v_training_missing_ratio": training_v_missing_ratio,
            "v_clip_bounds": v_clip_bounds_serializable,
            "v_clip_low": float(v_clip_low),
            "v_clip_high": float(v_clip_high),
            "v_quantile_clip_enabled": bool(v_quantile_clip_enabled),
            "v_log_transform_enabled": bool(v_log_transform_enabled),
            "v_log_offsets": v_log_offsets_serializable,
            "v_log_epsilon": float(v_log_epsilon),
            "v_post_impute_nan_ratio": float(v_post_impute_nan_ratio),
            "v_imputer_warning_count": int(v_imputer_warning_count),
            "v_imputer_warnings": v_imputer_warning_messages,
            "n_splits": int(args.n_splits),
            "gap": int(args.gap),
            "min_val_size": int(args.min_val_size),
            "optimize_for": args.optimize_for,
            "oof_rmse": float(rmse_oof),
            "coverage": float(coverage),
            "oof_metrics": oof_best_metrics_serializable,
            "fold_metrics": fold_logs_serializable,
            "pp_aggregate": args.pp_aggregate,
            "model_n_jobs": int(args.model_n_jobs),
            "e_column_count": int(e_column_count),
            "e_post_impute_nan_ratio": float(e_post_impute_nan_ratio),
            "e_imputer_warning_count": int(e_imputer_warning_count),
            "e_imputer_warnings": e_imputer_warning_messages,
            "random_seed": seed,
            "feature_list_path": str(feature_list_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"[ok] saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
