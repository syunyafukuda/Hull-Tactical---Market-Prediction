#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSR-proxy 学習スクリプト:
- data/raw/{train.parquet|train.csv} を自動検出（--train-file で明示可）
- 目的変数(--target-col)、ID列(--id-col) を除き、説明変数を構成
- LightGBM を使用（ベースラインを踏襲）
- 時系列CVで RMSE と MSR プロキシを同時評価
- post-process（mult, lo, hi）をfold内予測でグリッド探索し、fold平均で最適化
- 学習済みモデルとメタ情報（MSR関連を含む）を artifacts/MSR-proxy/ に保存
"""

import argparse
import inspect
import json
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

from preprocess.M_group.m_group import MGroupImputer  # noqa: E402
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
            pass
        try:
            params[key] = float(value_str)
            continue
        except ValueError:
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
    ap.add_argument("--out-dir", type=str, default="artifacts/Preprocessing_M")
    ap.add_argument("--m-policy", type=str, default="ffill_bfill", choices=list(MGroupImputer.SUPPORTED_POLICIES))
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
        default="date_id",
        help="Calendar column consumed by seasonal/time policies (set empty string to disable).",
    )
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--metrics-path", type=str, default=None, help="optional path to dump CV metrics JSON")
    # 時系列CV
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=0, help="optional gap between train and val for leakage safety")
    ap.add_argument("--min-val-size", type=int, default=0, help="skip folds whose validation size after gap is below this threshold")
    # post-process grid
    ap.add_argument("--mult-grid", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--lo-grid", type=float, nargs="*", default=[0.8, 0.9, 1.0])
    ap.add_argument("--hi-grid", type=float, nargs="*", default=[1.0, 1.1, 1.2])
    ap.add_argument("--optimize-for", type=str, default="msr", choices=["msr", "msr_down", "vmsr"]) 
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--lam-grid", type=float, nargs="*", default=[0.0, 0.25, 0.5], help="penalty lambda grid for vMSR")
    ap.add_argument("--pp-aggregate", type=str, default="refit", choices=["refit", "median", "vote"], help="strategy to combine fold-level post-process parameters for OOF evaluation")
    # LGBM params (軽いチューニング枠)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--min-data-in-leaf", type=int, default=20)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-freq", type=int, default=1)
    args = ap.parse_args()

    raw_policy_params = [item for item in (args.m_policy_param or []) if item]
    policy_params = parse_policy_params(raw_policy_params)
    calendar_col = args.m_calendar_col.strip() if args.m_calendar_col is not None else "date_id"
    if calendar_col == "":
        calendar_col = None

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = infer_train_file(data_dir, args.train_file)
    print(f"[info] train file: {train_path}")

    df = load_table(train_path)
    # date順に並べておく（CVの時系列整合のため）
    if "date_id" in df.columns:
        df = df.sort_values("date_id").reset_index(drop=True)
    # 余計なラグ生成は行わない（予測で未使用のためI/OとRAMを節約）
    generated_lagged: list[str] = []
    # testヘッダ読み込み（列交差用）
    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] test file (for column alignment): {test_path}")
    df_test_head = load_table(test_path)

    if args.target_col not in df.columns:
        raise KeyError(f"target column '{args.target_col}' not found in {list(df.columns)[:20]}...")
    if calendar_col and calendar_col not in df.columns:
        raise KeyError(f"calendar column '{calendar_col}' not found in train data")
    if calendar_col and calendar_col not in df_test_head.columns:
        raise KeyError(f"calendar column '{calendar_col}' not found in test data")

    # separate y
    y = df[args.target_col].astype(float)
    # 基本除外列（漏洩/非説明）
    base_drop = {
        args.target_col,
        args.id_col,
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
        "is_scored",
    }
    if calendar_col:
        base_drop.discard(calendar_col)
    # 学習側で一旦除外
    X_all = df.drop(columns=[c for c in base_drop if c in df.columns])
    # test 側の列と交差（共通列のみ使用）
    test_cols = set(df_test_head.columns.tolist())
    # 学習時に生成した lagged_* は推論で再現しないため、一旦除外
    use_cols = [c for c in X_all.columns if c in test_cols and c not in base_drop and not c.startswith("lagged_")]
    if calendar_col and calendar_col in df.columns and calendar_col in test_cols and calendar_col not in use_cols:
        use_cols.append(calendar_col)
    if not use_cols:
        raise RuntimeError("No common feature columns between train and test after drops.")
    X = X_all[use_cols]

    # detect categorical vs numeric
    # dtype 判定は環境差に強い指定へ
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if calendar_col and calendar_col in num_cols:
        num_cols.remove(calendar_col)
    if calendar_col and calendar_col in cat_cols:
        cat_cols.remove(calendar_col)

    m_cols = [c for c in num_cols if isinstance(c, str) and c.startswith("M")]
    mask_cols: list[str] = []
    if args.m_policy == "mask_plus_mean":
        mask_cols = [f"{col}_missing_flag" for col in m_cols]
    num_cols_with_masks = num_cols + [c for c in mask_cols if c not in num_cols]

    # simple preprocessing: one-hot encode categoricals, passthrough numerics
    # Handle OneHotEncoder compatibility for scikit-learn <1.2 and >=1.2
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
        columns=m_cols,
        policy=args.m_policy,
        rolling_window=args.m_rolling_window,
        ema_alpha=args.m_ema_alpha,
        calendar_column=calendar_col,
        policy_params=policy_params,
    )
    if not HAS_LGBM or LGBMRegressor is None:  # type: ignore[truthy-function]
        raise RuntimeError("LightGBM is required but not installed. Please install 'lightgbm'.")
    # LightGBM モデル（パラメタは軽量なデフォルト）
    model = LGBMRegressor(  # type: ignore[operator]
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.bagging_fraction,
        colsample_bytree=args.feature_fraction,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        bagging_freq=args.bagging_freq,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("m_imputer", m_imputer), ("pre", pre), ("model", model)])

    # 進捗表示用のコールバック準備
    n_rounds = int(model.get_params().get("n_estimators", 100))
    log_period = max(1, n_rounds // 10)
    callbacks = []
    if HAS_LGBM and lgb is not None:
        try:
            # 学習ログ（評価値）を一定間隔で表示
            callbacks.append(lgb.log_evaluation(period=log_period))

            # 進捗（イテレーション/全体%）の簡易表示
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
    # OOFは未予測位置をNaNで保持し、後段の評価・探索でマスク
    oof_pred = np.full(len(X), np.nan, dtype=float)
    fold_logs: list[dict] = []

    # 手動でインデックスを時系列に沿って割り当て
    X_np = X.reset_index(drop=True)
    y_np = y.reset_index(drop=True)

    # optimize_for はループ外で一度だけ固定
    optimize_for = args.optimize_for if args.optimize_for in ("msr", "msr_down", "vmsr") else "msr"

    # 学習: 各foldで fit -> val 予測 -> RMSE と MSR を評価
    fold_id = 0
    val_sizes: list[int] = []
    for train_idx, val_idx in tscv.split(X_np):
        fold_id += 1
        if args.gap > 0:
            # train末尾からgap分を捨て、val先頭もgap分を捨てる
            if len(train_idx) > args.gap:
                train_idx = train_idx[:-args.gap]
            if len(val_idx) > args.gap:
                val_idx = val_idx[args.gap:]
        # gap適用により空foldになった場合はスキップ
        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"[warn][fold {fold_id}] empty train/val after gap; skip fold.")
            continue
        if args.min_val_size and len(val_idx) < args.min_val_size:
            print(
                f"[warn][fold {fold_id}] validation size {len(val_idx)} < min_val_size={args.min_val_size}; skip fold."
            )
            continue
        # 健全性チェック用にvalサイズを収集
        val_sizes.append(len(val_idx))
        X_tr, X_va = X_np.iloc[train_idx], X_np.iloc[val_idx]
        y_tr, y_va = y_np.iloc[train_idx], y_np.iloc[val_idx]

        # 再インスタンス化（foldごとに独立学習）
        model_f = LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            subsample=args.bagging_fraction,
            colsample_bytree=args.feature_fraction,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            bagging_freq=args.bagging_freq,
            random_state=42,
            n_jobs=-1,
        )
        pipe_f = Pipeline([("m_imputer", clone(m_imputer)), ("pre", clone(pre)), ("model", model_f)])

        fit_kwargs_f: dict = {}
        if callbacks:
            fit_kwargs_f["model__callbacks"] = callbacks

        # 例外安全: 一部環境で callbacks が型不一致になる場合のフォールバック
        try:
            pipe_f.fit(X_tr, y_tr, **fit_kwargs_f)
        except TypeError:
            pipe_f.fit(X_tr, y_tr)
        yhat_va = pipe_f.predict(X_va)
        yhat_va = _to_1d_np(yhat_va)
        oof_pred[val_idx] = yhat_va
        rmse_va = float(np.sqrt(mean_squared_error(y_va, yhat_va)))

        # fold内で post-process グリッド探索（MSR最適化）
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
        # lamはbestを再計算する場合0/lamの扱いに注意（grid側で反映済み）
        # vmsrの際は最良lamを含む行を抽出できるように、再計算時にもlamを入れる
        if optimize_for == "vmsr":
            # grid_resからbest_paramsに一致し、スコア最大の行を拾う
            # 複数一致があっても最大vmsrのもの
            candidates = [r for r in grid_res if r["mult"] == best_params.mult and r["lo"] == best_params.lo and r["hi"] == best_params.hi]
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
        fold_logs.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "rmse_val": rmse_va,
                "best_mult": best_params.mult,
                "best_lo": best_params.lo,
                "best_hi": best_params.hi,
                "best_lam": lam_best,
                **{f"best_{k}": v for k, v in best_metrics.items()},
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

    # foldのvalサイズ集計を出力（gap設定が妥当かの健全性チェック）
    if val_sizes:
        vs = np.array(val_sizes)
        vmin, vmed, vmax = int(vs.min()), float(np.median(vs)), int(vs.max())
        print(f"[cv] val_size stats: min={vmin} median={vmed:.1f} max={vmax} n_folds={len(val_sizes)} gap={args.gap}")
        if vmed > 0.0:
            small_folds = [int(v) for v in vs if v < 0.5 * vmed]
            if small_folds:
                print(f"[warn] validation size below 50% of median detected: {small_folds}")

    # OOF全体のRMSE（予測が存在する位置のみ）
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

    # グリッド探索: 全foldのOOFでまとめて post-process 最適化（optional）
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

    # 全データで最終学習（推論用モデル）
    # 最終学習も callbacks が使えない環境に備えフォールバック
    try:
        pipe.fit(X, y, **({"model__eval_metric": "rmse"} if callbacks else {}))
    except TypeError:
        pipe.fit(X, y)

    m_column_count = len(m_cols)
    m_post_impute_nan_ratio = 0.0
    imputer_warning_messages: List[str] = []
    m_imputer_warning_count = 0
    try:
        if m_column_count:
            imputer_fitted = pipe.named_steps.get("m_imputer")
            if imputer_fitted is not None:
                transformed_train = cast(pd.DataFrame, imputer_fitted.transform(X))
                m_post_impute_nan_ratio = float(
                    np.isnan(transformed_train[m_cols].to_numpy(dtype=float, copy=False)).mean()
                )
                state_dict = getattr(imputer_fitted, "_state_", {})
                if isinstance(state_dict, dict):
                    warnings_list = state_dict.get("warnings", [])
                    if isinstance(warnings_list, list):
                        imputer_warning_messages = list(dict.fromkeys(str(msg) for msg in warnings_list if isinstance(msg, str)))
                        m_imputer_warning_count = len(imputer_warning_messages)
    except Exception:
        m_post_impute_nan_ratio = float("nan")
        imputer_warning_messages = []
        m_imputer_warning_count = 0

    # save model and meta（MSR設定も保存）
    model_path = out_dir / "model_pre_m.pkl"
    if not args.no_artifacts:
        joblib.dump(pipe, model_path)

    # ライブラリのバージョン（再現性向上）
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

    meta = {
        "train_path": str(train_path),
        "target_col": args.target_col,
        "id_col": args.id_col,
        "model_type": type(model).__name__,
        "numeric_cols": num_cols_with_masks,
        "categorical_cols": cat_cols,
        "oof_rmse": float(rmse_oof),
        "feature_count": len(num_cols_with_masks) + len(cat_cols),
        "feature_columns": use_cols,
        "test_path_for_alignment": str(test_path),
        "generated_lagged_features": generated_lagged,
        "cv": fold_logs_serializable,
        "oof_best_params": {"mult": best_params_global.mult, "lo": best_params_global.lo, "hi": best_params_global.hi, "lam": lam_best_global},
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
        "random_state": 42,
        "postprocess_defaults": {"clip_min": 0.0, "clip_max": 2.0, "use_post_process": True},
        "sklearn_version": sk_ver,
        "lightgbm_version": lgb_ver,
        "m_policy": args.m_policy,
        "m_rolling_window": int(args.m_rolling_window),
        "m_ema_alpha": float(args.m_ema_alpha),
        "m_policy_params": policy_params,
        "m_calendar_col": calendar_col,
        "m_mask_cols": mask_cols,
        "m_column_count": m_column_count,
        "m_post_impute_nan_ratio": float(m_post_impute_nan_ratio),
        "m_imputer_warning_count": int(m_imputer_warning_count),
        "m_imputer_warnings": imputer_warning_messages,
    }
    if not args.no_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ok] saved: {model_path}, model_meta.json")

    # 参考: OOFグリッド結果とfold最良の分布をCSVで残す
    try:
        if not args.no_artifacts:
            if grid_all:
                pd.DataFrame(grid_all).to_csv(out_dir / "oof_grid_results.csv", index=False)
            pd.DataFrame(fold_logs).to_csv(out_dir / "cv_fold_logs.csv", index=False)
            print("[ok] saved: oof_grid_results.csv, cv_fold_logs.csv")
    except Exception:
        pass

    if args.metrics_path:
        metrics_out = {
            "m_policy": args.m_policy,
            "m_rolling_window": int(args.m_rolling_window),
            "m_ema_alpha": float(args.m_ema_alpha),
            "m_policy_params": policy_params,
            "m_calendar_col": calendar_col,
            "m_mask_cols": mask_cols,
            "n_splits": int(args.n_splits),
            "gap": int(args.gap),
            "min_val_size": int(args.min_val_size),
            "optimize_for": args.optimize_for,
            "oof_rmse": float(rmse_oof),
            "coverage": float(coverage),
            "oof_metrics": oof_best_metrics_serializable,
            "fold_metrics": fold_logs_serializable,
            "pp_aggregate": args.pp_aggregate,
            "m_column_count": m_column_count,
            "m_post_impute_nan_ratio": float(m_post_impute_nan_ratio),
            "m_imputer_warning_count": int(m_imputer_warning_count),
            "m_imputer_warnings": imputer_warning_messages,
        }
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"[ok] saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
