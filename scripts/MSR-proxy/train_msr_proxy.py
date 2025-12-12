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
import json
import sys
from pathlib import Path

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

# ensure project root on sys.path for `scripts.utils_msr` import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
    ap.add_argument("--out-dir", type=str, default="artifacts/MSR-proxy")
    # 時系列CV
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=0, help="optional gap between train and val for leakage safety")
    # post-process grid
    ap.add_argument("--mult-grid", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--lo-grid", type=float, nargs="*", default=[0.8, 0.9, 1.0])
    ap.add_argument("--hi-grid", type=float, nargs="*", default=[1.0, 1.1, 1.2])
    ap.add_argument("--optimize-for", type=str, default="msr", choices=["msr", "msr_down", "vmsr"]) 
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--lam-grid", type=float, nargs="*", default=[0.0, 0.25, 0.5], help="penalty lambda grid for vMSR")
    # LGBM params (軽いチューニング枠)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--min-data-in-leaf", type=int, default=20)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-freq", type=int, default=1)
    args = ap.parse_args()

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
    # 学習側で一旦除外
    X_all = df.drop(columns=[c for c in base_drop if c in df.columns])
    # test 側の列と交差（共通列のみ使用）
    test_cols = set(df_test_head.columns.tolist())
    # 学習時に生成した lagged_* は推論で再現しないため、一旦除外
    use_cols = [c for c in X_all.columns if c in test_cols and c not in base_drop and not c.startswith("lagged_")]
    if not use_cols:
        raise RuntimeError("No common feature columns between train and test after drops.")
    X = X_all[use_cols]

    # detect categorical vs numeric
    # dtype 判定は環境差に強い指定へ
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # simple preprocessing: one-hot encode categoricals, passthrough numerics
    # Handle OneHotEncoder compatibility for scikit-learn <1.2 and >=1.2
    try:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", cat_encoder, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
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
    pipe = Pipeline([("pre", pre), ("model", model)])

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
        pipe_f = Pipeline([("pre", clone(pre)), ("model", model_f)])

        fit_kwargs_f: dict = {}
        if callbacks:
            fit_kwargs_f["model__callbacks"] = callbacks
            fit_kwargs_f["model__eval_set"] = [(X_va, y_va)]
            fit_kwargs_f["model__eval_metric"] = "rmse"

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
            y_true=y_va.values,
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

        best_metrics = evaluate_msr_proxy(yhat_va, y_va.values, best_params, eps=args.eps, lam=lam_best)
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

    # OOF全体のRMSE（予測が存在する位置のみ）
    mask = np.isfinite(oof_pred)
    if not np.any(mask):
        print("[warn] no OOF predictions available; check CV settings.")
        rmse_oof = float("nan")
    else:
        rmse_oof = float(np.sqrt(mean_squared_error(y_np.values[mask], oof_pred[mask])))
    coverage = float(np.mean(mask)) if len(mask) else 0.0
    print(f"[metric][oof] rmse={rmse_oof:.6f} coverage={coverage:.3f}")

    # グリッド探索: 全foldのOOFでまとめて post-process 最適化（optional）
    if not np.any(mask):
        print("[warn] OOF empty after gap; skip OOF optimization.")
        best_params_global = PostProcessParams()
        grid_all = []
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
        lam_best_global = 0.0
    else:
        best_params_global, grid_all = grid_search_msr(
            y_pred=oof_pred[mask],
            y_true=y_np.values[mask],
            mult_grid=args.mult_grid,
            lo_grid=args.lo_grid,
            hi_grid=args.hi_grid,
            eps=args.eps,
            optimize_for=optimize_for,
            lam_grid=args.lam_grid if optimize_for == "vmsr" else [0.0],
        )
        if optimize_for == "vmsr":
            candidates = [r for r in grid_all if r["mult"] == best_params_global.mult and r["lo"] == best_params_global.lo and r["hi"] == best_params_global.hi]
            if candidates:
                best_row = max(candidates, key=lambda r: r.get("vmsr", -1e18))
                lam_best_global = float(best_row.get("vmsr_lam", 0.0))
            else:
                lam_best_global = float(args.lam_grid[0]) if args.lam_grid else 0.0
        else:
            lam_best_global = 0.0
        best_global_metrics = evaluate_msr_proxy(oof_pred[mask], y_np.values[mask], best_params_global, eps=args.eps, lam=lam_best_global)
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

    # save model and meta（MSR設定も保存）
    model_path = out_dir / "model_msr_proxy.pkl"
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

    meta = {
        "train_path": str(train_path),
        "target_col": args.target_col,
        "id_col": args.id_col,
        "model_type": type(model).__name__,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "oof_rmse": float(rmse_oof),
        "feature_count": len(num_cols) + len(cat_cols),
        "feature_columns": use_cols,
        "test_path_for_alignment": str(test_path),
        "generated_lagged_features": generated_lagged,
        "cv": fold_logs,
        "oof_best_params": {"mult": best_params_global.mult, "lo": best_params_global.lo, "hi": best_params_global.hi, "lam": lam_best_global},
        "oof_best_metrics": best_global_metrics,
        "optimize_for": args.optimize_for,
        "mult_grid": list(map(float, args.mult_grid)),
        "lo_grid": list(map(float, args.lo_grid)),
        "hi_grid": list(map(float, args.hi_grid)),
        "eps": float(args.eps),
        "n_splits": int(args.n_splits),
        "gap": int(args.gap),
        "random_state": 42,
        "postprocess_defaults": {"clip_min": 0.0, "clip_max": 2.0, "use_post_process": True},
        "sklearn_version": sk_ver,
        "lightgbm_version": lgb_ver,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] saved: {model_path}, model_meta.json")

    # 参考: OOFグリッド結果とfold最良の分布をCSVで残す
    try:
        pd.DataFrame(grid_all).to_csv(out_dir / "oof_grid_results.csv", index=False)
        pd.DataFrame(fold_logs).to_csv(out_dir / "cv_fold_logs.csv", index=False)
        print("[ok] saved: oof_grid_results.csv, cv_fold_logs.csv")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
