#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最小学習スクリプト（ベースライン）:
- data/raw/{train.parquet|train.csv} を自動検出（--train-file で明示可）
- 目的変数(--target-col)、ID列(--id-col) を除き、説明変数を構成
- LightGBM のみを使用（Ridge等のフォールバックは削除）
- 学習時に --test-file を指定し、train/test の共通列のみを使用してカラム整合性を担保
- 学習済みモデルとメタ情報を artifacts/simple_baseline/ に保存
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from lightgbm import LGBMRegressor  # type: ignore
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None  # type: ignore
    lgb = None  # type: ignore
    HAS_LGBM = False

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


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
    ap.add_argument("--out-dir", type=str, default="artifacts/simple_baseline")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = infer_train_file(data_dir, args.train_file)
    print(f"[info] train file: {train_path}")

    df = load_table(train_path)
    # date順に並べて安全にラグ特徴を生成（リーク防止: shift(1)）
    if "date_id" in df.columns:
        df = df.sort_values("date_id").reset_index(drop=True)
        lag_sources = [
            "forward_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
        ]
        generated_lagged: list[str] = []
        for c in lag_sources:
            if c in df.columns:
                lag_col = f"lagged_{c}"
                df[lag_col] = df[c].shift(1)
                generated_lagged.append(lag_col)
    else:
        generated_lagged = []
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
    use_cols = [c for c in X_all.columns if c in test_cols and c not in base_drop]
    if not use_cols:
        raise RuntimeError("No common feature columns between train and test after drops.")
    X = X_all[use_cols]

    # detect categorical vs numeric
    num_cols = X.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # simple preprocessing: one-hot encode categoricals, passthrough numerics
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if not HAS_LGBM or LGBMRegressor is None:  # type: ignore[truthy-function]
        raise RuntimeError("LightGBM is required but not installed. Please install 'lightgbm'.")
    # LightGBM モデル（パラメタは軽量なデフォルト）
    model = LGBMRegressor(  # type: ignore[operator]
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
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

    print(f"[info] fit model: {type(model).__name__} | num={len(num_cols)} cat={len(cat_cols)} | features={len(use_cols)}")
    # LightGBM の eval_set/metric と callbacks を Pipeline 経由で渡す
    fit_kwargs: dict = {}
    if callbacks:
        fit_kwargs["model__callbacks"] = callbacks
        fit_kwargs["model__eval_set"] = [(X, y)]  # 学習データでのRMSE推移を出す
        fit_kwargs["model__eval_metric"] = "rmse"
    pipe.fit(X, y, **fit_kwargs)

    # quick in-sample RMSE for sanity
    pred = pipe.predict(X)
    # 互換性のため squared 引数は使わず RMSE を算出
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    print(f"[metric] in-sample RMSE: {rmse:.6f}")

    # 学習後に段階的なRMSEをログ（大枠のエポック単位）
    try:
        pre_fitted = pipe.named_steps["pre"]
        model_fitted: LGBMRegressor = pipe.named_steps["model"]  # type: ignore
        Xt = pre_fitted.transform(X)
        step = max(1, n_rounds // 10)
        for i in range(step, n_rounds + 1, step):
            yhat_i = model_fitted.predict(Xt, num_iteration=i)
            rmse_i = float(np.sqrt(mean_squared_error(y, yhat_i)))
            print(f"[metric][train] iter={i:4d}/{n_rounds} rmse={rmse_i:.6f}")
        if (n_rounds % step) != 0 and (n_rounds // step) * step != n_rounds:
            yhat_last = model_fitted.predict(Xt, num_iteration=n_rounds)
            rmse_last = float(np.sqrt(mean_squared_error(y, yhat_last)))
            print(f"[metric][train] iter={n_rounds:4d}/{n_rounds} rmse={rmse_last:.6f}")
    except Exception:
        # 任意ログなので失敗しても継続
        pass

    # save model and meta
    model_path = out_dir / "model_simple.pkl"
    joblib.dump(pipe, model_path)

    meta = {
        "train_path": str(train_path),
        "target_col": args.target_col,
        "id_col": args.id_col,
        "model_type": type(model).__name__,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "in_sample_rmse": float(rmse),
        "feature_count": len(num_cols) + len(cat_cols),
        "feature_columns": use_cols,
        "test_path_for_alignment": str(test_path),
        "generated_lagged_features": generated_lagged,
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ok] saved: {model_path}, model_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
