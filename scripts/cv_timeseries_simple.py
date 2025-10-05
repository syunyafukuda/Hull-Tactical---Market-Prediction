#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
簡易 時系列CV スクリプト（汎用モジュール）:
- 本スクリプトは提出ライン（simple_baseline 等）に依存しない汎用の評価コンポーネントです。
- date_id に基づく時間順分割でリークを防ぎつつ RMSE を評価します。
- 学習スクリプトに準拠した前処理（元の forward/risk_free/excess は除外、lagged_* は shift(1) で利用）。
- 評価結果を artifacts/cv_simple.json に保存します。
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None  # type: ignore
    HAS_LGBM = False

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def make_folds_by_date(unique_dates: np.ndarray, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_date_ids, val_date_ids) pairs in chronological order."""
    unique_sorted = np.sort(unique_dates)
    # 均等分割に近い形でスライス
    fold_sizes = np.full(n_folds, len(unique_sorted) // n_folds, dtype=int)
    fold_sizes[: len(unique_sorted) % n_folds] += 1
    splits = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        splits.append(unique_sorted[start:end])
        start = end

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_folds):
        # 逐次的に拡張学習 + 次ブロックを検証に使う（walk-forward風）
        train_dates = np.concatenate(splits[:i])
        val_dates = splits[i]
        folds.append((train_dates, val_dates))
    return folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/raw")
    ap.add_argument("--train-file", type=str, default="train.csv")
    ap.add_argument("--target-col", type=str, default="market_forward_excess_returns")
    ap.add_argument("--id-col", type=str, default="date_id")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", type=str, default="artifacts/cv_simple.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_path = Path(args.train_file)
    if not train_path.is_absolute():
        train_path = data_dir / train_path

    df = load_table(train_path)
    if args.id_col not in df.columns:
        raise KeyError(f"id_col '{args.id_col}' not found in train")
    if args.target_col not in df.columns:
        raise KeyError(f"target_col '{args.target_col}' not found in train")

    # 時系列順に並べ、学習時のラグを生成
    df = df.sort_values(args.id_col).reset_index(drop=True)
    for c in ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]:
        if c in df.columns:
            df[f"lagged_{c}"] = df[c].shift(1)

    # 特徴量抽出（元列は除外、lagged_* は活かす）
    base_drop = {
        args.target_col,
        args.id_col,
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
        "is_scored",
    }
    y_all = df[args.target_col].astype(float)
    X_all = df.drop(columns=[c for c in base_drop if c in df.columns])

    # カテゴリ/数値に分割
    num_cols = X_all.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
    cat_cols = [c for c in X_all.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if not HAS_LGBM or LGBMRegressor is None:
        raise RuntimeError("LightGBM is required but not installed. Please install 'lightgbm'.")
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    # フォールド作成
    unique_dates = df[args.id_col].dropna().astype(int).unique()
    folds = make_folds_by_date(unique_dates, args.n_folds)

    results = []
    for i, (train_dates, val_dates) in enumerate(folds, start=1):
        tr_idx = df[args.id_col].isin(train_dates).values
        va_idx = df[args.id_col].isin(val_dates).values

        X_tr, y_tr = X_all.loc[tr_idx], y_all.loc[tr_idx]
        X_va, y_va = X_all.loc[va_idx], y_all.loc[va_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        rmse = float(mean_squared_error(y_va, y_pred, squared=False))
        results.append({"fold": i, "rmse": rmse, "train_days": int(len(train_dates)), "val_days": int(len(val_dates))})
        print(f"[fold {i}] rmse={rmse:.6f} | train_days={len(train_dates)} val_days={len(val_dates)}")

    summary = {
        "n_folds": args.n_folds,
        "folds": results,
        "rmse_mean": float(np.mean([r["rmse"] for r in results])) if results else None,
        "rmse_std": float(np.std([r["rmse"] for r in results])) if results else None,
        "target_col": args.target_col,
        "id_col": args.id_col,
        "feature_count": int(X_all.shape[1]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] saved CV summary -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
