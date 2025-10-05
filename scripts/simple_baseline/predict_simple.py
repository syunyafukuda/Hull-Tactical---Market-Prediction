#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最小推論・提出スクリプト（ベースライン）:
- data/raw/{test.parquet|test.csv} を自動検出（--test-file で明示可）
- artifacts/simple_baseline/model_simple.pkl と model_meta.json をロード
- 学習時に決めた feature_columns に合わせてカラム整形し予測
- 提出ファイルを {submission.parquet|submission.csv} で書き出し
- 列名やファイル名は引数で変更可（競技仕様に合わせて調整）
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


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
        # heuristic: prefer names containing 'test'
        found_sorted = sorted(found, key=lambda p: ("test" not in p.stem.lower(), p.name))
        if found_sorted:
            return found_sorted[0]
    raise FileNotFoundError(f"No test file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/raw")
    ap.add_argument("--test-file", type=str, default=None)
    ap.add_argument("--artifacts-dir", type=str, default="artifacts/simple_baseline")
    ap.add_argument("--id-col", type=str, default="date_id")
    ap.add_argument("--pred-col", type=str, default="prediction")
    ap.add_argument("--out-parquet", type=str, default="artifacts/simple_baseline/submission.parquet")
    ap.add_argument("--out-csv", type=str, default="artifacts/simple_baseline/submission.csv", help="also write csv (default: artifacts/simple_baseline/submission.csv)")
    ap.add_argument("--no-csv", action="store_true", help="do not write CSV alongside parquet")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifacts_dir)
    model_path = art_dir / "model_simple.pkl"
    meta_path = art_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}. Run train_simple.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] test file: {test_path}")
    df_test = load_table(test_path)

    if args.id_col not in df_test.columns:
        # allow fallback to meta-defined id_col (e.g., date_id)
        id_col = meta.get("id_col", args.id_col)
        if id_col in df_test.columns:
            args.id_col = id_col
        else:
            raise KeyError(f"id column '{args.id_col}' not in test columns: {list(df_test.columns)[:20]}...")

    pipe = joblib.load(model_path)

    # 学習時に使用した特徴列に合わせて整形
    feat_cols = meta.get("feature_columns")
    num_cols = set(meta.get("numeric_cols", []))
    cat_cols = set(meta.get("categorical_cols", []))
    if not feat_cols:
        raise KeyError("feature_columns not found in meta. Re-train with updated train_simple.py")

    # 非説明列は明示的に落とす
    drop_non_features = {args.id_col, "is_scored"}
    df_feat = df_test.copy()
    df_feat = df_feat.drop(columns=[c for c in drop_non_features if c in df_feat.columns], errors="ignore")

    # 欠けている特徴列はダミー生成（数値はNaN、カテゴリは'"missing"'）
    for c in feat_cols:
        if c not in df_feat.columns:
            if c in num_cols:
                df_feat[c] = np.nan
            elif c in cat_cols:
                df_feat[c] = "missing"
            else:
                # 型不明の場合はNaNで生成（OneHot対象ならunknownとして無視される）
                df_feat[c] = np.nan

    # 学習時の列順に並べる
    X_infer = df_feat[feat_cols]

    print("[info] predicting...")
    yhat = pipe.predict(X_infer)

    sub = pd.DataFrame({args.id_col: df_test[args.id_col].values, args.pred_col: yhat.astype(float)})

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out_parquet, index=False)
    print(f"[ok] wrote {out_parquet} [{len(sub)} rows]")

    # CSVも原則出力（--no-csv で無効化可能）
    if not args.no_csv:
        out_csv = Path(args.out_csv) if args.out_csv else Path("artifacts/simple_baseline/submission.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
