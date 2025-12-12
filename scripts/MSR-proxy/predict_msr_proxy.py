#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSR-proxy 推論・提出スクリプト:
- data/raw/{test.parquet|test.csv} を自動検出（--test-file で明示可）
- artifacts/MSR-proxy/model_msr_proxy.pkl と model_meta.json をロード
- 学習時に決めた feature_columns に合わせてカラム整形し予測
- 任意で post-process（mult, lo, hi）を適用し、prediction 列に出力
- 提出ファイルを {submission.parquet|submission.csv} で書き出し
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ensure project root on sys.path for `scripts.utils_msr`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.utils_msr import PostProcessParams, to_signal  # noqa: E402


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
    ap.add_argument("--artifacts-dir", type=str, default="artifacts/MSR-proxy")
    ap.add_argument("--id-col", type=str, default="date_id")
    ap.add_argument("--pred-col", type=str, default="prediction")
    ap.add_argument("--out-parquet", type=str, default="artifacts/MSR-proxy/submission.parquet")
    ap.add_argument("--out-csv", type=str, default="artifacts/MSR-proxy/submission.csv", help="also write csv (default: artifacts/MSR-proxy/submission.csv)")
    ap.add_argument("--no-csv", action="store_true", help="do not write CSV alongside parquet")
    # post-process override（未指定なら meta の best を使う）
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:
        bool_action = None  # fallback
    if bool_action:
        ap.add_argument("--use-post-process", action=bool_action, default=None, help="apply post-process to map raw pred -> signal (default: meta.postprocess_defaults or True)")
        ap.add_argument("--force-clip", action=bool_action, default=None, help="clip final output to [clip-min, clip-max] (default: meta.postprocess_defaults or True)")
    else:
        # 古環境向け: --use-post-process / --no-use-post-process の相互排他グループ
        grp = ap.add_mutually_exclusive_group()
        grp.add_argument("--use-post-process", dest="use_post_process", action="store_true", help="apply post-process")
        grp.add_argument("--no-use-post-process", dest="use_post_process", action="store_false", help="do not apply post-process")
        ap.set_defaults(use_post_process=None)

        # 古環境向け: --force-clip / --no-force-clip の相互排他グループ
        grp2 = ap.add_mutually_exclusive_group()
        grp2.add_argument("--force-clip", dest="force_clip", action="store_true", help="clip output to bounds")
        grp2.add_argument("--no-force-clip", dest="force_clip", action="store_false", help="do not clip output")
        ap.set_defaults(force_clip=None)
    ap.add_argument("--clip-min", type=float, default=None, help="lower clip bound for final output (default: meta.postprocess_defaults or 0.0)")
    ap.add_argument("--clip-max", type=float, default=None, help="upper clip bound for final output (default: meta.postprocess_defaults or 2.0)")
    ap.add_argument("--mult", type=float, default=None, help="override mult")
    ap.add_argument("--lo", type=float, default=None, help="override lo")
    ap.add_argument("--hi", type=float, default=None, help="override hi")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifacts_dir)
    model_path = art_dir / "model_msr_proxy.pkl"
    meta_path = art_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}. Run train_msr_proxy.py first.")
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
        raise KeyError("feature_columns not found in meta. Re-train with updated train_msr_proxy.py")

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
    # id順に並べて予測（戻すために元インデックスも保持）
    df_feat_sorted = df_feat.join(df_test[[args.id_col]]).sort_values(args.id_col).reset_index()
    X_infer = df_feat_sorted[feat_cols]

    print("[info] predicting...")
    yhat = pipe.predict(X_infer)

    # optional: map to signal via post-process
    pp_defaults = meta.get("postprocess_defaults", {"clip_min": 0.0, "clip_max": 2.0, "use_post_process": True})
    use_pp = (args.use_post_process if args.use_post_process is not None else pp_defaults.get("use_post_process", True))
    force_clip = (args.force_clip if args.force_clip is not None else True)
    clip_min = (args.clip_min if args.clip_min is not None else pp_defaults.get("clip_min", 0.0))
    clip_max = (args.clip_max if args.clip_max is not None else pp_defaults.get("clip_max", 2.0))

    if use_pp:
        best = meta.get("oof_best_params", {})
        params = PostProcessParams(
            mult=float(args.mult if args.mult is not None else best.get("mult", 1.0)),
            lo=float(args.lo if args.lo is not None else best.get("lo", 0.0)),
            hi=float(args.hi if args.hi is not None else best.get("hi", 2.0)),
        )
        signal = to_signal(yhat, params)
        # 出力は規約に合わせてクリップ（既定ON）
        if force_clip:
            lo_c = float(clip_min)
            hi_c = float(clip_max)
            if lo_c >= hi_c:
                lo_c, hi_c = 0.0, 2.0
            signal = np.clip(signal, lo_c, hi_c)
        pred_out = signal.astype(float)
        print(f"[info] applied post-process: mult={params.mult} lo={params.lo} hi={params.hi}")
    else:
        pred_out = yhat.astype(float)

    # 元の順序へ戻す
    out_df = pd.DataFrame({"_idx": df_feat_sorted["index"].values, args.id_col: df_feat_sorted[args.id_col].values, args.pred_col: pred_out})
    out_df = out_df.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)
    sub = out_df

    # 予測レンジの検証（force_clip時）
    if force_clip:
        assert sub[args.pred_col].between(clip_min, clip_max).all(), (
            f"Predictions out of range [{clip_min}, {clip_max}] despite force_clip=True"
        )

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out_parquet, index=False)
    print(f"[ok] wrote {out_parquet} [{len(sub)} rows]")

    # CSVも原則出力（--no-csv で無効化可能）
    if not args.no_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
