#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSR-proxy 推論・提出スクリプト (E 系特徴量版):
- data/raw/{test.parquet|test.csv} を自動検出（--test-file で明示可）
- artifacts/Preprocessing_E/model_pre_e.pkl と model_meta.json をロード
- 学習時に決めた feature_columns に合わせてカラム整形し予測
- 任意で post-process（mult, lo, hi）を適用し、prediction 列に出力
- 提出ファイルを {submission.parquet|submission.csv} で書き出し
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ensure custom transformers are importable for joblib.load
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from preprocess.E_group.e_group import EGroupImputer  # noqa: F401,E402
from preprocess.M_group.m_group import MGroupImputer  # noqa: F401,E402
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


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/raw")
    ap.add_argument("--test-file", type=str, default=None)
    ap.add_argument("--artifacts-dir", type=str, default="artifacts/Preprocessing_E")
    ap.add_argument("--id-col", type=str, default="date_id")
    ap.add_argument("--pred-col", type=str, default="prediction")
    ap.add_argument("--out-parquet", type=str, default="artifacts/Preprocessing_E/submission.parquet")
    ap.add_argument("--out-csv", type=str, default="artifacts/Preprocessing_E/submission.csv", help="also write csv (default: artifacts/Preprocessing_E/submission.csv)")
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
    model_path = art_dir / "model_pre_e.pkl"
    meta_path = art_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}. Run train_pre_e.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_list_candidates = []
    feature_list_path_raw = meta.get("feature_list_path")
    if feature_list_path_raw:
        candidate = Path(feature_list_path_raw)
        feature_list_candidates.append(candidate)
        if not candidate.is_absolute():
            feature_list_candidates.append((art_dir / candidate.name).resolve())
    feature_list_spec: dict[str, Any] | None = None
    for candidate in feature_list_candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fp:
                feature_list_spec = json.load(fp)
            break
    if feature_list_spec is None:
        feature_list_spec = {
            "pipeline_input_columns": meta.get("pipeline_input_columns", meta.get("feature_columns", [])),
            "model_feature_columns": meta.get("feature_columns", []),
            "calendar_column": meta.get("e_calendar_col"),
            "e_columns": meta.get("e_columns", []),
            "e_generated_columns": meta.get("e_generated_columns", []),
            "numeric_feature_columns": meta.get("numeric_cols", []),
            "numeric_feature_columns_with_masks": meta.get("numeric_cols", []),
            "categorical_feature_columns": meta.get("categorical_cols", []),
        }

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

    meta_id_col = meta.get("id_col")
    if meta_id_col and meta_id_col != args.id_col:
        raise ValueError(f"Inference id_col '{args.id_col}' does not match training meta id_col '{meta_id_col}'.")

    feat_cols = list(feature_list_spec.get("model_feature_columns", []))
    if not feat_cols:
        raise KeyError("model_feature_columns not found in feature specification. Re-train artifacts to regenerate metadata.")
    pipeline_input_columns = list(feature_list_spec.get("pipeline_input_columns", feat_cols))
    calendar_col = feature_list_spec.get("calendar_column") or meta.get("e_calendar_col")
    numeric_cols = list(feature_list_spec.get("numeric_feature_columns_with_masks", meta.get("numeric_cols", [])))
    categorical_cols = list(feature_list_spec.get("categorical_feature_columns", meta.get("categorical_cols", [])))
    e_columns_meta = list(feature_list_spec.get("e_columns", []))
    if not e_columns_meta:
        e_columns_meta = [c for c in feat_cols if isinstance(c, str) and c.startswith("E")]
    if not pipeline_input_columns:
        raise ValueError("pipeline_input_columns resolved to empty list. Re-train artifacts to regenerate metadata.")

    # Ensure calendar column is present when required by policy
    if calendar_col and calendar_col not in df_test.columns:
        raise KeyError(
            f"calendar column '{calendar_col}' required by training policy is missing in inference data."
        )

    pipe = joblib.load(model_path)
    if not hasattr(pipe, "named_steps"):
        raise TypeError("Loaded artifact is not a sklearn Pipeline with named_steps.")
    m_imputer_loaded = pipe.named_steps.get("m_imputer")
    if m_imputer_loaded is None:
        raise KeyError("m_imputer step missing in loaded pipeline. Re-train artifacts.")
    trained_m_policy = getattr(m_imputer_loaded, "policy_requested", None)
    meta_m_policy = meta.get("m_policy")
    if meta_m_policy and trained_m_policy and trained_m_policy != meta_m_policy:
        raise ValueError(
            f"Loaded pipeline M policy '{trained_m_policy}' does not match meta recorded policy '{meta_m_policy}'."
        )
    trained_m_columns = list(getattr(m_imputer_loaded, "columns_", []) or [])
    m_columns_meta = list(feature_list_spec.get("m_columns", meta.get("m_columns", [])))
    if m_columns_meta:
        if trained_m_columns and set(trained_m_columns) != set(m_columns_meta):
            raise ValueError(
                "Mismatch between meta-recorded M columns and the fitted M imputer columns."
            )
    elif trained_m_columns:
        raise ValueError("Model exposes M columns but metadata is missing m_columns definition.")

    imputer_loaded = pipe.named_steps.get("e_imputer")
    if imputer_loaded is None:
        raise KeyError("e_imputer step missing in loaded pipeline. Re-train artifacts.")
    trained_policy = getattr(imputer_loaded, "policy_requested", None)
    if trained_policy and meta.get("e_policy") and trained_policy != meta.get("e_policy"):
        raise ValueError(
            f"Loaded pipeline policy '{trained_policy}' does not match meta recorded policy '{meta.get('e_policy')}'."
        )
    trained_columns = list(getattr(imputer_loaded, "columns_", []) or [])
    if e_columns_meta:
        if trained_columns and set(trained_columns) != set(e_columns_meta):
            raise ValueError(
                "Mismatch between meta-recorded E columns and the fitted imputer columns."
            )
    elif trained_columns:
        raise ValueError("Model exposes E columns but metadata is missing e_columns definition.")

    drop_non_features = {"is_scored"}
    if calendar_col:
        drop_non_features.discard(calendar_col)
    df_feat = df_test.drop(columns=[c for c in drop_non_features if c in df_test.columns], errors="ignore").copy()
    required_columns = list(dict.fromkeys(pipeline_input_columns))
    for col in required_columns:
        if col in df_feat.columns:
            continue
        if calendar_col and col == calendar_col:
            raise KeyError(f"Required calendar column '{calendar_col}' missing from inference data.")
        if col in numeric_cols or col in e_columns_meta:
            df_feat[col] = np.nan
        elif col in categorical_cols:
            df_feat[col] = "missing"
        else:
            df_feat[col] = np.nan

    unexpected_cols = set(df_feat.columns) - set(required_columns) - {args.id_col}
    if unexpected_cols:
        df_feat = df_feat.drop(columns=sorted(unexpected_cols))

    if e_columns_meta:
        df_feat.loc[:, e_columns_meta] = df_feat.loc[:, e_columns_meta].apply(pd.to_numeric, errors="coerce")
    if m_columns_meta:
        df_feat.loc[:, m_columns_meta] = df_feat.loc[:, m_columns_meta].apply(pd.to_numeric, errors="coerce")

    # 統一された列順を構築
    df_model_input = df_feat.loc[:, required_columns].copy()
    orig_idx = np.arange(len(df_model_input))
    sort_indexer = df_model_input[args.id_col].to_numpy().argsort(kind="mergesort")
    df_sorted = df_model_input.iloc[sort_indexer].reset_index(drop=True)
    X_infer = df_sorted[pipeline_input_columns]

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
    out_df = pd.DataFrame({
        "_idx": orig_idx[sort_indexer],
        args.id_col: df_sorted[args.id_col].values,
        args.pred_col: pred_out,
    })
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

    out_csv: Path | None = None
    if not args.no_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv}")

    checksum_map: dict[str, dict[str, str]] = {
        "parquet": {"path": str(out_parquet), "sha256": _sha256_file(out_parquet)},
    }
    if out_csv is not None:
        checksum_map["csv"] = {"path": str(out_csv), "sha256": _sha256_file(out_csv)}

    try:
        git_commit = (
            subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
            .stdout.strip()
        )
    except Exception:
        git_commit = None

    policy_version = {
        "e": {
            "name": meta.get("e_policy"),
            "params": meta.get("e_policy_params"),
            "calendar_col": meta.get("e_calendar_col"),
            "random_seed": meta.get("random_seed"),
        },
        "m": {
            "name": meta.get("m_policy"),
            "params": meta.get("m_policy_params"),
            "calendar_col": meta.get("m_calendar_col"),
            "random_seed": meta.get("random_seed"),
        },
    }

    pred_meta = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "model_path": str(model_path),
        "model_meta_path": str(meta_path),
        "model_version": meta.get("model_type"),
        "policy_version": policy_version,
        "feature_list_path": feature_list_path_raw,
        "pipeline_input_columns": pipeline_input_columns,
        "model_feature_columns": feat_cols,
        "m_columns": m_columns_meta,
        "m_generated_columns": feature_list_spec.get("m_generated_columns", meta.get("m_generated_columns", [])),
        "e_columns": e_columns_meta,
        "e_generated_columns": feature_list_spec.get("e_generated_columns", []),
        "e_all_nan_columns": meta.get("e_all_nan_columns", []),
        "e_all_nan_strategy": meta.get("e_all_nan_strategy"),
        "e_all_nan_fill_value": meta.get("e_all_nan_fill_value"),
        "inference_id_col": args.id_col,
        "prediction_files": checksum_map,
        "git_commit": git_commit,
    }
    pred_meta_path = art_dir / "pred_meta.json"
    pred_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_meta_path, "w", encoding="utf-8") as f:
        json.dump(pred_meta, f, indent=2)
    print(f"[ok] wrote {pred_meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
