#!/usr/bin/env python
"""SU8 モデルの推論エントリーポイント。

学習済みの ``artifacts/SU8/inference_bundle.pkl`` を読み込み、指定されたテストデータに
対して推論を実行し、提出形式の ``submission.csv``（および parquet 版）を出力する。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def _ensure_numpy_bitgenerator_aliases() -> None:
    """Register legacy numpy BitGenerator names so joblib pickle loading works across versions."""
    try:
        import numpy as _np
    except Exception:
        return

    generator_cls = getattr(_np.random, "MT19937", None)
    if generator_cls is None:
        return

    alias_keys = (
        "<class 'numpy.random._mt19937.MT19937'>",
        "numpy.random._mt19937.MT19937",
        "numpy.random.bit_generator.MT19937",
        "MT19937",
    )

    registries: list[dict[str, Any]] = []
    try:
        import numpy.random._pickle as _np_random_pickle  # type: ignore
    except Exception:
        _np_random_pickle = None  # type: ignore

    if _np_random_pickle is not None:
        registry = getattr(_np_random_pickle, "BitGenerators", None)
        if isinstance(registry, dict):
            registries.append(registry)

    try:
        import numpy.random.bit_generator as _np_bitgen  # type: ignore
    except Exception:
        _np_bitgen = None  # type: ignore

    if _np_bitgen is not None:
        for attr_name in ("_bit_generators", "BitGenerators"):
            registry = getattr(_np_bitgen, attr_name, None)
            if isinstance(registry, dict):
                registries.append(registry)

    seen: set[int] = set()
    unique_registries: list[dict[str, Any]] = []
    for registry in registries:
        registry_id = id(registry)
        if registry_id not in seen:
            unique_registries.append(registry)
            seen.add(registry_id)

    for registry in unique_registries:
        for key in alias_keys:
            registry.setdefault(key, generator_cls)

    if not unique_registries:
        try:
            import numpy.random._pickle as _np_random_pickle  # type: ignore[attr-defined]
        except Exception:
            return
        registry = getattr(_np_random_pickle, "BitGenerators", None)
        if isinstance(registry, dict):
            for key in alias_keys:
                registry.setdefault(key, generator_cls)


_ensure_numpy_bitgenerator_aliases()

# Import classes needed for unpickling
from preprocess.E_group.e_group import EGroupImputer  # noqa: E402,F401
from preprocess.I_group.i_group import IGroupImputer  # noqa: E402,F401
from preprocess.M_group.m_group import MGroupImputer  # noqa: E402,F401
from preprocess.P_group.p_group import PGroupImputer  # noqa: E402,F401
from preprocess.S_group.s_group import SGroupImputer  # noqa: E402,F401
from src.feature_generation.su5.feature_su5 import (  # noqa: E402,F401
    SU5Config,
    SU5FeatureGenerator,
)
from src.feature_generation.su8.feature_su8 import (  # noqa: E402,F401
    SU8Config,
    SU8FeatureAugmenter,
    SU8FeatureGenerator,
)
from src.feature_generation.su8.train_su8 import (  # noqa: E402,F401
    SU8FullFeatureAugmenter,
)


def infer_test_file(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not candidate.exists():
            raise FileNotFoundError(f"--test-file not found: {candidate}")
        return candidate
    candidates = [
        data_dir / "test.parquet",
        data_dir / "test.csv",
        data_dir / "Test.parquet",
        data_dir / "Test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    for ext in ("parquet", "csv"):
        found = sorted(
            (p for p in data_dir.glob(f"*.{ext}")),
            key=lambda p: ("test" not in p.stem.lower(), p.name),
        )
        if found:
            return found[0]
    raise FileNotFoundError(f"No test file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, max_missing: int | None = None
) -> pd.DataFrame:
    """Ensure test data has the same columns as training data."""
    missing = [col for col in columns if col not in frame.columns]
    if max_missing is not None and len(missing) > max_missing:
        preview = ", ".join(str(col) for col in missing[:5])
        raise RuntimeError(
            f"Detected {len(missing)} missing feature columns > max-missing-columns={max_missing}. preview={preview}"
        )
    for col in missing:
        frame[col] = np.nan
    if missing:
        preview = ", ".join(str(col) for col in missing[:5])
        print(f"[warn] added {len(missing)} placeholder feature columns (preview: {preview})")
    return frame.reindex(columns=list(columns))


def _coerce_numeric_like_columns(frame: pd.DataFrame) -> list[str]:
    converted: list[str] = []
    object_cols = frame.select_dtypes(include="object").columns
    for col in object_cols:
        original = frame[col]
        if not isinstance(original, pd.Series):
            continue
        converted_series = pd.to_numeric(original, errors="coerce")
        if not isinstance(converted_series, pd.Series):
            continue
        if converted_series.notna().any() or original.notna().sum() == 0:
            frame[col] = converted_series
            converted.append(str(col))
    if converted:
        preview = ", ".join(converted[:5])
        print(f"[info] coerced {len(converted)} object columns to numeric (preview: {preview})")
    return converted


def _extract_required_calendar_columns(meta: Mapping[str, Any]) -> set[str]:
    required: set[str] = set()
    imputer_meta = meta.get("imputer_metadata") if isinstance(meta, Mapping) else None
    if isinstance(imputer_meta, Mapping):
        for info in imputer_meta.values():
            if not isinstance(info, Mapping):
                continue
            calendar_value = info.get("calendar_column")
            if isinstance(calendar_value, str) and calendar_value.strip():
                required.add(calendar_value.strip())
    return required


@dataclass(frozen=True)
class PostProcessParams:
    mult: float
    lo: float
    hi: float


DEFAULT_POSTPROCESS_PARAMS = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)


def _coerce_postprocess_params(mapping: Mapping[str, Any] | None) -> PostProcessParams | None:
    if not isinstance(mapping, Mapping):
        return None
    try:
        mult_val = float(mapping["mult"])
    except (KeyError, TypeError, ValueError):
        return None
    lo_val: float | None = None
    hi_val: float | None = None
    for lo_key in ("lo", "clip_min", "clip_lo"):
        value = mapping.get(lo_key)
        if value is not None:
            try:
                lo_val = float(value)
                break
            except (TypeError, ValueError):
                continue
    for hi_key in ("hi", "clip_max", "clip_hi"):
        value = mapping.get(hi_key)
        if value is not None:
            try:
                hi_val = float(value)
                break
            except (TypeError, ValueError):
                continue
    if lo_val is None or hi_val is None or not (lo_val < hi_val):
        return None
    return PostProcessParams(mult=mult_val, lo=lo_val, hi=hi_val)


def _resolve_postprocess_params(meta: Mapping[str, Any]) -> PostProcessParams:
    """Resolve post-process params in a conservative way for SU8.

    SU8 ラインでは保守的なパラメータに固定する。
    具体的には、学習メタデータに保存された `oof_best_params` などは
    参照せず、デフォルト値 (mult=1.0, lo=0.9, hi=1.1) を
    直接返すようにする。
    """
    # 最も保守的な設定に固定
    return PostProcessParams(mult=1.0, lo=0.9, hi=1.1)


def to_signal(pred: np.ndarray, params: PostProcessParams) -> np.ndarray:
    values = np.asarray(pred, dtype=float) * params.mult + 1.0
    return np.clip(values, params.lo, params.hi)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_bundle_compat(meta: Mapping[str, Any]) -> None:
    lib_versions = meta.get("library_versions") if isinstance(meta, Mapping) else None
    if isinstance(lib_versions, Mapping):
        for lib_name, expected_version in lib_versions.items():
            if not isinstance(lib_name, str):
                continue
            try:
                module = importlib.import_module(lib_name)
                actual_version = getattr(module, "__version__", None)
            except Exception:
                actual_version = None
            if expected_version is None or actual_version is None:
                print(
                    f"[warn] unable to verify version for {lib_name}: expected={expected_version} actual={actual_version}"
                )
                continue
            if str(actual_version) != str(expected_version):
                print(
                    f"[warn] library version mismatch for {lib_name}: expected {expected_version}, running {actual_version}"
                )

    for key in ("config_hash", "config_digest"):
        expected_hash = meta.get(key)
        if not isinstance(expected_hash, str):
            continue
        config_path_str = meta.get("config_path")
        if not isinstance(config_path_str, str):
            continue
        config_path = Path(config_path_str)
        if not config_path.exists():
            print(f"[warn] unable to verify config hash; file not found: {config_path}")
            continue
        actual_hash = _hash_file(config_path)
        if actual_hash != expected_hash:
            print(
                f"[warn] config hash mismatch: expected {expected_hash}, current {actual_hash} (path={config_path})"
            )

    preprocess_hash = meta.get("preprocess_config_hash")
    preprocess_path_str = meta.get("preprocess_config_path")
    if isinstance(preprocess_hash, str) and isinstance(preprocess_path_str, str):
        preprocess_path = Path(preprocess_path_str)
        if preprocess_path.exists():
            actual_pp_hash = _hash_file(preprocess_path)
            if actual_pp_hash != preprocess_hash:
                print(
                    f"[warn] preprocess config hash mismatch: expected {preprocess_hash}, current {actual_pp_hash}"
                )
        else:
            print(
                f"[warn] unable to verify preprocess config hash; file not found: {preprocess_path}"
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the SU8 pipeline bundle."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--test-file", type=str, default=None, help="Explicit path to the test file"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/SU8",
        help="Directory storing SU8 artifacts",
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        default=None,
        help="Override path to inference_bundle.pkl",
    )
    parser.add_argument(
        "--meta-path", type=str, default=None, help="Override path to model_meta.json"
    )
    parser.add_argument(
        "--feature-list-path",
        type=str,
        default=None,
        help="Override path to feature_list.json",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="ID column name (defaults to metadata value)",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="prediction",
        help="Output prediction column name",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default="artifacts/SU8/submission.parquet",
        help="Submission parquet output path",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/SU8/submission.csv",
        help="Submission csv output path",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Suppress CSV output and write parquet only",
    )
    parser.add_argument(
        "--max-missing-columns",
        type=int,
        default=20,
        help="Maximum number of placeholder feature columns tolerated before aborting",
    )
    parser.add_argument(
        "--ensure-finite",
        action="store_true",
        help="Raise an error if predictions contain NaN or infinite values instead of auto-replacing",
    )
    parser.add_argument(
        "--replace-nonfinite-with",
        type=float,
        default=0.0,
        help="Replacement value to use for non-finite predictions when --ensure-finite is not set",
    )
    parser.add_argument(
        "--clip-low",
        type=float,
        default=None,
        help="Optional lower bound applied to predictions",
    )
    parser.add_argument(
        "--clip-high",
        type=float,
        default=None,
        help="Optional upper bound applied to predictions",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _predict_with_task(pipe: Any, X: pd.DataFrame, meta: Mapping[str, Any]) -> np.ndarray:
    task_type = str(meta.get("task_type", "regression")).lower()
    if task_type in {"regression", "regressor"}:
        raw_pred = pipe.predict(X)
    elif task_type in {"probability", "proba", "binary_proba"}:
        proba = pipe.predict_proba(X)
        proba_array = np.asarray(proba)
        if proba_array.ndim == 2 and proba_array.shape[1] >= 2:
            raw_pred = proba_array[:, 1]
        else:
            raw_pred = proba_array.ravel()
    elif task_type in {"classifier", "classification"}:
        proba = pipe.predict_proba(X)
        raw_pred = np.asarray(proba)
        if raw_pred.ndim == 2 and raw_pred.shape[1] >= 2:
            raw_pred = raw_pred[:, 1]
        else:
            raw_pred = raw_pred.ravel()
    else:
        print(f"[warn] unknown task_type '{task_type}'; falling back to predict().")
        raw_pred = pipe.predict(X)
    return np.asarray(raw_pred, dtype=float)


def _post_process_predictions(
    prediction: np.ndarray,
    *,
    ensure_finite: bool,
    replace_nonfinite_with: float,
    clip_low: float | None,
    clip_high: float | None,
) -> np.ndarray:
    if not np.isfinite(prediction).all():
        if ensure_finite:
            raise ValueError(
                "Predictions contain non-finite values and --ensure-finite is set."
            )
        print("[warn] detected non-finite predictions; applying replacement")
        prediction = np.nan_to_num(
            prediction,
            nan=replace_nonfinite_with,
            posinf=replace_nonfinite_with,
            neginf=replace_nonfinite_with,
        )
    if clip_low is not None or clip_high is not None:
        low = clip_low if clip_low is not None else -np.inf
        high = clip_high if clip_high is not None else np.inf
        prediction = np.clip(prediction, low, high)
    return prediction


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifacts_dir)
    bundle_path = (
        Path(args.bundle_path) if args.bundle_path else art_dir / "inference_bundle.pkl"
    )
    meta_path = Path(args.meta_path) if args.meta_path else art_dir / "model_meta.json"
    feature_list_path = (
        Path(args.feature_list_path)
        if args.feature_list_path
        else art_dir / "feature_list.json"
    )

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"inference bundle not found: {bundle_path}. Run train_su8.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"model meta not found: {meta_path}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"feature list not found: {feature_list_path}")

    meta = _load_json(meta_path)
    feature_list = _load_json(feature_list_path)
    feature_cols = feature_list.get("pipeline_input_columns")
    if not feature_cols or not isinstance(feature_cols, list):
        raise KeyError("pipeline_input_columns not found in feature_list.json")

    id_col = args.id_col or meta.get("id_col", "date_id")
    pred_col = args.pred_col
    postprocess_params = _resolve_postprocess_params(
        meta if isinstance(meta, Mapping) else {}
    )
    print(
        f"[info] post-process params: mult={postprocess_params.mult}, lo={postprocess_params.lo}, hi={postprocess_params.hi}"
    )

    _check_bundle_compat(meta)

    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] test file: {test_path}")
    test_df = load_table(test_path)
    if id_col not in test_df.columns:
        raise KeyError(f"ID column '{id_col}' not found in test data")

    required_calendar_cols = _extract_required_calendar_columns(meta)
    missing_calendar_cols = [
        col for col in required_calendar_cols if col not in test_df.columns
    ]
    if missing_calendar_cols:
        raise KeyError(
            "Missing calendar columns required by preprocessing: "
            + ", ".join(sorted(map(str, missing_calendar_cols)))
        )

    working_df = test_df.reset_index(drop=True).copy()
    working_df["__original_order__"] = np.arange(len(working_df))
    working_sorted = working_df.sort_values(id_col).reset_index(drop=True)
    feature_frame = working_sorted.drop(columns=["__original_order__"])

    X_test = _ensure_columns(
        feature_frame, feature_cols, max_missing=args.max_missing_columns
    )
    _coerce_numeric_like_columns(X_test)

    print("[info] loading inference bundle")
    pipe = joblib.load(bundle_path)

    print("[info] predicting...")
    prediction = _predict_with_task(pipe, X_test, meta)
    prediction = _post_process_predictions(
        prediction,
        ensure_finite=args.ensure_finite,
        replace_nonfinite_with=args.replace_nonfinite_with,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )
    signal_prediction = to_signal(prediction, postprocess_params).astype(
        np.float32, copy=False
    )

    if "is_scored" not in working_sorted.columns:
        raise KeyError(
            "Expected 'is_scored' column in test data for submission filtering. "
            "This column should indicate which rows are evaluated in the competition. "
            "Please ensure your test data includes an 'is_scored' column, with True/1 for rows to be scored. "
            "Refer to the competition documentation or data preparation instructions for details."
        )
    scored_mask_sorted = working_sorted["is_scored"].to_numpy()
    try:
        scored_mask_sorted = scored_mask_sorted.astype(bool)
    except ValueError as exc:
        raise ValueError(
            "Unable to interpret 'is_scored' column as boolean mask."
        ) from exc

    submission_sorted = pd.DataFrame(
        {
            id_col: working_sorted[id_col].to_numpy(),
            pred_col: signal_prediction,
            "raw_prediction": np.asarray(prediction, dtype=float),
            "__original_order__": working_sorted["__original_order__"].to_numpy(),
            "__is_scored__": scored_mask_sorted,
        }
    )

    submission_scored = submission_sorted.loc[submission_sorted["__is_scored__"]].copy()
    if submission_scored.empty:
        raise ValueError("No scored rows found in test data; cannot produce submission.")

    submission_scored = submission_scored.sort_values(id_col).reset_index(drop=True)
    if not np.isfinite(submission_scored[pred_col]).all():
        raise ValueError(
            "Submission contains non-finite predictions after signal transformation."
        )
    if not submission_scored[id_col].is_unique:
        raise ValueError("Duplicate ID values detected in submission output.")

    submission_scored[id_col] = submission_scored[id_col].astype("int64", copy=False)
    submission_scored[pred_col] = submission_scored[pred_col].astype("float32", copy=False)
    submission = submission_scored[[id_col, pred_col]]
    expected_count = int(np.count_nonzero(scored_mask_sorted))
    if len(submission) != expected_count:
        raise RuntimeError(
            f"Submission row count {len(submission)} does not match scored rows {expected_count}."
        )

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out_parquet, index=False)
    print(f"[ok] wrote {out_parquet} [{len(submission)} rows]")

    if not args.no_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
