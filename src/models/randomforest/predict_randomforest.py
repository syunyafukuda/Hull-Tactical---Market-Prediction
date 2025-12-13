#!/usr/bin/env python
"""RandomForest model inference script.

Loads the trained ``artifacts/models/randomforest/inference_bundle.pkl`` and runs inference
on the specified test data, outputting a competition-format ``submission.csv``.

Usage:
    python -m src.models.randomforest.predict_randomforest --artifacts-dir artifacts/models/randomforest
"""

from __future__ import annotations

import argparse
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
        sys.path.insert(0, str(path))


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


_ensure_numpy_bitgenerator_aliases()

# Import classes needed for unpickling
from src.preprocess.E_group.e_group import EGroupImputer  # noqa: E402,F401
from src.preprocess.I_group.i_group import IGroupImputer  # noqa: E402,F401
from src.preprocess.M_group.m_group import MGroupImputer  # noqa: E402,F401
from src.preprocess.P_group.p_group import PGroupImputer  # noqa: E402,F401
from src.preprocess.S_group.s_group import SGroupImputer  # noqa: E402,F401
from src.feature_generation.su1.feature_su1 import (  # noqa: E402,F401
    SU1Config,
    SU1FeatureGenerator,
)
from src.feature_generation.su5.feature_su5 import (  # noqa: E402,F401
    SU5Config,
    SU5FeatureGenerator,
)
from src.feature_generation.su5.train_su5 import (  # noqa: E402,F401
    SU5FeatureAugmenter,
)


def infer_test_file(data_dir: Path, explicit: str | None) -> Path:
    """Infer the test file path."""
    if explicit:
        candidate = Path(explicit)
        if not candidate.exists():
            raise FileNotFoundError(f"--test-file not found: {candidate}")
        return candidate
    candidates = [
        data_dir / "test.parquet",
        data_dir / "test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No test file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
    """Load a table from parquet or csv."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def _load_json(path: Path) -> Any:
    """Load a JSON file."""
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
    """Coerce object columns to numeric."""
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


@dataclass(frozen=True)
class PostProcessParams:
    """Post-processing parameters for signal transformation."""
    mult: float
    lo: float
    hi: float


def _resolve_postprocess_params(
    meta: Mapping[str, Any],
    *,
    override_mult: float | None = None,
    override_lo: float | None = None,
    override_hi: float | None = None,
) -> PostProcessParams:
    """Resolve post-process params from model metadata."""
    # Try oof_best_params first, then post_process_params
    pp_params = meta.get("oof_best_params", meta.get("post_process_params", {}))
    mult = pp_params.get("mult", pp_params.get("best_mult", 1.0))
    lo = pp_params.get("lo", pp_params.get("best_lo", 0.9))
    hi = pp_params.get("hi", pp_params.get("best_hi", 1.1))
    
    # Apply overrides
    if override_mult is not None:
        mult = override_mult
    if override_lo is not None:
        lo = override_lo
    if override_hi is not None:
        hi = override_hi
    
    return PostProcessParams(mult=mult, lo=lo, hi=hi)


def to_signal(pred: np.ndarray, params: PostProcessParams) -> np.ndarray:
    """Transform prediction to signal format."""
    values = np.asarray(pred, dtype=float) * params.mult + 1.0
    return np.clip(values, params.lo, params.hi)


def _post_process_predictions(
    prediction: np.ndarray,
    *,
    ensure_finite: bool,
    replace_nonfinite_with: float,
    clip_low: float | None,
    clip_high: float | None,
) -> np.ndarray:
    """Post-process predictions with finite value handling and clipping."""
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with the RandomForest pipeline bundle."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Explicit path to the test file",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/models/randomforest",
        help="Directory storing RandomForest artifacts",
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        default=None,
        help="Override path to inference_bundle.pkl",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=None,
        help="Override path to model_meta.json",
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
        default="date_id",
        help="ID column name",
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
        default=None,
        help="Submission parquet output path (default: artifacts_dir/submission.parquet)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Submission csv output path (default: artifacts_dir/submission.csv)",
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
        help="Raise an error if predictions contain NaN or infinite values",
    )
    parser.add_argument(
        "--replace-nonfinite-with",
        type=float,
        default=0.0,
        help="Replacement value to use for non-finite predictions",
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
    parser.add_argument(
        "--pp-mult",
        type=float,
        default=None,
        help="Override post-process multiplier",
    )
    parser.add_argument(
        "--pp-lo",
        type=float,
        default=None,
        help="Override post-process lower clip bound",
    )
    parser.add_argument(
        "--pp-hi",
        type=float,
        default=None,
        help="Override post-process upper clip bound",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Main entry point."""
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

    # Validate paths
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"inference bundle not found: {bundle_path}. Run train_randomforest.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"model meta not found: {meta_path}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"feature list not found: {feature_list_path}")

    # Load metadata
    meta = _load_json(meta_path)
    feature_list = _load_json(feature_list_path)
    pipeline_input_cols = feature_list.get("pipeline_input_columns")
    if not pipeline_input_cols or not isinstance(pipeline_input_cols, list):
        raise KeyError("pipeline_input_columns not found in feature_list.json")

    id_col = args.id_col
    pred_col = args.pred_col
    postprocess_params = _resolve_postprocess_params(
        meta if isinstance(meta, Mapping) else {},
        override_mult=args.pp_mult,
        override_lo=args.pp_lo,
        override_hi=args.pp_hi,
    )
    print(
        f"[info] post-process params: mult={postprocess_params.mult}, "
        f"lo={postprocess_params.lo}, hi={postprocess_params.hi}"
    )

    # Load test data
    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] test file: {test_path}")
    test_df = load_table(test_path)
    if id_col not in test_df.columns:
        raise KeyError(f"ID column '{id_col}' not found in test data")

    # Prepare working dataframe
    working_df = test_df.reset_index(drop=True).copy()
    working_df["__original_order__"] = np.arange(len(working_df))
    working_sorted = working_df.sort_values(id_col).reset_index(drop=True)

    # Ensure required columns
    X_test_source = (
        working_sorted[pipeline_input_cols].copy() if all(c in working_sorted.columns for c in pipeline_input_cols) 
        else working_sorted.drop(columns=["__original_order__"], errors="ignore")
    )
    if not isinstance(X_test_source, pd.DataFrame):
        raise ValueError("X_test_source must be a DataFrame")
    X_test = _ensure_columns(
        X_test_source,
        pipeline_input_cols,
        max_missing=args.max_missing_columns,
    )
    _coerce_numeric_like_columns(X_test)

    # Load inference bundle
    print("[info] loading inference bundle")
    bundle = joblib.load(bundle_path)
    
    # RandomForest bundle is a direct Pipeline (not a dict)
    if isinstance(bundle, dict):
        pipeline = bundle.get("pipeline")
        print("[info] bundle format: dict with pipeline")
    else:
        # Assume bundle is the pipeline directly
        pipeline = bundle
        print("[info] bundle format: direct Pipeline")
    
    if pipeline is None:
        raise ValueError("Pipeline is None - invalid bundle structure")
    
    # Predict
    print("[info] predicting...")
    prediction = pipeline.predict(X_test)
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

    # Filter to scored rows
    if "is_scored" not in working_sorted.columns:
        raise KeyError(
            "Expected 'is_scored' column in test data for submission filtering. "
            "This column should indicate which rows are evaluated in the competition."
        )
    scored_mask = working_sorted["is_scored"].to_numpy().astype(bool)

    # Build submission dataframe
    submission_sorted = pd.DataFrame(
        {
            id_col: working_sorted[id_col].to_numpy(),
            pred_col: signal_prediction,
            "raw_prediction": np.asarray(prediction, dtype=float),
            "__is_scored__": scored_mask,
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
    
    expected_count = int(np.count_nonzero(scored_mask))
    if len(submission) != expected_count:
        raise RuntimeError(
            f"Submission row count {len(submission)} does not match scored rows {expected_count}."
        )

    # Write outputs
    out_parquet = Path(args.out_parquet) if args.out_parquet else art_dir / "submission.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out_parquet, index=False)
    print(f"[ok] wrote {out_parquet} [{len(submission)} rows]")

    if not args.no_csv:
        out_csv = Path(args.out_csv) if args.out_csv else art_dir / "submission.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv}")

    # Summary
    print("\n[summary] Prediction stats:")
    print(f"  - Raw prediction: min={prediction.min():.6f}, max={prediction.max():.6f}, mean={prediction.mean():.6f}")
    print(f"  - Signal: min={signal_prediction.min():.6f}, max={signal_prediction.max():.6f}, mean={signal_prediction.mean():.6f}")
    print(f"  - Submission rows: {len(submission)}")
    
    # Print submission preview
    print("\n[preview] submission.csv:")
    print(submission.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
