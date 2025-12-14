#!/usr/bin/env python
"""LightGBM prediction script for Hull Sharpe optimized model.

This script loads the trained model from artifacts/models/lgbm-sharpe-wf-opt/
and generates submission.csv for Kaggle submission.

Usage:
    python -m src.metrics.lgbm.predict_lgbm \
        --artifacts-dir artifacts/models/lgbm-sharpe-wf-opt
"""

from __future__ import annotations

import argparse
import json
import sys
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
    """Register legacy numpy BitGenerator names for pickle compatibility."""
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
        import numpy.random._pickle as _np_random_pickle
    except Exception:
        _np_random_pickle = None

    if _np_random_pickle is not None:
        registry = getattr(_np_random_pickle, "BitGenerators", None)
        if isinstance(registry, dict):
            registries.append(registry)

    for registry in registries:
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
from src.feature_generation.su5.train_su5 import SU5FeatureAugmenter  # noqa: E402,F401


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
        data_dir / "Test.parquet",
        data_dir / "Test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No test file found under {data_dir}")


def load_table(path: Path) -> pd.DataFrame:
    """Load a parquet or CSV file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def _load_json(path: Path) -> Any:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def to_signal(
    pred: np.ndarray,
    mult: float = 20.0,
    offset: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> np.ndarray:
    """Convert raw predictions to signal (position) for submission.
    
    position = pred * mult + offset, clipped to [clip_min, clip_max]
    
    Default mult=20.0 is the optimized value for Hull Sharpe evaluation.
    """
    values = np.asarray(pred, dtype=float) * mult + offset
    return np.clip(values, clip_min, clip_max)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with the optimized LGBM model."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--test-file", type=str, default=None,
        help="Explicit path to the test file",
    )
    parser.add_argument(
        "--artifacts-dir", type=str,
        default="artifacts/models/lgbm-sharpe-wf-opt",
        help="Directory storing model artifacts",
    )
    parser.add_argument(
        "--bundle-path", type=str, default=None,
        help="Override path to inference_bundle.pkl",
    )
    parser.add_argument(
        "--id-col", type=str, default="date_id",
        help="ID column name",
    )
    parser.add_argument(
        "--pred-col", type=str, default="prediction",
        help="Output prediction column name",
    )
    parser.add_argument(
        "--out-csv", type=str, default=None,
        help="Submission CSV output path (default: <artifacts-dir>/submission.csv)",
    )
    # Signal transformation parameters (optimized defaults)
    parser.add_argument(
        "--signal-mult", type=float, default=20.0,
        help="Multiplier for prediction->position (optimized=20.0)",
    )
    parser.add_argument(
        "--signal-offset", type=float, default=1.0,
        help="Offset for prediction->position",
    )
    parser.add_argument(
        "--clip-min", type=float, default=0.0,
        help="Minimum position value",
    )
    parser.add_argument(
        "--clip-max", type=float, default=2.0,
        help="Maximum position value",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Main prediction function."""
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifacts_dir)
    bundle_path = Path(args.bundle_path) if args.bundle_path else art_dir / "inference_bundle.pkl"
    
    out_csv = args.out_csv if args.out_csv else str(art_dir / "submission.csv")
    out_csv_path = Path(out_csv)

    if not bundle_path.exists():
        raise FileNotFoundError(f"inference bundle not found: {bundle_path}")

    # Load test data
    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] test file: {test_path}")
    test_df = load_table(test_path)
    
    if args.id_col not in test_df.columns:
        raise KeyError(f"ID column '{args.id_col}' not found in test data")

    # Sort by ID for consistent ordering
    working_df = test_df.reset_index(drop=True).copy()
    working_df["__original_order__"] = np.arange(len(working_df))
    working_sorted = working_df.sort_values(args.id_col).reset_index(drop=True)

    # Load model
    print(f"[info] loading model from {bundle_path}")
    bundle = joblib.load(bundle_path)

    # Handle both dict format (new) and Pipeline format (legacy)
    if isinstance(bundle, dict):
        augmenter = bundle.get("augmenter")
        core_pipeline = bundle.get("core_pipeline")
        excluded_features = bundle.get("excluded_features", [])
        print(f"[info] Bundle format: dict with augmenter + core_pipeline")
        print(f"[info] Excluded features: {len(excluded_features)}")
    else:
        # Legacy Pipeline format
        augmenter = None
        core_pipeline = bundle
        excluded_features = []
        print("[info] Bundle format: direct Pipeline (legacy)")

    # Prepare features (drop non-feature columns)
    drop_cols = ["__original_order__", "market_forward_excess_returns", "is_scored", "row_id"]
    X_test = working_sorted.drop(columns=[c for c in drop_cols if c in working_sorted.columns])

    # Apply augmenter if present
    if augmenter is not None:
        print("[info] applying feature augmentation...")
        X_augmented = augmenter.transform(X_test)
        # Apply feature exclusion
        if excluded_features:
            cols_to_drop = [c for c in X_augmented.columns if c in excluded_features]
            X_augmented = X_augmented.drop(columns=cols_to_drop, errors="ignore")
            print(f"[info] features after exclusion: {X_augmented.shape[1]}")
        X_final = X_augmented
    else:
        X_final = X_test

    # Predict
    print("[info] predicting...")
    raw_prediction = core_pipeline.predict(X_final)
    raw_prediction = np.asarray(raw_prediction, dtype=float).ravel()

    # Handle non-finite values
    if not np.isfinite(raw_prediction).all():
        print("[warn] detected non-finite predictions; replacing with 0.0")
        raw_prediction = np.nan_to_num(raw_prediction, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply signal transformation (prediction -> position)
    print(f"[info] signal transform: mult={args.signal_mult}, offset={args.signal_offset}")
    signal_prediction = to_signal(
        raw_prediction,
        mult=args.signal_mult,
        offset=args.signal_offset,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    ).astype(np.float32, copy=False)

    # Filter to scored rows
    if "is_scored" not in working_sorted.columns:
        raise KeyError(
            "Expected 'is_scored' column in test data for submission filtering."
        )
    scored_mask = working_sorted["is_scored"].astype(bool).to_numpy()

    # Build submission DataFrame
    submission_full = pd.DataFrame({
        args.id_col: working_sorted[args.id_col].to_numpy(),
        args.pred_col: signal_prediction,
        "__is_scored__": scored_mask,
    })

    submission_scored = submission_full.loc[submission_full["__is_scored__"]].copy()
    if submission_scored.empty:
        raise ValueError("No scored rows found in test data.")

    submission_scored = submission_scored.sort_values(args.id_col).reset_index(drop=True)
    
    # Validate
    if not np.isfinite(submission_scored[args.pred_col]).all():
        raise ValueError("Submission contains non-finite predictions.")
    if not submission_scored[args.id_col].is_unique:
        raise ValueError("Duplicate ID values detected in submission.")

    # Format output
    submission_scored[args.id_col] = submission_scored[args.id_col].astype("int64")
    submission_scored[args.pred_col] = submission_scored[args.pred_col].astype("float32")
    submission = submission_scored[[args.id_col, args.pred_col]]

    expected_count = int(np.count_nonzero(scored_mask))
    if len(submission) != expected_count:
        raise RuntimeError(
            f"Submission row count {len(submission)} does not match scored rows {expected_count}."
        )

    # Save
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv_path, index=False)
    print(f"[ok] wrote {out_csv_path} [{len(submission)} rows]")

    # Print summary statistics
    print(f"\n[info] Submission Statistics:")
    print(f"  Mean prediction: {submission[args.pred_col].mean():.4f}")
    print(f"  Std prediction:  {submission[args.pred_col].std():.4f}")
    print(f"  Min prediction:  {submission[args.pred_col].min():.4f}")
    print(f"  Max prediction:  {submission[args.pred_col].max():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
