#!/usr/bin/env python
"""Two-Head LightGBM prediction script.

This script loads the trained two-head models (forward_model + rf_model)
and generates submission.csv using the position formula:

    position = clip((x - rf_pred) / (forward_pred - rf_pred), 0, 2)

Usage:
    python -m src.models.lgbm.predict_lgbm_two_head \
        --artifacts-dir artifacts/models/lgbm-two-head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

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
from src.feature_generation.su5.train_su5 import (  # noqa: E402,F401
    SU5FeatureAugmenter,
    load_su1_config,
    load_su5_config,
    load_preprocess_policies,
)
from src.models.common.signals_two_head import (  # noqa: E402
    TwoHeadPositionConfig,
    map_positions_from_forward_rf,
)
from src.models.common.feature_loader import get_excluded_features  # noqa: E402


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with Two-Head LGBM models."
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
        default="artifacts/models/lgbm-two-head",
        help="Directory storing model artifacts",
    )
    parser.add_argument(
        "--feature-config", type=str, 
        default="configs/feature_generation/feature_generation.yaml",
        help="Path to feature_generation.yaml",
    )
    parser.add_argument(
        "--preprocess-config", type=str,
        default="configs/preprocess/preprocess.yaml",
        help="Path to preprocess.yaml",
    )
    parser.add_argument(
        "--feature-tier", type=str, default="tier3",
        help="Feature tier for exclusion",
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
    # Two-head specific parameters
    parser.add_argument(
        "--x", type=float, default=None,
        help="Override x parameter (default: load from position_config.json)",
    )
    parser.add_argument(
        "--clip-min", type=float, default=0.0,
        help="Minimum position value",
    )
    parser.add_argument(
        "--clip-max", type=float, default=2.0,
        help="Maximum position value",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Main prediction function."""
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    art_dir = Path(args.artifacts_dir)
    
    out_csv = args.out_csv if args.out_csv else str(art_dir / "submission.csv")
    out_csv_path = Path(out_csv)

    # Check required files exist
    forward_model_path = art_dir / "forward_model.pkl"
    rf_model_path = art_dir / "rf_model.pkl"
    position_config_path = art_dir / "position_config.json"
    
    for path in [forward_model_path, rf_model_path, position_config_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

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

    # Load models
    print(f"[info] loading forward model from {forward_model_path}")
    forward_bundle = joblib.load(forward_model_path)
    
    print(f"[info] loading rf model from {rf_model_path}")
    rf_bundle = joblib.load(rf_model_path)
    
    # Load position config
    position_config = _load_json(position_config_path)
    x_value = args.x if args.x is not None else position_config.get("x", 0.0)
    clip_min = args.clip_min if args.clip_min != 0.0 else position_config.get("clip_min", 0.0)
    clip_max = args.clip_max if args.clip_max != 2.0 else position_config.get("clip_max", 2.0)
    epsilon = position_config.get("epsilon", 1e-8)
    
    print(f"[info] position config: x={x_value:.6f}, clip=[{clip_min}, {clip_max}]")

    # Load models (core_pipeline without augmenter, trained on tier3-excluded features)
    forward_pipeline = forward_bundle
    rf_pipeline = rf_bundle
    
    # Load augmenter (if exists - new format)
    augmenter_path = art_dir / "augmenter.pkl"
    excluded_path = art_dir / "excluded_features.json"
    
    if augmenter_path.exists():
        print(f"[info] loading augmenter from {augmenter_path}")
        augmenter = joblib.load(augmenter_path)
        
        # Load excluded features
        if excluded_path.exists():
            with open(excluded_path, "r") as f:
                excluded_data = json.load(f)
            excluded_features = set(excluded_data.get("excluded", []))
            print(f"[info] excluded features: {len(excluded_features)}")
        else:
            excluded_features = set()
        
        # Prepare test data
        drop_cols = ["__original_order__", "is_scored", "row_id"]
        X_test_raw = working_sorted.drop(columns=[c for c in drop_cols if c in working_sorted.columns])
        
        # Apply augmenter
        print("[info] applying feature augmentation...")
        X_augmented = augmenter.fit_transform(X_test_raw)
        
        # Apply feature exclusion
        if excluded_features:
            cols_to_drop = [c for c in X_augmented.columns if c in excluded_features]
            X_augmented = X_augmented.drop(columns=cols_to_drop, errors="ignore")
            print(f"[info] features after exclusion: {X_augmented.shape[1]}")
        
        # Drop target/id columns
        final_drop = ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
        X_test = X_augmented.drop(columns=[c for c in final_drop if c in X_augmented.columns], errors="ignore")
        print(f"[info] final features for prediction: {X_test.shape[1]}")
    else:
        # Legacy format: pipeline includes augmenter
        print("[info] legacy format: using pipeline with built-in augmenter")
        drop_cols = ["__original_order__", "is_scored", "row_id"]
        X_test = working_sorted.drop(columns=[c for c in drop_cols if c in working_sorted.columns])
    
    print(f"[info] Test data shape: {X_test.shape}")

    # Predict with both models (full pipeline including augmenter)
    print("[info] predicting forward_returns...")
    forward_pred = forward_pipeline.predict(X_test)
    forward_pred = np.asarray(forward_pred, dtype=float).ravel()
    
    print("[info] predicting risk_free_rate...")
    rf_pred = rf_pipeline.predict(X_test)
    rf_pred = np.asarray(rf_pred, dtype=float).ravel()

    # Handle non-finite values
    for name, pred in [("forward", forward_pred), ("rf", rf_pred)]:
        if not np.isfinite(pred).all():
            print(f"[warn] detected non-finite {name} predictions; replacing with mean")
            mean_val = np.nanmean(pred[np.isfinite(pred)])
            pred[~np.isfinite(pred)] = mean_val

    # Compute positions using two-head formula
    print("[info] computing positions using two-head formula...")
    positions = map_positions_from_forward_rf(
        forward_pred=forward_pred,
        rf_pred=rf_pred,
        x=x_value,
        clip_min=clip_min,
        clip_max=clip_max,
        epsilon=epsilon,
    ).astype(np.float32, copy=False)

    if args.verbose:
        print(f"\n[debug] Forward predictions: mean={np.mean(forward_pred):.6f}, std={np.std(forward_pred):.6f}")
        print(f"[debug] RF predictions: mean={np.mean(rf_pred):.6f}, std={np.std(rf_pred):.6f}")
        print(f"[debug] Positions: mean={np.mean(positions):.4f}, std={np.std(positions):.4f}")

    # Filter to scored rows
    if "is_scored" not in working_sorted.columns:
        raise KeyError(
            "Expected 'is_scored' column in test data for submission filtering."
        )
    scored_mask = working_sorted["is_scored"].astype(bool).to_numpy()

    # Build submission DataFrame
    submission_full = pd.DataFrame({
        args.id_col: working_sorted[args.id_col].to_numpy(),
        args.pred_col: positions,
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
    print(f"  Mean position: {submission[args.pred_col].mean():.4f}")
    print(f"  Std position:  {submission[args.pred_col].std():.4f}")
    print(f"  Min position:  {submission[args.pred_col].min():.4f}")
    print(f"  Max position:  {submission[args.pred_col].max():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
