#!/usr/bin/env python
"""XGBoost training script using the unified model framework.

This script trains an XGBoost model using the FS_compact feature set (116 features)
and provides OOF evaluation consistent with other model types.

Key features:
- Uses FS_compact feature exclusion (tier3/excluded.json)
- Reuses existing preprocessing pipeline (M/E/I/P/S group imputers)
- Compatible with the unified CV evaluation framework
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    XGBRegressor = None  # type: ignore
    HAS_XGB = False

# Path setup
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# Import from existing modules
from src.feature_generation.su5.train_su5 import (  # noqa: E402
    SU5FeatureAugmenter,
    _prepare_features,
    build_pipeline,
    infer_test_file,
    infer_train_file,
    load_preprocess_policies,
    load_su1_config,
    load_su5_config,
    load_table,
)
from src.models.common.cv_utils import (  # noqa: E402
    compute_fold_metrics,
    evaluate_oof_predictions,
)
from src.models.common.feature_loader import (  # noqa: E402
    get_excluded_features,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Train XGBoost with FS_compact features.")
    ap.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing train/test files",
    )
    ap.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Explicit path to training file",
    )
    ap.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Explicit path to test file",
    )
    ap.add_argument(
        "--config-path",
        type=str,
        default="configs/feature_generation.yaml",
        help="Path to feature_generation.yaml",
    )
    ap.add_argument(
        "--preprocess-config",
        type=str,
        default="configs/preprocess.yaml",
        help="Path to preprocess.yaml",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="market_forward_excess_returns",
    )
    ap.add_argument(
        "--id-col",
        type=str,
        default="date_id",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/models/xgboost",
    )
    ap.add_argument(
        "--numeric-fill-value",
        type=float,
        default=0.0,
    )
    # CV settings
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--min-val-size", type=int, default=0)
    # XGBoost hyperparameters
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--min-child-weight", type=int, default=32)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--verbosity", type=int, default=0)
    # Feature selection
    ap.add_argument(
        "--feature-tier",
        type=str,
        default="tier3",
        choices=["tier0", "tier1", "tier2", "tier3"],
        help="Feature tier to use (tier3 = FS_compact)",
    )
    ap.add_argument(
        "--no-feature-exclusion",
        action="store_true",
        help="Skip feature exclusion (use all features)",
    )
    # Output control
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="Run CV only, no final training"
    )
    return ap.parse_args(argv)


def sanitize_feature_names(feature_cols: List[str]) -> List[str]:
    """Sanitize feature names for XGBoost compatibility.

    XGBoost warns about special characters like [, ], <, > in feature names.
    Replace them with underscores.

    Parameters
    ----------
    feature_cols : List[str]
        Original feature column names.

    Returns
    -------
    List[str]
        Sanitized feature column names.
    """
    return [re.sub(r"[\[\]<>]", "_", col) for col in feature_cols]


def _to_1d(pred: Any) -> np.ndarray:
    """Convert prediction to 1D numpy array."""
    array = np.asarray(pred)
    if array.ndim > 1:
        array = array.ravel()
    return array.astype(float, copy=False)


def apply_feature_exclusion_to_augmented(
    df: pd.DataFrame,
    tier: str = "tier3",
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply feature exclusion to augmented DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after SU1/SU5 feature augmentation.
    tier : str
        Feature tier for exclusion.
    verbose : bool
        Print exclusion info.

    Returns
    -------
    pd.DataFrame
        DataFrame with excluded features removed.
    """
    excluded = get_excluded_features(tier)
    columns_to_drop = [col for col in df.columns if col in excluded]

    if verbose and columns_to_drop:
        print(f"[info] Excluding {len(columns_to_drop)} features based on {tier}")

    return df.drop(columns=columns_to_drop, errors="ignore")


def main(argv: Sequence[str] | None = None) -> int:
    """Main training function."""
    args = parse_args(argv)

    if not HAS_XGB:
        print("[error] XGBoost is not installed")
        return 1

    # Setup paths
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if not args.no_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load configurations
    su1_config = load_su1_config(args.config_path)
    su5_config = load_su5_config(args.config_path)
    preprocess_settings = load_preprocess_policies(args.preprocess_config)

    # Load data
    train_path = infer_train_file(data_dir, args.train_file)
    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] train file: {train_path}")
    print(f"[info] test file : {test_path}")

    train_df = load_table(train_path)
    test_df = load_table(test_path)

    # Sort by date
    if args.id_col in train_df.columns:
        train_df = train_df.sort_values(args.id_col).reset_index(drop=True)
    if args.id_col in test_df.columns:
        test_df = test_df.sort_values(args.id_col).reset_index(drop=True)

    if args.target_col not in train_df.columns:
        raise KeyError(f"Target column '{args.target_col}' not found")

    # Prepare features
    X, y, feature_cols = _prepare_features(
        train_df,
        test_df,
        target_col=args.target_col,
        id_col=args.id_col,
    )

    # XGBoost model kwargs
    model_kwargs = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "min_child_weight": args.min_child_weight,
        "random_state": args.random_state,
        "n_jobs": -1,
        "verbosity": args.verbosity,
    }

    # Build base pipeline
    base_pipeline = build_pipeline(
        su1_config,
        su5_config,
        preprocess_settings,
        numeric_fill_value=args.numeric_fill_value,
        model_kwargs=model_kwargs,
        random_state=args.random_state,
    )

    # CV setup
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X_np = X.reset_index(drop=True)
    y_np = y.reset_index(drop=True)
    y_np_array = y_np.to_numpy()

    # Pre-fit augmenter for CV
    print("[info] Pre-fitting SU1/SU5 feature augmenter...")
    su5_prefit = SU5FeatureAugmenter(
        su1_config,
        su5_config,
        fill_value=args.numeric_fill_value,
    )
    su5_prefit.fit(X_np)

    # Build fold_indices array
    fold_indices_full = np.full(len(X_np), -1, dtype=int)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np)):
        fold_indices_full[train_idx] = fold_idx
        fold_indices_full[val_idx] = fold_idx

    # Transform with fold_indices
    print("[info] Generating augmented features...")
    X_augmented_all = su5_prefit.transform(X_np, fold_indices=fold_indices_full)

    # Apply feature exclusion based on tier
    if not args.no_feature_exclusion:
        print(f"[info] Applying {args.feature_tier} feature exclusion...")
        n_before = X_augmented_all.shape[1]
        X_augmented_all = apply_feature_exclusion_to_augmented(
            X_augmented_all, tier=args.feature_tier, verbose=True
        )
        n_after = X_augmented_all.shape[1]
        print(
            f"[info] Features: {n_before} -> {n_after} ({n_before - n_after} excluded)"
        )
    else:
        print(f"[info] Using all {X_augmented_all.shape[1]} features (no exclusion)")

    # Sanitize feature names for XGBoost
    original_columns = X_augmented_all.columns.tolist()
    sanitized_columns = sanitize_feature_names(original_columns)
    X_augmented_all.columns = sanitized_columns

    # Build core pipeline (without augmenter)
    core_pipeline_template = cast(Pipeline, Pipeline(base_pipeline.steps[1:]))

    # CV loop
    oof_pred = np.full(len(X_np), np.nan, dtype=float)
    fold_logs: List[Dict[str, Any]] = []

    print(f"\n{'=' * 60}")
    print(f"Starting {args.n_splits}-fold TimeSeriesSplit CV")
    print(f"{'=' * 60}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np), start=1):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        # Apply gap
        if args.gap > 0:
            if len(train_idx) > args.gap:
                train_idx = train_idx[: -args.gap]
            if len(val_idx) > args.gap:
                val_idx = val_idx[args.gap :]

        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"[warn][fold {fold_idx}] skipped: empty train/val after gap")
            continue
        if args.min_val_size and len(val_idx) < args.min_val_size:
            print(f"[warn][fold {fold_idx}] skipped: val size {len(val_idx)} < min")
            continue

        X_train = X_augmented_all.iloc[train_idx]
        y_train = y_np.iloc[train_idx]
        X_valid = X_augmented_all.iloc[val_idx]
        y_valid = y_np.iloc[val_idx]

        # Clone and fit pipeline with early stopping
        pipe = cast(Pipeline, clone(core_pipeline_template))
        fit_kwargs: Dict[str, Any] = {}

        # XGBoost early stopping: pass eval_set through pipeline's model__ prefix
        fit_kwargs["model__eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["model__early_stopping_rounds"] = 50
        fit_kwargs["model__verbose"] = False

        pipe.fit(X_train, y_train, **fit_kwargs)

        # Predict
        pred = pipe.predict(X_valid)
        pred = _to_1d(pred)

        # Compute metrics
        val_rmse = float(math.sqrt(mean_squared_error(y_valid, pred)))
        train_pred = pipe.predict(X_train)
        train_rmse = float(math.sqrt(mean_squared_error(y_train, _to_1d(train_pred))))

        oof_pred[val_idx] = pred

        # MSR metrics
        fold_metrics = compute_fold_metrics(
            y_valid.to_numpy(),
            pred,
            target_returns=y_valid.to_numpy(),
        )

        fold_logs.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "val_msr": fold_metrics.get("msr", float("nan")),
                "val_msr_down": fold_metrics.get("msr_down", float("nan")),
            }
        )

        print(
            f"[fold {fold_idx}] train_rmse={train_rmse:.6f} | "
            f"val_rmse={val_rmse:.6f} | val_msr={fold_metrics.get('msr', 0):.4f}"
        )

    # Overall OOF metrics
    valid_mask = ~np.isnan(oof_pred)
    if valid_mask.any():
        oof_rmse = float(
            math.sqrt(mean_squared_error(y_np_array[valid_mask], oof_pred[valid_mask]))
        )
        oof_metrics = evaluate_oof_predictions(
            oof_pred[valid_mask],
            y_np_array[valid_mask],
            oof_indices=np.where(valid_mask)[0],
            target_returns=y_np_array,
        )
    else:
        oof_rmse = float("nan")
        oof_metrics = {}

    print(f"\n{'=' * 60}")
    print(f"[OOF] RMSE = {oof_rmse:.6f}")
    print(f"[OOF] MSR  = {oof_metrics.get('oof_msr', float('nan')):.6f}")
    print(f"{'=' * 60}\n")

    # Save artifacts
    if not args.no_artifacts and not args.dry_run:
        print("[info] Final training on all data...")

        # Retrain on all data
        final_pipeline = cast(Pipeline, clone(base_pipeline))
        final_pipeline.fit(X_np, y_np)

        # Save model
        model_path = out_dir / "inference_bundle.pkl"
        joblib.dump(final_pipeline, model_path)
        print(f"[info] Saved model to {model_path}")

        # Save OOF predictions
        oof_df = pd.DataFrame(
            {
                "index": np.arange(len(oof_pred)),
                "actual": y_np_array,
                "prediction": oof_pred,
            }
        )
        oof_path = out_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f"[info] Saved OOF predictions to {oof_path}")

        # Save fold logs
        fold_log_path = out_dir / "cv_fold_logs.csv"
        if fold_logs:
            fieldnames = list(fold_logs[0].keys())
            with open(fold_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(fold_logs)
            print(f"[info] Saved fold logs to {fold_log_path}")

        # Save metadata
        meta = {
            "model_type": "xgboost",
            "feature_tier": args.feature_tier,
            "n_features": X_augmented_all.shape[1],
            "oof_rmse": oof_rmse,
            "oof_msr": oof_metrics.get("oof_msr", float("nan")),
            "n_splits": args.n_splits,
            "gap": args.gap,
            "hyperparameters": model_kwargs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = out_dir / "model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[info] Saved metadata to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
