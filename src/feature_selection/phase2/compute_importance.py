#!/usr/bin/env python
"""Compute feature importance for Tier1 feature set.

This script evaluates the Tier1 feature set (after Phase 1 filtering) and outputs:
1. Fold-wise feature importance (gain and split)
2. Aggregated importance statistics (mean, std, normalized)

The output is used in Phase 2-1 to identify low-importance features as deletion candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from datetime import datetime, timezone

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LGBM = False

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[1]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# Import from train_su5.py
from src.feature_generation.su5.train_su5 import (  # noqa: E402
    build_pipeline,
    load_su1_config,
    load_su5_config,
    load_preprocess_policies,
    infer_train_file,
    infer_test_file,
    load_table,
    _prepare_features,
    SU5FeatureAugmenter,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Compute LGBM feature importance for Tier1 feature set."
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
        "--out-dir",
        type=str,
        default="results/feature_selection",
        help="Output directory for results",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="market_forward_excess_returns",
        help="Target column name",
    )
    ap.add_argument(
        "--id-col",
        type=str,
        default="date_id",
        help="ID column name",
    )
    ap.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for TimeSeriesSplit",
    )
    ap.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Gap between train and validation in each fold",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    ap.add_argument(
        "--verbosity",
        type=int,
        default=-1,
        help="LightGBM verbosity level",
    )
    ap.add_argument(
        "--numeric-fill-value",
        type=float,
        default=0.0,
        help="Value for numeric imputation",
    )
    # LightGBM hyperparameters
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--min-data-in-leaf", type=int, default=32)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument(
        "--exclude-features",
        type=str,
        default=None,
        help="Path to JSON file containing features to exclude (e.g., tier1/excluded.json)",
    )
    return ap.parse_args(argv)


def compute_fold_importance(
    model: Any,
    feature_names: List[str],
    fold_idx: int,
) -> pd.DataFrame:
    """Compute feature importance for a single fold.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        fold_idx: Current fold index
        
    Returns:
        DataFrame with columns: feature_name, importance_gain, importance_split, fold
    """
    if not hasattr(model, "feature_importances_"):
        # Return empty DataFrame if no importance available
        return pd.DataFrame(columns=["feature_name", "importance_gain", "importance_split", "fold"])
    
    # Get importances
    importance_gain = model.feature_importances_  # default is 'gain'
    
    # Get split importance - check if booster_ attribute exists (LightGBM specific)
    try:
        if hasattr(model, "booster_"):
            importance_split = model.booster_.feature_importance(importance_type='split')
        else:
            # Fallback: use gain importance for split as well
            print(f"[warn][fold {fold_idx}] Model does not have booster_ attribute, using gain for split importance")
            importance_split = importance_gain.copy()
    except Exception as e:
        print(f"[warn][fold {fold_idx}] Could not get split importance: {e}")
        print(f"[warn][fold {fold_idx}] Falling back to gain importance for split metric")
        importance_split = importance_gain.copy()
    
    # Ensure same length
    n_features = len(feature_names)
    if len(importance_gain) != n_features or len(importance_split) != n_features:
        print(f"[warn][fold {fold_idx}] importance length mismatch: "
              f"gain={len(importance_gain)}, split={len(importance_split)}, "
              f"features={n_features}")
        # Truncate or pad as needed
        min_len = min(len(importance_gain), len(importance_split), n_features)
        importance_gain = importance_gain[:min_len]
        importance_split = importance_split[:min_len]
        feature_names = feature_names[:min_len]
    
    df = pd.DataFrame({
        "feature_name": feature_names,
        "importance_gain": importance_gain,
        "importance_split": importance_split,
        "fold": fold_idx,
    })
    
    return df


def aggregate_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance across folds.
    
    Args:
        importance_df: DataFrame with fold-wise importance
        
    Returns:
        DataFrame with aggregated statistics (mean, std, min, max) and normalized mean
    """
    if importance_df.empty:
        return pd.DataFrame()
    
    agg_dict = {
        "importance_gain": ["mean", "std", "min", "max"],
        "importance_split": ["mean", "std", "min", "max"],
    }
    
    grouped = importance_df.groupby("feature_name").agg(agg_dict)
    
    # Flatten column names more robustly
    new_columns = []
    for col in grouped.columns:
        metric, stat = col  # col is a tuple (metric, stat)
        # Extract the type from metric: 'importance_gain' -> 'gain'
        if "_" in metric:
            metric_type = metric.split("_", 1)[1]  # Split on first underscore only
        else:
            metric_type = metric
        new_columns.append(f"{stat}_{metric_type}")
    
    grouped.columns = new_columns
    grouped = grouped.reset_index()
    
    # Add normalized mean gain (share of total)
    if "mean_gain" in grouped.columns:
        total_gain = grouped["mean_gain"].sum()
        if total_gain > 0:
            grouped["mean_gain_normalized"] = grouped["mean_gain"] / total_gain
        else:
            grouped["mean_gain_normalized"] = 0.0
        
        # Sort by mean gain importance (descending)
        grouped = grouped.sort_values("mean_gain", ascending=False)
    
    return grouped


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    if not HAS_LGBM:
        print("[error] LightGBM is not installed. Cannot proceed.")
        return 1
    
    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Output directory: {out_dir}")
    
    # Load configurations
    print("[info] Loading configurations...")
    config_path = Path(args.config_path)
    preprocess_config_path = Path(args.preprocess_config)
    
    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}")
        return 1
    if not preprocess_config_path.exists():
        print(f"[error] Preprocess config file not found: {preprocess_config_path}")
        return 1
    
    su1_cfg = load_su1_config(str(config_path))
    su5_cfg = load_su5_config(str(config_path))
    preprocess_policies = load_preprocess_policies(str(preprocess_config_path))
    
    # Infer data files
    data_dir = Path(args.data_dir)
    train_path = infer_train_file(data_dir, args.train_file)
    test_path = infer_test_file(data_dir, args.test_file)
    
    print(f"[info] Train file: {train_path}")
    print(f"[info] Test file: {test_path}")
    
    # Load data
    print("[info] Loading data...")
    train_df = load_table(train_path)
    test_df = load_table(test_path)
    
    # Prepare features
    print("[info] Preparing features...")
    X_np, y_np, _ = _prepare_features(
        train_df,
        test_df,
        target_col=args.target_col,
        id_col=args.id_col,
    )
    
    # Load exclusion list if provided
    excluded_features = set()
    if args.exclude_features:
        exclude_path = Path(args.exclude_features)
        if exclude_path.exists():
            print(f"[info] Loading feature exclusion list from: {exclude_path}")
            with exclude_path.open("r", encoding="utf-8") as fh:
                exclude_data = json.load(fh)
                if "candidates" in exclude_data:
                    excluded_features = {item["feature_name"] for item in exclude_data["candidates"]}
                print(f"[info] Excluding {len(excluded_features)} features")
        else:
            print(f"[warn] Exclusion file not found: {exclude_path}")
    
    # Build SU5 augmenter
    print("[info] Building SU5 feature augmenter...")
    su5_augmenter = SU5FeatureAugmenter(
        su1_config=su1_cfg,
        su5_config=su5_cfg,
    )
    
    # Fit augmenter on train data
    print("[info] Fitting augmenter on training data...")
    su5_prefit = su5_augmenter.fit(X_np)
    
    # Transform train data
    print("[info] Augmenting training features...")
    X_augmented = su5_prefit.transform(X_np)
    
    # Apply exclusions if any
    if excluded_features:
        cols_before = list(X_augmented.columns)
        cols_to_keep = [c for c in cols_before if c not in excluded_features]
        X_augmented = X_augmented[cols_to_keep]
        print(f"[info] Excluded {len(cols_before) - len(cols_to_keep)} features")
        print(f"[info] Remaining features: {len(X_augmented.columns)}")
    
    X_augmented_all = X_augmented
    y_np_array = y_np.to_numpy().ravel()
    
    # Build pipeline
    print("[info] Building pipeline...")
    model_kwargs = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "random_state": args.random_state,
        "n_jobs": -1,
        "verbosity": args.verbosity,
    }
    
    core_pipeline_template = build_pipeline(
        su1_cfg,
        su5_cfg,
        preprocess_policies,
        numeric_fill_value=args.numeric_fill_value,
        model_kwargs=model_kwargs,
        random_state=args.random_state,
    )
    
    # Perform CV for importance calculation
    print(f"[info] Performing {args.n_splits}-fold TimeSeriesSplit CV...")
    tscv = TimeSeriesSplit(n_splits=args.n_splits, gap=args.gap)
    
    importance_dfs: List[pd.DataFrame] = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_augmented_all), start=1):
        print(f"[info][fold {fold_idx}] Training on {len(train_idx)} samples, validating on {len(val_idx)} samples...")
        
        X_train = X_augmented_all.iloc[train_idx]
        y_train = y_np.iloc[train_idx]
        X_valid = X_augmented_all.iloc[val_idx]
        y_valid = y_np.iloc[val_idx]
        
        # Clone and fit pipeline
        from typing import cast
        pipe = cast(Pipeline, clone(core_pipeline_template))
        pipe.fit(X_train, y_train)
        
        # Extract feature importance
        model_step = pipe.named_steps.get("model")
        if model_step is not None and hasattr(model_step, "feature_importances_"):
            # Get feature names from preprocess step
            preprocess_step = pipe.named_steps.get("preprocess")
            if preprocess_step is not None:
                try:
                    # Try to get feature names using get_feature_names_out()
                    if hasattr(preprocess_step, "get_feature_names_out"):
                        feature_names = list(preprocess_step.get_feature_names_out())
                    else:
                        # Fallback: use generic names
                        n_features = len(model_step.feature_importances_)
                        feature_names = [f"feature_{i}" for i in range(n_features)]
                        print(f"[warn][fold {fold_idx}] Preprocess step lacks get_feature_names_out(), using generic names")
                except (NotFittedError, AttributeError, TypeError, ValueError) as e:
                    # Fallback: use generic names
                    n_features = len(model_step.feature_importances_)
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                    print(f"[warn][fold {fold_idx}] Failed to extract feature names: {e}")
                    print(f"[warn][fold {fold_idx}] Using generic feature names")
            else:
                n_features = len(model_step.feature_importances_)
                feature_names = [f"feature_{i}" for i in range(n_features)]
                print(f"[warn][fold {fold_idx}] No preprocess step found, using generic feature names")
            
            fold_importance = compute_fold_importance(model_step, feature_names, fold_idx)
            importance_dfs.append(fold_importance)
            print(f"[ok][fold {fold_idx}] Computed importance for {len(feature_names)} features")
        else:
            print(f"[warn][fold {fold_idx}] Model does not have feature_importances_ attribute")
    
    # Save importance data
    if importance_dfs:
        # Concatenate all folds
        all_importance = pd.concat(importance_dfs, ignore_index=True)
        
        # Determine file prefix
        file_prefix = "tier1"
        
        # Save fold-wise importance
        importance_path = out_dir / f"{file_prefix}_importance.csv"
        all_importance.to_csv(importance_path, index=False)
        print(f"[ok] Saved fold-wise importance: {importance_path}")
        print(f"[info] Total features tracked: {all_importance['feature_name'].nunique()}")
        
        # Aggregate importance
        summary = aggregate_importance(all_importance)
        summary_path = out_dir / f"{file_prefix}_importance_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[ok] Saved importance summary: {summary_path}")
        print(f"[info] Summary shape: {summary.shape}")
        
        # Display top and bottom features
        if not summary.empty and "mean_gain" in summary.columns:
            print("\n[info] Top 10 features by mean gain:")
            print(summary.head(10)[["feature_name", "mean_gain", "std_gain", "mean_gain_normalized"]])
            print("\n[info] Bottom 10 features by mean gain:")
            print(summary.tail(10)[["feature_name", "mean_gain", "std_gain", "mean_gain_normalized"]])
    else:
        print("[error] No importance data collected.")
        return 1
    
    print("[ok] Importance computation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
