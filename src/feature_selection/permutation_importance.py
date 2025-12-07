#!/usr/bin/env python
"""Compute permutation importance for Phase 2-1 candidate features.

This script:
1. Loads candidate features from Phase 2-1 (low-importance features)
2. Trains model once per fold with Tier1 feature set
3. For each candidate, shuffles the column and measures ΔRMSE
4. Outputs results showing which features have near-zero impact on RMSE

This is used in Phase 2-2 to confirm which features can be safely deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import math

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LGBM = False

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
        description="Compute permutation importance for candidate features."
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
        "--exclude-features",
        type=str,
        required=True,
        help="Path to JSON file with Tier1 excluded features",
    )
    ap.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to JSON file with Phase 2-1 candidate features",
    )
    ap.add_argument(
        "--out-path",
        type=str,
        default="results/feature_selection/phase2_permutation_results.csv",
        help="Output CSV path for permutation results",
    )
    ap.add_argument(
        "--n-permutations",
        type=int,
        default=5,
        help="Number of permutations per feature per fold",
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
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    if not HAS_LGBM:
        print("[error] LightGBM is not installed. Cannot proceed.")
        return 1
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load candidate features
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"[error] Candidates file not found: {candidates_path}")
        return 1
    
    print(f"[info] Loading candidate features from: {candidates_path}")
    with candidates_path.open("r", encoding="utf-8") as fh:
        candidates_data = json.load(fh)
    
    if "candidates" not in candidates_data:
        print("[error] Invalid candidates file format: missing 'candidates' key")
        return 1
    
    candidate_features = [item["feature_name"] for item in candidates_data["candidates"]]
    print(f"[info] Loaded {len(candidate_features)} candidate features for permutation test")
    
    # Load exclusion list
    excluded_features = set()
    exclude_path = Path(args.exclude_features)
    if exclude_path.exists():
        print(f"[info] Loading feature exclusion list from: {exclude_path}")
        with exclude_path.open("r", encoding="utf-8") as fh:
            exclude_data = json.load(fh)
            if "candidates" in exclude_data:
                excluded_features = {item["feature_name"] for item in exclude_data["candidates"]}
            print(f"[info] Excluding {len(excluded_features)} features")
    else:
        print(f"[error] Exclusion file not found: {exclude_path}")
        return 1
    
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
    train_path = Path(args.train_file) if args.train_file else infer_train_file(data_dir)
    
    print(f"[info] Train file: {train_path}")
    
    # Load data
    print("[info] Loading data...")
    train_df = load_table(train_path)
    
    # Prepare features
    print("[info] Preparing features...")
    X_np, y_np, _ = _prepare_features(
        train_df,
        target_col=args.target_col,
        id_col=args.id_col,
    )
    
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
    
    # Apply exclusions (Tier1 features)
    cols_before = list(X_augmented.columns)
    cols_to_keep = [c for c in cols_before if c not in excluded_features]
    X_augmented = X_augmented[cols_to_keep]
    print(f"[info] Applied Tier1 exclusions: {len(cols_before)} -> {len(X_augmented.columns)} features")
    
    # Verify candidate features are present
    available_candidates = [c for c in candidate_features if c in X_augmented.columns]
    missing_candidates = [c for c in candidate_features if c not in X_augmented.columns]
    
    if missing_candidates:
        print(f"[warn] {len(missing_candidates)} candidate features not found in data, skipping them")
        print(f"[warn] Missing features: {missing_candidates[:10]}...")  # Show first 10
    
    candidate_features = available_candidates
    print(f"[info] Testing {len(candidate_features)} candidate features")
    
    if not candidate_features:
        print("[error] No candidate features found in the data")
        return 1
    
    X_augmented_all = X_augmented
    y_np_array = y_np.to_numpy().ravel()
    
    # Build pipeline
    print("[info] Building pipeline...")
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_data_in_leaf,
        "subsample": args.bagging_fraction,
        "subsample_freq": args.bagging_freq,
        "colsample_bytree": args.feature_fraction,
        "random_state": args.random_seed,
        "verbosity": args.verbosity,
        "force_col_wise": True,
    }
    
    core_pipeline_template = build_pipeline(
        preprocess_policies=preprocess_policies,
        model_class=LGBMRegressor,
        model_params=lgbm_params,
        numeric_fill_value=args.numeric_fill_value,
    )
    
    # Perform CV with permutation importance
    print(f"[info] Performing {args.n_splits}-fold TimeSeriesSplit CV with permutation importance...")
    tscv = TimeSeriesSplit(n_splits=args.n_splits, gap=args.gap)
    
    # Store results: {feature_name: {fold_1: [delta1, delta2, ...], fold_2: [...], ...}}
    permutation_results: Dict[str, Dict[str, List[float]]] = {
        feat: {} for feat in candidate_features
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_augmented_all), start=1):
        print(f"\n[info][fold {fold_idx}] Training on {len(train_idx)} samples, validating on {len(val_idx)} samples...")
        
        X_train = X_augmented_all.iloc[train_idx]
        y_train = y_np.iloc[train_idx]
        X_valid = X_augmented_all.iloc[val_idx].copy()  # Copy for mutation
        y_valid = y_np.iloc[val_idx]
        
        # Clone and fit pipeline
        pipe = cast(Pipeline, clone(core_pipeline_template))
        pipe.fit(X_train, y_train)
        
        # Get baseline RMSE
        y_pred_baseline = pipe.predict(X_valid)
        rmse_baseline = math.sqrt(mean_squared_error(y_valid, y_pred_baseline))
        print(f"[info][fold {fold_idx}] Baseline RMSE: {rmse_baseline:.6f}")
        
        # Permutation test for each candidate
        for feat_idx, feature_name in enumerate(candidate_features, start=1):
            if feature_name not in X_valid.columns:
                print(f"[warn][fold {fold_idx}] Feature '{feature_name}' not found in validation data, skipping")
                continue
            
            delta_rmses = []
            
            for perm_idx in range(args.n_permutations):
                # Create a copy and shuffle the feature
                X_valid_shuffled = X_valid.copy()
                X_valid_shuffled[feature_name] = np.random.permutation(X_valid_shuffled[feature_name].values)
                
                # Predict with shuffled feature
                y_pred_shuffled = pipe.predict(X_valid_shuffled)
                rmse_shuffled = math.sqrt(mean_squared_error(y_valid, y_pred_shuffled))
                
                # Calculate delta (positive = feature is important, zero/negative = not important)
                delta_rmse = rmse_shuffled - rmse_baseline
                delta_rmses.append(delta_rmse)
            
            # Store results for this fold
            fold_key = f"fold_{fold_idx}"
            permutation_results[feature_name][fold_key] = delta_rmses
            
            # Report progress
            mean_delta = np.mean(delta_rmses)
            if (feat_idx % 10 == 0) or (feat_idx == len(candidate_features)):
                print(f"[info][fold {fold_idx}] Tested {feat_idx}/{len(candidate_features)} features...")
    
    # Aggregate results across folds
    print("\n[info] Aggregating permutation results...")
    results_list = []
    
    for feature_name in candidate_features:
        fold_results = permutation_results[feature_name]
        
        # Collect all deltas across all folds and permutations
        all_deltas = []
        fold_means = {}
        
        for fold_key in sorted(fold_results.keys()):
            deltas = fold_results[fold_key]
            all_deltas.extend(deltas)
            fold_means[fold_key] = np.mean(deltas)
        
        if not all_deltas:
            continue
        
        mean_delta = np.mean(all_deltas)
        std_delta = np.std(all_deltas)
        
        # Determine decision (initial threshold, to be refined in analysis)
        # ΔRMSE ≈ 0 means feature is not important
        decision = "remove" if abs(mean_delta) < 1e-5 and std_delta < 1e-5 else "keep"
        
        result_row = {
            "feature_name": feature_name,
            "mean_delta_rmse": mean_delta,
            "std_delta_rmse": std_delta,
        }
        
        # Add per-fold means
        for fold_idx in range(1, args.n_splits + 1):
            fold_key = f"fold_{fold_idx}"
            result_row[f"fold_{fold_idx}_delta"] = fold_means.get(fold_key, np.nan)
        
        result_row["decision"] = decision
        results_list.append(result_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results_list)
    
    # Sort by mean_delta_rmse (ascending - lowest impact first)
    results_df = results_df.sort_values("mean_delta_rmse", ascending=True)
    
    # Save results
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"[ok] Saved permutation results: {out_path}")
    print(f"[info] Results shape: {results_df.shape}")
    
    # Display summary
    print("\n[info] Permutation Importance Summary:")
    print(f"  Total features tested: {len(results_df)}")
    print(f"  Features marked 'remove' (initial threshold): {(results_df['decision'] == 'remove').sum()}")
    print(f"  Features marked 'keep' (initial threshold): {(results_df['decision'] == 'keep').sum()}")
    
    print("\n[info] Top 10 features by impact (highest ΔRMSE):")
    print(results_df.tail(10)[["feature_name", "mean_delta_rmse", "std_delta_rmse", "decision"]])
    
    print("\n[info] Bottom 10 features by impact (lowest ΔRMSE):")
    print(results_df.head(10)[["feature_name", "mean_delta_rmse", "std_delta_rmse", "decision"]])
    
    print("\n[ok] Permutation importance computation complete.")
    print("[info] Review the results to determine final threshold for feature deletion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
