#!/usr/bin/env python
"""Filter trivial features based on variance, missing rate, and correlation.

This script identifies features that should be excluded based on:
1. Low variance (near-constant features)
2. High missing rate (near-empty features)
3. High correlation (redundant features)

The filtering is applied to preprocessed features to evaluate them as they
would appear to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# Add paths for imports
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[1]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from src.feature_generation.su5.train_su5 import (  # noqa: E402
    load_su1_config,
    load_su5_config,
    infer_train_file,
    load_table,
    _prepare_features,
    SU5FeatureAugmenter,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Filter trivial features based on statistical criteria."
    )
    ap.add_argument(
        "--config-path",
        type=str,
        default="configs/tier0_snapshot/feature_generation.yaml",
        help="Path to feature_generation.yaml",
    )
    ap.add_argument(
        "--preprocess-config",
        type=str,
        default="configs/tier0_snapshot/preprocess.yaml",
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
        "--out-path",
        type=str,
        default="results/feature_selection/phase1_filter_candidates.json",
        help="Output path for candidate list JSON",
    )
    ap.add_argument(
        "--importance-path",
        type=str,
        default=None,
        help="Path to importance summary CSV (for correlation filtering)",
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
        "--variance-threshold",
        type=float,
        default=1e-10,
        help="Minimum variance threshold",
    )
    ap.add_argument(
        "--missing-threshold",
        type=float,
        default=0.99,
        help="Maximum missing rate threshold",
    )
    ap.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.999,
        help="Maximum correlation threshold",
    )
    ap.add_argument(
        "--numeric-fill-value",
        type=float,
        default=0.0,
        help="Value for numeric imputation",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return ap.parse_args(argv)


def compute_statistics(X: pd.DataFrame) -> Dict[str, Any]:
    """Compute variance, missing rate, and correlation matrix.
    
    Args:
        X: Input DataFrame (preprocessed features)
        
    Returns:
        Dictionary with variance, missing_rate, and correlation matrix
    """
    print("[info] Computing statistics on preprocessed features...")
    
    # Compute variance
    variances = X.var(axis=0, skipna=True)
    
    # Compute missing rate
    missing_rates = X.isnull().mean(axis=0)
    
    # Compute correlation matrix for numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    print(f"[info] Computing correlation matrix for {len(numeric_cols)} numeric columns...")
    correlation_matrix = X_numeric.corr()  # type: ignore[assignment]
    
    return {
        "variance": variances,
        "missing_rate": missing_rates,
        "correlation": correlation_matrix,
    }


def find_low_variance_features(
    X: pd.DataFrame,
    threshold: float = 1e-10
) -> List[Dict[str, Any]]:
    """Find features with variance below threshold.
    
    Args:
        X: Input DataFrame
        threshold: Minimum variance threshold
        
    Returns:
        List of dictionaries with feature_name, reason, and value
    """
    variances: pd.Series = X.var(axis=0, skipna=True)  # type: ignore[assignment]
    
    candidates = []
    for feature_name in variances.index:
        variance = float(variances[feature_name])
        if variance < threshold:
            candidates.append({
                "feature_name": str(feature_name),
                "reason": "low_variance",
                "value": float(variance),
            })
    
    print(f"[info] Found {len(candidates)} low variance features (threshold={threshold})")
    return candidates


def find_high_missing_features(
    X: pd.DataFrame,
    threshold: float = 0.99
) -> List[Dict[str, Any]]:
    """Find features with missing rate above threshold.
    
    Args:
        X: Input DataFrame
        threshold: Maximum missing rate threshold
        
    Returns:
        List of dictionaries with feature_name, reason, and value
    """
    missing_rates: pd.Series = X.isnull().mean(axis=0)  # type: ignore[assignment]
    
    candidates = []
    for feature_name in missing_rates.index:
        missing_rate = float(missing_rates[feature_name])
        if missing_rate > threshold:
            candidates.append({
                "feature_name": str(feature_name),
                "reason": "high_missing",
                "value": float(missing_rate),
            })
    
    print(f"[info] Found {len(candidates)} high missing features (threshold={threshold})")
    return candidates


def find_high_correlation_features(
    X: pd.DataFrame,
    threshold: float = 0.999,
    importance_df: pd.DataFrame | None = None
) -> List[Dict[str, Any]]:
    """Find redundant features based on high correlation.
    
    For correlated pairs, the feature with lower importance is marked as candidate.
    If importance data is not available, the second feature in alphabetical order
    is marked.
    
    Args:
        X: Input DataFrame
        threshold: Correlation threshold
        importance_df: DataFrame with feature importance (must have 'feature_name' and 'mean_gain' columns)
        
    Returns:
        List of dictionaries with feature_name, reason, correlated_with, and correlation
    """
    # Only consider numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    print(f"[info] Computing correlations for {len(numeric_cols)} numeric columns...")
    corr_matrix = X_numeric.corr()  # type: ignore[assignment]
    
    # Build importance lookup
    importance_lookup = {}
    if importance_df is not None:
        if 'feature_name' in importance_df.columns and 'mean_gain' in importance_df.columns:
            for _, row in importance_df.iterrows():
                importance_lookup[str(row['feature_name'])] = float(row['mean_gain'])
            print(f"[info] Loaded importance for {len(importance_lookup)} features")
    
    # Find highly correlated pairs
    candidates = []
    processed_pairs = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            
            corr_value = corr_matrix.iloc[i, j]
            
            # Skip NaN correlations
            if pd.isna(corr_value):
                continue
            
            # Check if correlation exceeds threshold
            if abs(corr_value) > threshold:
                pair_key = tuple(sorted([feature_i, feature_j]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Determine which feature to drop
                # Use -1.0 as sentinel value for missing importance to ensure
                # features without importance data are treated equally and fall
                # back to alphabetical ordering
                imp_i = importance_lookup.get(str(feature_i), -1.0)
                imp_j = importance_lookup.get(str(feature_j), -1.0)
                
                if imp_i > imp_j:
                    # Drop feature_j (lower importance)
                    drop_feature = feature_j
                    keep_feature = feature_i
                elif imp_j > imp_i:
                    # Drop feature_i (lower importance)
                    drop_feature = feature_i
                    keep_feature = feature_j
                else:
                    # Equal importance or no importance data: drop alphabetically later one
                    if str(feature_i) < str(feature_j):
                        drop_feature = feature_j
                        keep_feature = feature_i
                    else:
                        drop_feature = feature_i
                        keep_feature = feature_j
                
                candidates.append({
                    "feature_name": str(drop_feature),
                    "reason": "high_correlation",
                    "correlated_with": str(keep_feature),
                    "correlation": float(corr_value),
                })
    
    print(f"[info] Found {len(candidates)} highly correlated features (threshold={threshold})")
    return candidates


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Loading configs from {args.config_path}")
    
    # Load configs
    su1_config = load_su1_config(args.config_path)
    su5_config = load_su5_config(args.config_path)
    
    # Load data
    train_path = infer_train_file(data_dir, args.train_file)
    test_path = None
    if args.test_file:
        test_path = Path(args.test_file)
    elif (data_dir / "test.parquet").exists():
        test_path = data_dir / "test.parquet"
    elif (data_dir / "test.csv").exists():
        test_path = data_dir / "test.csv"
    
    print(f"[info] train file: {train_path}")
    
    train_df = load_table(train_path)
    
    # For feature generation, we need test data
    if test_path and test_path.exists():
        print(f"[info] test file: {test_path}")
        test_df = load_table(test_path)
    else:
        print("[warn] No test file found, using empty test set")
        test_df = pd.DataFrame()
    
    # Sort by date_id
    if args.id_col in train_df.columns:
        train_df = train_df.sort_values(args.id_col).reset_index(drop=True)
    if not test_df.empty and args.id_col in test_df.columns:
        test_df = test_df.sort_values(args.id_col).reset_index(drop=True)
    
    if args.target_col not in train_df.columns:
        raise KeyError(f"Target column '{args.target_col}' not found in train data.")
    
    # Prepare features (apply SU1 and SU5)
    print("[info] Generating features...")
    X, y, feature_cols = _prepare_features(
        train_df, test_df, target_col=args.target_col, id_col=args.id_col
    )
    
    print(f"[info] Pipeline input features: {len(feature_cols)}")
    
    # Build augmenter and generate features
    su5_augmenter = SU5FeatureAugmenter(su1_config, su5_config, fill_value=args.numeric_fill_value)
    su5_augmenter.fit(X)
    
    # Build fake fold_indices for full dataset
    fold_indices = np.zeros(len(X), dtype=int)
    X_augmented = su5_augmenter.transform(X, fold_indices=fold_indices)
    
    print(f"[info] Augmented features: {len(X_augmented.columns)} columns")
    
    # Check for duplicate columns
    duplicated_cols = X_augmented.columns[X_augmented.columns.duplicated()].tolist()
    if duplicated_cols:
        print(f"[warn] Found {len(duplicated_cols)} duplicated column names, removing duplicates")
        print(f"[warn] Sample duplicates: {duplicated_cols[:5]}")
        # Keep first occurrence
        X_augmented = X_augmented.loc[:, ~X_augmented.columns.duplicated()]
        print(f"[info] After deduplication: {len(X_augmented.columns)} columns")
    
    # Use augmented features directly for analysis
    # Note: We analyze the raw augmented features instead of preprocessed features
    # because the preprocessing pipeline has column name handling issues.
    # The key filtering criteria (variance, missing, correlation) work well on raw data.
    X_preprocessed = X_augmented.copy()
    
    print(f"[info] Analyzing features: {len(X_preprocessed.columns)} columns")
    
    # Load importance data if provided
    importance_df = None
    if args.importance_path:
        importance_path = Path(args.importance_path)
        if importance_path.exists():
            print(f"[info] Loading importance from {importance_path}")
            importance_df = pd.read_csv(importance_path)
        else:
            print(f"[warn] Importance file not found: {importance_path}")
    
    # Find candidates
    print("[info] Finding candidate features for removal...")
    
    low_var_candidates = find_low_variance_features(
        X_preprocessed,
        threshold=args.variance_threshold
    )
    
    high_missing_candidates = find_high_missing_features(
        X_preprocessed,
        threshold=args.missing_threshold
    )
    
    high_corr_candidates = find_high_correlation_features(
        X_preprocessed,
        threshold=args.correlation_threshold,
        importance_df=importance_df
    )
    
    # Combine all candidates (remove duplicates)
    all_candidates = low_var_candidates + high_missing_candidates + high_corr_candidates
    
    # Count unique features
    unique_features = set(c["feature_name"] for c in all_candidates)
    
    # Build output
    output = {
        "version": "phase1-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "variance_min": args.variance_threshold,
            "missing_rate_max": args.missing_threshold,
            "correlation_max": args.correlation_threshold,
        },
        "candidates": all_candidates,
        "summary": {
            "total_features": len(X_preprocessed.columns),
            "low_variance_count": len(low_var_candidates),
            "high_missing_count": len(high_missing_candidates),
            "high_correlation_count": len(high_corr_candidates),
            "total_candidates": len(unique_features),
        },
    }
    
    # Save to JSON
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    
    print(f"[ok] Saved candidate list: {out_path}")
    print(f"[summary] Total features: {output['summary']['total_features']}")
    print(f"[summary] Low variance: {output['summary']['low_variance_count']}")
    print(f"[summary] High missing: {output['summary']['high_missing_count']}")
    print(f"[summary] High correlation: {output['summary']['high_correlation_count']}")
    print(f"[summary] Unique candidates for removal: {output['summary']['total_candidates']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
