#!/usr/bin/env python
"""Baseline evaluation script for Tier0 feature set.

This script evaluates the Tier0 baseline (SU1 + SU5 + Brushup) and outputs:
1. OOF RMSE and MSR metrics
2. Fold-wise feature importance (gain and split)
3. Aggregated importance statistics (mean, std, min, max)
4. Fold-wise evaluation logs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, cast

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LGBM = False

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
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

# Import MSR utilities
from scripts.utils_msr import evaluate_msr_proxy, grid_search_msr  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Evaluate Tier0 baseline and generate feature importance reports."
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
        "--out-dir",
        type=str,
        default="results/feature_selection",
        help="Output directory for results",
    )
    ap.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/tier1",
        help="Directory for tier artifacts (inference_bundle.pkl, feature_list.json, model_meta.json)",
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
        help="Path to JSON file containing features to exclude (from filter_trivial.py)",
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
        return pd.DataFrame(columns=pd.Index(["feature_name", "importance_gain", "importance_split", "fold"]))
    
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
        DataFrame with aggregated statistics (mean, std, min, max)
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
    
    # Sort by mean gain importance (descending)
    if "mean_gain" in grouped.columns:
        grouped = grouped.sort_values("mean_gain", ascending=False)
    
    return grouped


def evaluate_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate OOF predictions with RMSE and MSR metrics.
    
    Args:
        y_true: True target values
        y_pred: OOF predictions
        fold_logs: List of fold-wise metrics
        
    Returns:
        Dictionary with evaluation results
    """
    valid_mask = ~np.isnan(y_pred)
    
    if not valid_mask.any():
        return {
            "oof_rmse": float("nan"),
            "oof_coverage": 0.0,
            "oof_msr": float("nan"),
            "oof_msr_down": float("nan"),
            "fold_count": len(fold_logs),
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Calculate RMSE
    oof_rmse = float(math.sqrt(mean_squared_error(y_true_valid, y_pred_valid)))
    coverage = float(np.mean(valid_mask))
    
    # Calculate MSR using grid search
    best_params, _ = grid_search_msr(
        y_pred=y_pred_valid,
        y_true=y_true_valid,
        mult_grid=[0.5, 0.75, 1.0, 1.25, 1.5],
        lo_grid=[0.8, 0.9, 1.0],
        hi_grid=[1.0, 1.1, 1.2],
        eps=1e-8,
        optimize_for="msr",
    )
    
    metrics = evaluate_msr_proxy(
        y_pred_valid,
        y_true_valid,
        best_params,
        eps=1e-8,
    )
    
    return {
        "oof_rmse": oof_rmse,
        "oof_coverage": coverage,
        "oof_msr": float(metrics["msr"]),
        "oof_msr_down": float(metrics["msr_down"]),
        "oof_best_mult": float(best_params.mult),
        "oof_best_lo": float(best_params.lo),
        "oof_best_hi": float(best_params.hi),
        "fold_count": len(fold_logs),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    """Write rows to CSV file."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_exclude_features(exclude_path: str | Path) -> set[str]:
    """Load features to exclude from JSON file.
    
    Args:
        exclude_path: Path to JSON file with candidate features
        
    Returns:
        Set of feature names to exclude
    """
    path = Path(exclude_path)
    if not path.exists():
        raise FileNotFoundError(f"Exclude features file not found: {path}")
    
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    
    if not isinstance(data, dict) or "candidates" not in data:
        raise ValueError(f"Invalid exclude features file format: {path}")
    
    # Extract unique feature names from candidates
    exclude_set = set()
    for candidate in data["candidates"]:
        if isinstance(candidate, dict) and "feature_name" in candidate:
            exclude_set.add(candidate["feature_name"])
    
    return exclude_set


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    if not HAS_LGBM:
        print("[error] LightGBM is required but not installed.")
        print("[error] Please install with: pip install lightgbm")
        return 1
    
    # Setup paths
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Loading configs from {args.config_path}")
    
    # Load configs
    su1_config = load_su1_config(args.config_path)
    su5_config = load_su5_config(args.config_path)
    preprocess_settings = load_preprocess_policies(args.preprocess_config)
    
    # Load data
    train_path = infer_train_file(data_dir, args.train_file)
    test_path = infer_test_file(data_dir, args.test_file)
    print(f"[info] train file: {train_path}")
    print(f"[info] test file: {test_path}")
    
    train_df = load_table(train_path)
    test_df = load_table(test_path)
    
    # Sort by date_id
    if args.id_col in train_df.columns:
        train_df = train_df.sort_values(args.id_col).reset_index(drop=True)
    if args.id_col in test_df.columns:
        test_df = test_df.sort_values(args.id_col).reset_index(drop=True)
    
    if args.target_col not in train_df.columns:
        raise KeyError(f"Target column '{args.target_col}' not found in train data.")
    
    # Prepare features
    X, y, feature_cols = _prepare_features(
        train_df, test_df, target_col=args.target_col, id_col=args.id_col
    )
    
    print(f"[info] Pipeline input features: {len(feature_cols)}")
    
    # Build pipeline
    model_kwargs = {
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
    
    base_pipeline = build_pipeline(
        su1_config,
        su5_config,
        preprocess_settings,
        numeric_fill_value=args.numeric_fill_value,
        model_kwargs=model_kwargs,
        random_state=args.random_state,
    )
    
    # Setup CV
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X_np = X.reset_index(drop=True)
    y_np = y.reset_index(drop=True)
    y_np_array = y_np.to_numpy()
    
    # Pre-fit augmenter for CV
    su5_prefit = SU5FeatureAugmenter(su1_config, su5_config, fill_value=args.numeric_fill_value)
    su5_prefit.fit(X_np)
    
    # Build fold_indices array
    fold_indices_full = np.full(len(X_np), -1, dtype=int)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np)):
        fold_indices_full[train_idx] = fold_idx
        fold_indices_full[val_idx] = fold_idx
    
    # Transform with fold_indices
    X_augmented_all = su5_prefit.transform(X_np, fold_indices=fold_indices_full)
    
    # Load and apply feature exclusions if specified
    if args.exclude_features:
        exclude_set = _load_exclude_features(args.exclude_features)
        print(f"[info] Loaded {len(exclude_set)} features to exclude from {args.exclude_features}")
        
        # Filter out excluded features
        cols_to_keep = [col for col in X_augmented_all.columns if col not in exclude_set]
        cols_excluded = [col for col in X_augmented_all.columns if col in exclude_set]
        
        print(f"[info] Excluding {len(cols_excluded)} features")
        print(f"[info] Keeping {len(cols_to_keep)} features")
        
        if cols_excluded:
            print(f"[info] Sample excluded features: {cols_excluded[:5]}")
        
        X_augmented_all = X_augmented_all[cols_to_keep]
    
    # Get core pipeline (excluding augmenter which is the first step)
    # The augmenter is pre-fit separately, so we only need steps from index 1 onwards
    core_pipeline_template = cast(Pipeline, Pipeline(base_pipeline.steps[1:]))
    
    # Storage for results
    oof_pred = np.full(len(X_np), np.nan, dtype=float)
    fold_logs: List[Dict[str, Any]] = []
    importance_dfs: List[pd.DataFrame] = []
    
    print(f"[info] Starting {args.n_splits}-fold CV evaluation...")
    
    # CV loop
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np), start=1):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        
        # Apply gap if specified
        # Note: This follows the same gap logic as train_su5.py
        # Gap removes the last samples from training and first samples from validation
        # to prevent potential information leakage in time series data
        if args.gap > 0:
            if len(train_idx) > args.gap:
                train_idx = train_idx[:-args.gap]
            if len(val_idx) > args.gap:
                val_idx = val_idx[args.gap:]
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"[warn][fold {fold_idx}] skipped due to empty train/val")
            continue
        
        X_train = X_augmented_all.iloc[train_idx]
        y_train = y_np.iloc[train_idx]
        X_valid = X_augmented_all.iloc[val_idx]
        y_valid = y_np.iloc[val_idx]
        
        # Clone and fit pipeline
        pipe = cast(Pipeline, clone(core_pipeline_template))
        pipe.fit(X_train, y_train)
        
        # Predict
        pred = pipe.predict(X_valid)
        pred = np.asarray(pred).ravel().astype(float)
        
        # Calculate metrics
        rmse = float(math.sqrt(mean_squared_error(y_valid, pred)))
        oof_pred[val_idx] = pred
        
        # MSR evaluation
        best_params, _ = grid_search_msr(
            y_pred=pred,
            y_true=y_valid.to_numpy(),
            mult_grid=[0.5, 0.75, 1.0, 1.25, 1.5],
            lo_grid=[0.8, 0.9, 1.0],
            hi_grid=[1.0, 1.1, 1.2],
            eps=1e-8,
            optimize_for="msr",
        )
        
        fold_metrics = evaluate_msr_proxy(
            pred, y_valid.to_numpy(), best_params, eps=1e-8
        )
        
        # Log fold results
        fold_logs.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "rmse": rmse,
            "msr": float(fold_metrics["msr"]),
            "msr_down": float(fold_metrics["msr_down"]),
            "best_mult": float(best_params.mult),
            "best_lo": float(best_params.lo),
            "best_hi": float(best_params.hi),
        })
        
        print(f"[fold {fold_idx}] rmse={rmse:.6f} | msr={fold_metrics['msr']:.6f}")
        
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
                except Exception as e:
                    # Fallback: use generic names
                    n_features = len(model_step.feature_importances_)
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                    print(f"[warn][fold {fold_idx}] Failed to extract feature names: {e}")
                    print(f"[warn][fold {fold_idx}] Using generic feature names, which may make analysis harder")
            else:
                n_features = len(model_step.feature_importances_)
                feature_names = [f"feature_{i}" for i in range(n_features)]
                print(f"[warn][fold {fold_idx}] No preprocess step found, using generic feature names")
            
            fold_importance = compute_fold_importance(model_step, feature_names, fold_idx)
            importance_dfs.append(fold_importance)
    
    # Evaluate OOF
    print("[info] Evaluating OOF predictions...")
    oof_results = evaluate_oof(y_np_array, oof_pred, fold_logs)
    
    print(f"[metric][oof] rmse={oof_results['oof_rmse']:.6f}")
    print(f"[metric][oof] msr={oof_results['oof_msr']:.6f}")
    print(f"[metric][oof] coverage={oof_results['oof_coverage']:.2%}")
    
    # Determine output file prefix based on whether features were excluded
    file_prefix = "tier1" if args.exclude_features else "tier0"
    
    # Save evaluation results
    evaluation_path = out_dir / f"{file_prefix}_evaluation.json"
    with evaluation_path.open("w", encoding="utf-8") as fh:
        json.dump(oof_results, fh, indent=2, ensure_ascii=False)
    print(f"[ok] saved evaluation: {evaluation_path}")
    
    # Save fold logs
    if fold_logs:
        fold_logs_path = out_dir / f"{file_prefix}_fold_logs.csv"
        fieldnames = ["fold", "train_size", "val_size", "rmse", "msr", "msr_down",
                      "best_mult", "best_lo", "best_hi"]
        _write_csv(fold_logs_path, fold_logs, fieldnames=fieldnames)
        print(f"[ok] saved fold logs: {fold_logs_path}")
    
    # Save importance data
    if importance_dfs:
        # Concatenate all folds
        all_importance = pd.concat(importance_dfs, ignore_index=True)
        
        # Save fold-wise importance
        importance_path = out_dir / f"{file_prefix}_importance.csv"
        all_importance.to_csv(importance_path, index=False)
        print(f"[ok] saved importance: {importance_path}")
        print(f"[info] Total features tracked: {all_importance['feature_name'].nunique()}")
        
        # Aggregate importance
        summary = aggregate_importance(all_importance)
        summary_path = out_dir / f"{file_prefix}_importance_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[ok] saved importance summary: {summary_path}")
    else:
        print("[warn] No importance data collected.")
    
    # =========================================
    # Tier0 artifacts output
    # =========================================
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    print("[info] Training final model on full dataset for inference bundle...")
    
    # Get feature names from augmenter
    augmented_feature_names = list(X_augmented_all.columns)
    
    # Separate pipeline input columns (original) from generated columns
    original_cols = list(X_np.columns)
    su1_cols = [c for c in augmented_feature_names if c not in original_cols and "/" in c and not c.startswith("co_miss")]
    su5_cols = [c for c in augmented_feature_names if c.startswith("co_miss")]
    
    # Build and fit final pipeline on all data
    final_pipeline = cast(Pipeline, clone(core_pipeline_template))
    final_pipeline.fit(X_augmented_all, y_np)
    
    # Get model feature names after preprocessing
    model_step = final_pipeline.named_steps.get("model")
    preprocess_step = final_pipeline.named_steps.get("preprocess")
    model_feature_names = []
    if preprocess_step is not None and hasattr(preprocess_step, "get_feature_names_out"):
        try:
            model_feature_names = list(preprocess_step.get_feature_names_out())
        except Exception:
            pass
    
    # Create inference bundle that includes augmenter
    inference_bundle = {
        "augmenter": su5_prefit,
        "pipeline": final_pipeline,
        "feature_names": augmented_feature_names,
        "model_feature_names": model_feature_names,
    }
    
    # Save inference bundle
    bundle_path = artifacts_dir / "inference_bundle.pkl"
    joblib.dump(inference_bundle, bundle_path)
    print(f"[ok] saved inference bundle: {bundle_path}")
    
    # Get current commit hash
    try:
        import subprocess
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True
        ).strip()
        branch_name = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=str(PROJECT_ROOT),
            text=True
        ).strip()
    except Exception:
        commit_hash = "unknown"
        branch_name = "unknown"
    
    # Update feature_list.json
    feature_list = {
        "version": "tier0-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_input_columns": original_cols,
        "su1_generated_columns": su1_cols,
        "su5_generated_columns": su5_cols,
        "model_input_columns": model_feature_names if model_feature_names else augmented_feature_names,
        "total_feature_count": len(augmented_feature_names),
        "source_commit": commit_hash,
        "source_branch": branch_name,
    }
    
    feature_list_path = artifacts_dir / "feature_list.json"
    with feature_list_path.open("w", encoding="utf-8") as fh:
        json.dump(feature_list, fh, indent=2, ensure_ascii=False)
    print(f"[ok] saved feature list: {feature_list_path}")
    print(f"[info] Pipeline input: {len(original_cols)} cols, SU1: {len(su1_cols)} cols, SU5: {len(su5_cols)} cols")
    
    # Update model_meta.json
    model_meta = {
        "version": "tier0-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_settings": {
            "cv_strategy": "TimeSeriesSplit",
            "n_splits": args.n_splits,
            "gap": args.gap,
            "primary_metric": "RMSE",
            "secondary_metric": "MSR",
        },
        "model_params": {
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "random_state": args.random_state,
        },
        "metrics": {
            "oof_rmse": oof_results["oof_rmse"],
            "oof_msr": oof_results["oof_msr"],
            "oof_msr_down": oof_results["oof_msr_down"],
            "oof_coverage": oof_results["oof_coverage"],
        },
        "post_process_params": {
            "best_mult": oof_results.get("oof_best_mult"),
            "best_lo": oof_results.get("oof_best_lo"),
            "best_hi": oof_results.get("oof_best_hi"),
        },
        "source_commit": commit_hash,
        "source_branch": branch_name,
        "inference_bundle": str(bundle_path),
    }
    
    model_meta_path = artifacts_dir / "model_meta.json"
    with model_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(model_meta, fh, indent=2, ensure_ascii=False)
    print(f"[ok] saved model meta: {model_meta_path}")
    
    print("[ok] Baseline evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
