#!/usr/bin/env python
"""Two-Head LightGBM training script.

This script trains TWO separate LightGBM models:
1. forward_model: predicts forward_returns
2. rf_model: predicts risk_free_rate

Then combines predictions using:
    position = clip((x - rf_pred) / (forward_pred - rf_pred), 0, 2)

The x parameter is optimized via grid search on OOF predictions.

Based on Kaggle discussion/608349.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    LGBMRegressor = None  # type: ignore
    lgb = None  # type: ignore
    HAS_LGBM = False

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
from src.models.common.feature_loader import (  # noqa: E402
    get_excluded_features,
)
from src.models.common.signals_two_head import (  # noqa: E402
    TwoHeadPositionConfig,
    compute_hull_sharpe_two_head,
    map_positions_from_forward_rf,
    optimize_x_parameter,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Train Two-Head LightGBM (forward + rf prediction)."
    )
    ap.add_argument(
        "--config", type=str, default="configs/evaluation/two_head.yaml",
        help="Path to two_head.yaml configuration",
    )
    ap.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Directory containing train/test files",
    )
    ap.add_argument(
        "--train-file", type=str, default=None,
        help="Explicit path to training file",
    )
    ap.add_argument(
        "--test-file", type=str, default=None,
        help="Explicit path to test file",
    )
    ap.add_argument(
        "--feature-config", type=str, default="configs/feature_generation/feature_generation.yaml",
        help="Path to feature_generation.yaml",
    )
    ap.add_argument(
        "--preprocess-config", type=str, default="configs/preprocess/preprocess.yaml",
        help="Path to preprocess.yaml",
    )
    ap.add_argument(
        "--id-col", type=str, default="date_id",
    )
    ap.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output directory (default: from config)",
    )
    ap.add_argument(
        "--numeric-fill-value", type=float, default=0.0,
    )
    # CV settings
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--min-val-size", type=int, default=0)
    # Feature selection
    ap.add_argument(
        "--feature-tier", type=str, default=None,
        help="Feature tier to use (default: from config)",
    )
    ap.add_argument(
        "--no-feature-exclusion", action="store_true",
        help="Skip feature exclusion (use all features)",
    )
    # Output control
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Run CV only, no final training")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def load_two_head_config(config_path: str) -> Dict[str, Any]:
    """Load two-head configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def _to_1d(pred: Any) -> np.ndarray:
    """Convert prediction to 1D numpy array."""
    array = np.asarray(pred)
    if array.ndim > 1:
        array = array.ravel()
    return array.astype(float, copy=False)


def _initialise_callbacks(model: Any) -> List[Any]:
    """Initialize LightGBM training callbacks."""
    callbacks: List[Any] = []
    if not HAS_LGBM or lgb is None:
        return callbacks
    
    n_estimators = int(model.get_params().get("n_estimators", 100))
    log_period = max(1, n_estimators // 10)
    
    try:
        callbacks.append(lgb.log_evaluation(period=log_period))
    except Exception:
        pass
    
    return callbacks


def apply_feature_exclusion_to_augmented(
    df: pd.DataFrame,
    tier: str = "tier3",
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply feature exclusion to augmented DataFrame."""
    excluded = get_excluded_features(tier)
    columns_to_drop = [col for col in df.columns if col in excluded]
    
    if verbose and columns_to_drop:
        print(f"[info] Excluding {len(columns_to_drop)} features based on {tier}")
    
    return df.drop(columns=columns_to_drop, errors="ignore")


def train_single_head(
    X_augmented_all: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    splitter: TimeSeriesSplit,
    core_pipeline_template: Pipeline,
    callbacks: List[Any],
    gap: int = 0,
    min_val_size: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Train a single head and return OOF predictions.
    
    Parameters
    ----------
    X_augmented_all : pd.DataFrame
        Augmented features for all data.
    y : pd.Series
        Target values.
    target_name : str
        Name of target for logging.
    splitter : TimeSeriesSplit
        CV splitter.
    core_pipeline_template : Pipeline
        Template pipeline to clone.
    callbacks : list
        LightGBM callbacks.
    gap : int
        Gap between train and val.
    min_val_size : int
        Minimum validation size.
    verbose : bool
        Print progress.
        
    Returns
    -------
    tuple
        (oof_predictions, fold_logs)
    """
    X_np = X_augmented_all.reset_index(drop=True)
    y_np = y.reset_index(drop=True)
    
    oof_pred = np.full(len(X_np), np.nan, dtype=float)
    fold_logs: List[Dict[str, Any]] = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_np), start=1):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        
        # Apply gap
        if gap > 0:
            if len(train_idx) > gap:
                train_idx = train_idx[:-gap]
            if len(val_idx) > gap:
                val_idx = val_idx[gap:]
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            if verbose:
                print(f"[{target_name}][fold {fold_idx}] skipped: empty")
            continue
        if min_val_size and len(val_idx) < min_val_size:
            if verbose:
                print(f"[{target_name}][fold {fold_idx}] skipped: val < min")
            continue
        
        X_train = X_np.iloc[train_idx]
        y_train = y_np.iloc[train_idx]
        X_valid = X_np.iloc[val_idx]
        y_valid = y_np.iloc[val_idx]
        
        # Clone and fit
        pipe = cast(Pipeline, clone(core_pipeline_template))
        fit_kwargs: Dict[str, Any] = {}
        if callbacks:
            fit_kwargs["model__callbacks"] = callbacks
            fit_kwargs["model__eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["model__eval_metric"] = "rmse"
        
        pipe.fit(X_train, y_train, **fit_kwargs)
        
        # Predict
        pred = pipe.predict(X_valid)
        pred = _to_1d(pred)
        oof_pred[val_idx] = pred
        
        # Metrics
        val_rmse = float(math.sqrt(mean_squared_error(y_valid, pred)))
        train_pred = pipe.predict(X_train)
        train_rmse = float(math.sqrt(mean_squared_error(y_train, _to_1d(train_pred))))
        
        fold_logs.append({
            "target": target_name,
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })
        
        if verbose:
            print(
                f"[{target_name}][fold {fold_idx}] "
                f"train_rmse={train_rmse:.6f} | val_rmse={val_rmse:.6f}"
            )
    
    return oof_pred, fold_logs


def main(argv: Sequence[str] | None = None) -> int:
    """Main training function."""
    args = parse_args(argv)
    
    if not HAS_LGBM:
        print("[error] LightGBM is not installed")
        return 1
    
    # Load two-head config
    print(f"[info] Loading config from {args.config}")
    config = load_two_head_config(args.config)
    
    two_head_cfg = config.get("two_head", {})
    model_cfg = config.get("model", {})
    features_cfg = config.get("features", {})
    output_cfg = config.get("output", {})
    
    # Target columns
    forward_col = two_head_cfg.get("targets", {}).get("forward", "forward_returns")
    rf_col = two_head_cfg.get("targets", {}).get("rf", "risk_free_rate")
    
    # Position mapping config
    pos_cfg = two_head_cfg.get("position_mapping", {})
    clip_min = pos_cfg.get("clip_min", 0.0)
    clip_max = pos_cfg.get("clip_max", 2.0)
    
    # x optimization grid
    x_opt_cfg = two_head_cfg.get("x_optimization", {})
    x_min = x_opt_cfg.get("min", -0.002)
    x_max = x_opt_cfg.get("max", 0.002)
    x_steps = x_opt_cfg.get("steps", 41)
    x_grid = np.linspace(x_min, x_max, x_steps)
    
    # Feature tier
    feature_tier = args.feature_tier or features_cfg.get("tier", "tier3")
    
    # Output directory
    out_dir = Path(args.out_dir or output_cfg.get("artifacts_dir", "artifacts/models/lgbm-two-head"))
    if not args.no_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    
    # Load configurations
    su1_config = load_su1_config(args.feature_config)
    su5_config = load_su5_config(args.feature_config)
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
    
    # Verify target columns exist
    for col in [forward_col, rf_col]:
        if col not in train_df.columns:
            raise KeyError(f"Target column '{col}' not found in training data")
    
    print(f"[info] forward_returns column: {forward_col}")
    print(f"[info] risk_free_rate column : {rf_col}")
    
    # Prepare features using market_forward_excess_returns as dummy target
    # We'll replace the target later for each head
    X, _, feature_cols = _prepare_features(
        train_df, test_df,
        target_col="market_forward_excess_returns",
        id_col=args.id_col,
    )
    
    # Extract true target values
    y_forward = train_df[forward_col].copy()
    y_rf = train_df[rf_col].copy()
    
    # Model hyperparameters
    model_params = model_cfg.get("params", {})
    model_kwargs = {
        "learning_rate": model_params.get("learning_rate", 0.05),
        "n_estimators": model_params.get("n_estimators", 600),
        "num_leaves": model_params.get("num_leaves", 63),
        "min_data_in_leaf": model_params.get("min_data_in_leaf", 32),
        "feature_fraction": model_params.get("feature_fraction", 0.9),
        "bagging_fraction": model_params.get("bagging_fraction", 0.9),
        "bagging_freq": model_params.get("bagging_freq", 1),
        "random_state": model_params.get("random_state", 42),
        "n_jobs": model_params.get("n_jobs", -1),
        "verbosity": model_params.get("verbosity", -1),
    }
    
    # Build base pipeline
    base_pipeline = build_pipeline(
        su1_config,
        su5_config,
        preprocess_settings,
        numeric_fill_value=args.numeric_fill_value,
        model_kwargs=model_kwargs,
        random_state=model_kwargs.get("random_state", 42),
    )
    callbacks = _initialise_callbacks(base_pipeline.named_steps["model"])
    
    # CV setup
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X_np = X.reset_index(drop=True)
    
    # Pre-fit augmenter for CV
    print("[info] Pre-fitting SU1/SU5 feature augmenter...")
    su5_prefit = SU5FeatureAugmenter(
        su1_config, su5_config,
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
    
    # Apply feature exclusion
    if not args.no_feature_exclusion:
        print(f"[info] Applying {feature_tier} feature exclusion...")
        n_before = X_augmented_all.shape[1]
        X_augmented_all = apply_feature_exclusion_to_augmented(
            X_augmented_all, tier=feature_tier, verbose=True
        )
        n_after = X_augmented_all.shape[1]
        print(f"[info] Features: {n_before} -> {n_after}")
    
    # Build core pipeline template
    core_pipeline_template = cast(Pipeline, Pipeline(base_pipeline.steps[1:]))
    
    # ========================================
    # Train Head 1: forward_returns
    # ========================================
    print(f"\n{'='*60}")
    print(f"Training HEAD 1: {forward_col}")
    print(f"{'='*60}\n")
    
    oof_forward, logs_forward = train_single_head(
        X_augmented_all,
        y_forward,
        target_name="forward",
        splitter=splitter,
        core_pipeline_template=core_pipeline_template,
        callbacks=callbacks,
        gap=args.gap,
        min_val_size=args.min_val_size,
        verbose=True,
    )
    
    # ========================================
    # Train Head 2: risk_free_rate
    # ========================================
    print(f"\n{'='*60}")
    print(f"Training HEAD 2: {rf_col}")
    print(f"{'='*60}\n")
    
    oof_rf, logs_rf = train_single_head(
        X_augmented_all,
        y_rf,
        target_name="rf",
        splitter=splitter,
        core_pipeline_template=core_pipeline_template,
        callbacks=callbacks,
        gap=args.gap,
        min_val_size=args.min_val_size,
        verbose=True,
    )
    
    # ========================================
    # Optimize x parameter
    # ========================================
    print(f"\n{'='*60}")
    print(f"Optimizing x parameter")
    print(f"{'='*60}\n")
    
    # Use only valid OOF indices
    valid_mask = ~np.isnan(oof_forward) & ~np.isnan(oof_rf)
    print(f"[info] Valid OOF samples: {valid_mask.sum()} / {len(valid_mask)}")
    
    # Get true values
    forward_true = y_forward.to_numpy()
    rf_true = y_rf.to_numpy()
    
    best_x, best_hull_sharpe, x_results = optimize_x_parameter(
        forward_oof=oof_forward[valid_mask],
        rf_oof=oof_rf[valid_mask],
        forward_true=forward_true[valid_mask],
        rf_true=rf_true[valid_mask],
        x_grid=x_grid,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    
    print(f"[result] Best x = {best_x:.6f}")
    print(f"[result] Best Hull Sharpe = {best_hull_sharpe:.4f}")
    
    # Compute positions with best x
    final_positions = map_positions_from_forward_rf(
        oof_forward[valid_mask],
        oof_rf[valid_mask],
        x=best_x,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    
    # Final Hull Sharpe metrics
    final_metrics = compute_hull_sharpe_two_head(
        final_positions,
        forward_true[valid_mask],
        rf_true[valid_mask],
    )
    
    print(f"\n{'='*60}")
    print(f"Final OOF Hull Sharpe Metrics (x={best_x:.6f})")
    print(f"{'='*60}")
    print(f"  Hull Sharpe   : {final_metrics['hull_sharpe']:.4f}")
    print(f"  Raw Sharpe    : {final_metrics['raw_sharpe']:.4f}")
    print(f"  Vol Ratio     : {final_metrics['vol_ratio']:.4f}")
    print(f"  Vol Penalty   : {final_metrics['vol_penalty']:.4f}")
    print(f"  Return Penalty: {final_metrics['return_penalty']:.4f}")
    print(f"  Mean Position : {final_metrics['mean_position']:.4f}")
    print(f"  Std Position  : {final_metrics['std_position']:.4f}")
    print(f"{'='*60}\n")
    
    # ========================================
    # Save artifacts
    # ========================================
    if not args.no_artifacts and not args.dry_run:
        print("[info] Training final models on all data...")
        
        # Prepare final training data (X_augmented_all with tier3 exclusion applied)
        # Drop target columns from X_augmented_all
        drop_cols_for_model = [forward_col, rf_col, "market_forward_excess_returns", "date_id"]
        X_final_train = X_augmented_all.drop(
            columns=[c for c in drop_cols_for_model if c in X_augmented_all.columns],
            errors="ignore"
        )
        print(f"[info] Final training features: {X_final_train.shape[1]}")
        
        # Train final forward model using core_pipeline (no augmenter)
        # This ensures tier3 exclusion is applied consistently with CV
        final_forward_pipeline = cast(Pipeline, clone(core_pipeline_template))
        fit_kwargs_final: Dict[str, Any] = {}
        if callbacks:
            fit_kwargs_final["model__callbacks"] = callbacks
            fit_kwargs_final["model__eval_metric"] = "rmse"
        final_forward_pipeline.fit(X_final_train, y_forward, **fit_kwargs_final)
        
        # Train final rf model
        final_rf_pipeline = cast(Pipeline, clone(core_pipeline_template))
        final_rf_pipeline.fit(X_final_train, y_rf, **fit_kwargs_final)
        
        # Save models (core_pipeline without augmenter)
        forward_model_path = out_dir / "forward_model.pkl"
        rf_model_path = out_dir / "rf_model.pkl"
        joblib.dump(final_forward_pipeline, forward_model_path)
        joblib.dump(final_rf_pipeline, rf_model_path)
        print(f"[info] Saved forward model to {forward_model_path}")
        print(f"[info] Saved rf model to {rf_model_path}")
        
        # Save augmenter separately for inference
        augmenter_path = out_dir / "augmenter.pkl"
        joblib.dump(su5_prefit, augmenter_path)
        print(f"[info] Saved augmenter to {augmenter_path}")
        
        # Save excluded features list for inference
        excluded_features = list(get_excluded_features(feature_tier))
        excluded_path = out_dir / "excluded_features.json"
        with open(excluded_path, "w", encoding="utf-8") as f:
            json.dump({"tier": feature_tier, "excluded": excluded_features}, f, indent=2)
        print(f"[info] Saved excluded features to {excluded_path}")
        
        # Save position config with optimized x
        position_config = TwoHeadPositionConfig(
            x=best_x,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        config_path = out_dir / "position_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(position_config.to_dict(), f, indent=2)
        print(f"[info] Saved position config to {config_path}")
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            "index": np.arange(len(oof_forward)),
            "forward_pred": oof_forward,
            "rf_pred": oof_rf,
            "forward_true": forward_true,
            "rf_true": rf_true,
        })
        # Add positions for valid samples
        oof_df["position"] = np.nan
        oof_df.loc[valid_mask, "position"] = final_positions
        
        oof_path = out_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f"[info] Saved OOF predictions to {oof_path}")
        
        # Save fold logs
        all_logs = logs_forward + logs_rf
        fold_log_path = out_dir / "cv_fold_logs.csv"
        if all_logs:
            fieldnames = list(all_logs[0].keys())
            with open(fold_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_logs)
            print(f"[info] Saved fold logs to {fold_log_path}")
        
        # Save x optimization results
        x_results_path = out_dir / "x_optimization.csv"
        if x_results:
            fieldnames = list(x_results[0].keys())
            with open(x_results_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(x_results)
            print(f"[info] Saved x optimization results to {x_results_path}")
        
        # Save metadata
        meta = {
            "model_type": "lightgbm-two-head",
            "feature_tier": feature_tier,
            "n_features": X_augmented_all.shape[1],
            "targets": {
                "forward": forward_col,
                "rf": rf_col,
            },
            "best_x": best_x,
            "hull_sharpe": final_metrics["hull_sharpe"],
            "raw_sharpe": final_metrics["raw_sharpe"],
            "vol_ratio": final_metrics["vol_ratio"],
            "vol_penalty": final_metrics["vol_penalty"],
            "return_penalty": final_metrics["return_penalty"],
            "mean_position": final_metrics["mean_position"],
            "n_splits": args.n_splits,
            "gap": args.gap,
            "hyperparameters": model_kwargs,
            "x_optimization": {
                "min": float(x_min),
                "max": float(x_max),
                "steps": x_steps,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = out_dir / "model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[info] Saved metadata to {meta_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
