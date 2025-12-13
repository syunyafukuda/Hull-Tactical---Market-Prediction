#!/usr/bin/env python
"""Lasso training script using the unified model framework.

This script trains a Lasso model (L1 regularization) using the FS_compact feature set (116 features)
and provides OOF evaluation consistent with other model types.

Key features:
- Uses FS_compact feature exclusion (tier3/excluded.json)
- Reuses existing preprocessing pipeline (M/E/I/P/S group imputers)
- Includes StandardScaler for feature normalization (required for Lasso)
- Outputs coefficient diagnostics for feature importance analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
from src.preprocess.E_group.e_group import EGroupImputer  # noqa: E402
from src.preprocess.I_group.i_group import IGroupImputer  # noqa: E402
from src.preprocess.M_group.m_group import MGroupImputer  # noqa: E402
from src.preprocess.P_group.p_group import PGroupImputer  # noqa: E402
from src.preprocess.S_group.s_group import SGroupImputer  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Train Lasso with FS_compact features.")
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
        default="artifacts/models/lasso",
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
    # Lasso hyperparameters
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--max-iter", type=int, default=10000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument(
        "--selection", type=str, default="cyclic", choices=["cyclic", "random"]
    )
    ap.add_argument("--random-state", type=int, default=42)
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


def build_lasso_pipeline(
    su1_config: Any,
    su5_config: Any,
    preprocess_settings: Dict[str, Any],
    *,
    numeric_fill_value: float,
    model_kwargs: Dict[str, Any],
    random_state: int,
) -> Pipeline:
    """Build Lasso pipeline with SU1/SU5 augmenter, preprocessing, and StandardScaler.

    This is analogous to build_pipeline in train_su5.py but uses Lasso
    instead of tree-based models. StandardScaler is included for proper
    coefficient interpretation.
    """
    augmenter = SU5FeatureAugmenter(
        su1_config, su5_config, fill_value=numeric_fill_value
    )

    m_cfg = preprocess_settings.get("m_group", {})
    e_cfg = preprocess_settings.get("e_group", {})
    i_cfg = preprocess_settings.get("i_group", {})
    p_cfg = preprocess_settings.get("p_group", {})
    s_cfg = preprocess_settings.get("s_group", {})

    m_imputer = MGroupImputer(
        columns=None,
        policy=str(m_cfg.get("policy", "ffill_bfill")),
        rolling_window=int(m_cfg.get("rolling_window", 5)),
        ema_alpha=float(m_cfg.get("ema_alpha", 0.3)),
        calendar_column=m_cfg.get("calendar_column"),
        policy_params=m_cfg.get("policy_params", {}),
        random_state=random_state,
    )
    e_imputer = EGroupImputer(
        columns=None,
        policy=str(e_cfg.get("policy", "ffill_bfill")),
        rolling_window=int(e_cfg.get("rolling_window", 5)),
        ema_alpha=float(e_cfg.get("ema_alpha", 0.3)),
        calendar_column=e_cfg.get("calendar_column"),
        policy_params=e_cfg.get("policy_params", {}),
        random_state=random_state,
        all_nan_strategy=str(e_cfg.get("all_nan_strategy", "keep_nan")),
        all_nan_fill=float(e_cfg.get("all_nan_fill", 0.0)),
    )
    i_imputer = IGroupImputer(
        columns=None,
        policy=str(i_cfg.get("policy", "ffill_bfill")),
        rolling_window=int(i_cfg.get("rolling_window", 5)),
        ema_alpha=float(i_cfg.get("ema_alpha", 0.3)),
        calendar_column=i_cfg.get("calendar_column"),
        policy_params=i_cfg.get("policy_params", {}),
        random_state=random_state,
        clip_quantile_low=float(i_cfg.get("clip_quantile_low", 0.001)),
        clip_quantile_high=float(i_cfg.get("clip_quantile_high", 0.999)),
        enable_quantile_clip=bool(i_cfg.get("enable_quantile_clip", True)),
    )
    p_imputer = PGroupImputer(
        columns=None,
        policy=str(p_cfg.get("policy", "ffill_bfill")),
        rolling_window=int(p_cfg.get("rolling_window", 5)),
        ema_alpha=float(p_cfg.get("ema_alpha", 0.3)),
        calendar_column=p_cfg.get("calendar_column"),
        policy_params=p_cfg.get("policy_params", {}),
        random_state=random_state,
        mad_clip_scale=float(p_cfg.get("mad_clip_scale", 4.0)),
        mad_clip_min_samples=int(p_cfg.get("mad_clip_min_samples", 25)),
        enable_mad_clip=bool(p_cfg.get("enable_mad_clip", True)),
        fallback_quantile_low=float(p_cfg.get("fallback_quantile_low", 0.005)),
        fallback_quantile_high=float(p_cfg.get("fallback_quantile_high", 0.995)),
    )
    s_imputer = SGroupImputer(
        columns=None,
        policy=str(s_cfg.get("policy", "ffill_bfill")),
        rolling_window=int(s_cfg.get("rolling_window", 5)),
        ema_alpha=float(s_cfg.get("ema_alpha", 0.3)),
        calendar_column=s_cfg.get("calendar_column"),
        policy_params=s_cfg.get("policy_params", {}),
        random_state=random_state,
        mad_clip_scale=float(s_cfg.get("mad_clip_scale", 4.0)),
        mad_clip_min_samples=int(s_cfg.get("mad_clip_min_samples", 25)),
        enable_mad_clip=bool(s_cfg.get("enable_mad_clip", True)),
        fallback_quantile_low=float(s_cfg.get("fallback_quantile_low", 0.005)),
        fallback_quantile_high=float(s_cfg.get("fallback_quantile_high", 0.995)),
    )

    # Preprocessing: SimpleImputer for remaining NaNs
    final_imputer = SimpleImputer(strategy="constant", fill_value=numeric_fill_value)

    # StandardScaler for feature normalization (required for Lasso)
    scaler = StandardScaler()

    # Lasso model
    model = Lasso(**model_kwargs)

    steps = [
        ("augment", augmenter),
        ("m_imputer", m_imputer),
        ("e_imputer", e_imputer),
        ("i_imputer", i_imputer),
        ("p_imputer", p_imputer),
        ("s_imputer", s_imputer),
        ("final_imputer", final_imputer),
        ("scaler", scaler),
        ("model", model),
    ]
    return Pipeline(steps=steps)


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

    # Lasso model kwargs
    model_kwargs = {
        "alpha": args.alpha,
        "fit_intercept": True,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "selection": args.selection,
        "random_state": args.random_state,
    }

    # Build base pipeline with Lasso
    base_pipeline = build_lasso_pipeline(
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

    # Capture augmented columns before tier exclusion for feature_list.json
    augmented_columns_before_exclusion = X_augmented_all.columns.tolist()

    # Identify SU1/SU5 generated columns
    su1_generated_columns = [
        c for c in augmented_columns_before_exclusion if c not in feature_cols
    ]
    su5_generated_columns: List[
        str
    ] = []  # SU5 currently doesn't add new columns in this pipeline

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

    # Store original column names for feature_list.json
    original_columns = X_augmented_all.columns.tolist()

    # Extract imputers and scaler from base_pipeline for manual CV preprocessing
    # base_pipeline.steps: [augment, m_imputer, e_imputer, i_imputer, p_imputer, s_imputer, final_imputer, scaler, model]
    imputer_step_names = [
        "m_imputer",
        "e_imputer",
        "i_imputer",
        "p_imputer",
        "s_imputer",
        "final_imputer",
        "scaler",
    ]

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

        X_train = X_augmented_all.iloc[train_idx].copy()
        y_train = y_np.iloc[train_idx]
        X_valid = X_augmented_all.iloc[val_idx].copy()
        y_valid = y_np.iloc[val_idx]

        # Manual imputation and scaling: fit on train, transform both train and valid
        pipeline_steps = dict(base_pipeline.steps)
        for step_name in imputer_step_names:
            imputer = clone(pipeline_steps[step_name])
            X_train = imputer.fit_transform(X_train)  # type: ignore[union-attr]
            X_valid = imputer.transform(X_valid)  # type: ignore[union-attr]
            # Convert back to DataFrame if numpy array
            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train, columns=pd.Index(original_columns))
            if isinstance(X_valid, np.ndarray):
                X_valid = pd.DataFrame(X_valid, columns=pd.Index(original_columns))

        # Clone Lasso model
        lasso_model = clone(pipeline_steps["model"])

        # Fit
        lasso_model.fit(X_train, y_train)  # type: ignore[union-attr]

        # Predict
        pred = lasso_model.predict(X_valid)  # type: ignore[union-attr]
        pred = _to_1d(pred)

        # Compute metrics
        val_rmse = float(math.sqrt(mean_squared_error(y_valid, pred)))
        train_pred = lasso_model.predict(X_train)  # type: ignore[union-attr]
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
        final_pipeline = build_lasso_pipeline(
            su1_config,
            su5_config,
            preprocess_settings,
            numeric_fill_value=args.numeric_fill_value,
            model_kwargs=model_kwargs,
            random_state=args.random_state,
        )
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

        # Extract and save Lasso coefficients
        lasso_model_final = final_pipeline.named_steps["model"]
        coefficients = lasso_model_final.coef_
        nonzero_mask = coefficients != 0
        nonzero_count = np.sum(nonzero_mask)

        coefficients_df = pd.DataFrame(
            {
                "feature": [
                    original_columns[i]
                    for i in range(len(original_columns))
                    if nonzero_mask[i]
                ],
                "coefficient": coefficients[nonzero_mask],
            }
        ).sort_values("coefficient", key=abs, ascending=False)

        coef_path = out_dir / "coefficients.csv"
        coefficients_df.to_csv(coef_path, index=False)
        print(f"[info] Saved coefficients to {coef_path}")
        print(
            f"[info] Non-zero coefficients: {nonzero_count} / {len(original_columns)} ({100 * nonzero_count / len(original_columns):.1f}%)"
        )

        # Save metadata
        meta = {
            "model_type": "lasso",
            "feature_tier": args.feature_tier,
            "n_features": X_augmented_all.shape[1],
            "n_nonzero_coefficients": int(nonzero_count),
            "sparsity": float(nonzero_count / len(original_columns)),
            "oof_rmse": oof_rmse,
            "oof_msr": oof_metrics.get("oof_msr", float("nan")),
            "n_splits": args.n_splits,
            "gap": args.gap,
            "hyperparameters": model_kwargs,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # Fields required for Kaggle submission notebook
            "id_col": args.id_col,
            "target_col": args.target_col,
            "oof_best_params": {
                "mult": 1.0,
                "lo": 0.9,
                "hi": 1.1,
            },
        }
        meta_path = out_dir / "model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[info] Saved metadata to {meta_path}")

        # Save feature_list.json for Kaggle submission
        # Get git commit/branch info
        try:
            import subprocess

            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(PROJECT_ROOT),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            git_branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(PROJECT_ROOT),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            git_commit = "unknown"
            git_branch = "unknown"

        # Model input columns are the final columns after tier exclusion
        model_input_columns = original_columns

        feature_list = {
            "version": f"lasso-{args.feature_tier}-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_input_columns": sorted(feature_cols),
            "su1_generated_columns": sorted(
                [c for c in su1_generated_columns if c in model_input_columns]
            ),
            "su5_generated_columns": sorted(
                [c for c in su5_generated_columns if c in model_input_columns]
            ),
            "model_input_columns": model_input_columns,
            "total_feature_count": len(model_input_columns),
            "source_commit": git_commit,
            "source_branch": git_branch,
        }
        feature_list_path = out_dir / "feature_list.json"
        with open(feature_list_path, "w", encoding="utf-8") as f:
            json.dump(feature_list, f, indent=2)
        print(f"[info] Saved feature list to {feature_list_path}")

        # Generate test predictions and save submission.csv
        print("[info] Generating test predictions...")
        # Use only the feature columns that were used in training
        test_features = test_df[feature_cols].copy()
        test_pred = final_pipeline.predict(test_features)
        test_pred = _to_1d(test_pred)

        # Apply signal transformation: pred * mult + 1.0, clipped to [0.9, 1.1]
        # This converts excess returns predictions to competition signal format
        signal_mult = 1.0
        signal_lo = 0.9
        signal_hi = 1.1
        signal_pred = np.clip(test_pred * signal_mult + 1.0, signal_lo, signal_hi)

        # Filter to is_scored==True rows only (competition requirement)
        if "is_scored" in test_df.columns:
            is_scored_mask = test_df["is_scored"].astype(bool).to_numpy()
            signal_pred_scored = signal_pred[is_scored_mask]
            id_values = (
                test_df.loc[is_scored_mask, args.id_col].to_numpy()
                if args.id_col in test_df.columns
                else np.arange(len(signal_pred_scored))
            )
            print(f"[info] Filtered to {len(signal_pred_scored)} scored rows")
        else:
            signal_pred_scored = signal_pred
            id_values = (
                test_df[args.id_col].to_numpy()
                if args.id_col in test_df.columns
                else np.arange(len(signal_pred_scored))
            )

        # Build submission DataFrame with standard column names
        submission_df = pd.DataFrame(
            {
                args.id_col: id_values,
                "prediction": signal_pred_scored,
            }
        )

        submission_path = out_dir / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"[info] Saved submission to {submission_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
