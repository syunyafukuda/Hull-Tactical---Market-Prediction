#!/usr/bin/env python
"""LightGBM training script with Hull Sharpe × Walk-Forward CV evaluation.

This script trains a LightGBM model using the FS_compact feature set (116 features)
and provides OOF evaluation with both RMSE and Hull Competition Sharpe metrics.

Key features:
- Uses FS_compact feature exclusion (tier3/excluded.json)
- Reuses existing preprocessing pipeline (M/E/I/P/S group imputers)
- Compatible with the unified CV evaluation framework
- Supports Walk-Forward CV for time-series validation
- Computes official Hull Competition Sharpe metric
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import joblib
import numpy as np
import pandas as pd
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
from src.models.common.cv_utils import (  # noqa: E402
    compute_fold_metrics,
    evaluate_oof_predictions,
)
from src.models.common.feature_loader import (  # noqa: E402
    get_excluded_features,
)
from src.models.common.signals import (  # noqa: E402
    AlphaBetaPositionConfig,
    map_predictions_to_positions,
)
from src.models.common.walk_forward import (  # noqa: E402
    WalkForwardConfig,
    make_walk_forward_splits,
)
from src.metrics.hull_sharpe import (  # noqa: E402
    compute_hull_sharpe,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Train LightGBM with FS_compact features.")
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
        default="artifacts/models/lgbm",
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
    ap.add_argument(
        "--cv-mode",
        type=str,
        default="timeseries_split",
        choices=["timeseries_split", "walk_forward"],
        help="CV strategy: timeseries_split or walk_forward",
    )
    # Walk-Forward settings
    ap.add_argument("--wf-train-window", type=int, default=6000)
    ap.add_argument("--wf-val-window", type=int, default=1000)
    ap.add_argument("--wf-step", type=int, default=1000)
    ap.add_argument(
        "--wf-mode",
        type=str,
        default="expanding",
        choices=["expanding", "rolling"],
    )
    # Hull Sharpe evaluation
    ap.add_argument(
        "--eval-sharpe",
        action="store_true",
        help="Compute Hull Competition Sharpe in CV",
    )
    # Legacy args (deprecated, use alpha/beta instead)
    ap.add_argument(
        "--sharpe-mult",
        type=float,
        default=100.0,
        help="[DEPRECATED] Use --alpha instead. Multiplier for prediction->position mapping",
    )
    ap.add_argument(
        "--sharpe-offset",
        type=float,
        default=1.0,
        help="[DEPRECATED] Use --beta instead. Offset for prediction->position mapping",
    )
    # Alpha-Beta position mapping (preferred)
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha (scale) for alpha-beta position mapping. Overrides --sharpe-mult.",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta (offset) for alpha-beta position mapping. Overrides --sharpe-offset.",
    )
    ap.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Minimum position for clipping (default: 0.0 = 100%% cash)",
    )
    ap.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Maximum position for clipping (default: 2.0 = 200%% market)",
    )
    ap.add_argument(
        "--winsor-pct",
        type=float,
        default=None,
        help="Winsorize predictions at this percentile (e.g., 0.01 = 1st/99th)",
    )
    ap.add_argument(
        "--position-mapping-config",
        type=str,
        default=None,
        help="Path to walk_forward.yaml for position_mapping section",
    )
    # Model hyperparameters
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--min-data-in-leaf", type=int, default=32)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--verbosity", type=int, default=-1)
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


def _get_forward_returns_and_rf(
    train_df: pd.DataFrame,
    val_indices: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract forward_returns and risk_free_rate for Hull Sharpe calculation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Original training DataFrame.
    val_indices : np.ndarray
        Indices into train_df for validation set.
    y_true : np.ndarray
        Actual values (fallback for forward_returns).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (forward_returns, risk_free_rate)
    """
    # Get forward_returns
    if "sp500_forward_returns" in train_df.columns:
        forward_returns = train_df.iloc[val_indices]["sp500_forward_returns"].to_numpy()
    else:
        forward_returns = y_true  # fallback to target

    # Get risk_free_rate
    if "federal_funds_rate" in train_df.columns:
        risk_free_rate = train_df.iloc[val_indices]["federal_funds_rate"].to_numpy()
    else:
        # Use approximate daily rate (~4% annual)
        risk_free_rate = np.full(len(val_indices), 0.04 / 252)

    return forward_returns, risk_free_rate


def _compute_sharpe_for_fold(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    train_df: pd.DataFrame,
    val_indices: np.ndarray,
    position_config: AlphaBetaPositionConfig,
) -> Dict[str, float]:
    """Compute Hull Competition Sharpe for a CV fold.

    Parameters
    ----------
    y_pred : np.ndarray
        Model predictions (excess returns).
    y_true : np.ndarray
        Actual values.
    train_df : pd.DataFrame
        Original training DataFrame with forward_returns and rf columns.
    val_indices : np.ndarray
        Indices into train_df for validation set.
    position_config : AlphaBetaPositionConfig
        Position mapping configuration (alpha, beta, clip, winsor).

    Returns
    -------
    Dict[str, float]
        Hull Sharpe metrics.
    """
    # Map predictions to positions using alpha-beta config
    positions = map_predictions_to_positions(
        y_pred,
        alpha=position_config.alpha,
        beta=position_config.beta,
        clip_min=position_config.clip_min,
        clip_max=position_config.clip_max,
        winsor_pct=position_config.winsor_pct,
    )

    # Get forward_returns and risk_free_rate
    forward_returns, risk_free_rate = _get_forward_returns_and_rf(
        train_df, val_indices, y_true
    )

    # Compute Hull Sharpe
    result = compute_hull_sharpe(
        positions, forward_returns, risk_free_rate, validate=False
    )

    return result.to_dict()


def _load_position_mapping_config(config_path: str | None) -> Dict[str, Any]:
    """Load position_mapping section from walk_forward.yaml."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"[warn] Position mapping config not found: {path}")
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("position_mapping", {})
    except Exception as e:
        print(f"[warn] Failed to load position mapping config: {e}")
        return {}


def _build_position_config(args: argparse.Namespace) -> AlphaBetaPositionConfig:
    """Build AlphaBetaPositionConfig from args and optional config file.

    Priority: CLI args > config file > defaults
    """
    # Load from config file if specified
    file_config = _load_position_mapping_config(
        getattr(args, "position_mapping_config", None)
    )

    # Determine alpha: CLI --alpha > config file > legacy --sharpe-mult conversion
    if args.alpha is not None:
        alpha = args.alpha
    elif "alpha" in file_config:
        alpha = float(file_config["alpha"])
    else:
        # Legacy: convert sharpe_mult to alpha (sharpe_mult=100 → alpha≈0.25)
        # Note: This is approximate; prefer using --alpha directly
        alpha = args.sharpe_mult / 400.0  # Heuristic conversion

    # Determine beta: CLI --beta > config file > legacy --sharpe-offset
    if args.beta is not None:
        beta = args.beta
    elif "beta" in file_config:
        beta = float(file_config["beta"])
    else:
        beta = args.sharpe_offset

    # Other params: CLI > config file > defaults
    # clip_min: CLI (if specified) > config file > default 0.0
    if args.clip_min is not None:
        clip_min = args.clip_min
    elif "clip_min" in file_config:
        clip_min = float(file_config["clip_min"])
    else:
        clip_min = 0.0

    # clip_max: CLI (if specified) > config file > default 2.0
    if args.clip_max is not None:
        clip_max = args.clip_max
    elif "clip_max" in file_config:
        clip_max = float(file_config["clip_max"])
    else:
        clip_max = 2.0

    winsor_pct = args.winsor_pct
    if winsor_pct is None and "winsor_pct" in file_config:
        config_winsor = file_config["winsor_pct"]
        # Handle null/None in YAML
        winsor_pct = float(config_winsor) if config_winsor is not None else None

    return AlphaBetaPositionConfig(
        alpha=alpha,
        beta=beta,
        clip_min=clip_min,
        clip_max=clip_max,
        winsor_pct=winsor_pct,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Main training function."""
    args = parse_args(argv)

    if not HAS_LGBM:
        print("[error] LightGBM is not installed")
        return 1

    # Setup paths
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if not args.no_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build position mapping config (alpha-beta)
    position_config = _build_position_config(args)
    if args.eval_sharpe:
        print(
            f"[info] Position mapping: alpha={position_config.alpha:.4f}, "
            f"beta={position_config.beta:.4f}, "
            f"clip=[{position_config.clip_min}, {position_config.clip_max}], "
            f"winsor={position_config.winsor_pct}"
        )

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

    # Model kwargs
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

    # Build base pipeline
    base_pipeline = build_pipeline(
        su1_config,
        su5_config,
        preprocess_settings,
        numeric_fill_value=args.numeric_fill_value,
        model_kwargs=model_kwargs,
        random_state=args.random_state,
    )
    callbacks = _initialise_callbacks(base_pipeline.named_steps["model"])

    # CV setup - determine CV mode
    X_np = X.reset_index(drop=True)
    y_np = y.reset_index(drop=True)
    y_np_array = y_np.to_numpy()
    n_samples = len(X_np)

    if args.cv_mode == "walk_forward":
        # Walk-Forward CV
        wf_config = WalkForwardConfig(
            train_window=args.wf_train_window,
            val_window=args.wf_val_window,
            step=args.wf_step,
            mode=args.wf_mode,
            min_folds=2,  # allow 2+ folds
            gap=args.gap,
        )
        print(f"[info] Walk-Forward CV mode: {wf_config.mode}")
        print(
            f"[info] train_window={wf_config.train_window}, val_window={wf_config.val_window}, step={wf_config.step}"
        )

        try:
            wf_folds = make_walk_forward_splits(n_samples, wf_config)
            cv_splits = [(fold.train_indices, fold.val_indices) for fold in wf_folds]
            n_splits_actual = len(cv_splits)
        except ValueError as e:
            print(f"[warn] Walk-Forward split failed: {e}")
            print("[info] Falling back to TimeSeriesSplit")
            args.cv_mode = "timeseries_split"
            cv_splits = list(TimeSeriesSplit(n_splits=args.n_splits).split(X_np))
            n_splits_actual = len(cv_splits)
    else:
        # TimeSeriesSplit CV
        splitter = TimeSeriesSplit(n_splits=args.n_splits)
        cv_splits = list(splitter.split(X_np))
        n_splits_actual = len(cv_splits)

    # Pre-fit augmenter for CV
    print("[info] Pre-fitting SU1/SU5 feature augmenter...")
    su5_prefit = SU5FeatureAugmenter(
        su1_config,
        su5_config,
        fill_value=args.numeric_fill_value,
    )
    su5_prefit.fit(X_np)

    # Build fold_indices array for proper CV isolation
    fold_indices_full = np.full(len(X_np), -1, dtype=int)
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
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

    # Build core pipeline (without augmenter)
    core_pipeline_template = cast(Pipeline, Pipeline(base_pipeline.steps[1:]))

    # CV loop
    oof_pred = np.full(len(X_np), np.nan, dtype=float)
    fold_logs: List[Dict[str, Any]] = []
    sharpe_results: List[Dict[str, Any]] = []

    cv_mode_display = (
        "Walk-Forward" if args.cv_mode == "walk_forward" else "TimeSeriesSplit"
    )
    print(f"\n{'=' * 60}")
    print(f"Starting {n_splits_actual}-fold {cv_mode_display} CV")
    if args.eval_sharpe:
        print(
            f"Hull Sharpe evaluation enabled (alpha={position_config.alpha:.4f}, beta={position_config.beta:.4f})"
        )
    print(f"{'=' * 60}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        # Gap is already applied in Walk-Forward mode via config
        if args.cv_mode != "walk_forward" and args.gap > 0:
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

        # Clone and fit pipeline
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

        # Hull Sharpe metrics (if enabled)
        sharpe_dict: Dict[str, float] = {}
        if args.eval_sharpe:
            sharpe_dict = _compute_sharpe_for_fold(
                y_pred=pred,
                y_true=y_valid.to_numpy(),
                train_df=train_df,
                val_indices=val_idx,
                position_config=position_config,
            )
            sharpe_results.append(
                {
                    "fold_idx": fold_idx,
                    "train_start": int(train_idx[0]),
                    "train_end": int(train_idx[-1]),
                    "val_start": int(val_idx[0]),
                    "val_end": int(val_idx[-1]),
                    **sharpe_dict,
                }
            )

        fold_log_entry = {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_start": int(train_idx[0]),
            "train_end": int(train_idx[-1]),
            "val_start": int(val_idx[0]),
            "val_end": int(val_idx[-1]),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_msr": fold_metrics.get("msr", float("nan")),
            "val_msr_down": fold_metrics.get("msr_down", float("nan")),
        }

        # Add Hull Sharpe to fold log if enabled
        if args.eval_sharpe and sharpe_dict:
            fold_log_entry["hull_sharpe"] = sharpe_dict.get("final_score", float("nan"))
            fold_log_entry["raw_sharpe"] = sharpe_dict.get("raw_sharpe", float("nan"))
            fold_log_entry["vol_ratio"] = sharpe_dict.get("vol_ratio", float("nan"))
            fold_log_entry["vol_penalty"] = sharpe_dict.get("vol_penalty", float("nan"))
            fold_log_entry["return_penalty"] = sharpe_dict.get(
                "return_penalty", float("nan")
            )

        fold_logs.append(fold_log_entry)

        # Print fold summary
        if args.eval_sharpe and sharpe_dict:
            print(
                f"[fold {fold_idx}] train_rmse={train_rmse:.6f} | "
                f"val_rmse={val_rmse:.6f} | hull_sharpe={sharpe_dict.get('final_score', 0):.4f} | "
                f"vol_ratio={sharpe_dict.get('vol_ratio', 0):.3f}"
            )
        else:
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

    # Compute overall Hull Sharpe for OOF predictions
    oof_sharpe_summary: Dict[str, float] = {}
    if args.eval_sharpe and sharpe_results:
        sharpe_scores = [r["final_score"] for r in sharpe_results]
        oof_sharpe_summary = {
            "mean_sharpe": float(np.mean(sharpe_scores)),
            "min_sharpe": float(np.min(sharpe_scores)),
            "max_sharpe": float(np.max(sharpe_scores)),
            "std_sharpe": float(np.std(sharpe_scores)),
            "mean_rmse": float(np.mean([log["val_rmse"] for log in fold_logs])),
            "mean_vol_ratio": float(np.mean([r["vol_ratio"] for r in sharpe_results])),
            "mean_vol_penalty": float(
                np.mean([r["vol_penalty"] for r in sharpe_results])
            ),
            "mean_return_penalty": float(
                np.mean([r["return_penalty"] for r in sharpe_results])
            ),
        }

    print(f"\n{'=' * 60}")
    print(f"[OOF] RMSE = {oof_rmse:.6f}")
    print(f"[OOF] MSR  = {oof_metrics.get('oof_msr', float('nan')):.6f}")
    if args.eval_sharpe and oof_sharpe_summary:
        print(f"[OOF] Hull Sharpe (mean) = {oof_sharpe_summary['mean_sharpe']:.4f}")
        print(f"[OOF] Hull Sharpe (min)  = {oof_sharpe_summary['min_sharpe']:.4f}")
        print(f"[OOF] Hull Sharpe (std)  = {oof_sharpe_summary['std_sharpe']:.4f}")
    print(f"{'=' * 60}\n")

    # Save artifacts
    if not args.no_artifacts and not args.dry_run:
        print("[info] Final training on all data...")

        # Retrain on all data using the same augmented/excluded features as CV
        # Use core_pipeline (without augmenter) trained on X_augmented_all
        final_core_pipeline = cast(Pipeline, clone(core_pipeline_template))
        fit_kwargs_final: Dict[str, Any] = {}
        if callbacks:
            fit_kwargs_final["model__callbacks"] = callbacks
            fit_kwargs_final["model__eval_metric"] = "rmse"
        final_core_pipeline.fit(X_augmented_all, y_np, **fit_kwargs_final)

        # Create a bundle with augmenter + core pipeline for inference
        # The augmenter is pre-fit and can be used at inference time
        inference_bundle = {
            "augmenter": su5_prefit,
            "core_pipeline": final_core_pipeline,
            "feature_tier": args.feature_tier,
            "excluded_features": get_excluded_features(args.feature_tier)
            if not args.no_feature_exclusion
            else [],
        }

        # Save model bundle
        model_path = out_dir / "inference_bundle.pkl"
        joblib.dump(inference_bundle, model_path)
        print(f"[info] Saved model bundle to {model_path}")
        print(
            f"[info] Bundle contains: augmenter + core_pipeline ({X_augmented_all.shape[1]} features)"
        )

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

        # Save Walk-Forward fold logs (if using walk_forward mode)
        if args.cv_mode == "walk_forward" and sharpe_results:
            wf_log_path = out_dir / "walk_forward_folds.csv"
            fieldnames = list(sharpe_results[0].keys())
            with open(wf_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sharpe_results)
            print(f"[info] Saved Walk-Forward fold logs to {wf_log_path}")

        # Save Hull Sharpe summary (if eval_sharpe enabled)
        if args.eval_sharpe and oof_sharpe_summary:
            sharpe_summary = {
                "cv_mode": args.cv_mode,
                "config": {
                    "train_window": args.wf_train_window
                    if args.cv_mode == "walk_forward"
                    else None,
                    "val_window": args.wf_val_window
                    if args.cv_mode == "walk_forward"
                    else None,
                    "step": args.wf_step if args.cv_mode == "walk_forward" else None,
                    "mode": args.wf_mode if args.cv_mode == "walk_forward" else None,
                    "n_splits": n_splits_actual,
                    # Alpha-beta position mapping config
                    "alpha": position_config.alpha,
                    "beta": position_config.beta,
                    "clip_min": position_config.clip_min,
                    "clip_max": position_config.clip_max,
                    "winsor_pct": position_config.winsor_pct,
                },
                "n_folds": len(sharpe_results),
                "metrics": oof_sharpe_summary,
            }
            sharpe_summary_path = out_dir / "hull_sharpe_summary.json"
            with open(sharpe_summary_path, "w", encoding="utf-8") as f:
                json.dump(sharpe_summary, f, indent=2)
            print(f"[info] Saved Hull Sharpe summary to {sharpe_summary_path}")

        # Save metadata
        meta = {
            "model_type": "lightgbm",
            "feature_tier": args.feature_tier,
            "n_features": X_augmented_all.shape[1],
            "cv_mode": args.cv_mode,
            "oof_rmse": oof_rmse,
            "oof_msr": oof_metrics.get("oof_msr", float("nan")),
            "n_splits": n_splits_actual,
            "gap": args.gap,
            "hyperparameters": model_kwargs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.eval_sharpe and oof_sharpe_summary:
            meta["oof_hull_sharpe_mean"] = oof_sharpe_summary.get(
                "mean_sharpe", float("nan")
            )
            meta["oof_hull_sharpe_min"] = oof_sharpe_summary.get(
                "min_sharpe", float("nan")
            )
        meta_path = out_dir / "model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[info] Saved metadata to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
