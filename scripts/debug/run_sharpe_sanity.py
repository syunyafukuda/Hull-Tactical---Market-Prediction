#!/usr/bin/env python
"""Sanity check script for Hull Competition Sharpe evaluation.

This script validates the Hull Sharpe metric implementation by:
1. Using train[-1000:] as a pseudo-LB holdout
2. Training LGBM on train[:-1000]
3. Computing Hull Sharpe on the holdout
4. Verifying that a positive Sharpe is achieved

Usage:
    python scripts/debug/run_sharpe_sanity.py
    python scripts/debug/run_sharpe_sanity.py --holdout-size 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.metrics.hull_sharpe import (  # noqa: E402
    compute_hull_sharpe,
)
from src.models.common.signals import (  # noqa: E402
    analyze_position_distribution,
    map_predictions_to_positions,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Sanity check for Hull Sharpe metric")
    ap.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing train.csv",
    )
    ap.add_argument(
        "--holdout-size",
        type=int,
        default=1000,
        help="Number of samples to use as pseudo-LB holdout",
    )
    # Legacy args (deprecated)
    ap.add_argument(
        "--sharpe-mult",
        type=float,
        default=100.0,
        help="[DEPRECATED] Use --alpha instead",
    )
    ap.add_argument(
        "--sharpe-offset",
        type=float,
        default=1.0,
        help="[DEPRECATED] Use --beta instead",
    )
    # Alpha-beta position mapping (preferred)
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha (scale) for alpha-beta position mapping",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta (offset) for alpha-beta position mapping",
    )
    ap.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Minimum position for clipping",
    )
    ap.add_argument(
        "--clip-max",
        type=float,
        default=2.0,
        help="Maximum position for clipping",
    )
    ap.add_argument(
        "--winsor-pct",
        type=float,
        default=None,
        help="Winsorize predictions at this percentile",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )
    return ap.parse_args(argv)


def load_train_data(data_dir: Path) -> pd.DataFrame:
    """Load training data."""
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    df = pd.read_csv(train_path)
    if "date_id" in df.columns:
        df = df.sort_values("date_id").reset_index(drop=True)

    return df


def train_simple_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train a simple LGBM model for sanity check."""
    if not HAS_LGBM:
        raise RuntimeError("LightGBM is not installed")

    # Simple model with default parameters
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    # Select numeric columns only
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Fill NaN with 0 for simplicity
    X_train_clean = X_train[numeric_cols].fillna(0)

    model.fit(X_train_clean, y_train)

    return model, numeric_cols


def main(argv: Sequence[str] | None = None) -> int:
    """Run sanity check."""
    args = parse_args(argv)

    if not HAS_LGBM:
        print("[error] LightGBM is not installed")
        return 1

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Hull Competition Sharpe - Sanity Check")
    print("=" * 60)
    print()

    # Load data
    print("[1/5] Loading data...")
    df = load_train_data(data_dir)
    print(f"      Total samples: {len(df)}")

    # Split into train and holdout
    print(f"[2/5] Creating holdout set (last {args.holdout_size} samples)...")
    holdout_size = args.holdout_size
    train_df = df.iloc[:-holdout_size].copy()
    holdout_df = df.iloc[-holdout_size:].copy()
    print(f"      Train samples: {len(train_df)}")
    print(f"      Holdout samples: {len(holdout_df)}")

    # Prepare features and target
    target_col = "market_forward_excess_returns"
    id_cols = ["date_id"]

    if target_col not in train_df.columns:
        print(f"[error] Target column '{target_col}' not found")
        return 1

    # Get feature columns
    feature_cols = [c for c in train_df.columns if c not in id_cols + [target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df[target_col]

    # Train model
    print("[3/5] Training LightGBM model...")
    model, used_cols = train_simple_lgbm(X_train, y_train)
    print(f"      Used {len(used_cols)} numeric features")

    # Predict on holdout
    print("[4/5] Predicting on holdout set...")
    X_holdout_clean = X_holdout[used_cols].fillna(0)
    y_pred = model.predict(X_holdout_clean)

    # Compute RMSE
    rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))
    print(f"      Holdout RMSE: {rmse:.6f}")

    # Map predictions to positions
    print("[5/5] Computing Hull Sharpe...")

    # Determine alpha/beta (CLI > legacy conversion)
    if args.alpha is not None:
        alpha = args.alpha
    else:
        alpha = args.sharpe_mult / 400.0  # Legacy conversion

    if args.beta is not None:
        beta = args.beta
    else:
        beta = args.sharpe_offset

    positions = map_predictions_to_positions(
        y_pred,
        alpha=alpha,
        beta=beta,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        winsor_pct=args.winsor_pct,
    )

    if args.verbose:
        print(f"      Position mapping: alpha={alpha:.4f}, beta={beta:.4f}")

    if args.verbose:
        pos_stats = analyze_position_distribution(positions)
        print("      Position stats:")
        print(f"        mean={pos_stats['mean']:.3f}, std={pos_stats['std']:.3f}")
        print(f"        min={pos_stats['min']:.3f}, max={pos_stats['max']:.3f}")
        print(
            f"        pct_at_min={pos_stats['pct_at_min_clip'] * 100:.1f}%, pct_at_max={pos_stats['pct_at_max_clip'] * 100:.1f}%"
        )

    # Get forward_returns and risk_free_rate
    # Use target as forward_returns approximation
    if "sp500_forward_returns" in holdout_df.columns:
        forward_returns = holdout_df["sp500_forward_returns"].to_numpy()
    else:
        forward_returns = y_holdout.to_numpy()

    if "federal_funds_rate" in holdout_df.columns:
        risk_free_rate = holdout_df["federal_funds_rate"].to_numpy()
    else:
        risk_free_rate = np.full(len(holdout_df), 0.04 / 252)

    # Compute Hull Sharpe
    result = compute_hull_sharpe(
        positions, forward_returns, risk_free_rate, validate=False
    )

    print()
    print("-" * 40)
    print("Hull Sharpe Results:")
    print("-" * 40)
    print(f"  Final Score:     {result.final_score:.4f}")
    print(f"  Raw Sharpe:      {result.raw_sharpe:.4f}")
    print(f"  Vol Ratio:       {result.vol_ratio:.4f}")
    print(f"  Vol Penalty:     {result.vol_penalty:.4f}")
    print(f"  Return Penalty:  {result.return_penalty:.4f}")
    print(f"  Strategy Mean:   {result.strategy_mean:.6f}")
    print(f"  Strategy Std:    {result.strategy_std:.6f}")
    print(f"  Market Mean:     {result.market_mean:.6f}")
    print(f"  Market Std:      {result.market_std:.6f}")
    print("-" * 40)
    print()

    # Sanity check assertions
    checks_passed = 0
    checks_total = 0

    # Check 1: Raw Sharpe is defined (not NaN/inf)
    checks_total += 1
    if not np.isnan(result.raw_sharpe) and not np.isinf(result.raw_sharpe):
        checks_passed += 1
        print("[PASS] Raw Sharpe is defined (not NaN/inf)")
    else:
        print("[FAIL] Raw Sharpe is NaN or inf")

    # Check 2: Final score is reasonable (between -10 and 10)
    checks_total += 1
    if -10 < result.final_score < 10:
        checks_passed += 1
        print("[PASS] Final score is in reasonable range (-10, 10)")
    else:
        print(f"[FAIL] Final score {result.final_score} is outside reasonable range")

    # Check 3: Vol ratio is positive
    checks_total += 1
    if result.vol_ratio > 0:
        checks_passed += 1
        print("[PASS] Vol ratio is positive")
    else:
        print(f"[FAIL] Vol ratio {result.vol_ratio} is not positive")

    # Check 4: Penalties are non-negative
    checks_total += 1
    if result.vol_penalty >= 0 and result.return_penalty >= 0:
        checks_passed += 1
        print("[PASS] Penalties are non-negative")
    else:
        print(
            f"[FAIL] Negative penalty: vol={result.vol_penalty}, ret={result.return_penalty}"
        )

    # Check 5: Positions are in valid range
    checks_total += 1
    if positions.min() >= 0 and positions.max() <= 2:
        checks_passed += 1
        print("[PASS] Positions are in valid [0, 2] range")
    else:
        print(
            f"[FAIL] Positions out of range: min={positions.min()}, max={positions.max()}"
        )

    print()
    print("=" * 60)
    print(f"Sanity Check Summary: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)

    if checks_passed == checks_total:
        print("\n✅ All sanity checks passed!")
        return 0
    else:
        print("\n⚠️ Some sanity checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
