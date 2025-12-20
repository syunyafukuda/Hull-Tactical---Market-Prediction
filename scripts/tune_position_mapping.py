#!/usr/bin/env python3
"""Alpha/Beta grid search for position mapping optimization.

This script searches for optimal alpha/beta parameters for the
prediction → position mapping, based on Kaggle discussion/611071.

Usage:
    python scripts/tune_position_mapping.py \
        --oof-path artifacts/models/lgbm-sharpe-wf-opt/oof_predictions.csv \
        --train-path data/raw/train.csv \
        --output results/position_sweep/alpha_beta_search.csv

Examples:
    # Reproduce do-nothing baseline
    python scripts/tune_position_mapping.py --alpha-grid 0 --beta-grid 0.806

    # Full grid search
    python scripts/tune_position_mapping.py --full-grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common.signals import map_predictions_to_positions  # noqa: E402


def hull_sharpe_simple(
    positions: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """Simplified Hull Sharpe calculation.
    
    Returns the annualized Sharpe ratio without Vol Ratio penalty.
    """
    portfolio_returns = positions * forward_returns
    excess_returns = portfolio_returns - risk_free_rate / annualization
    
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)
    
    if std_return < 1e-10:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(annualization)
    return sharpe


def main():
    parser = argparse.ArgumentParser(
        description="Alpha/Beta grid search for position mapping"
    )
    parser.add_argument(
        "--oof-path",
        type=str,
        default="artifacts/models/lgbm-sharpe-wf-opt/oof_predictions.csv",
        help="Path to OOF predictions CSV",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/raw/train.csv",
        help="Path to train.csv with forward_returns",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/position_sweep/alpha_beta_search.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
        help="Alpha values to search",
    )
    parser.add_argument(
        "--beta-grid",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 0.806, 1.0, 1.2],
        help="Beta values to search",
    )
    parser.add_argument(
        "--clip-min-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4],
        help="Clip min values to search",
    )
    parser.add_argument(
        "--clip-max-grid",
        type=float,
        nargs="+",
        default=[1.6, 1.8, 2.0],
        help="Clip max values to search",
    )
    parser.add_argument(
        "--winsor-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05],
        help="Winsor percentile values to search (0 = disabled)",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full grid (all combinations)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only alpha/beta, fixed clip/winsor",
    )
    args = parser.parse_args()

    # Load OOF predictions
    print(f"Loading OOF predictions from {args.oof_path}...")
    oof_df = pd.read_csv(args.oof_path)
    
    if "prediction" not in oof_df.columns:
        raise ValueError("OOF CSV must have 'prediction' column")
    
    # Load forward returns from train.csv
    print(f"Loading forward returns from {args.train_path}...")
    train_df = pd.read_csv(args.train_path)
    
    if "forward_returns" not in train_df.columns:
        raise ValueError("Train CSV must have 'forward_returns' column")
    
    # Align with OOF predictions
    # Assuming OOF predictions are for the same rows as train.csv
    if "row_id" in oof_df.columns and "row_id" in train_df.columns:
        # Merge on row_id
        merged = oof_df.merge(train_df[["row_id", "forward_returns"]], on="row_id")
        forward_returns = merged["forward_returns"].values
        pred_excess = merged["prediction"].values
    else:
        # Assume same order - use index from OOF
        forward_returns = train_df["forward_returns"].values[:len(oof_df)]
        pred_excess = oof_df["prediction"].values
    
    # Remove NaN values (OOF predictions have NaN for training-only samples)
    valid_mask = ~np.isnan(pred_excess) & ~np.isnan(forward_returns)
    pred_excess = pred_excess[valid_mask]
    forward_returns = forward_returns[valid_mask]
    
    print(f"  Valid samples: {len(pred_excess)} (after removing NaN)")
    print(f"  Prediction stats: mean={pred_excess.mean():.6f}, "
          f"std={pred_excess.std():.6f}, "
          f"min={pred_excess.min():.6f}, max={pred_excess.max():.6f}")
    print(f"  Forward returns stats: mean={forward_returns.mean():.6f}, "
          f"std={forward_returns.std():.6f}")

    # Setup grids
    if args.quick:
        clip_min_grid = [0.0]
        clip_max_grid = [2.0]
        winsor_grid = [0.0]
    else:
        clip_min_grid = args.clip_min_grid
        clip_max_grid = args.clip_max_grid
        winsor_grid = args.winsor_grid

    # Grid search
    print("\nStarting grid search...")
    results = []
    
    total_combos = (
        len(args.alpha_grid)
        * len(args.beta_grid)
        * len(clip_min_grid)
        * len(clip_max_grid)
        * len(winsor_grid)
    )
    print(f"  Total combinations: {total_combos}")

    for i, (alpha, beta, clip_min, clip_max, winsor) in enumerate(
        itertools.product(
            args.alpha_grid,
            args.beta_grid,
            clip_min_grid,
            clip_max_grid,
            winsor_grid,
        )
    ):
        if clip_min >= clip_max:
            continue  # Invalid config
        
        winsor_pct = winsor if winsor > 0 else None
        
        positions = map_predictions_to_positions(
            pred_excess,
            alpha=alpha,
            beta=beta,
            clip_min=clip_min,
            clip_max=clip_max,
            winsor_pct=winsor_pct,
        )
        
        sharpe = hull_sharpe_simple(positions, forward_returns)
        
        results.append({
            "alpha": alpha,
            "beta": beta,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "winsor_pct": winsor,
            "sharpe": sharpe,
            "pos_mean": positions.mean(),
            "pos_std": positions.std(),
            "pos_min": positions.min(),
            "pos_max": positions.max(),
            "pct_at_clip_min": (positions == clip_min).mean(),
            "pct_at_clip_max": (positions == clip_max).mean(),
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total_combos} combinations...")

    # Sort by Sharpe (descending)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)
    print(results_df.head(10).to_string(index=False))

    # Print do-nothing baseline
    print("\n" + "=" * 60)
    print("DO-NOTHING BASELINE (alpha=0)")
    print("=" * 60)
    donothings = results_df[results_df["alpha"] == 0.0].head(5)
    if not donothings.empty:
        print(donothings.to_string(index=False))
        best_donothing = donothings.iloc[0]
        print(f"\n→ Best do-nothing: beta={best_donothing['beta']:.3f}, "
              f"sharpe={best_donothing['sharpe']:.4f}")
    
    # Print best config
    best = results_df.iloc[0]
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"  alpha:     {best['alpha']}")
    print(f"  beta:      {best['beta']}")
    print(f"  clip_min:  {best['clip_min']}")
    print(f"  clip_max:  {best['clip_max']}")
    print(f"  winsor:    {best['winsor_pct']}")
    print(f"  sharpe:    {best['sharpe']:.4f}")
    print(f"  pos_mean:  {best['pos_mean']:.4f}")
    print(f"  pos_std:   {best['pos_std']:.4f}")

    # Save best config as JSON
    best_config = {
        "alpha": float(best["alpha"]),
        "beta": float(best["beta"]),
        "clip_min": float(best["clip_min"]),
        "clip_max": float(best["clip_max"]),
        "winsor_pct": float(best["winsor_pct"]) if best["winsor_pct"] > 0 else None,
        "sharpe": float(best["sharpe"]),
    }
    config_path = output_path.parent / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to {config_path}")


if __name__ == "__main__":
    main()
