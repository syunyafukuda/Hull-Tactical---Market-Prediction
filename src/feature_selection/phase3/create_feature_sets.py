#!/usr/bin/env python
"""Create feature sets configuration for model selection phase.

This script creates the feature_sets.json configuration that defines
multiple feature set variants (FS_full, FS_compact, FS_topK) for
the model selection phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Create feature sets configuration."
    )
    ap.add_argument(
        "--tier2-excluded",
        type=str,
        default="configs/feature_selection/tier2/excluded.json",
        help="Path to Tier2 excluded.json",
    )
    ap.add_argument(
        "--tier3-excluded",
        type=str,
        default="configs/feature_selection/tier3/excluded.json",
        help="Path to Tier3 excluded.json (optional)",
    )
    ap.add_argument(
        "--tier2-evaluation",
        type=str,
        default="results/feature_selection/tier2/evaluation.json",
        help="Path to Tier2 evaluation.json (optional)",
    )
    ap.add_argument(
        "--tier3-evaluation",
        type=str,
        default="results/feature_selection/tier3/evaluation.json",
        help="Path to Tier3 evaluation.json (optional)",
    )
    ap.add_argument(
        "--tier2-importance",
        type=str,
        default="results/feature_selection/tier2/importance_summary.csv",
        help="Path to Tier2 importance_summary.csv",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top features for FS_topK (default: 50)",
    )
    ap.add_argument(
        "--recommended",
        type=str,
        default="FS_compact",
        choices=["FS_full", "FS_compact", "FS_topK"],
        help="Recommended feature set (default: FS_compact)",
    )
    ap.add_argument(
        "--out-file",
        type=str,
        default="configs/feature_selection/feature_sets.json",
        help="Output path for feature_sets.json",
    )
    return ap.parse_args(argv)


def count_features(excluded_json_path: str, total_features: int = 577) -> int:
    """Count number of features after exclusions."""
    try:
        with open(excluded_json_path, "r") as f:
            data = json.load(f)
        n_excluded = len(data.get("candidates", []))
        return total_features - n_excluded
    except FileNotFoundError:
        return 0


def load_evaluation_rmse(eval_json_path: str) -> Optional[float]:
    """Load OOF RMSE from evaluation.json."""
    try:
        with open(eval_json_path, "r") as f:
            data = json.load(f)
        return data.get("oof_rmse")
    except FileNotFoundError:
        return None


def create_topk_excluded(
    tier2_excluded_path: str,
    importance_csv_path: str,
    topk: int,
    output_path: str,
) -> int:
    """Create excluded.json for FS_topK by keeping only top K features."""
    # Load Tier2 excluded features
    with open(tier2_excluded_path, "r") as f:
        tier2_data = json.load(f)
    
    # Load importance
    importance_df = pd.read_csv(importance_csv_path)
    
    # Sort by mean_gain descending
    importance_df = importance_df.sort_values('mean_gain', ascending=False)
    
    # Get top K features
    top_features = set(importance_df.head(topk)['feature'].tolist())
    
    # Get all features that are NOT in top K (these will be excluded)
    all_features = set(importance_df['feature'].tolist())
    to_exclude = all_features - top_features
    
    # Start with Tier2 exclusions
    all_candidates = tier2_data.get("candidates", []).copy()
    
    # Add features not in top K
    for feature in sorted(to_exclude):
        importance_series = cast(pd.Series, importance_df[
            importance_df['feature'] == feature
        ]['mean_gain'])
        mean_gain = float(importance_series.iloc[0]) if len(importance_series) > 0 else 0.0
        
        # Only add if not already in tier2 exclusions
        already_excluded = any(
            c["feature_name"] == feature for c in all_candidates
        )
        if not already_excluded:
            all_candidates.append({
                "feature_name": feature,
                "reason": "not_in_top_k",
                "mean_gain": mean_gain,
            })
    
    # Create topK output
    topk_output = {
        "version": "tier_topK-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_tier": "tier2",
        "top_k": topk,
        "candidates": all_candidates,
        "summary": {
            "tier2_exclusions": len(tier2_data.get("candidates", [])),
            "not_in_top_k": len(to_exclude),
            "total_exclusions": len(all_candidates),
        }
    }
    
    # Create output directory
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    with open(output_path, "w") as f:
        json.dump(topk_output, f, indent=2)
    
    return 577 - len(all_candidates)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    print("=" * 80)
    print("Creating Feature Sets Configuration")
    print("=" * 80)
    print(f"Tier2 excluded: {args.tier2_excluded}")
    print(f"Tier3 excluded: {args.tier3_excluded}")
    print(f"Top K: {args.topk}")
    print(f"Recommended: {args.recommended}")
    print(f"Output file: {args.out_file}")
    print()
    
    # Count features for each tier
    tier2_n_features = count_features(args.tier2_excluded)
    tier3_n_features = count_features(args.tier3_excluded)
    
    # Load evaluation metrics
    tier2_rmse = load_evaluation_rmse(args.tier2_evaluation)
    tier3_rmse = load_evaluation_rmse(args.tier3_evaluation)
    
    # Create topK excluded.json
    topk_excluded_path = "configs/feature_selection/tier_topK/excluded.json"
    print(f"Creating top-{args.topk} feature set...")
    topk_n_features = create_topk_excluded(
        args.tier2_excluded,
        args.tier2_importance,
        args.topk,
        topk_excluded_path,
    )
    print(f"Top-{args.topk} feature set created: {topk_n_features} features")
    print()
    
    # Create feature sets configuration
    feature_sets: Dict[str, Any] = {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_sets": {
            "FS_full": {
                "description": "Tier2 full feature set",
                "excluded_json": args.tier2_excluded,
                "n_features": tier2_n_features,
                "oof_rmse": tier2_rmse,
            },
        },
        "recommended": args.recommended,
    }
    
    # Add FS_compact if Tier3 exists
    if tier3_n_features > 0:
        feature_sets["feature_sets"]["FS_compact"] = {
            "description": "Tier3 after correlation clustering",
            "excluded_json": args.tier3_excluded,
            "n_features": tier3_n_features,
            "oof_rmse": tier3_rmse,
        }
    
    # Add FS_topK
    feature_sets["feature_sets"]["FS_topK"] = {
        "description": f"Top {args.topk} features by importance",
        "excluded_json": topk_excluded_path,
        "n_features": topk_n_features,
        "oof_rmse": None,  # Not evaluated yet
    }
    
    # Create output directory
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save feature sets configuration
    print(f"Saving feature sets configuration to {args.out_file}...")
    with open(args.out_file, "w") as f:
        json.dump(feature_sets, f, indent=2)
    
    print()
    print("Feature Sets Summary:")
    for name, config in feature_sets["feature_sets"].items():
        rmse_str = f"{config['oof_rmse']:.6f}" if config['oof_rmse'] is not None else "N/A"
        print(f"  {name}: {config['n_features']} features, RMSE={rmse_str}")
    print(f"\nRecommended: {feature_sets['recommended']}")
    print()
    print("Feature sets configuration created successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
