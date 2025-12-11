#!/usr/bin/env python
"""Select cluster representatives based on feature importance.

This script reads the correlation clustering results and selects one 
representative feature from each cluster based on LGBM feature importance
(mean_gain). Features not selected become deletion candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, cast

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
        description="Select cluster representatives based on feature importance."
    )
    ap.add_argument(
        "--clusters-json",
        type=str,
        required=True,
        help="Path to correlation_clusters.json",
    )
    ap.add_argument(
        "--importance-csv",
        type=str,
        required=True,
        help="Path to importance_summary.csv from Tier2",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results/feature_selection/phase3",
        help="Output directory for results",
    )
    return ap.parse_args(argv)


def is_raw_feature(feature_name: str) -> bool:
    """Check if a feature is a raw feature (D, E, I, M, P, S, V)."""
    # Raw features typically start with these prefixes
    raw_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V']
    # Check if it's a simple feature like "E1", "M5", etc.
    if len(feature_name) >= 2:
        if feature_name[0] in raw_prefixes and feature_name[1:].isdigit():
            return True
    return False


def select_cluster_representative(
    cluster_features: List[str],
    importance_df: pd.DataFrame,
) -> str:
    """Select representative from cluster based on importance.
    
    Priority:
    1. Maximum mean_gain
    2. Raw features preferred if importance is similar
    
    Args:
        cluster_features: List of features in the cluster
        importance_df: DataFrame with feature importance
        
    Returns:
        Name of the selected representative feature
    """
    # Filter importance for cluster features
    cluster_importance: pd.DataFrame = importance_df[
        importance_df['feature_name'].isin(cluster_features)
    ].copy()  # type: ignore[reportAssignmentType]
    
    if cluster_importance.empty:
        # If no importance data, just return first feature
        return cluster_features[0]
    
    # Sort by mean_gain descending
    cluster_importance = cluster_importance.sort_values(by='mean_gain', ascending=False)
    
    # Select the feature with max mean_gain
    representative = str(cluster_importance.iloc[0]['feature_name'])
    
    return representative


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 3-2: Select Cluster Representatives")
    print("=" * 80)
    print(f"Clusters JSON: {args.clusters_json}")
    print(f"Importance CSV: {args.importance_csv}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # Load clustering results
    print("Loading clustering results...")
    with open(args.clusters_json, "r") as f:
        clustering = json.load(f)
    
    print(f"Loaded {clustering['n_clusters']} clusters")
    print(f"Loaded {clustering['n_singletons']} singleton features")
    print()
    
    # Load importance data
    print("Loading feature importance...")
    importance_df = pd.read_csv(args.importance_csv)
    print(f"Loaded importance for {len(importance_df)} features")
    print()
    
    # Select representatives for each cluster
    print("Selecting cluster representatives...")
    representatives = []
    to_remove = []
    total_removed = 0
    
    for cluster in clustering['clusters']:
        cluster_id = cluster['cluster_id']
        features = cluster['features']
        
        # Select representative
        representative = select_cluster_representative(features, importance_df)
        
        # Get importance value
        rep_importance = cast(pd.Series, importance_df[
            importance_df['feature_name'] == representative
        ]['mean_gain'])
        rep_gain = float(rep_importance.iloc[0]) if len(rep_importance) > 0 else 0.0
        
        # Update cluster info
        cluster['representative'] = representative
        
        # Record representative
        representatives.append({
            "cluster_id": cluster_id,
            "feature": representative,
            "mean_gain": rep_gain,
        })
        
        # Record features to remove (all except representative)
        for feature in features:
            if feature != representative:
                feat_importance = cast(pd.Series, importance_df[
                    importance_df['feature_name'] == feature
                ]['mean_gain'])
                feat_gain = float(feat_importance.iloc[0]) if len(feat_importance) > 0 else 0.0
                
                to_remove.append({
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "mean_gain": feat_gain,
                })
                total_removed += 1
        
        print(f"  Cluster {cluster_id}: Selected '{representative}' "
              f"(mean_gain={rep_gain:.2f}) from {len(features)} features")
    
    print()
    print(f"Total features to remove: {total_removed}")
    print()
    
    # Create output structure
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_clusters": args.clusters_json,
        "source_importance": args.importance_csv,
        "representatives": representatives,
        "to_remove": to_remove,
        "total_removed": total_removed,
        "clusters_updated": clustering['clusters'],  # Include updated clusters with representatives
    }
    
    # Save results
    output_file = out_dir / "cluster_representatives.json"
    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print("Representative selection complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
