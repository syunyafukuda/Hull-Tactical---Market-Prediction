#!/usr/bin/env python
"""Correlation-based clustering for Tier2 features.

This script identifies groups of highly correlated features (|ρ| > 0.95) 
in the Tier2 feature set using hierarchical clustering and outputs cluster 
assignments for subsequent representative selection.
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
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
    load_table,
    _prepare_features,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Perform correlation clustering on Tier2 features."
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
        help="Path to Tier2 excluded features JSON",
    )
    ap.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for clustering (default: 0.95)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results/feature_selection/phase3",
        help="Output directory for results",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="market_forward_excess_returns",
        help="Target column name",
    )
    return ap.parse_args(argv)


def load_excluded_features(exclude_path: str) -> set[str]:
    """Load excluded features from JSON file."""
    with open(exclude_path, "r") as f:
        data = json.load(f)
    
    excluded = set()
    if "candidates" in data:
        for item in data["candidates"]:
            excluded.add(item["feature_name"])
    
    return excluded


def compute_correlation_clusters(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Perform hierarchical clustering based on correlation matrix.
    
    Args:
        correlation_matrix: Correlation matrix of features
        threshold: Correlation threshold for clustering
        
    Returns:
        Dictionary containing cluster assignments and metadata
    """
    # Convert correlation to distance: distance = 1 - |ρ|
    distance_matrix = 1 - np.abs(correlation_matrix.values)
    
    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering using Ward's method
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # Cut dendrogram at distance threshold
    distance_threshold = 1 - threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Group features by cluster
    feature_names = correlation_matrix.columns.tolist()
    clusters_dict: Dict[int, List[str]] = {}
    
    for feature, cluster_id in zip(feature_names, cluster_labels):
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(feature)
    
    # Build output structure
    clusters = []
    singleton_features = []
    
    for cluster_id, features in sorted(clusters_dict.items()):
        if len(features) == 1:
            singleton_features.extend(features)
        else:
            # Calculate max correlation within cluster
            cluster_corr = correlation_matrix.loc[features, features]
            max_corr = np.abs(cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)]).max()
            
            clusters.append({
                "cluster_id": int(cluster_id),
                "features": sorted(features),
                "representative": None,  # Will be filled by select_representatives.py
                "max_correlation": float(max_corr),
            })
    
    return {
        "threshold": threshold,
        "n_clusters": len(clusters),
        "n_singletons": len(singleton_features),
        "clusters": clusters,
        "singleton_features": sorted(singleton_features),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 3-1: Correlation Clustering")
    print("=" * 80)
    print(f"Config path: {args.config_path}")
    print(f"Preprocess config: {args.preprocess_config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Exclude features: {args.exclude_features}")
    print(f"Correlation threshold: {args.correlation_threshold}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # Load excluded features
    print("Loading excluded features...")
    excluded_features = load_excluded_features(args.exclude_features)
    print(f"Loaded {len(excluded_features)} excluded features")
    print()
    
    # Load configurations
    print("Loading configurations...")
    su1_cfg = load_su1_config(args.config_path)
    su5_cfg = load_su5_config(args.config_path)
    preprocess_policies = load_preprocess_policies(args.preprocess_config)
    print("Configurations loaded")
    print()
    
    # Load training data
    print("Loading training data...")
    train_file = args.train_file or infer_train_file(args.data_dir)
    df_train = load_table(train_file)
    print(f"Loaded {len(df_train)} rows from {train_file}")
    print()
    
    # Build pipeline and transform data
    print("Building preprocessing pipeline...")
    pipeline = build_pipeline(
        su1_cfg=su1_cfg,
        su5_cfg=su5_cfg,
        preprocess_policies=preprocess_policies,
    )
    
    print("Fitting and transforming data...")
    X_train, y_train = _prepare_features(
        df_train,
        target_col=args.target_col,
        pipeline=pipeline,
    )
    
    # Filter to Tier2 features (exclude removed features)
    all_features = X_train.columns.tolist()
    tier2_features = [f for f in all_features if f not in excluded_features]
    
    print(f"Total features: {len(all_features)}")
    print(f"Tier2 features: {len(tier2_features)}")
    print()
    
    # Select only Tier2 features
    X_tier2 = X_train[tier2_features]
    
    # Compute correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = X_tier2.corr()
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    print()
    
    # Perform clustering
    print(f"Performing hierarchical clustering (threshold={args.correlation_threshold})...")
    clustering_result = compute_correlation_clusters(
        corr_matrix,
        threshold=args.correlation_threshold,
    )
    
    print(f"Found {clustering_result['n_clusters']} clusters")
    print(f"Found {clustering_result['n_singletons']} singleton features")
    print()
    
    # Display cluster summary
    if clustering_result['clusters']:
        print("Cluster summary:")
        for cluster in clustering_result['clusters']:
            print(f"  Cluster {cluster['cluster_id']}: "
                  f"{len(cluster['features'])} features, "
                  f"max_corr={cluster['max_correlation']:.4f}")
        print()
    
    # Add metadata
    clustering_result["created_at"] = datetime.now(timezone.utc).isoformat()
    clustering_result["config"] = {
        "config_path": args.config_path,
        "preprocess_config": args.preprocess_config,
        "exclude_features": args.exclude_features,
        "correlation_threshold": args.correlation_threshold,
    }
    
    # Save results
    output_file = out_dir / "correlation_clusters.json"
    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(clustering_result, f, indent=2)
    
    print("Correlation clustering complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
