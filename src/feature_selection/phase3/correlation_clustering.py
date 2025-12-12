#!/usr/bin/env python
"""Variable clustering for Tier2 features.

This script identifies groups of correlated features in the Tier2 feature set 
using the VarClus algorithm (variable clustering based on PCA) and outputs 
cluster assignments for subsequent representative selection.
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
from varclushi import VarClusHi

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
    infer_test_file,
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
        help="(Deprecated - not used by VarClus) Correlation threshold for clustering (default: 0.95)",
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
    X_data: pd.DataFrame,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Perform variable clustering using VarClus algorithm.
    
    Args:
        X_data: Feature matrix (DataFrame)
        threshold: Not used in VarClus (kept for API compatibility)
        
    Returns:
        Dictionary containing cluster assignments and metadata
    """
    # Run VarClus algorithm
    # maxeigval2=1 means split clusters if second eigenvalue > 1
    # This ensures clusters are homogeneous (features within cluster explain variance well)
    vc = VarClusHi(X_data, maxeigval2=1, maxclus=None)
    vc.varclus()
    
    # Extract cluster information
    rsquare_df = vc.rsquare
    info_df = vc.info
    
    # Build clusters dictionary
    clusters_dict: Dict[int, List[str]] = {}
    for _, row in rsquare_df.iterrows():
        cluster_id = int(row['Cluster'])
        feature = row['Variable']
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(feature)
    
    # Build output structure
    clusters = []
    singleton_features = []
    
    # Compute correlation matrix for max_correlation calculation
    corr_matrix = X_data.corr()
    
    for cluster_id, features in sorted(clusters_dict.items()):
        if len(features) == 1:
            singleton_features.extend(features)
        else:
            # Calculate max correlation within cluster
            cluster_corr = corr_matrix.loc[features, features]
            max_corr = np.abs(cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)]).max()
            
            # Get VarProp (proportion of variance explained by 1st PC) for this cluster
            cluster_info = info_df[info_df['Cluster'] == cluster_id]
            var_prop = float(cluster_info['VarProp'].iloc[0]) if not cluster_info.empty else 0.0
            
            clusters.append({
                "cluster_id": int(cluster_id),
                "features": sorted(features),
                "representative": None,  # Will be filled by select_representatives.py
                "max_correlation": float(max_corr),
                "variance_explained": var_prop,  # Additional VarClus metric
            })
    
    return {
        "method": "VarClus",
        "maxeigval2": 1,
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
    data_dir = Path(args.data_dir)
    train_file = args.train_file or infer_train_file(data_dir, None)
    test_file = infer_test_file(data_dir, None)
    df_train = load_table(Path(train_file) if isinstance(train_file, str) else train_file)
    df_test = load_table(Path(test_file) if isinstance(test_file, str) else test_file)
    print(f"Loaded {len(df_train)} rows from {train_file}")
    print()
    
    # Build pipeline and transform data
    print("Building preprocessing pipeline...")
    build_pipeline(
        su1_config=su1_cfg,
        su5_config=su5_cfg,
        preprocess_settings=preprocess_policies,
        numeric_fill_value=0.0,
        model_kwargs={},
        random_state=42,
    )
    
    print("Fitting and transforming data...")
    X_train, y_train, _ = _prepare_features(
        df_train,
        df_test,
        target_col=args.target_col,
        id_col="row_id",
    )
    
    # Filter to Tier2 features (exclude removed features)
    all_features = X_train.columns.tolist()
    tier2_features = [f for f in all_features if f not in excluded_features]
    
    print(f"Total features: {len(all_features)}")
    print(f"Tier2 features: {len(tier2_features)}")
    print()
    
    # Select only Tier2 features
    X_tier2 = X_train[tier2_features]
    
    # Perform VarClus clustering
    print(f"Performing VarClus variable clustering...")
    clustering_result = compute_correlation_clusters(
        X_tier2,
        threshold=args.correlation_threshold,  # Not used by VarClus, kept for compatibility
    )
    
    print(f"Found {clustering_result['n_clusters']} clusters")
    print(f"Found {clustering_result['n_singletons']} singleton features")
    print()
    
    # Display cluster summary
    if clustering_result['clusters']:
        print("Cluster summary:")
        for cluster in clustering_result['clusters']:
            var_expl = cluster.get('variance_explained', 0.0)
            print(f"  Cluster {cluster['cluster_id']}: "
                  f"{len(cluster['features'])} features, "
                  f"max_corr={cluster['max_correlation']:.4f}, "
                  f"var_explained={var_expl:.4f}")
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
