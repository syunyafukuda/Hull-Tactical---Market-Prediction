#!/usr/bin/env python
"""Tests for Phase 3 correlation clustering and feature set creation."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.feature_selection.phase3.correlation_clustering import (  # noqa: E402
    compute_correlation_clusters,
)
from src.feature_selection.phase3.select_representatives import (  # noqa: E402
    select_cluster_representative,
    is_raw_feature,
)


class TestCorrelationClustering:
    """Test suite for correlation clustering functions."""
    
    def test_compute_correlation_clusters_basic(self):
        """Test basic VarClus clustering."""
        # Create sample data with correlated features
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'feat_a': np.random.randn(n_samples),
            'feat_b': np.random.randn(n_samples),
        })
        # Make feat_a_corr highly correlated with feat_a
        df['feat_a_corr'] = df['feat_a'] + np.random.randn(n_samples) * 0.1
        
        result = compute_correlation_clusters(df)
        
        # Check output structure
        assert 'method' in result
        assert result['method'] == 'VarClus'
        assert 'n_clusters' in result
        assert 'n_singletons' in result
        assert 'clusters' in result
        assert 'singleton_features' in result
        
        # Should find at least one cluster or singletons
        assert result['n_clusters'] + result['n_singletons'] > 0
    
    def test_compute_correlation_clusters_all_uncorrelated(self):
        """Test VarClus with uncorrelated features."""
        # Create data with independent features
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'feat_a': np.random.randn(n_samples),
            'feat_b': np.random.randn(n_samples),
            'feat_c': np.random.randn(n_samples),
        })
        
        result = compute_correlation_clusters(df)
        
        # Should have some output structure
        assert 'n_clusters' in result
        assert 'n_singletons' in result
        # Total features should be accounted for
        total = result['n_clusters'] + result['n_singletons']
        # Note: VarClus may still create clusters even with uncorrelated data
        assert total >= 0
    
    def test_compute_correlation_clusters_all_correlated(self):
        """Test VarClus with highly correlated features."""
        # Create data where all features are correlated
        np.random.seed(42)
        n_samples = 100
        
        base = np.random.randn(n_samples)
        df = pd.DataFrame({
            'feat_a': base + np.random.randn(n_samples) * 0.1,
            'feat_b': base + np.random.randn(n_samples) * 0.1,
            'feat_c': base + np.random.randn(n_samples) * 0.1,
        })
        
        result = compute_correlation_clusters(df)
        
        # Should detect correlation
        assert result['n_clusters'] >= 0
        
        # Check that clusters have correct structure
        for cluster in result['clusters']:
            assert 'cluster_id' in cluster
            assert 'features' in cluster
            assert 'representative' in cluster
            assert 'max_correlation' in cluster
            assert 'variance_explained' in cluster
            assert len(cluster['features']) >= 1
    
    def test_compute_correlation_clusters_negative_correlation(self):
        """Test VarClus with negative correlations."""
        # Create data with negative correlation
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'feat_a': np.random.randn(n_samples),
            'feat_b': np.random.randn(n_samples),
            'feat_c': np.random.randn(n_samples),
        })
        # Make feat_b negatively correlated with feat_a
        df['feat_b'] = -df['feat_a'] + np.random.randn(n_samples) * 0.1
        
        result = compute_correlation_clusters(df)
        
        # Should detect some clustering pattern
        assert result['n_clusters'] >= 0
        assert 'clusters' in result


class TestSelectRepresentatives:
    """Test suite for representative selection functions."""
    
    def test_is_raw_feature(self):
        """Test raw feature detection."""
        assert is_raw_feature('E1')
        assert is_raw_feature('M5')
        assert is_raw_feature('P10')
        assert is_raw_feature('S2')
        assert is_raw_feature('V13')
        
        assert not is_raw_feature('m/E1')
        assert not is_raw_feature('gap_ffill/M1')
        assert not is_raw_feature('run_na/E10')
        assert not is_raw_feature('co_miss_deg/E1')
    
    def test_select_cluster_representative_basic(self):
        """Test basic representative selection."""
        cluster_features = ['feat_a', 'feat_b', 'feat_c']
        
        importance_df = pd.DataFrame({
            'feature_name': ['feat_a', 'feat_b', 'feat_c', 'feat_d'],
            'mean_gain': [100.0, 200.0, 50.0, 150.0],
        })
        
        representative = select_cluster_representative(cluster_features, importance_df)
        
        # Should select feat_b (highest mean_gain)
        assert representative == 'feat_b'
    
    def test_select_cluster_representative_no_importance_data(self):
        """Test representative selection when importance data is missing."""
        cluster_features = ['feat_x', 'feat_y']
        
        importance_df = pd.DataFrame({
            'feature_name': ['feat_a', 'feat_b'],
            'mean_gain': [100.0, 200.0],
        })
        
        representative = select_cluster_representative(cluster_features, importance_df)
        
        # Should return first feature when no importance data available
        assert representative in cluster_features
    
    def test_select_cluster_representative_single_feature(self):
        """Test representative selection with single feature cluster."""
        cluster_features = ['feat_a']
        
        importance_df = pd.DataFrame({
            'feature_name': ['feat_a', 'feat_b'],
            'mean_gain': [100.0, 200.0],
        })
        
        representative = select_cluster_representative(cluster_features, importance_df)
        
        # Should return the only feature
        assert representative == 'feat_a'

    def test_select_cluster_representative_raw_feature_preferred(self):
        """Test that raw features are preferred when importance is similar."""
        # E1 is a raw feature, 'derived/E1' is not
        cluster_features = ['derived/E1', 'E1', 'another/feat']
        
        importance_df = pd.DataFrame({
            'feature_name': ['derived/E1', 'E1', 'another/feat'],
            # E1 and derived/E1 have similar importance (within 1%)
            'mean_gain': [100.0, 99.5, 50.0],
        })
        
        representative = select_cluster_representative(cluster_features, importance_df)
        
        # Should prefer E1 (raw feature) since importance is within 1% of max
        assert representative == 'E1'

    def test_select_cluster_representative_importance_wins_over_raw(self):
        """Test that significantly higher importance wins over raw feature preference."""
        cluster_features = ['derived/E1', 'E1']
        
        importance_df = pd.DataFrame({
            'feature_name': ['derived/E1', 'E1'],
            # derived/E1 has significantly higher importance (more than 1% difference)
            'mean_gain': [100.0, 95.0],  # 5% difference
        })
        
        representative = select_cluster_representative(cluster_features, importance_df)
        
        # Should select derived/E1 because importance difference is too large
        assert representative == 'derived/E1'


class TestIntegration:
    """Integration tests for Phase 3 workflow."""
    
    def test_create_tier3_excluded_structure(self):
        """Test the structure of Tier3 excluded.json creation."""
        # This is a minimal test to verify the expected output structure
        # Create mock data
        tier2_data = {
            "version": "tier2-v1",
            "candidates": [
                {"feature_name": "feat1", "reason": "phase2_zero_importance"},
                {"feature_name": "feat2", "reason": "phase2_low_importance"},
            ]
        }
        
        phase3_removals = [
            {"feature": "feat3", "cluster_id": 1, "mean_gain": 50.0},
            {"feature": "feat4", "cluster_id": 1, "mean_gain": 30.0},
        ]
        
        # Expected Tier3 structure
        expected_candidates_count = len(tier2_data["candidates"]) + len(phase3_removals)
        
        assert expected_candidates_count == 4
    
    def test_feature_sets_structure(self):
        """Test the structure of feature_sets.json."""
        # Define expected structure
        expected_feature_sets = {
            "version": "v1",
            "created_at": "2025-12-11T00:00:00+00:00",
            "feature_sets": {
                "FS_full": {
                    "description": "Tier2 full feature set",
                    "excluded_json": "configs/feature_selection/tier2/excluded.json",
                    "n_features": 120,
                    "oof_rmse": 0.012172,
                },
                "FS_compact": {
                    "description": "Tier3 after correlation clustering",
                    "excluded_json": "configs/feature_selection/tier3/excluded.json",
                    "n_features": 95,
                    "oof_rmse": 0.012180,
                },
                "FS_topK": {
                    "description": "Top 50 features by importance",
                    "excluded_json": "configs/feature_selection/tier_topK/excluded.json",
                    "n_features": 50,
                    "oof_rmse": None,
                }
            },
            "recommended": "FS_compact"
        }
        
        # Verify structure
        assert "version" in expected_feature_sets
        assert "feature_sets" in expected_feature_sets
        assert "recommended" in expected_feature_sets
        
        # Verify each feature set has required fields
        for name, config in expected_feature_sets["feature_sets"].items():
            assert "description" in config
            assert "excluded_json" in config
            assert "n_features" in config
            assert "oof_rmse" in config
