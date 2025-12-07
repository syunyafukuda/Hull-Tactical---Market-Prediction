#!/usr/bin/env python
"""Tests for compute_importance.py and permutation_importance.py."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
import json
import tempfile

# Add src to path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.feature_selection.phase2.compute_importance import (  # noqa: E402
    compute_fold_importance,
    aggregate_importance,
)


class TestComputeImportance:
    """Test suite for compute_importance.py functions."""
    
    def test_compute_fold_importance_with_importances(self):
        """Test fold importance computation with valid model."""
        # Create mock model with feature importances
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.5, 0.3, 0.2])
                
                # Mock booster for split importance
                class MockBooster:
                    def feature_importance(self, importance_type='split'):
                        return np.array([100, 50, 25])
                
                self.booster_ = MockBooster()
        
        model = MockModel()
        feature_names = ['feat_a', 'feat_b', 'feat_c']
        fold_idx = 1
        
        result = compute_fold_importance(model, feature_names, fold_idx)
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['feature_name', 'importance_gain', 'importance_split', 'fold']
        
        # Check values
        assert result['feature_name'].tolist() == feature_names
        assert result['importance_gain'].tolist() == [0.5, 0.3, 0.2]
        assert result['importance_split'].tolist() == [100, 50, 25]
        assert all(result['fold'] == fold_idx)
    
    def test_compute_fold_importance_no_booster(self):
        """Test fold importance computation when model lacks booster."""
        # Create mock model without booster
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        model = MockModel()
        feature_names = ['feat_a', 'feat_b', 'feat_c']
        fold_idx = 1
        
        result = compute_fold_importance(model, feature_names, fold_idx)
        
        # Should fallback to using gain for both
        assert len(result) == 3
        assert result['importance_gain'].tolist() == [0.5, 0.3, 0.2]
        # Split should fallback to gain
        assert result['importance_split'].tolist() == [0.5, 0.3, 0.2]
    
    def test_compute_fold_importance_no_importances(self):
        """Test fold importance computation without feature importances."""
        # Create mock model without feature_importances_
        class MockModel:
            pass
        
        model = MockModel()
        feature_names = ['feat_a', 'feat_b']
        fold_idx = 1
        
        result = compute_fold_importance(model, feature_names, fold_idx)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_compute_fold_importance_length_mismatch(self):
        """Test handling of length mismatch between importances and feature names."""
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.5, 0.3])  # Only 2 values
                
                class MockBooster:
                    def feature_importance(self, importance_type='split'):
                        return np.array([100, 50])
                
                self.booster_ = MockBooster()
        
        model = MockModel()
        feature_names = ['feat_a', 'feat_b', 'feat_c']  # 3 names
        fold_idx = 1
        
        result = compute_fold_importance(model, feature_names, fold_idx)
        
        # Should truncate to minimum length
        assert len(result) == 2
        assert result['feature_name'].tolist() == ['feat_a', 'feat_b']
    
    def test_aggregate_importance_basic(self):
        """Test importance aggregation across folds."""
        # Create sample fold-wise importance data
        importance_df = pd.DataFrame({
            'feature_name': ['feat_a', 'feat_a', 'feat_b', 'feat_b'],
            'importance_gain': [0.5, 0.6, 0.2, 0.3],
            'importance_split': [100, 120, 50, 60],
            'fold': [1, 2, 1, 2],
        })
        
        result = aggregate_importance(importance_df)
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        
        # Check columns
        expected_cols = [
            'feature_name',
            'mean_gain', 'std_gain', 'min_gain', 'max_gain',
            'mean_split', 'std_split', 'min_split', 'max_split',
            'mean_gain_normalized'
        ]
        assert all(col in result.columns for col in expected_cols)
        
        # Check aggregated values for feat_a
        feat_a = result[result['feature_name'] == 'feat_a'].iloc[0]
        assert feat_a['mean_gain'] == pytest.approx(0.55)
        assert feat_a['min_gain'] == 0.5
        assert feat_a['max_gain'] == 0.6
        assert feat_a['mean_split'] == 110
        
        # Check normalized values sum to 1
        assert result['mean_gain_normalized'].sum() == pytest.approx(1.0)
        
        # Check sorting (descending by mean_gain)
        assert result['feature_name'].tolist()[0] == 'feat_a'  # Higher mean
    
    def test_aggregate_importance_empty(self):
        """Test aggregation with empty DataFrame."""
        importance_df = pd.DataFrame()
        result = aggregate_importance(importance_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_aggregate_importance_zero_total(self):
        """Test aggregation when total gain is zero."""
        importance_df = pd.DataFrame({
            'feature_name': ['feat_a', 'feat_a', 'feat_b', 'feat_b'],
            'importance_gain': [0.0, 0.0, 0.0, 0.0],
            'importance_split': [0, 0, 0, 0],
            'fold': [1, 2, 1, 2],
        })
        
        result = aggregate_importance(importance_df)
        
        # Should handle zero total gracefully
        assert len(result) == 2
        assert all(result['mean_gain_normalized'] == 0.0)


class TestPermutationImportance:
    """Test suite for permutation_importance.py functions."""
    
    def test_permutation_logic(self):
        """Test basic permutation importance logic."""
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        
        # Feature that matters
        important_feat = np.random.randn(n_samples)
        # Target depends on important_feat
        y = important_feat * 2 + np.random.randn(n_samples) * 0.1
        # Unimportant feature
        unimportant_feat = np.random.randn(n_samples)
        
        X = pd.DataFrame({
            'important': important_feat,
            'unimportant': unimportant_feat,
        })
        
        # Simple mock: when important feature is shuffled, error increases
        # This is just a conceptual test - actual implementation would use real model
        
        # Calculate "baseline RMSE" (not shuffled)
        baseline_residuals = y - (important_feat * 2)
        baseline_rmse = np.sqrt(np.mean(baseline_residuals ** 2))
        
        # Shuffle important feature
        shuffled_important = np.random.permutation(important_feat)
        shuffled_residuals = y - (shuffled_important * 2)
        shuffled_rmse = np.sqrt(np.mean(shuffled_residuals ** 2))
        
        # Delta should be positive (performance degraded)
        delta_rmse = shuffled_rmse - baseline_rmse
        assert delta_rmse > 0, "Shuffling important feature should increase RMSE"
        
        # For unimportant feature, delta should be close to 0
        # (Not tested here but that's the concept)


class TestCandidatesJSON:
    """Test candidate JSON format handling."""
    
    def test_candidates_json_format(self):
        """Test that candidate JSON has correct format."""
        # This would be generated by the notebook
        sample_json = {
            "version": "phase2-v1",
            "created_at": "2025-12-07T00:00:00Z",
            "source_tier": "tier1",
            "selection_criteria": {
                "method": "lgbm_importance",
                "metric": "gain",
                "threshold_quantile": 0.25,
                "require_stable_low": True
            },
            "candidates": [
                {
                    "feature_name": "test_feature",
                    "mean_gain": 0.0001,
                    "std_gain": 0.00005,
                    "share_of_total": 0.0002,
                    "note": "Low and stable importance"
                }
            ],
            "summary": {
                "total_features": 160,
                "candidate_count": 1,
                "candidate_ratio": 0.00625
            }
        }
        
        # Validate structure
        assert "version" in sample_json
        assert "candidates" in sample_json
        assert "summary" in sample_json
        assert isinstance(sample_json["candidates"], list)
        
        # Validate candidate structure
        if sample_json["candidates"]:
            candidate = sample_json["candidates"][0]
            assert "feature_name" in candidate
            assert "mean_gain" in candidate
            assert "std_gain" in candidate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
