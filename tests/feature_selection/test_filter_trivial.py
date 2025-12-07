#!/usr/bin/env python
"""Tests for filter_trivial.py feature filtering functions."""

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
sys.path.insert(0, str(PROJECT_ROOT))

from feature_selection.filter_trivial_phase1 import (  # noqa: E402
    find_low_variance_features,
    find_high_missing_features,
    find_high_correlation_features,
)
from src.feature_selection.evaluate_baseline import _load_exclude_features  # noqa: E402


class TestFilterTrivial:
    """Test suite for filter_trivial.py functions."""
    
    def test_find_low_variance_features(self):
        """Test low variance feature detection."""
        # Create test data with low variance features
        np.random.seed(42)
        df = pd.DataFrame({
            'constant': [1.0] * 100,  # Zero variance
            'near_constant': [1.0] * 99 + [1.00001],  # Very low variance
            'normal': np.random.randn(100),  # Normal variance
        })
        
        candidates = find_low_variance_features(df, threshold=1e-10)
        
        # Extract feature names
        candidate_names = {c['feature_name'] for c in candidates}
        
        # Should detect constant feature
        assert 'constant' in candidate_names
        # Should detect near-constant feature
        assert 'near_constant' in candidate_names
        # Should not detect normal variance feature
        assert 'normal' not in candidate_names
        
    def test_find_high_missing_features(self):
        """Test high missing rate feature detection."""
        # Create test data with high missing features
        df = pd.DataFrame({
            'mostly_missing': [np.nan] * 100,  # 100% missing
            'some_missing': [np.nan] * 50 + [1.0] * 50,  # 50% missing
            'no_missing': [1.0] * 100,  # No missing
        })
        
        candidates = find_high_missing_features(df, threshold=0.99)
        
        # Extract feature names
        candidate_names = {c['feature_name'] for c in candidates}
        
        # Should detect mostly missing feature
        assert 'mostly_missing' in candidate_names
        # Should not detect features below threshold
        assert 'some_missing' not in candidate_names
        assert 'no_missing' not in candidate_names
        
    def test_find_high_correlation_features(self):
        """Test high correlation feature detection."""
        # Create test data with highly correlated features
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            'feat_a': base,
            'feat_b': base + np.random.randn(100) * 0.0001,  # Almost identical to feat_a
            'feat_c': np.random.randn(100),  # Independent
        })
        
        # Without importance data (alphabetical order)
        candidates = find_high_correlation_features(df, threshold=0.999, importance_df=None)
        
        # Should detect one of the correlated pair
        assert len(candidates) > 0
        candidate_names = {c['feature_name'] for c in candidates}
        # feat_b should be dropped (alphabetically later)
        assert 'feat_b' in candidate_names
        assert 'feat_a' not in candidate_names
        
    def test_find_high_correlation_with_importance(self):
        """Test correlation filtering with importance data."""
        # Create test data
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            'feat_a': base,
            'feat_b': base + np.random.randn(100) * 0.0001,
            'feat_c': np.random.randn(100),
        })
        
        # Create importance data where feat_a has lower importance
        importance_df = pd.DataFrame({
            'feature_name': ['feat_a', 'feat_b', 'feat_c'],
            'mean_gain': [0.1, 0.5, 0.3],  # feat_a has lowest importance
        })
        
        candidates = find_high_correlation_features(
            df, threshold=0.999, importance_df=importance_df
        )
        
        # Should drop feat_a (lower importance)
        candidate_names = {c['feature_name'] for c in candidates}
        assert 'feat_a' in candidate_names
        assert 'feat_b' not in candidate_names
        
    def test_load_exclude_features(self):
        """Test loading exclude features from JSON."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_data = {
                "version": "phase1-v1",
                "candidates": [
                    {"feature_name": "feat1", "reason": "low_variance"},
                    {"feature_name": "feat2", "reason": "high_missing"},
                    {"feature_name": "feat1", "reason": "high_correlation"},  # Duplicate
                ]
            }
            json.dump(json_data, f)
            temp_path = f.name
        
        try:
            exclude_set = _load_exclude_features(temp_path)
            
            # Should have unique features
            assert len(exclude_set) == 2
            assert 'feat1' in exclude_set
            assert 'feat2' in exclude_set
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_load_exclude_features_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            _load_exclude_features('/nonexistent/path.json')
    
    def test_load_exclude_features_invalid_format(self):
        """Test error handling for invalid JSON format."""
        # Create temporary invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "format"}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                _load_exclude_features(temp_path)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
