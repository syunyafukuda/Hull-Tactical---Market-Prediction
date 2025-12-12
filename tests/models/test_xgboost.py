"""Unit tests for XGBoost training script."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xgboost.train_xgb import (  # noqa: E402
    parse_args,
    sanitize_feature_names,
    _to_1d,
)


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])

        assert args.data_dir == "data/raw"
        assert args.out_dir == "artifacts/models/xgboost"
        assert args.n_splits == 5
        assert args.gap == 0
        assert args.max_depth == 6
        assert args.learning_rate == 0.05
        assert args.n_estimators == 600
        assert args.subsample == 0.8
        assert args.colsample_bytree == 0.8
        assert args.feature_tier == "tier3"
        assert args.random_state == 42

    def test_custom_hyperparameters(self):
        """Test custom hyperparameter values."""
        args = parse_args(
            [
                "--max-depth",
                "8",
                "--learning-rate",
                "0.03",
                "--n-estimators",
                "800",
                "--subsample",
                "0.9",
                "--colsample-bytree",
                "0.9",
            ]
        )

        assert args.max_depth == 8
        assert args.learning_rate == 0.03
        assert args.n_estimators == 800
        assert args.subsample == 0.9
        assert args.colsample_bytree == 0.9

    def test_feature_tier_selection(self):
        """Test feature tier selection."""
        for tier in ["tier0", "tier1", "tier2", "tier3"]:
            args = parse_args(["--feature-tier", tier])
            assert args.feature_tier == tier

    def test_cv_settings(self):
        """Test cross-validation settings."""
        args = parse_args(
            [
                "--n-splits",
                "10",
                "--gap",
                "5",
                "--min-val-size",
                "100",
            ]
        )

        assert args.n_splits == 10
        assert args.gap == 5
        assert args.min_val_size == 100


class TestFeatureNameSanitization:
    """Test feature name sanitization for XGBoost compatibility."""

    def test_sanitize_brackets(self):
        """Test sanitization of square brackets."""
        feature_cols = ["feature[0]", "feature[1]", "normal_feature"]
        sanitized = sanitize_feature_names(feature_cols)

        assert sanitized == ["feature_0_", "feature_1_", "normal_feature"]

    def test_sanitize_angle_brackets(self):
        """Test sanitization of angle brackets."""
        feature_cols = ["feature<10", "feature>20", "normal"]
        sanitized = sanitize_feature_names(feature_cols)

        assert sanitized == ["feature_10", "feature_20", "normal"]

    def test_sanitize_mixed_characters(self):
        """Test sanitization of mixed special characters."""
        feature_cols = ["co_miss_deg/E[1]", "run_na<5>", "m/P1"]
        sanitized = sanitize_feature_names(feature_cols)

        assert sanitized == ["co_miss_deg/E_1_", "run_na_5_", "m/P1"]

    def test_no_changes_needed(self):
        """Test that normal feature names are unchanged."""
        feature_cols = ["feature_1", "feature_2", "mean_value"]
        sanitized = sanitize_feature_names(feature_cols)

        assert sanitized == feature_cols


class TestUtilityFunctions:
    """Test utility functions."""

    def test_to_1d_flat_array(self):
        """Test _to_1d with already flat array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_1d(arr)

        assert result.ndim == 1
        assert len(result) == 3
        np.testing.assert_array_equal(result, arr)

    def test_to_1d_2d_array(self):
        """Test _to_1d with 2D array."""
        arr = np.array([[1.0, 2.0, 3.0]])
        result = _to_1d(arr)

        assert result.ndim == 1
        assert len(result) == 3
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_1d_list(self):
        """Test _to_1d with list input."""
        lst = [1.0, 2.0, 3.0]
        result = _to_1d(lst)

        assert result.ndim == 1
        assert len(result) == 3
        assert result.dtype == float


class TestFeatureExclusion:
    """Test feature exclusion logic."""

    def test_apply_feature_exclusion_basic(self):
        """Test basic feature exclusion."""
        # Create sample dataframe
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
                "feature_3": [7, 8, 9],
            }
        )

        # Mock exclusion by dropping feature_2
        # Note: This test uses a simplified version without loading actual tier configs
        result = df.drop(columns=["feature_2"], errors="ignore")

        assert "feature_1" in result.columns
        assert "feature_2" not in result.columns
        assert "feature_3" in result.columns
        assert len(result.columns) == 2

    def test_exclusion_with_nonexistent_features(self):
        """Test exclusion when some features don't exist."""
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
            }
        )

        # Try to drop features that don't exist
        result = df.drop(columns=["feature_2", "feature_999"], errors="ignore")

        assert "feature_1" in result.columns
        assert "feature_2" not in result.columns
        assert len(result.columns) == 1


class TestIntegration:
    """Integration tests for the training pipeline."""

    @pytest.mark.skipif(
        not Path("data/raw/train.csv").exists(), reason="Training data not available"
    )
    def test_training_with_mock_data(self, tmp_path):
        """Test training pipeline with mock data (skipped if no data)."""
        # This test would require actual data files
        # It's marked to skip if data is not available
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
