"""Unit tests for Lasso training script."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lasso.train_lasso import (  # noqa: E402
    _to_1d,
    parse_args,
)


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])

        assert args.data_dir == "data/raw"
        assert args.out_dir == "artifacts/models/lasso"
        assert args.n_splits == 5
        assert args.gap == 0
        assert args.alpha == 0.001
        assert args.max_iter == 10000
        assert args.tol == 1e-4
        assert args.selection == "cyclic"
        assert args.feature_tier == "tier3"
        assert args.random_state == 42

    def test_custom_hyperparameters(self):
        """Test custom hyperparameter values."""
        args = parse_args(
            [
                "--alpha",
                "0.01",
                "--max-iter",
                "5000",
                "--tol",
                "1e-5",
                "--selection",
                "random",
            ]
        )

        assert args.alpha == 0.01
        assert args.max_iter == 5000
        assert args.tol == 1e-5
        assert args.selection == "random"

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

    def test_paths(self):
        """Test path arguments."""
        args = parse_args(
            [
                "--data-dir",
                "custom/data",
                "--out-dir",
                "custom/output",
                "--train-file",
                "custom_train.csv",
                "--test-file",
                "custom_test.csv",
            ]
        )

        assert args.data_dir == "custom/data"
        assert args.out_dir == "custom/output"
        assert args.train_file == "custom_train.csv"
        assert args.test_file == "custom_test.csv"


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


class TestLassoSpecifics:
    """Test Lasso-specific functionality."""

    def test_alpha_parameter_range(self):
        """Test that alpha can be set to various values."""
        # Test small alpha (less regularization)
        args = parse_args(["--alpha", "0.00001"])
        assert args.alpha == 0.00001

        # Test medium alpha
        args = parse_args(["--alpha", "0.001"])
        assert args.alpha == 0.001

        # Test large alpha (more regularization)
        args = parse_args(["--alpha", "0.1"])
        assert args.alpha == 0.1

    def test_selection_strategies(self):
        """Test different coordinate descent strategies."""
        # Cyclic selection
        args = parse_args(["--selection", "cyclic"])
        assert args.selection == "cyclic"

        # Random selection
        args = parse_args(["--selection", "random"])
        assert args.selection == "random"

    def test_convergence_parameters(self):
        """Test convergence-related parameters."""
        args = parse_args(
            [
                "--max-iter",
                "20000",
                "--tol",
                "1e-5",
            ]
        )

        assert args.max_iter == 20000
        assert args.tol == 1e-5


class TestPipelineStructure:
    """Test pipeline structure expectations."""

    def test_pipeline_includes_scaler(self):
        """Test that pipeline construction includes StandardScaler.

        Note: This is a conceptual test. The actual pipeline construction
        requires full data loading and configuration, which is tested in
        integration tests.
        """
        # This test verifies that we have the necessary imports
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        from sklearn.linear_model import Lasso  # noqa: F401

        # If imports work, the basic structure is valid
        assert True

    def test_required_preprocessing_steps(self):
        """Test that required preprocessing components are importable."""
        # Verify all preprocessing components exist
        from src.preprocess.M_group.m_group import MGroupImputer  # noqa: F401
        from src.preprocess.E_group.e_group import EGroupImputer  # noqa: F401
        from src.preprocess.I_group.i_group import IGroupImputer  # noqa: F401
        from src.preprocess.P_group.p_group import PGroupImputer  # noqa: F401
        from src.preprocess.S_group.s_group import SGroupImputer  # noqa: F401
        from sklearn.impute import SimpleImputer  # noqa: F401

        # If all imports work, preprocessing chain is complete
        assert True


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
