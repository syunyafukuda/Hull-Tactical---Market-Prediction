"""Unit tests for ElasticNet training script."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.elasticnet.train_elasticnet import (  # noqa: E402
    _to_1d,
    parse_args,
)


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])

        assert args.data_dir == "data/raw"
        assert args.out_dir == "artifacts/models/elasticnet"
        assert args.n_splits == 5
        assert args.gap == 0
        assert args.alpha == 0.001
        assert args.l1_ratio == 0.5
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
                "--l1-ratio",
                "0.7",
                "--max-iter",
                "5000",
                "--tol",
                "1e-5",
                "--selection",
                "random",
            ]
        )

        assert args.alpha == 0.01
        assert args.l1_ratio == 0.7
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

    def test_l1_ratio_range(self):
        """Test that l1_ratio values are properly parsed."""
        # Test Ridge-like configuration (low L1 ratio)
        args_ridge = parse_args(["--l1-ratio", "0.1"])
        assert args_ridge.l1_ratio == 0.1

        # Test Lasso-like configuration (high L1 ratio)
        args_lasso = parse_args(["--l1-ratio", "0.9"])
        assert args_lasso.l1_ratio == 0.9

        # Test balanced configuration
        args_balanced = parse_args(["--l1-ratio", "0.5"])
        assert args_balanced.l1_ratio == 0.5


class TestUtilityFunctions:
    """Test utility functions."""

    def test_to_1d_with_1d_array(self):
        """Test _to_1d with 1D array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_1d(arr)

        assert result.ndim == 1
        assert len(result) == 3
        assert np.allclose(result, arr)

    def test_to_1d_with_2d_array(self):
        """Test _to_1d with 2D array."""
        arr = np.array([[1.0], [2.0], [3.0]])
        result = _to_1d(arr)

        assert result.ndim == 1
        assert len(result) == 3
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_to_1d_with_list(self):
        """Test _to_1d with list input."""
        lst = [1.0, 2.0, 3.0]
        result = _to_1d(lst)

        assert result.ndim == 1
        assert len(result) == 3
        assert np.allclose(result, lst)

    def test_to_1d_preserves_dtype(self):
        """Test _to_1d converts to float."""
        arr = np.array([1, 2, 3], dtype=int)
        result = _to_1d(arr)

        assert result.dtype == float


class TestElasticNetConfiguration:
    """Test ElasticNet-specific configuration options."""

    def test_alpha_values(self):
        """Test different alpha (regularization strength) values."""
        # Test weak regularization
        args_weak = parse_args(["--alpha", "0.0001"])
        assert args_weak.alpha == 0.0001

        # Test moderate regularization
        args_moderate = parse_args(["--alpha", "0.01"])
        assert args_moderate.alpha == 0.01

        # Test strong regularization
        args_strong = parse_args(["--alpha", "0.1"])
        assert args_strong.alpha == 0.1

    def test_selection_methods(self):
        """Test coordinate descent selection methods."""
        # Test cyclic selection
        args_cyclic = parse_args(["--selection", "cyclic"])
        assert args_cyclic.selection == "cyclic"

        # Test random selection
        args_random = parse_args(["--selection", "random"])
        assert args_random.selection == "random"

    def test_convergence_settings(self):
        """Test convergence tolerance and iteration settings."""
        args = parse_args(
            [
                "--max-iter",
                "20000",
                "--tol",
                "1e-6",
            ]
        )

        assert args.max_iter == 20000
        assert args.tol == 1e-6


class TestPipelineConfiguration:
    """Test pipeline configuration options."""

    def test_feature_exclusion_flags(self):
        """Test feature exclusion control."""
        # Test with feature exclusion (default)
        args_with_exclusion = parse_args([])
        assert not args_with_exclusion.no_feature_exclusion

        # Test without feature exclusion
        args_without_exclusion = parse_args(["--no-feature-exclusion"])
        assert args_without_exclusion.no_feature_exclusion

    def test_output_control_flags(self):
        """Test output control flags."""
        # Test default (with artifacts)
        args_default = parse_args([])
        assert not args_default.no_artifacts

        # Test without artifacts
        args_no_artifacts = parse_args(["--no-artifacts"])
        assert args_no_artifacts.no_artifacts

        # Test dry-run mode
        args_dry_run = parse_args(["--dry-run"])
        assert args_dry_run.dry_run


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
