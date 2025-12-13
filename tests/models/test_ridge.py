"""Unit tests for Ridge training script."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ridge.train_ridge import (  # noqa: E402
    _to_1d,
    parse_args,
)


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])

        assert args.data_dir == "data/raw"
        assert args.out_dir == "artifacts/models/ridge"
        assert args.n_splits == 5
        assert args.gap == 0
        assert args.alpha == 1.0
        assert args.fit_intercept is True
        assert args.solver == "auto"
        assert args.feature_tier == "tier3"
        assert args.random_state == 42

    def test_custom_hyperparameters(self):
        """Test custom hyperparameter values."""
        args = parse_args(
            [
                "--alpha",
                "10.0",
                "--solver",
                "lsqr",
                "--max-iter",
                "1000",
                "--tol",
                "0.001",
            ]
        )

        assert args.alpha == 10.0
        assert args.solver == "lsqr"
        assert args.max_iter == 1000
        assert args.tol == 0.001

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


class TestPredictionConversion:
    """Test prediction array conversion utilities."""

    def test_1d_array_unchanged(self):
        """Test that 1D arrays remain unchanged."""
        pred = np.array([1.0, 2.0, 3.0])
        result = _to_1d(pred)

        np.testing.assert_array_equal(result, pred)
        assert result.dtype == np.float64

    def test_2d_array_flattened(self):
        """Test that 2D arrays are flattened."""
        pred = np.array([[1.0], [2.0], [3.0]])
        result = _to_1d(pred)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)
        assert result.ndim == 1

    def test_list_converted(self):
        """Test that lists are converted to 1D arrays."""
        pred = [1.0, 2.0, 3.0]
        result = _to_1d(pred)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)


class TestRidgeRegression:
    """Test Ridge regression model behavior."""

    def test_ridge_basic_fit_predict(self):
        """Test basic fit and predict with Ridge."""
        from sklearn.linear_model import Ridge

        # Simple linear relationship
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + np.random.randn(100) * 0.1

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        # Check predictions are reasonable
        assert predictions.shape == (100,)
        assert np.isfinite(predictions).all()

        # Check that coefficients are learned
        assert model.coef_ is not None
        assert model.coef_.shape == (5,)

    def test_ridge_with_standardscaler(self):
        """Test Ridge with StandardScaler preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Create dataset with different scales
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100) * 1000,  # Large scale
                "feature2": np.random.randn(100) * 0.001,  # Small scale
                "feature3": np.random.randn(100),  # Normal scale
            }
        )
        y = X["feature1"] * 0.001 + X["feature2"] * 1000 + X["feature3"] * 1.0

        # Without StandardScaler - may perform poorly
        model_no_scale = Ridge(alpha=1.0, random_state=42)
        model_no_scale.fit(X, y)
        pred_no_scale = model_no_scale.predict(X)

        # With StandardScaler - should perform better
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )
        pipeline.fit(X, y)
        pred_scaled = pipeline.predict(X)

        # Both should produce finite predictions
        assert np.isfinite(pred_no_scale).all()
        assert np.isfinite(pred_scaled).all()

        # Check that pipeline can predict on new data
        X_test = pd.DataFrame(
            {
                "feature1": [1000.0, 2000.0],
                "feature2": [0.001, 0.002],
                "feature3": [1.0, 2.0],
            }
        )
        pred_test = pipeline.predict(X_test)
        assert pred_test.shape == (2,)
        assert np.isfinite(pred_test).all()

    def test_ridge_regularization_effect(self):
        """Test that alpha controls regularization strength."""
        from sklearn.linear_model import Ridge

        # Create correlated features
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

        # Low regularization (alpha close to 0)
        model_low = Ridge(alpha=0.001, random_state=42)
        model_low.fit(X, y)

        # High regularization
        model_high = Ridge(alpha=1000.0, random_state=42)
        model_high.fit(X, y)

        # High regularization should have smaller coefficients
        coef_norm_low = np.linalg.norm(model_low.coef_)
        coef_norm_high = np.linalg.norm(model_high.coef_)

        assert coef_norm_high < coef_norm_low


class TestPipelineStructure:
    """Test the Ridge pipeline structure."""

    def test_build_ridge_pipeline_structure(self):
        """Test that build_ridge_pipeline creates correct structure."""
        from src.models.ridge.train_ridge import build_ridge_pipeline

        # Mock configs
        su1_config = {}
        su5_config = {}
        preprocess_settings = {
            "m_group": {},
            "e_group": {},
            "i_group": {},
            "p_group": {},
            "s_group": {},
        }
        model_kwargs = {"alpha": 1.0, "random_state": 42}

        pipeline = build_ridge_pipeline(
            su1_config,
            su5_config,
            preprocess_settings,
            numeric_fill_value=0.0,
            model_kwargs=model_kwargs,
            random_state=42,
        )

        # Check pipeline steps
        step_names = [name for name, _ in pipeline.steps]

        # Must include all preprocessing steps
        assert "augment" in step_names
        assert "m_imputer" in step_names
        assert "e_imputer" in step_names
        assert "i_imputer" in step_names
        assert "p_imputer" in step_names
        assert "s_imputer" in step_names
        assert "final_imputer" in step_names

        # CRITICAL: Must include StandardScaler before Ridge
        assert "scaler" in step_names
        assert "model" in step_names

        # Verify order: scaler must come before model
        scaler_idx = step_names.index("scaler")
        model_idx = step_names.index("model")
        assert scaler_idx < model_idx, (
            f"StandardScaler must come before Ridge model, but scaler at index {scaler_idx} "
            f"comes after model at index {model_idx}"
        )

        # Verify final step is Ridge
        from sklearn.linear_model import Ridge

        final_step = pipeline.steps[-1][1]
        assert isinstance(final_step, Ridge), (
            f"Final pipeline step should be Ridge, but got {type(final_step).__name__}"
        )


class TestFeatureExclusion:
    """Test feature exclusion functionality."""

    def test_apply_feature_exclusion(self):
        """Test that feature exclusion removes correct columns."""
        from src.models.ridge.train_ridge import apply_feature_exclusion_to_augmented

        # Create test dataframe
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3],
                "feature_b": [4, 5, 6],
                "feature_c": [7, 8, 9],
            }
        )

        # Mock exclusion - this will fail if tier3/excluded.json doesn't exist
        # But we can test the function structure
        try:
            result = apply_feature_exclusion_to_augmented(
                df, tier="tier3", verbose=False
            )
            # If successful, result should be a DataFrame
            assert isinstance(result, pd.DataFrame)
            # Number of columns should be <= original
            assert result.shape[1] <= df.shape[1]
        except FileNotFoundError:
            # Expected if excluded.json doesn't exist in test environment
            pytest.skip("Feature exclusion config not available in test environment")


class TestDataValidation:
    """Test data validation and error handling."""

    def test_to_1d_handles_empty_array(self):
        """Test that _to_1d handles empty arrays."""
        pred = np.array([])
        result = _to_1d(pred)

        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_to_1d_handles_single_value(self):
        """Test that _to_1d handles single values."""
        pred = np.array([42.0])
        result = _to_1d(pred)

        assert result.shape == (1,)
        np.testing.assert_array_equal(result, np.array([42.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
