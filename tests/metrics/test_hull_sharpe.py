"""Tests for Hull Competition Sharpe metric implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics.hull_sharpe import (
    ANNUALIZATION,
    VOL_TOLERANCE,
    HullSharpeResult,
    compute_hull_sharpe,
    evaluate_hull_metric,
    hull_sharpe_score,
    validate_prediction,
)


class TestValidatePrediction:
    """Tests for validate_prediction function."""
    
    def test_valid_prediction_scalar(self):
        """Valid scalar predictions should return True."""
        assert validate_prediction(0.0) is True
        assert validate_prediction(1.0) is True
        assert validate_prediction(2.0) is True
        assert validate_prediction(0.5) is True
        
    def test_invalid_prediction_below_range(self):
        """Predictions below 0 should return False."""
        assert validate_prediction(-0.1) is False
        assert validate_prediction(-1.0) is False
        
    def test_invalid_prediction_above_range(self):
        """Predictions above 2 should return False."""
        assert validate_prediction(2.1) is False
        assert validate_prediction(3.0) is False
        
    def test_valid_array(self):
        """Valid arrays should return True."""
        arr = np.array([0.0, 1.0, 2.0])
        assert validate_prediction(arr) is True
        
    def test_invalid_array_below(self):
        """Arrays with values below 0 should return False."""
        arr = np.array([-0.1, 1.0, 2.0])
        assert validate_prediction(arr) is False
        
    def test_invalid_array_above(self):
        """Arrays with values above 2 should return False."""
        arr = np.array([0.0, 1.0, 2.1])
        assert validate_prediction(arr) is False
        
    def test_nan_in_array(self):
        """Arrays with NaN should return False."""
        arr = np.array([0.0, np.nan, 1.0])
        assert validate_prediction(arr) is False


class TestComputeHullSharpe:
    """Tests for compute_hull_sharpe function."""
    
    def test_neutral_position(self):
        """Neutral position (1.0) should match market performance."""
        np.random.seed(42)
        n = 252  # one year of trading days
        
        forward_returns = np.random.normal(0.0005, 0.015, n)
        rf = np.full(n, 0.00015)  # ~4% annual
        position = np.ones(n)  # neutral
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        assert isinstance(result, HullSharpeResult)
        assert not np.isnan(result.raw_sharpe)
        # Neutral position should have no return penalty (matches market)
        assert result.return_penalty == pytest.approx(0.0, abs=0.001)
        
    def test_zero_position_positive_market(self):
        """Zero position (all cash) should underperform in positive market."""
        np.random.seed(42)
        n = 252
        
        # Positive market returns
        forward_returns = np.full(n, 0.001)  # 0.1% daily = ~25% annual
        rf = np.full(n, 0.00015)
        position = np.zeros(n)  # 100% cash
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Strategy should have lower returns than market
        assert result.strategy_mean < result.market_mean
        # Should get a return penalty
        assert result.return_penalty > 0
        
    def test_leveraged_position_negative_market(self):
        """Leveraged position (2.0) should amplify losses in down market."""
        n = 252
        
        # Negative market returns
        forward_returns = np.full(n, -0.001)  # -0.1% daily
        rf = np.full(n, 0.00015)
        position = np.full(n, 2.0)  # 2x leverage
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Leveraged losses should be larger magnitude
        assert result.strategy_mean < result.market_mean
        
    def test_vol_penalty_high_volatility(self):
        """High volatility ratio should trigger penalty."""
        np.random.seed(42)
        n = 252
        
        # Low vol market
        forward_returns = np.random.normal(0.0, 0.005, n)
        rf = np.full(n, 0.00015)
        
        # Oscillating position creates high strategy vol
        position = np.where(np.arange(n) % 2 == 0, 0.0, 2.0)
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Vol ratio > 1.2 should trigger penalty
        if result.vol_ratio > 1.0 + VOL_TOLERANCE:
            assert result.vol_penalty > 0
            
    def test_vol_penalty_low_volatility(self):
        """Low volatility ratio should also trigger penalty."""
        n = 252
        
        # High vol market
        np.random.seed(42)
        forward_returns = np.random.normal(0.0, 0.03, n)
        rf = np.full(n, 0.00015)
        
        # Constant position = lower strategy vol
        position = np.full(n, 0.5)  # 50% market exposure
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Vol ratio < 0.8 should trigger penalty
        if result.vol_ratio < 1.0 - VOL_TOLERANCE:
            assert result.vol_penalty > 0
            
    def test_annualization_factor(self):
        """Verify annualization factor is applied correctly."""
        assert ANNUALIZATION == pytest.approx(math.sqrt(252), abs=0.001)
        
    def test_result_dataclass_fields(self):
        """HullSharpeResult should have all expected fields."""
        n = 100
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        position = np.ones(n)
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        assert hasattr(result, "raw_sharpe")
        assert hasattr(result, "vol_penalty")
        assert hasattr(result, "return_penalty")
        assert hasattr(result, "final_score")
        assert hasattr(result, "strategy_mean")
        assert hasattr(result, "strategy_std")
        assert hasattr(result, "market_mean")
        assert hasattr(result, "market_std")
        assert hasattr(result, "vol_ratio")
        

class TestHullSharpeScore:
    """Tests for hull_sharpe_score convenience function."""
    
    def test_returns_float(self):
        """Should return a single float value."""
        n = 100
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        position = np.ones(n)
        
        score = hull_sharpe_score(position, forward_returns, rf)
        
        assert isinstance(score, float)
        
    def test_matches_compute_hull_sharpe(self):
        """Should match final_score from compute_hull_sharpe."""
        np.random.seed(42)
        n = 100
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        position = np.random.uniform(0.5, 1.5, n)
        
        full_result = compute_hull_sharpe(position, forward_returns, rf)
        score = hull_sharpe_score(position, forward_returns, rf)
        
        assert score == pytest.approx(full_result.final_score)


class TestEvaluateHullMetric:
    """Tests for evaluate_hull_metric function."""
    
    def test_default_mapping(self):
        """Test default prediction-to-position mapping."""
        n = 100
        np.random.seed(42)
        
        # Predictions around 0 (typical return predictions)
        y_pred_returns = np.random.normal(0.0, 0.005, n)
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        
        result = evaluate_hull_metric(y_pred_returns, forward_returns, rf)
        
        assert "final_score" in result
        assert "raw_sharpe" in result
        assert "vol_penalty" in result
        assert "return_penalty" in result
        
    def test_custom_mapping(self):
        """Test custom multiplier and offset."""
        n = 100
        np.random.seed(42)
        
        y_pred_returns = np.random.normal(0.0, 0.01, n)
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        
        result1 = evaluate_hull_metric(
            y_pred_returns, forward_returns, rf,
            mult=50.0, offset=1.0,
        )
        result2 = evaluate_hull_metric(
            y_pred_returns, forward_returns, rf,
            mult=100.0, offset=1.0,
        )
        
        # Different mappings should give different results
        assert result1["final_score"] != result2["final_score"]
        
    def test_result_includes_metadata(self):
        """Result should include position statistics."""
        n = 100
        np.random.seed(42)
        
        y_pred_returns = np.random.normal(0.0, 0.005, n)
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        
        result = evaluate_hull_metric(y_pred_returns, forward_returns, rf)
        
        assert "strategy_mean" in result
        assert "strategy_std" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_short_series(self):
        """Very short series should still compute."""
        n = 10
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        position = np.ones(n)
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        assert not np.isnan(result.final_score)
        
    def test_zero_volatility(self):
        """Zero volatility should be handled gracefully."""
        n = 100
        forward_returns = np.full(n, 0.001)  # constant returns
        rf = np.full(n, 0.0001)
        position = np.ones(n)
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Should not raise, may return inf or handle gracefully
        assert not np.isnan(result.raw_sharpe) or np.isinf(result.raw_sharpe)
        
    def test_all_cash_position(self):
        """All cash position (0.0) should work."""
        n = 100
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        position = np.zeros(n)
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Strategy returns = rf when position = 0
        assert not np.isnan(result.final_score)
        
    def test_all_leveraged_position(self):
        """All leveraged position (2.0) should work."""
        n = 100
        forward_returns = np.random.normal(0.0, 0.01, n)
        rf = np.full(n, 0.0001)
        position = np.full(n, 2.0)
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        assert not np.isnan(result.final_score)


class TestRegressionValues:
    """Regression tests with known expected values."""
    
    def test_known_sharpe_calculation(self):
        """Test against known calculation."""
        # Case with non-zero volatility: random returns
        np.random.seed(123)
        n = 252
        forward_returns = np.random.normal(0.001, 0.01, n)  # ~0.1% mean, 1% std
        rf = np.full(n, 0.0001)  # 0.01% daily
        position = np.ones(n)  # neutral
        
        result = compute_hull_sharpe(position, forward_returns, rf)
        
        # Neutral position: strategy = market
        # Check that strategy mean matches market mean
        assert result.strategy_mean == pytest.approx(result.market_mean, rel=0.01)
        # Should have no return penalty (strategy == market)
        assert result.return_penalty == pytest.approx(0.0, abs=0.001)
