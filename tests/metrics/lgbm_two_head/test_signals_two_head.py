"""Unit tests for two-head signal functions."""

from __future__ import annotations

import numpy as np

from src.models.common.signals_two_head import (
    TwoHeadPositionConfig,
    map_positions_from_forward_rf,
    map_positions_from_two_head_config,
    compute_hull_sharpe_two_head,
    optimize_x_parameter,
)


class TestTwoHeadPositionConfig:
    """Tests for TwoHeadPositionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TwoHeadPositionConfig()
        assert config.x == 0.0
        assert config.clip_min == 0.0
        assert config.clip_max == 2.0
        assert config.epsilon == 1e-8
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = TwoHeadPositionConfig(
            x=0.001,
            clip_min=0.5,
            clip_max=1.5,
            epsilon=1e-6,
        )
        assert config.x == 0.001
        assert config.clip_min == 0.5
        assert config.clip_max == 1.5
        assert config.epsilon == 1e-6
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = TwoHeadPositionConfig(x=0.001)
        d = config.to_dict()
        assert d["x"] == 0.001
        assert "clip_min" in d
        assert "clip_max" in d
        assert "epsilon" in d
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {"x": 0.002, "clip_min": 0.1, "clip_max": 1.9}
        config = TwoHeadPositionConfig.from_dict(d)
        assert config.x == 0.002
        assert config.clip_min == 0.1
        assert config.clip_max == 1.9
        assert config.epsilon == 1e-8  # default
    
    def test_from_dict_with_missing_keys(self):
        """Test deserialization with missing keys uses defaults."""
        d = {"x": 0.001}
        config = TwoHeadPositionConfig.from_dict(d)
        assert config.x == 0.001
        assert config.clip_min == 0.0
        assert config.clip_max == 2.0


class TestMapPositionsFromForwardRf:
    """Tests for map_positions_from_forward_rf function."""
    
    def test_basic_calculation(self):
        """Test basic position calculation."""
        forward = np.array([0.001, 0.002, 0.003])
        rf = np.array([0.0003, 0.0003, 0.0003])
        x = 0.001
        
        positions = map_positions_from_forward_rf(forward, rf, x=x)
        
        # position[0] = (0.001 - 0.0003) / (0.001 - 0.0003) = 1.0
        assert np.isclose(positions[0], 1.0)
        
        # position[1] = (0.001 - 0.0003) / (0.002 - 0.0003) ≈ 0.412
        expected_1 = (0.001 - 0.0003) / (0.002 - 0.0003)
        assert np.isclose(positions[1], expected_1, rtol=1e-5)
        
        # position[2] = (0.001 - 0.0003) / (0.003 - 0.0003) ≈ 0.259
        expected_2 = (0.001 - 0.0003) / (0.003 - 0.0003)
        assert np.isclose(positions[2], expected_2, rtol=1e-5)
    
    def test_clipping_max(self):
        """Test that positions are clipped to clip_max."""
        forward = np.array([0.0005])  # Small excess
        rf = np.array([0.0003])
        x = 0.002  # Large target return
        
        # (0.002 - 0.0003) / (0.0005 - 0.0003) = 0.0017 / 0.0002 = 8.5
        positions = map_positions_from_forward_rf(forward, rf, x=x, clip_max=2.0)
        assert positions[0] == 2.0
    
    def test_clipping_min(self):
        """Test that positions are clipped to clip_min."""
        forward = np.array([0.002])
        rf = np.array([0.0003])
        x = -0.001  # Negative target
        
        # (-0.001 - 0.0003) / (0.002 - 0.0003) = -0.0013 / 0.0017 ≈ -0.76
        positions = map_positions_from_forward_rf(forward, rf, x=x, clip_min=0.0)
        assert positions[0] == 0.0
    
    def test_division_by_zero_guard(self):
        """Test that division by zero is handled."""
        forward = np.array([0.0003])  # Same as rf
        rf = np.array([0.0003])
        x = 0.001
        
        # forward ≈ rf, so denom is near zero
        positions = map_positions_from_forward_rf(forward, rf, x=x)
        
        # Should not raise and should be a finite value
        assert np.isfinite(positions[0])
    
    def test_negative_excess(self):
        """Test behavior when forward < rf (negative market excess)."""
        forward = np.array([0.0001])
        rf = np.array([0.0003])
        x = 0.0002
        
        # (0.0002 - 0.0003) / (0.0001 - 0.0003) = -0.0001 / -0.0002 = 0.5
        positions = map_positions_from_forward_rf(forward, rf, x=x)
        assert np.isclose(positions[0], 0.5)
    
    def test_array_shapes(self):
        """Test that output has correct shape."""
        forward = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        rf = np.array([0.0003, 0.0003, 0.0003, 0.0003, 0.0003])
        
        positions = map_positions_from_forward_rf(forward, rf, x=0.001)
        assert positions.shape == forward.shape
    
    def test_x_equals_rf(self):
        """Test when x equals rf (position should be 0)."""
        forward = np.array([0.002])
        rf = np.array([0.0003])
        x = 0.0003  # Same as rf
        
        # (0.0003 - 0.0003) / (0.002 - 0.0003) = 0 / 0.0017 = 0
        positions = map_positions_from_forward_rf(forward, rf, x=x)
        assert np.isclose(positions[0], 0.0)


class TestMapPositionsFromTwoHeadConfig:
    """Tests for map_positions_from_two_head_config function."""
    
    def test_with_default_config(self):
        """Test with default configuration."""
        forward = np.array([0.001, 0.002])
        rf = np.array([0.0003, 0.0003])
        
        positions = map_positions_from_two_head_config(forward, rf)
        
        # x=0.0, so (0 - 0.0003) / (forward - 0.0003)
        # Should be negative, clipped to 0
        assert np.all(positions == 0.0)
    
    def test_with_custom_config(self):
        """Test with custom configuration."""
        forward = np.array([0.001])
        rf = np.array([0.0003])
        
        config = TwoHeadPositionConfig(x=0.001, clip_min=0.0, clip_max=2.0)
        positions = map_positions_from_two_head_config(forward, rf, config)
        
        assert np.isclose(positions[0], 1.0)


class TestComputeHullSharpeTwoHead:
    """Tests for compute_hull_sharpe_two_head function.
    
    This test class validates that compute_hull_sharpe_two_head uses
    the official Hull Competition Sharpe metric, including:
    - Raw Sharpe ratio (annualized)
    - Vol penalty (when vol_ratio outside [0.8, 1.2])
    - Return penalty (when strategy_return < market_return)
    """
    
    def test_neutral_positions(self):
        """Test with all neutral positions (position=1)."""
        n = 100
        positions = np.ones(n)
        forward_true = np.random.randn(n) * 0.01 + 0.0005
        rf_true = np.full(n, 0.0003)
        
        metrics = compute_hull_sharpe_two_head(positions, forward_true, rf_true)
        
        # All required keys present (official metric format)
        assert "raw_sharpe" in metrics
        assert "vol_ratio" in metrics
        assert "vol_penalty" in metrics
        assert "return_penalty" in metrics
        assert "hull_sharpe" in metrics
        assert "strategy_mean" in metrics
        assert "strategy_std" in metrics
        assert "market_mean" in metrics
        assert "market_std" in metrics
        
        # With position=1, vol_ratio should be close to 1
        assert np.isclose(metrics["vol_ratio"], 1.0, atol=0.1)
    
    def test_vol_penalty_high_leverage(self):
        """Test that vol penalty is applied when vol_ratio > 1.2."""
        n = 100
        # High leverage positions (2x) increase volatility
        positions = np.full(n, 2.0)
        forward_true = np.random.randn(n) * 0.01 + 0.001
        rf_true = np.full(n, 0.0003)
        
        metrics = compute_hull_sharpe_two_head(positions, forward_true, rf_true)
        
        # With 2x leverage, vol_ratio ≈ 2.0, so penalty should apply
        assert metrics["vol_ratio"] > 1.2
        assert metrics["vol_penalty"] > 0
        # hull_sharpe = raw_sharpe - vol_penalty - return_penalty
        expected_hull = metrics["raw_sharpe"] - metrics["vol_penalty"] - metrics["return_penalty"]
        assert np.isclose(metrics["hull_sharpe"], expected_hull, atol=1e-6)
    
    def test_vol_penalty_low_volatility(self):
        """Test that vol penalty is applied when vol_ratio < 0.8."""
        n = 100
        # Low leverage positions (0.3x) decrease volatility
        positions = np.full(n, 0.3)
        forward_true = np.random.randn(n) * 0.01 + 0.001
        rf_true = np.full(n, 0.0003)
        
        metrics = compute_hull_sharpe_two_head(positions, forward_true, rf_true)
        
        # With 0.3x leverage, vol_ratio ≈ 0.3, so penalty should apply
        assert metrics["vol_ratio"] < 0.8
        assert metrics["vol_penalty"] > 0
    
    def test_no_penalty_in_tolerance_band(self):
        """Test no vol penalty when vol_ratio in [0.8, 1.2]."""
        n = 100
        positions = np.ones(n)  # Neutral positions
        np.random.seed(42)  # For reproducibility
        forward_true = np.random.randn(n) * 0.01 + 0.001
        rf_true = np.full(n, 0.0003)
        
        metrics = compute_hull_sharpe_two_head(positions, forward_true, rf_true)
        
        # With position=1, vol_ratio = 1.0, no penalty
        assert 0.8 <= metrics["vol_ratio"] <= 1.2
        assert metrics["vol_penalty"] == 0.0


class TestOptimizeXParameter:
    """Tests for optimize_x_parameter function."""
    
    def test_returns_correct_types(self):
        """Test that function returns correct types."""
        np.random.seed(42)
        n = 100
        forward_oof = np.random.randn(n) * 0.005 + 0.001
        rf_oof = np.full(n, 0.0003)
        forward_true = forward_oof + np.random.randn(n) * 0.001
        rf_true = rf_oof
        
        best_x, best_sharpe, results = optimize_x_parameter(
            forward_oof, rf_oof, forward_true, rf_true
        )
        
        assert isinstance(best_x, float)
        assert isinstance(best_sharpe, float)
        assert isinstance(results, list)
        assert len(results) == 41  # Default grid size
    
    def test_custom_grid(self):
        """Test with custom x grid."""
        np.random.seed(42)
        n = 50
        forward_oof = np.random.randn(n) * 0.005 + 0.001
        rf_oof = np.full(n, 0.0003)
        forward_true = forward_oof
        rf_true = rf_oof
        
        x_grid = np.linspace(-0.001, 0.001, 11)
        best_x, best_sharpe, results = optimize_x_parameter(
            forward_oof, rf_oof, forward_true, rf_true,
            x_grid=x_grid
        )
        
        assert len(results) == 11
        assert all("x" in r and "hull_sharpe" in r for r in results)
    
    def test_best_x_is_optimal(self):
        """Test that returned best_x has highest hull_sharpe."""
        np.random.seed(42)
        n = 100
        forward_oof = np.random.randn(n) * 0.005 + 0.001
        rf_oof = np.full(n, 0.0003)
        forward_true = forward_oof
        rf_true = rf_oof
        
        best_x, best_sharpe, results = optimize_x_parameter(
            forward_oof, rf_oof, forward_true, rf_true
        )
        
        # Find max hull_sharpe in results
        max_sharpe_in_results = max(r["hull_sharpe"] for r in results)
        assert np.isclose(best_sharpe, max_sharpe_in_results)
