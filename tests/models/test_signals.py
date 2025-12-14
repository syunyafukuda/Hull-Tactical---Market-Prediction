"""Tests for signal/position mapping utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.common.signals import (
    PositionMapperConfig,
    analyze_position_distribution,
    calibrate_position_mapper,
    map_to_position,
    map_to_position_from_config,
)


class TestMapToPosition:
    """Tests for map_to_position function."""
    
    def test_default_mapping_center(self):
        """Zero prediction should map to neutral position (1.0)."""
        predictions = np.array([0.0])
        positions = map_to_position(predictions)
        
        assert positions[0] == pytest.approx(1.0)
        
    def test_default_mapping_negative(self):
        """Negative predictions should map below 1.0."""
        predictions = np.array([-0.01])  # -1%
        positions = map_to_position(predictions)
        
        # -0.01 * 100 + 1 = 0.0
        assert positions[0] == pytest.approx(0.0)
        
    def test_default_mapping_positive(self):
        """Positive predictions should map above 1.0."""
        predictions = np.array([0.01])  # +1%
        positions = map_to_position(predictions)
        
        # 0.01 * 100 + 1 = 2.0
        assert positions[0] == pytest.approx(2.0)
        
    def test_clipping_below(self):
        """Predictions below clip_min should be clipped."""
        predictions = np.array([-0.02])  # -2%
        positions = map_to_position(predictions)
        
        # -0.02 * 100 + 1 = -1.0 -> clipped to 0.0
        assert positions[0] == pytest.approx(0.0)
        
    def test_clipping_above(self):
        """Predictions above clip_max should be clipped."""
        predictions = np.array([0.02])  # +2%
        positions = map_to_position(predictions)
        
        # 0.02 * 100 + 1 = 3.0 -> clipped to 2.0
        assert positions[0] == pytest.approx(2.0)
        
    def test_custom_multiplier(self):
        """Custom multiplier should scale predictions."""
        predictions = np.array([0.01])
        positions = map_to_position(predictions, multiplier=50.0, offset=1.0)
        
        # 0.01 * 50 + 1 = 1.5
        assert positions[0] == pytest.approx(1.5)
        
    def test_custom_offset(self):
        """Custom offset should shift positions."""
        predictions = np.array([0.0])
        positions = map_to_position(predictions, multiplier=100.0, offset=0.5)
        
        # 0.0 * 100 + 0.5 = 0.5
        assert positions[0] == pytest.approx(0.5)
        
    def test_custom_clip_range(self):
        """Custom clip range should be respected."""
        predictions = np.array([0.0, 0.02, -0.02])
        positions = map_to_position(
            predictions, 
            multiplier=100.0, 
            offset=1.0, 
            clip_min=0.5, 
            clip_max=1.5
        )
        
        # 0.0 * 100 + 1 = 1.0 (within range)
        # 0.02 * 100 + 1 = 3.0 -> clipped to 1.5
        # -0.02 * 100 + 1 = -1.0 -> clipped to 0.5
        assert positions[0] == pytest.approx(1.0)
        assert positions[1] == pytest.approx(1.5)
        assert positions[2] == pytest.approx(0.5)
        
    def test_array_input(self):
        """Should handle array inputs correctly."""
        predictions = np.array([-0.01, 0.0, 0.01])
        positions = map_to_position(predictions)
        
        assert positions.shape == (3,)
        assert positions[0] == pytest.approx(0.0)
        assert positions[1] == pytest.approx(1.0)
        assert positions[2] == pytest.approx(2.0)


class TestMapToPositionFromConfig:
    """Tests for map_to_position_from_config function."""
    
    def test_default_config(self):
        """Default config should give same result as default map_to_position."""
        predictions = np.array([-0.01, 0.0, 0.01])
        
        positions_direct = map_to_position(predictions)
        positions_config = map_to_position_from_config(predictions)
        
        np.testing.assert_array_almost_equal(positions_direct, positions_config)
        
    def test_custom_config(self):
        """Custom config should be applied correctly."""
        predictions = np.array([0.0])
        config = PositionMapperConfig(
            multiplier=50.0,
            offset=0.5,
            clip_min=0.0,
            clip_max=2.0,
        )
        
        positions = map_to_position_from_config(predictions, config)
        
        # 0.0 * 50 + 0.5 = 0.5
        assert positions[0] == pytest.approx(0.5)


class TestPositionMapperConfig:
    """Tests for PositionMapperConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have expected values."""
        config = PositionMapperConfig()
        
        assert config.multiplier == 100.0
        assert config.offset == 1.0
        assert config.clip_min == 0.0
        assert config.clip_max == 2.0
        
    def test_custom_values(self):
        """Custom values should override defaults."""
        config = PositionMapperConfig(
            multiplier=50.0,
            offset=0.5,
            clip_min=0.1,
            clip_max=1.9,
        )
        
        assert config.multiplier == 50.0
        assert config.offset == 0.5
        assert config.clip_min == 0.1
        assert config.clip_max == 1.9


class TestCalibratePositionMapper:
    """Tests for calibrate_position_mapper function."""
    
    def test_calibration_produces_target_mean(self):
        """Calibration should produce target mean position."""
        np.random.seed(42)
        predictions = np.random.normal(0.005, 0.01, 1000)
        
        config = calibrate_position_mapper(
            predictions, target_mean=1.0
        )
        
        positions = map_to_position_from_config(predictions, config)
        
        # Mean should be close to target
        assert np.mean(positions) == pytest.approx(1.0, abs=0.1)
        
    def test_calibration_produces_target_range(self):
        """Calibration should produce target range (approximately)."""
        np.random.seed(42)
        predictions = np.random.normal(0.0, 0.01, 1000)
        
        config = calibrate_position_mapper(
            predictions, 
            target_mean=1.0, 
            target_range=(0.3, 1.7),
        )
        
        positions = map_to_position_from_config(predictions, config)
        
        # P5 and P95 should be close to target range
        p5 = np.percentile(positions, 5)
        p95 = np.percentile(positions, 95)
        
        assert p5 == pytest.approx(0.3, abs=0.2)
        assert p95 == pytest.approx(1.7, abs=0.2)
        
    def test_calibration_handles_zero_range(self):
        """Calibration should handle constant predictions."""
        predictions = np.full(100, 0.005)
        
        config = calibrate_position_mapper(predictions)
        
        # Should return valid config
        assert isinstance(config, PositionMapperConfig)
        assert config.multiplier > 0


class TestAnalyzePositionDistribution:
    """Tests for analyze_position_distribution function."""
    
    def test_returns_all_stats(self):
        """Should return all expected statistics."""
        positions = np.random.uniform(0, 2, 1000)
        
        stats = analyze_position_distribution(positions)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "p5" in stats
        assert "p25" in stats
        assert "p75" in stats
        assert "p95" in stats
        assert "pct_at_min_clip" in stats
        assert "pct_at_max_clip" in stats
        assert "pct_below_neutral" in stats
        assert "pct_above_neutral" in stats
        
    def test_stats_are_correct(self):
        """Statistics should be calculated correctly."""
        positions = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        
        stats = analyze_position_distribution(positions)
        
        assert stats["mean"] == pytest.approx(1.0)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(1.0)
        
    def test_clip_percentages(self):
        """Clipping percentages should be calculated correctly."""
        positions = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
        
        stats = analyze_position_distribution(positions)
        
        assert stats["pct_at_min_clip"] == pytest.approx(0.4)
        assert stats["pct_at_max_clip"] == pytest.approx(0.4)
        
    def test_below_above_neutral(self):
        """Below/above neutral percentages should be calculated correctly."""
        positions = np.array([0.5, 0.5, 1.5, 1.5, 1.5])
        
        stats = analyze_position_distribution(positions)
        
        assert stats["pct_below_neutral"] == pytest.approx(0.4)
        assert stats["pct_above_neutral"] == pytest.approx(0.6)
