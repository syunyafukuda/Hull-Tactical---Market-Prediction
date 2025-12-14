"""Tests for Walk-Forward cross-validation splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.common.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    estimate_fold_count,
    make_walk_forward_splits,
    make_walk_forward_splits_df,
)


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have expected values."""
        config = WalkForwardConfig()
        
        assert config.train_window == 6000
        assert config.val_window == 1000
        assert config.step == 1000
        assert config.mode == "expanding"
        assert config.min_folds == 3
        assert config.gap == 0
        
    def test_custom_values(self):
        """Custom values should override defaults."""
        config = WalkForwardConfig(
            train_window=5000,
            val_window=500,
            step=250,
            mode="rolling",
            min_folds=5,
            gap=10,
        )
        
        assert config.train_window == 5000
        assert config.val_window == 500
        assert config.step == 250
        assert config.mode == "rolling"
        assert config.min_folds == 5
        assert config.gap == 10


class TestMakeWalkForwardSplitsExpanding:
    """Tests for expanding window mode."""
    
    def test_basic_expanding_splits(self):
        """Basic expanding window should work correctly."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        n_samples = 200
        folds = make_walk_forward_splits(n_samples, config)
        
        assert len(folds) >= 1
        
        # First fold: train [0:100], val [100:120]
        assert folds[0].train_indices[0] == 0
        assert folds[0].train_indices[-1] == 99
        assert folds[0].val_indices[0] == 100
        assert folds[0].val_indices[-1] == 119
        
    def test_expanding_train_start_fixed(self):
        """In expanding mode, train start should be fixed at 0."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        n_samples = 200
        folds = make_walk_forward_splits(n_samples, config)
        
        for fold in folds:
            assert fold.train_indices[0] == 0
            
    def test_expanding_train_grows(self):
        """In expanding mode, training set should grow with each fold."""
        config = WalkForwardConfig(
            train_window=50,
            val_window=10,
            step=10,
            mode="expanding",
            min_folds=1,
        )
        
        n_samples = 100
        folds = make_walk_forward_splits(n_samples, config)
        
        if len(folds) >= 2:
            assert len(folds[1].train_indices) > len(folds[0].train_indices)
            
    def test_no_overlap_train_val(self):
        """Train and validation sets should not overlap."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        n_samples = 200
        folds = make_walk_forward_splits(n_samples, config)
        
        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)
            assert len(train_set & val_set) == 0


class TestMakeWalkForwardSplitsRolling:
    """Tests for rolling window mode."""
    
    def test_basic_rolling_splits(self):
        """Basic rolling window should work correctly."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="rolling",
            min_folds=1,
        )
        
        n_samples = 200
        folds = make_walk_forward_splits(n_samples, config)
        
        assert len(folds) >= 1
        
    def test_rolling_train_moves(self):
        """In rolling mode, train start should move with each fold."""
        config = WalkForwardConfig(
            train_window=50,
            val_window=10,
            step=10,
            mode="rolling",
            min_folds=1,
        )
        
        n_samples = 100
        folds = make_walk_forward_splits(n_samples, config)
        
        if len(folds) >= 2:
            assert folds[1].train_indices[0] > folds[0].train_indices[0]
            
    def test_rolling_train_size_constant(self):
        """In rolling mode, training set size should be constant."""
        config = WalkForwardConfig(
            train_window=50,
            val_window=10,
            step=10,
            mode="rolling",
            min_folds=1,
        )
        
        n_samples = 100
        folds = make_walk_forward_splits(n_samples, config)
        
        train_sizes = [len(f.train_indices) for f in folds]
        assert all(s == train_sizes[0] for s in train_sizes)


class TestGapParameter:
    """Tests for gap between train and validation."""
    
    def test_gap_zero(self):
        """Zero gap should have adjacent train/val."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
            gap=0,
        )
        
        n_samples = 200
        folds = make_walk_forward_splits(n_samples, config)
        
        # Val should start right after train
        assert folds[0].val_indices[0] == folds[0].train_indices[-1] + 1
        
    def test_gap_positive(self):
        """Positive gap should leave space between train/val."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
            gap=5,
        )
        
        n_samples = 200
        folds = make_walk_forward_splits(n_samples, config)
        
        # Val should start 5 positions after train ends
        expected_gap = folds[0].val_indices[0] - folds[0].train_indices[-1] - 1
        assert expected_gap == 5


class TestMinFolds:
    """Tests for minimum folds requirement."""
    
    def test_min_folds_satisfied(self):
        """Should succeed when min_folds is satisfied."""
        config = WalkForwardConfig(
            train_window=50,
            val_window=10,
            step=10,
            mode="expanding",
            min_folds=3,
        )
        
        n_samples = 100
        folds = make_walk_forward_splits(n_samples, config)
        
        assert len(folds) >= 3
        
    def test_min_folds_not_satisfied_raises(self):
        """Should raise when min_folds cannot be satisfied."""
        config = WalkForwardConfig(
            train_window=80,
            val_window=20,
            step=10,
            mode="expanding",
            min_folds=5,
        )
        
        n_samples = 100
        
        with pytest.raises(ValueError, match="min_folds"):
            make_walk_forward_splits(n_samples, config)


class TestWalkForwardFold:
    """Tests for WalkForwardFold dataclass."""
    
    def test_fold_properties(self):
        """Fold should have expected properties."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        folds = make_walk_forward_splits(200, config)
        fold = folds[0]
        
        assert fold.fold_idx == 0
        assert isinstance(fold.train_indices, np.ndarray)
        assert isinstance(fold.val_indices, np.ndarray)
        assert isinstance(fold.metadata, dict)
        
    def test_fold_range_properties(self):
        """Fold range properties should return correct values."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        folds = make_walk_forward_splits(200, config)
        fold = folds[0]
        
        train_range = fold.train_range
        val_range = fold.val_range
        
        assert train_range == (0, 99)
        assert val_range == (100, 119)


class TestMakeWalkForwardSplitsDf:
    """Tests for DataFrame-based split generation."""
    
    def test_with_date_column(self):
        """Should add date range to metadata."""
        df = pd.DataFrame({
            "date_id": list(range(1000, 1200)),
            "value": np.random.randn(200),
        })
        
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        result = make_walk_forward_splits_df(df, config)
        
        assert len(result) >= 1
        train_idx, val_idx, metadata = result[0]
        
        assert "train_date_range" in metadata
        assert "val_date_range" in metadata
        assert metadata["train_date_range"] == (1000, 1099)
        assert metadata["val_date_range"] == (1100, 1119)


class TestEstimateFoldCount:
    """Tests for fold count estimation."""
    
    def test_estimate_expanding(self):
        """Estimate should be accurate for expanding mode."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="expanding",
            min_folds=1,
        )
        
        n_samples = 200
        estimate = estimate_fold_count(n_samples, config)
        actual = len(make_walk_forward_splits(n_samples, config))
        
        assert estimate == actual
        
    def test_estimate_rolling(self):
        """Estimate should be accurate for rolling mode."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="rolling",
            min_folds=1,
        )
        
        n_samples = 200
        estimate = estimate_fold_count(n_samples, config)
        actual = len(make_walk_forward_splits(n_samples, config))
        
        assert estimate == actual
        
    def test_estimate_insufficient_data(self):
        """Estimate should return 0 for insufficient data."""
        config = WalkForwardConfig(
            train_window=500,
            val_window=100,
            step=50,
            mode="expanding",
            min_folds=1,
        )
        
        n_samples = 100
        estimate = estimate_fold_count(n_samples, config)
        
        assert estimate == 0


class TestInvalidMode:
    """Tests for invalid mode handling."""
    
    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        config = WalkForwardConfig(
            train_window=100,
            val_window=20,
            step=20,
            mode="invalid",  # type: ignore
            min_folds=1,
        )
        
        with pytest.raises(ValueError, match="Invalid mode"):
            make_walk_forward_splits(200, config)


class TestHullTacticalDatasetSize:
    """Tests with Hull Tactical dataset dimensions."""
    
    def test_hull_tactical_expanding(self):
        """Test with realistic Hull Tactical data size."""
        # Hull Tactical: ~8990 training samples
        n_samples = 8990
        
        config = WalkForwardConfig(
            train_window=6000,
            val_window=1000,
            step=1000,
            mode="expanding",
            min_folds=2,
        )
        
        folds = make_walk_forward_splits(n_samples, config)
        
        # Should get 2 folds:
        # Fold 0: train [0:6000], val [6000:7000]
        # Fold 1: train [0:7000], val [7000:8000]
        assert len(folds) >= 2
        
    def test_hull_tactical_rolling(self):
        """Test rolling mode with Hull Tactical size."""
        n_samples = 8990
        
        config = WalkForwardConfig(
            train_window=6000,
            val_window=1000,
            step=1000,
            mode="rolling",
            min_folds=2,
        )
        
        folds = make_walk_forward_splits(n_samples, config)
        
        # Should get 2 folds:
        # Fold 0: train [0:6000], val [6000:7000]
        # Fold 1: train [1000:7000], val [7000:8000]
        assert len(folds) >= 2
        
        # Verify rolling behavior
        if len(folds) >= 2:
            assert folds[0].train_indices[0] == 0
            assert folds[1].train_indices[0] == 1000
