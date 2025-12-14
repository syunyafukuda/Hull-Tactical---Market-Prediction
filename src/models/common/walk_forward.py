"""Walk-Forward Cross-Validation splitter.

This module provides walk-forward (rolling/expanding window) cross-validation
for time series data, which is more appropriate for financial data than
standard TimeSeriesSplit.

Key features:
- Expanding or rolling window modes
- Configurable train/validation window sizes
- Gap support to prevent data leakage
- Metadata output for debugging and logging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward cross-validation.
    
    Parameters
    ----------
    train_window : int
        Number of samples in training window.
    val_window : int
        Number of samples in validation window.
    step : int
        Step size for moving the window.
    mode : str
        "expanding" (train start fixed) or "rolling" (train start moves).
    min_folds : int
        Minimum number of folds required.
    gap : int
        Gap between train and validation (to prevent leakage).
    """
    
    train_window: int = 6000
    val_window: int = 1000
    step: int = 1000
    mode: str = "expanding"  # "expanding" or "rolling"
    min_folds: int = 3
    gap: int = 0


@dataclass
class WalkForwardFold:
    """Single fold result from walk-forward split."""
    
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    metadata: Dict[str, Any]
    
    @property
    def train_range(self) -> Tuple[int, int]:
        """Return (start_idx, end_idx) for training set."""
        return self.metadata.get("train_range", (0, 0))
    
    @property
    def val_range(self) -> Tuple[int, int]:
        """Return (start_idx, end_idx) for validation set."""
        return self.metadata.get("val_range", (0, 0))


def make_walk_forward_splits(
    n_samples: int,
    config: WalkForwardConfig | None = None,
) -> List[WalkForwardFold]:
    """Generate walk-forward cross-validation splits.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    config : WalkForwardConfig, optional
        Configuration for the splits. If None, uses defaults.
        
    Returns
    -------
    List[WalkForwardFold]
        List of fold objects containing train/val indices and metadata.
        
    Raises
    ------
    ValueError
        If the dataset is too small for the requested configuration.
        
    Examples
    --------
    >>> config = WalkForwardConfig(train_window=5000, val_window=1000, step=1000)
    >>> folds = make_walk_forward_splits(8990, config)
    >>> len(folds)
    3
    """
    if config is None:
        config = WalkForwardConfig()
    
    indices = np.arange(n_samples)
    folds: List[WalkForwardFold] = []
    
    fold_idx = 0
    
    if config.mode == "expanding":
        # Expanding window: train_start fixed at 0
        train_start = 0
        train_end = config.train_window
        
        while True:
            val_start = train_end + config.gap
            val_end = val_start + config.val_window
            
            if val_end > n_samples:
                break
            
            train_idx = indices[train_start:train_end]
            val_idx = indices[val_start:val_end]
            
            metadata = {
                "train_range": (int(train_start), int(train_end - 1)),
                "val_range": (int(val_start), int(val_end - 1)),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
            
            folds.append(WalkForwardFold(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata=metadata,
            ))
            
            fold_idx += 1
            train_end += config.step
            
    elif config.mode == "rolling":
        # Rolling window: both train_start and train_end move
        train_start = 0
        
        while True:
            train_end = train_start + config.train_window
            val_start = train_end + config.gap
            val_end = val_start + config.val_window
            
            if val_end > n_samples:
                break
            
            train_idx = indices[train_start:train_end]
            val_idx = indices[val_start:val_end]
            
            metadata = {
                "train_range": (int(train_start), int(train_end - 1)),
                "val_range": (int(val_start), int(val_end - 1)),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
            
            folds.append(WalkForwardFold(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata=metadata,
            ))
            
            fold_idx += 1
            train_start += config.step
            
    else:
        raise ValueError(f"Invalid mode: {config.mode}. Use 'expanding' or 'rolling'.")
    
    if len(folds) < config.min_folds:
        raise ValueError(
            f"Only {len(folds)} folds generated, but min_folds={config.min_folds}. "
            f"Consider reducing train_window ({config.train_window}) or "
            f"val_window ({config.val_window})."
        )
    
    return folds


def make_walk_forward_splits_df(
    df: pd.DataFrame,
    config: WalkForwardConfig | None = None,
    date_col: str = "date_id",
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Generate walk-forward splits with date-based metadata.
    
    This is a convenience function that adds date range information
    to the metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column (must be sorted by date).
    config : WalkForwardConfig, optional
        Configuration for the splits.
    date_col : str, optional
        Name of the date column. Default "date_id".
        
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]
        List of (train_idx, val_idx, metadata) tuples.
        Metadata includes date_range for train and val sets.
    """
    if config is None:
        config = WalkForwardConfig()
    
    n_samples = len(df)
    folds = make_walk_forward_splits(n_samples, config)
    
    result = []
    for fold in folds:
        # Add date range to metadata
        train_dates = df.iloc[fold.train_indices][date_col]
        val_dates = df.iloc[fold.val_indices][date_col]
        
        extended_metadata = fold.metadata.copy()
        extended_metadata["train_date_range"] = (
            int(train_dates.iloc[0]),
            int(train_dates.iloc[-1]),
        )
        extended_metadata["val_date_range"] = (
            int(val_dates.iloc[0]),
            int(val_dates.iloc[-1]),
        )
        
        result.append((
            fold.train_indices,
            fold.val_indices,
            extended_metadata,
        ))
    
    return result


def estimate_fold_count(
    n_samples: int,
    config: WalkForwardConfig,
) -> int:
    """Estimate the number of folds for given configuration.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples.
    config : WalkForwardConfig
        Configuration for the splits.
        
    Returns
    -------
    int
        Estimated number of folds.
    """
    if config.mode == "expanding":
        remaining = n_samples - config.train_window - config.gap - config.val_window
        if remaining < 0:
            return 0
        return 1 + (remaining // config.step)
    elif config.mode == "rolling":
        remaining = n_samples - config.train_window - config.gap - config.val_window
        if remaining < 0:
            return 0
        return 1 + (remaining // config.step)
    else:
        return 0
