#!/usr/bin/env python
"""Cross-validation utilities for model training.

This module provides unified CV splitting and evaluation functions
that are shared across all model types (LGBM, XGBoost, CatBoost, Ridge).

Key design decisions:
- Use TimeSeriesSplit for temporal consistency
- Support gap between train/validation to prevent leakage
- Provide consistent OOF evaluation metrics (RMSE, MSR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    
    n_splits: int = 5
    gap: int = 0
    min_val_size: int = 0
    random_state: int = 42


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_rmse: float
    val_rmse: float
    val_predictions: np.ndarray
    val_actuals: np.ndarray
    metadata: Dict[str, Any]


def create_cv_splits(
    n_samples: int,
    config: CVConfig | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create TimeSeriesSplit indices for cross-validation.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    config : CVConfig, optional
        CV configuration. If None, uses default settings.
        
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples for each fold.
    """
    if config is None:
        config = CVConfig()
    
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    all_indices = np.arange(n_samples)
    
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    
    for train_idx, val_idx in tscv.split(all_indices):
        # Apply gap
        if config.gap > 0:
            val_idx = val_idx[config.gap:]
        
        # Skip if validation is too small
        if len(val_idx) < config.min_val_size:
            continue
            
        splits.append((train_idx, val_idx))
    
    return splits


def create_cv_splits_with_dates(
    df: pd.DataFrame,
    date_col: str = "date_id",
    config: CVConfig | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Create CV splits with date range metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column.
    date_col : str
        Name of the date column.
    config : CVConfig, optional
        CV configuration.
        
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]
        List of (train_indices, val_indices, metadata) tuples.
    """
    if config is None:
        config = CVConfig()
    
    splits = create_cv_splits(len(df), config)
    
    result: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        dates = df[date_col].values
        
        metadata = {
            "fold_idx": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_date_range": (int(dates[train_idx[0]]), int(dates[train_idx[-1]])),
            "val_date_range": (int(dates[val_idx[0]]), int(dates[val_idx[-1]])),
        }
        
        result.append((train_idx, val_idx, metadata))
    
    return result


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_returns: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute evaluation metrics for a single fold.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    target_returns : np.ndarray, optional
        Market forward excess returns for MSR calculation.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with RMSE and optionally MSR metrics.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    metrics: Dict[str, float] = {
        "rmse": rmse,
        "mse": rmse ** 2,
    }
    
    # If target_returns provided, compute MSR metrics
    if target_returns is not None:
        target_returns = np.asarray(target_returns, dtype=float).ravel()
        
        # Simple signal: pred + 1.0 (no clipping for raw evaluation)
        signal = y_pred + 1.0
        trade_returns = (signal - 1.0) * target_returns
        
        eps = 1e-8
        mean_r = float(np.nanmean(trade_returns))
        std_r = float(np.nanstd(trade_returns))
        std_down = float(np.nanstd(np.minimum(trade_returns, 0.0)))
        
        metrics["msr"] = mean_r / (std_r + eps)
        metrics["msr_down"] = mean_r / (std_down + eps)
        metrics["trade_mean"] = mean_r
        metrics["trade_std"] = std_r
    
    return metrics


def evaluate_oof_predictions(
    oof_predictions: np.ndarray,
    oof_actuals: np.ndarray,
    oof_indices: np.ndarray | None = None,
    target_returns: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Evaluate out-of-fold predictions across all folds.
    
    Parameters
    ----------
    oof_predictions : np.ndarray
        All OOF predictions concatenated.
    oof_actuals : np.ndarray
        Corresponding actual values.
    oof_indices : np.ndarray, optional
        Original indices of OOF samples (for alignment).
    target_returns : np.ndarray, optional
        Full target returns array for MSR calculation.
        
    Returns
    -------
    Dict[str, Any]
        Aggregated OOF metrics.
    """
    oof_predictions = np.asarray(oof_predictions, dtype=float).ravel()
    oof_actuals = np.asarray(oof_actuals, dtype=float).ravel()
    
    # Compute OOF RMSE
    oof_rmse = float(np.sqrt(mean_squared_error(oof_actuals, oof_predictions)))
    
    result: Dict[str, Any] = {
        "oof_rmse": oof_rmse,
        "oof_mse": oof_rmse ** 2,
        "n_samples": len(oof_predictions),
    }
    
    # Compute OOF MSR if target returns provided
    if target_returns is not None and oof_indices is not None:
        oof_target = np.asarray(target_returns, dtype=float).ravel()[oof_indices]
        signal = oof_predictions + 1.0
        trade_returns = (signal - 1.0) * oof_target
        
        eps = 1e-8
        mean_r = float(np.nanmean(trade_returns))
        std_r = float(np.nanstd(trade_returns))
        std_down = float(np.nanstd(np.minimum(trade_returns, 0.0)))
        
        result["oof_msr"] = mean_r / (std_r + eps)
        result["oof_msr_down"] = mean_r / (std_down + eps)
    
    return result


def aggregate_fold_results(
    fold_results: List[FoldResult],
) -> Dict[str, Any]:
    """Aggregate results from all CV folds.
    
    Parameters
    ----------
    fold_results : List[FoldResult]
        Results from each fold.
        
    Returns
    -------
    Dict[str, Any]
        Aggregated statistics across folds.
    """
    if not fold_results:
        return {}
    
    train_rmses = [f.train_rmse for f in fold_results]
    val_rmses = [f.val_rmse for f in fold_results]
    
    # Concatenate all OOF predictions
    all_val_preds = np.concatenate([f.val_predictions for f in fold_results])
    all_val_actuals = np.concatenate([f.val_actuals for f in fold_results])
    all_val_indices = np.concatenate([f.val_indices for f in fold_results])
    
    # Sort by original index for proper ordering
    sort_order = np.argsort(all_val_indices)
    all_val_preds = all_val_preds[sort_order]
    all_val_actuals = all_val_actuals[sort_order]
    all_val_indices = all_val_indices[sort_order]
    
    # Compute overall OOF RMSE
    oof_rmse = float(np.sqrt(mean_squared_error(all_val_actuals, all_val_preds)))
    
    return {
        "n_folds": len(fold_results),
        "train_rmse_mean": float(np.mean(train_rmses)),
        "train_rmse_std": float(np.std(train_rmses)),
        "val_rmse_mean": float(np.mean(val_rmses)),
        "val_rmse_std": float(np.std(val_rmses)),
        "oof_rmse": oof_rmse,
        "oof_predictions": all_val_preds,
        "oof_actuals": all_val_actuals,
        "oof_indices": all_val_indices,
    }
