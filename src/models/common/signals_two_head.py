"""Two-Head Signal/Prediction to Position mapping utilities.

This module provides functions specifically for the two-head learning approach,
where forward_returns and risk_free_rate are predicted separately.

Based on Kaggle discussion/608349:
position = clip((x - rf_pred) / (forward_pred - rf_pred), clip_min, clip_max)

This is a standalone module that does NOT modify the existing signals.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class TwoHeadPositionConfig:
    """Two-head position mapping configuration.
    
    Based on Kaggle discussion/608349.
    
    Formula: position = clip((x - rf_pred) / (forward_pred - rf_pred), clip_min, clip_max)
    
    Parameters
    ----------
    x : float
        Target return level (same scale as forward_returns).
        Typical range: [-0.002, 0.002]
    clip_min : float
        Minimum position (0 = 100% cash).
    clip_max : float
        Maximum position (2 = 200% market).
    epsilon : float
        Minimum denominator to avoid division by zero.
    """
    x: float = 0.0
    clip_min: float = 0.0
    clip_max: float = 2.0
    epsilon: float = 1e-8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "x": self.x,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "epsilon": self.epsilon,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TwoHeadPositionConfig":
        """Create config from dictionary."""
        return cls(
            x=float(d.get("x", 0.0)),
            clip_min=float(d.get("clip_min", 0.0)),
            clip_max=float(d.get("clip_max", 2.0)),
            epsilon=float(d.get("epsilon", 1e-8)),
        )


def map_positions_from_forward_rf(
    forward_pred: np.ndarray,
    rf_pred: np.ndarray,
    x: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute positions from forward_returns and risk_free_rate predictions.
    
    Based on Kaggle discussion/608349:
    position = clip((x - rf_pred) / (forward_pred - rf_pred), clip_min, clip_max)
    
    Parameters
    ----------
    forward_pred : np.ndarray
        Predicted forward_returns.
    rf_pred : np.ndarray
        Predicted risk_free_rate.
    x : float
        Target return level (to be optimized via grid search).
        Intuition: x is the "target return" you want to achieve.
        - x > rf → bullish (position > 0)
        - x < rf → bearish (position < 0, clipped to 0)
    clip_min : float
        Minimum position (0 = 100% cash).
    clip_max : float
        Maximum position (2 = 200% market).
    epsilon : float
        Minimum |denominator| to avoid division by zero.
        
    Returns
    -------
    np.ndarray
        Position values in [clip_min, clip_max].
        
    Notes
    -----
    - When forward_pred ≈ rf_pred, the position becomes unstable.
      We use epsilon to guard against this.
    - The formula simplifies to Kelly-like position sizing:
      position = (expected_excess_return) / (market_excess_return)
    
    Examples
    --------
    >>> forward = np.array([0.001, 0.002, 0.003])
    >>> rf = np.array([0.0003, 0.0003, 0.0003])
    >>> positions = map_positions_from_forward_rf(forward, rf, x=0.001)
    >>> # position[0] = (0.001 - 0.0003) / (0.001 - 0.0003) = 1.0
    >>> # position[1] = (0.001 - 0.0003) / (0.002 - 0.0003) ≈ 0.41
    >>> # position[2] = (0.001 - 0.0003) / (0.003 - 0.0003) ≈ 0.26
    """
    forward_pred = np.asarray(forward_pred, dtype=float)
    rf_pred = np.asarray(rf_pred, dtype=float)
    
    # Denominator: (forward - rf)
    denom = forward_pred - rf_pred
    
    # Guard against division by zero
    # Use sign to preserve direction when |denom| < epsilon
    denom_safe = np.where(
        np.abs(denom) < epsilon,
        np.sign(denom) * epsilon,
        denom
    )
    # Handle exact zero case (sign returns 0 for 0)
    denom_safe = np.where(denom_safe == 0, epsilon, denom_safe)
    
    # Numerator: (x - rf)
    numer = x - rf_pred
    
    # Position calculation
    raw_position = numer / denom_safe
    
    # Clip to valid range
    return np.clip(raw_position, clip_min, clip_max)


def map_positions_from_two_head_config(
    forward_pred: np.ndarray,
    rf_pred: np.ndarray,
    config: TwoHeadPositionConfig | None = None,
) -> np.ndarray:
    """Map predictions to positions using TwoHeadPositionConfig.
    
    Parameters
    ----------
    forward_pred : np.ndarray
        Predicted forward_returns.
    rf_pred : np.ndarray
        Predicted risk_free_rate.
    config : TwoHeadPositionConfig, optional
        Configuration object. Uses defaults if None.
        
    Returns
    -------
    np.ndarray
        Position values in [config.clip_min, config.clip_max].
    """
    if config is None:
        config = TwoHeadPositionConfig()
    return map_positions_from_forward_rf(
        forward_pred,
        rf_pred,
        x=config.x,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        epsilon=config.epsilon,
    )


def compute_hull_sharpe_two_head(
    positions: np.ndarray,
    forward_true: np.ndarray,
    rf_true: np.ndarray,
    annualization: float = 252.0,
) -> Dict[str, float]:
    """Compute Hull Sharpe score for two-head positions.
    
    Parameters
    ----------
    positions : np.ndarray
        Position values in [0, 2].
    forward_true : np.ndarray
        True forward_returns.
    rf_true : np.ndarray
        True risk_free_rate.
    annualization : float
        Annualization factor (252 for daily).
        
    Returns
    -------
    dict
        Sharpe metrics including vol_ratio penalty.
        Keys: sharpe_raw, vol_ratio, vol_penalty, hull_sharpe,
              mean_return, std_return, mean_position, std_position
    """
    # Excess returns
    excess_true = forward_true - rf_true
    
    # Strategy returns
    strategy_returns = positions * excess_true
    
    # Statistics
    mean_return = float(np.mean(strategy_returns))
    std_return = float(np.std(strategy_returns, ddof=1))
    
    # Sharpe
    if std_return > 1e-10:
        sharpe = (mean_return / std_return) * np.sqrt(annualization)
    else:
        sharpe = 0.0
    
    # Vol ratio
    market_std = float(np.std(excess_true, ddof=1))
    if market_std > 1e-10:
        vol_ratio = std_return / market_std
    else:
        vol_ratio = 1.0
    
    # Vol penalty (Hull Competition rule)
    if vol_ratio > 1.2:
        vol_penalty = (vol_ratio - 1.2) * 100
    else:
        vol_penalty = 0.0
    
    # Hull Sharpe
    hull_sharpe = sharpe - vol_penalty
    
    return {
        "sharpe_raw": float(sharpe),
        "vol_ratio": float(vol_ratio),
        "vol_penalty": float(vol_penalty),
        "hull_sharpe": float(hull_sharpe),
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_position": float(np.mean(positions)),
        "std_position": float(np.std(positions)),
    }


def optimize_x_parameter(
    forward_oof: np.ndarray,
    rf_oof: np.ndarray,
    forward_true: np.ndarray,
    rf_true: np.ndarray,
    x_grid: np.ndarray | None = None,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> tuple[float, float, list[Dict[str, Any]]]:
    """Find optimal x parameter via grid search.
    
    Parameters
    ----------
    forward_oof : np.ndarray
        OOF predictions for forward_returns.
    rf_oof : np.ndarray
        OOF predictions for risk_free_rate.
    forward_true : np.ndarray
        True forward_returns.
    rf_true : np.ndarray
        True risk_free_rate.
    x_grid : np.ndarray, optional
        Grid of x values to search. Default: linspace(-0.002, 0.002, 41).
    clip_min : float
        Minimum position.
    clip_max : float
        Maximum position.
        
    Returns
    -------
    tuple
        (best_x, best_hull_sharpe, results_list)
    """
    if x_grid is None:
        x_grid = np.linspace(-0.002, 0.002, 41)
    
    results = []
    best_x = 0.0
    best_hull_sharpe = float("-inf")
    
    for x in x_grid:
        positions = map_positions_from_forward_rf(
            forward_oof, rf_oof, x=x, clip_min=clip_min, clip_max=clip_max
        )
        metrics = compute_hull_sharpe_two_head(
            positions, forward_true, rf_true
        )
        result = {
            "x": float(x),
            **metrics,
        }
        results.append(result)
        
        if metrics["hull_sharpe"] > best_hull_sharpe:
            best_hull_sharpe = metrics["hull_sharpe"]
            best_x = float(x)
    
    return best_x, best_hull_sharpe, results
