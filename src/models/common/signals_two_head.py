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

# Re-use official Hull Competition Sharpe metric
from src.metrics.hull_sharpe import compute_hull_sharpe, HullSharpeResult


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
) -> Dict[str, float]:
    """Compute Hull Sharpe score for two-head positions.
    
    This function uses the official Hull Competition Sharpe metric
    from src/metrics/hull_sharpe.py to ensure consistency with LB scoring.
    
    Parameters
    ----------
    positions : np.ndarray
        Position values in [0, 2].
    forward_true : np.ndarray
        True forward_returns.
    rf_true : np.ndarray
        True risk_free_rate.
        
    Returns
    -------
    dict
        Official Hull Sharpe metrics including:
        - hull_sharpe: final score (= raw_sharpe - vol_penalty - return_penalty)
        - raw_sharpe: annualized Sharpe ratio
        - vol_ratio: strategy_std / market_std
        - vol_penalty: penalty when vol_ratio outside [0.8, 1.2]
        - return_penalty: penalty when strategy_return < market_return
        - strategy_mean, strategy_std, market_mean, market_std
        - mean_position, std_position
        
    Notes
    -----
    The official strategy return formula is:
        strategy_returns = rf * (1 - position) + position * forward
    
    This represents:
        - (1 - position) invested in risk-free asset
        - position invested in S&P 500
    """
    positions = np.asarray(positions, dtype=float)
    forward_true = np.asarray(forward_true, dtype=float)
    rf_true = np.asarray(rf_true, dtype=float)
    
    # Use official Hull Competition Sharpe metric
    result: HullSharpeResult = compute_hull_sharpe(
        prediction=positions,
        forward_returns=forward_true,
        risk_free_rate=rf_true,
        validate=False,  # positions already clipped
    )
    
    return {
        "hull_sharpe": result.final_score,
        "raw_sharpe": result.raw_sharpe,
        "vol_ratio": result.vol_ratio,
        "vol_penalty": result.vol_penalty,
        "return_penalty": result.return_penalty,
        "strategy_mean": result.strategy_mean,
        "strategy_std": result.strategy_std,
        "market_mean": result.market_mean,
        "market_std": result.market_std,
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
