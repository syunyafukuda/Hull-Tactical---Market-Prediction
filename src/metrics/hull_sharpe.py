"""Hull Competition Sharpe metric implementation.

This module implements the official Hull Competition Sharpe metric
as specified in the Kaggle competition evaluation notebook.

The metric consists of:
1. Raw Sharpe ratio (annualized)
2. Volatility penalty (when strategy vol differs from market vol by >20%)
3. Return penalty (when strategy return is below market return)

Reference: Kaggle Hull Competition Sharpe Notebook (metric/hull-competition-sharpe)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Constants from official metric
ANNUALIZATION = np.sqrt(252.0)
MAX_INVESTMENT = 2.0
MIN_INVESTMENT = 0.0
VOL_TOLERANCE = 0.20  # ±20% corridor around market volatility
RETURN_FLOOR = 0.0    # Penalty starts when strategy return < market return
EPS = 1e-12


@dataclass(frozen=True)
class HullSharpeResult:
    """Result container for Hull Competition Sharpe calculation."""
    
    final_score: float
    raw_sharpe: float
    vol_ratio: float
    vol_penalty: float
    return_penalty: float
    strategy_mean: float
    strategy_std: float
    market_mean: float
    market_std: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "final_score": self.final_score,
            "raw_sharpe": self.raw_sharpe,
            "vol_ratio": self.vol_ratio,
            "vol_penalty": self.vol_penalty,
            "return_penalty": self.return_penalty,
            "strategy_mean": self.strategy_mean,
            "strategy_std": self.strategy_std,
            "market_mean": self.market_mean,
            "market_std": self.market_std,
        }


def validate_prediction(prediction: np.ndarray, raise_on_error: bool = False) -> bool:
    """Validate that predictions are in valid range [0, 2].
    
    Parameters
    ----------
    prediction : np.ndarray
        Array of predictions (investment ratios).
    raise_on_error : bool, optional
        If True, raise ValueError on invalid prediction.
        If False (default), return False for invalid predictions.
        
    Returns
    -------
    bool
        True if prediction is valid, False otherwise.
        
    Raises
    ------
    ValueError
        If raise_on_error=True and predictions are invalid.
    """
    prediction = np.asarray(prediction, dtype=float)
    
    if np.isnan(prediction).any():
        if raise_on_error:
            raise ValueError("prediction contains NaN values")
        return False
    
    if prediction.min() < MIN_INVESTMENT:
        if raise_on_error:
            raise ValueError(
                f"prediction contains values below {MIN_INVESTMENT}: "
                f"min={prediction.min()}"
            )
        return False
    
    if prediction.max() > MAX_INVESTMENT:
        if raise_on_error:
            raise ValueError(
                f"prediction contains values above {MAX_INVESTMENT}: "
                f"max={prediction.max()}"
            )
        return False
    
    return True


def compute_hull_sharpe(
    prediction: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    *,
    validate: bool = True,
) -> HullSharpeResult:
    """Compute the Hull Competition Sharpe metric.
    
    This implements the official Kaggle evaluation metric for the
    Hull Tactical Market Prediction competition.
    
    Parameters
    ----------
    prediction : np.ndarray
        Investment ratio predictions in range [0, 2].
    forward_returns : np.ndarray
        S&P 500 next-day returns.
    risk_free_rate : np.ndarray
        Risk-free rate (Federal Funds Rate).
    validate : bool, optional
        Whether to validate prediction range. Default True.
        
    Returns
    -------
    HullSharpeResult
        Named tuple containing final_score and all components.
        
    Notes
    -----
    The strategy return formula:
        strategy_returns = rf * (1 - position) + position * forward_returns
        
    This represents:
        - (1 - position) invested in risk-free asset
        - position invested in S&P 500
        
    The final score includes:
        - Raw Sharpe: annualized Sharpe ratio of strategy excess returns
        - Vol Penalty: penalty when strategy vol differs from market vol by >20%
        - Return Penalty: penalty when strategy return < market return
    """
    # Convert to numpy arrays
    prediction = np.asarray(prediction, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)
    
    # Validate
    if validate:
        validate_prediction(prediction, raise_on_error=True)
    
    # Clip to valid range (defensive, should already be valid)
    position = np.clip(prediction, MIN_INVESTMENT, MAX_INVESTMENT)
    
    # Compute strategy returns
    # Strategy: (1 - position) in risk-free + position in market
    strategy_returns = risk_free_rate * (1.0 - position) + position * forward_returns
    
    # Excess returns over risk-free rate
    strategy_excess = strategy_returns - risk_free_rate
    market_excess = forward_returns - risk_free_rate
    
    # Basic statistics
    strategy_mean = float(np.nanmean(strategy_excess))
    strategy_std = float(np.nanstd(strategy_excess, ddof=0))
    market_mean = float(np.nanmean(market_excess))
    market_std = float(np.nanstd(market_excess, ddof=0))
    
    # Raw Sharpe (annualized)
    raw_sharpe = float(ANNUALIZATION * strategy_mean / (strategy_std + EPS))
    
    # Volatility penalty
    # Penalty when strategy vol differs from market vol by more than 20%
    vol_ratio = strategy_std / (market_std + EPS)
    upper_bound = 1.0 + VOL_TOLERANCE
    lower_bound = 1.0 - VOL_TOLERANCE
    
    if vol_ratio > upper_bound:
        vol_penalty = (vol_ratio - upper_bound) * 100.0
    elif vol_ratio < lower_bound:
        vol_penalty = (lower_bound - vol_ratio) * 100.0
    else:
        vol_penalty = 0.0
    
    # Return penalty
    # Penalty when strategy return is below market return
    if strategy_mean < market_mean + RETURN_FLOOR:
        return_penalty = (market_mean - strategy_mean) * 1000.0
    else:
        return_penalty = 0.0
    
    # Final score
    final_score = raw_sharpe - vol_penalty - return_penalty
    
    return HullSharpeResult(
        final_score=float(final_score),
        raw_sharpe=raw_sharpe,
        vol_ratio=float(vol_ratio),
        vol_penalty=float(vol_penalty),
        return_penalty=float(return_penalty),
        strategy_mean=strategy_mean,
        strategy_std=strategy_std,
        market_mean=market_mean,
        market_std=market_std,
    )


def hull_sharpe_score(
    prediction: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    *,
    validate: bool = True,
) -> float:
    """Compute only the final Hull Competition Sharpe score.
    
    Convenience function that returns only the final score.
    For detailed breakdown, use compute_hull_sharpe().
    
    Parameters
    ----------
    prediction : np.ndarray
        Investment ratio predictions in range [0, 2].
    forward_returns : np.ndarray
        S&P 500 next-day returns.
    risk_free_rate : np.ndarray
        Risk-free rate (Federal Funds Rate).
    validate : bool, optional
        Whether to validate prediction range. Default True.
        
    Returns
    -------
    float
        Final Hull Competition Sharpe score.
    """
    result = compute_hull_sharpe(
        prediction, forward_returns, risk_free_rate, validate=validate
    )
    return result.final_score


def evaluate_hull_metric(
    y_pred_returns: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    *,
    mult: float = 1.0,
    offset: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> Dict[str, float]:
    """Evaluate Hull metric from raw model predictions.
    
    This function handles the conversion from model predictions
    (excess return predictions) to investment ratios [0, 2],
    then computes the Hull Sharpe metric.
    
    Parameters
    ----------
    y_pred_returns : np.ndarray
        Model predictions (excess return scale).
    forward_returns : np.ndarray
        S&P 500 next-day returns.
    risk_free_rate : np.ndarray
        Risk-free rate (Federal Funds Rate).
    mult : float, optional
        Multiplier for predictions. Default 1.0.
    offset : float, optional
        Offset added after multiplying. Default 1.0.
    clip_min : float, optional
        Minimum investment ratio. Default 0.0.
    clip_max : float, optional
        Maximum investment ratio. Default 2.0.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with all metric components plus RMSE.
        
    Notes
    -----
    The prediction mapping is:
        position = clip(y_pred * mult + offset, clip_min, clip_max)
        
    Default mapping (mult=1, offset=1) assumes:
        y_pred=0 → position=1 (market weight)
        y_pred<0 → position<1 (underweight)
        y_pred>0 → position>1 (overweight)
    """
    from sklearn.metrics import mean_squared_error
    
    y_pred_returns = np.asarray(y_pred_returns, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)
    
    # Map predictions to investment ratios
    prediction = y_pred_returns * mult + offset
    prediction = np.clip(prediction, clip_min, clip_max)
    
    # Compute Hull Sharpe
    result = compute_hull_sharpe(
        prediction, forward_returns, risk_free_rate, validate=False
    )
    
    # Also compute RMSE (for reference)
    market_excess = forward_returns - risk_free_rate
    rmse = float(np.sqrt(mean_squared_error(market_excess, y_pred_returns)))
    
    # Return combined metrics
    metrics = result.to_dict()
    metrics["rmse"] = rmse
    metrics["mult"] = mult
    metrics["offset"] = offset
    metrics["clip_min"] = clip_min
    metrics["clip_max"] = clip_max
    
    return metrics
