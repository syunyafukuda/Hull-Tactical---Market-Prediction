"""Signal/Prediction to Position mapping utilities.

This module provides functions to convert model predictions (regression targets)
to trading positions for the Hull Competition evaluation.

The Hull Competition expects positions in [0, 2]:
- 0: 100% cash (0% market exposure)
- 1: 100% market (neutral position)
- 2: 200% market (2x leverage)

Model predictions (e.g., RMSE-optimized returns) need to be mapped to this range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PositionMapperConfig:
    """Configuration for prediction-to-position mapping.
    
    Parameters
    ----------
    multiplier : float
        Scaling factor for predictions. Default 100.0.
    offset : float
        Offset to add after scaling. Default 1.0.
    clip_min : float
        Minimum position value. Default 0.0.
    clip_max : float
        Maximum position value. Default 2.0.
    """
    
    multiplier: float = 100.0
    offset: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 2.0


def map_to_position(
    predictions: np.ndarray,
    multiplier: float = 100.0,
    offset: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> np.ndarray:
    """Map model predictions to trading positions.
    
    Applies: position = clip(prediction * multiplier + offset, clip_min, clip_max)
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (e.g., predicted returns).
    multiplier : float, optional
        Scaling factor. Default 100.0.
        Intuition: if predictions are in return space (e.g., 0.01 = 1%),
        multiplier=100 converts to percentage points.
    offset : float, optional
        Offset after scaling. Default 1.0.
        Intuition: 1.0 = neutral position (100% market exposure).
    clip_min : float, optional
        Minimum position. Default 0.0 (100% cash).
    clip_max : float, optional
        Maximum position. Default 2.0 (200% market/2x leverage).
        
    Returns
    -------
    np.ndarray
        Positions in [clip_min, clip_max] range.
        
    Examples
    --------
    >>> predictions = np.array([-0.01, 0.0, 0.01, 0.02])
    >>> positions = map_to_position(predictions)
    >>> # -0.01 * 100 + 1 = 0.0 (100% cash)
    >>> # 0.0 * 100 + 1 = 1.0 (neutral)
    >>> # 0.01 * 100 + 1 = 2.0 (max leverage)
    >>> # 0.02 * 100 + 1 = 3.0 -> clipped to 2.0
    >>> positions
    array([0., 1., 2., 2.])
    """
    positions = predictions * multiplier + offset
    return np.clip(positions, clip_min, clip_max)


def map_to_position_from_config(
    predictions: np.ndarray,
    config: PositionMapperConfig | None = None,
) -> np.ndarray:
    """Map predictions to positions using config object.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions.
    config : PositionMapperConfig, optional
        Mapper configuration. Uses defaults if None.
        
    Returns
    -------
    np.ndarray
        Positions in [config.clip_min, config.clip_max] range.
    """
    if config is None:
        config = PositionMapperConfig()
    
    return map_to_position(
        predictions,
        multiplier=config.multiplier,
        offset=config.offset,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
    )


def calibrate_position_mapper(
    predictions: np.ndarray,
    target_mean: float = 1.0,
    target_range: Tuple[float, float] = (0.1, 1.9),
) -> PositionMapperConfig:
    """Auto-calibrate mapper parameters to achieve target position statistics.
    
    This is useful when you want positions to have a specific mean and range
    regardless of the prediction distribution.
    
    Parameters
    ----------
    predictions : np.ndarray
        Sample predictions to calibrate on.
    target_mean : float, optional
        Target mean position. Default 1.0 (neutral).
    target_range : Tuple[float, float], optional
        Target (percentile_5, percentile_95) of positions.
        Default (0.1, 1.9) = use most of [0, 2] range.
        
    Returns
    -------
    PositionMapperConfig
        Calibrated configuration.
        
    Notes
    -----
    Uses robust statistics (median, percentiles) to avoid outlier effects.
    """
    pred_median = np.median(predictions)
    pred_p5 = np.percentile(predictions, 5)
    pred_p95 = np.percentile(predictions, 95)
    pred_range = pred_p95 - pred_p5
    
    target_low, target_high = target_range
    target_spread = target_high - target_low
    
    # Calculate multiplier to achieve target spread
    if pred_range > 0:
        multiplier = target_spread / pred_range
    else:
        multiplier = 100.0  # fallback
    
    # Calculate offset to achieve target mean
    # position = pred * mult + offset
    # target_mean = pred_median * mult + offset
    # offset = target_mean - pred_median * mult
    offset = target_mean - pred_median * multiplier
    
    return PositionMapperConfig(
        multiplier=float(multiplier),
        offset=float(offset),
        clip_min=0.0,
        clip_max=2.0,
    )


def analyze_position_distribution(
    positions: np.ndarray,
) -> dict:
    """Analyze the distribution of positions.
    
    Useful for debugging and understanding the position mapping.
    
    Parameters
    ----------
    positions : np.ndarray
        Position values (should be in [0, 2]).
        
    Returns
    -------
    dict
        Statistics about the position distribution.
    """
    return {
        "mean": float(np.mean(positions)),
        "std": float(np.std(positions)),
        "min": float(np.min(positions)),
        "max": float(np.max(positions)),
        "median": float(np.median(positions)),
        "p5": float(np.percentile(positions, 5)),
        "p25": float(np.percentile(positions, 25)),
        "p75": float(np.percentile(positions, 75)),
        "p95": float(np.percentile(positions, 95)),
        "pct_at_min_clip": float(np.mean(positions == 0.0)),
        "pct_at_max_clip": float(np.mean(positions == 2.0)),
        "pct_below_neutral": float(np.mean(positions < 1.0)),
        "pct_above_neutral": float(np.mean(positions > 1.0)),
    }


# -----------------------------------------------------------------------------
# Alpha-Beta Position Mapping (Kaggle discussion/611071)
# -----------------------------------------------------------------------------


@dataclass
class AlphaBetaPositionConfig:
    """Alpha-Beta position mapping configuration.
    
    Based on Kaggle discussion/611071 (do-nothing baseline).
    
    Formula: position = clip(beta + alpha * pred_excess, clip_min, clip_max)
    
    Parameters
    ----------
    alpha : float
        Scaling factor for predictions.
        - alpha=0: Ignore predictions (do-nothing)
        - alpha=1: Use raw predictions
        - alpha=0.25: Dampen predictions (recommended)
    beta : float
        Offset/intercept.
        - beta=1.0: Market neutral
        - beta=0.806: Do-nothing optimal for Public LB
    clip_min : float
        Minimum position (0 = 100% cash).
    clip_max : float
        Maximum position (2 = 200% market).
    winsor_pct : float | None
        If provided, winsorize predictions at this percentile.
    """
    alpha: float = 0.25
    beta: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 2.0
    winsor_pct: float | None = None


def map_predictions_to_positions(
    pred_excess: np.ndarray,
    alpha: float = 0.25,
    beta: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
    winsor_pct: float | None = None,
) -> np.ndarray:
    """Affine transform + clipping for predicted excess returns.
    
    Implements the Kaggle discussion/611071 do-nothing baseline approach.
    
    Formula: position = clip(beta + alpha * pred_excess, clip_min, clip_max)
    
    Parameters
    ----------
    pred_excess : np.ndarray
        Predicted excess returns from model.
    alpha : float
        Scaling factor for predictions.
        - alpha=0: Ignore predictions (do-nothing)
        - alpha=1: Use raw predictions
        - alpha=0.25: Dampen predictions (recommended)
    beta : float
        Offset/intercept.
        - beta=1.0: Market neutral
        - beta=0.806: Do-nothing optimal for Public LB
    clip_min : float
        Minimum position (0 = 100% cash).
    clip_max : float
        Maximum position (2 = 200% market).
    winsor_pct : float, optional
        If provided, winsorize predictions at this percentile
        (e.g., 0.01 = clip at 1st/99th percentile).
        
    Returns
    -------
    np.ndarray
        Position values in [clip_min, clip_max].
        
    Examples
    --------
    >>> pred = np.array([-0.02, 0.0, 0.01, 0.05])
    >>> # Do-nothing: alpha=0, beta=0.806
    >>> map_predictions_to_positions(pred, alpha=0, beta=0.806)
    array([0.806, 0.806, 0.806, 0.806])
    >>> # With prediction signal
    >>> map_predictions_to_positions(pred, alpha=0.25, beta=1.0)
    array([0.995, 1.   , 1.0025, 1.0125])
    """
    pred = np.asarray(pred_excess, dtype=float).copy()
    
    # Winsorization (optional)
    if winsor_pct is not None and winsor_pct > 0:
        lower = np.quantile(pred, winsor_pct)
        upper = np.quantile(pred, 1 - winsor_pct)
        pred = np.clip(pred, lower, upper)
    
    # Affine transform
    position = beta + alpha * pred
    
    # Clip to valid range
    return np.clip(position, clip_min, clip_max)


def map_positions_from_alpha_beta_config(
    pred_excess: np.ndarray,
    config: AlphaBetaPositionConfig | None = None,
) -> np.ndarray:
    """Map predictions to positions using AlphaBetaPositionConfig.
    
    Parameters
    ----------
    pred_excess : np.ndarray
        Predicted excess returns.
    config : AlphaBetaPositionConfig, optional
        Configuration object. Uses defaults if None.
        
    Returns
    -------
    np.ndarray
        Position values in [config.clip_min, config.clip_max].
    """
    if config is None:
        config = AlphaBetaPositionConfig()
    return map_predictions_to_positions(
        pred_excess,
        alpha=config.alpha,
        beta=config.beta,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        winsor_pct=config.winsor_pct,
    )

