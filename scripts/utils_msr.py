#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSR（Mean-to-Std Ratio）プロキシの共通ユーティリティ。

提供機能:
- 予測値からシグナル生成（signal = clip(pred*mult + 1.0, lo, hi)）
- シグナルと実リターンからトレードリターン生成（r = (signal-1.0) * target）
- MSR/Downside MSR の計算（mean(r) / (std(r)+eps), mean(r) / (std(min(r,0))+eps)）
- 単一設定/グリッド探索での評価
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PostProcessParams:
    mult: float = 1.0
    lo: float = 0.0
    hi: float = 2.0


def to_signal(pred: np.ndarray, params: PostProcessParams) -> np.ndarray:
    """pred -> signal = clip(pred*mult + 1.0, lo, hi)

    pred: shape (N,)
    returns: shape (N,)
    """
    p = np.asarray(pred, dtype=float)
    s = p * params.mult + 1.0
    s = np.clip(s, params.lo, params.hi)
    return s


def returns_from_signal(signal: np.ndarray, target_returns: np.ndarray) -> np.ndarray:
    """Compute trade returns: r_t = (signal_t - 1.0) * market_forward_excess_returns_t"""
    s = np.asarray(signal, dtype=float)
    y = np.asarray(target_returns, dtype=float)
    return (s - 1.0) * y


def msr_ratio(r: np.ndarray, eps: float = 1e-8) -> float:
    r = np.asarray(r, dtype=float)
    mean = float(np.nanmean(r))
    std = float(np.nanstd(r))
    return mean / (std + eps)


def msr_downside_ratio(r: np.ndarray, eps: float = 1e-8) -> float:
    r = np.asarray(r, dtype=float)
    mean = float(np.nanmean(r))
    downside = np.minimum(r, 0.0)
    std_down = float(np.nanstd(downside))
    return mean / (std_down + eps)


def vmsr_ratio(r: np.ndarray, y_true: np.ndarray, lam: float = 0.25, eps: float = 1e-8) -> float:
    """Volatility-penalized MSR proxy.

    vMSR = mean(r) / ( std(r) * (1 + lam * max(0, std(r)/std(y) - 1)) + eps )
    - y_true は市場の将来超過リターン（target）
    - lam は過剰ボラに対する罰則強度
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y_true, dtype=float)
    sig_r = float(np.nanstd(r))
    sig_m = float(np.nanstd(y))
    penalty = 1.0 + float(lam) * max(0.0, sig_r / (sig_m + eps) - 1.0)
    return float(np.nanmean(r)) / (sig_r * penalty + eps)


def evaluate_msr_proxy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    params: PostProcessParams,
    eps: float = 1e-8,
    lam: Optional[float] = None,
) -> dict:
    """Evaluate RMSE + MSR metrics for a given post-process parameter.

    Returns dict with: rmse, msr, msr_down, mean, std, std_down
    """
    from sklearn.metrics import mean_squared_error

    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    signal = to_signal(y_pred, params)
    r = returns_from_signal(signal, y_true)
    msr = msr_ratio(r, eps=eps)
    msr_d = msr_downside_ratio(r, eps=eps)
    vmsr = vmsr_ratio(r, y_true, lam=lam if lam is not None else 0.0, eps=eps)
    out = {
        "rmse": rmse,
        "msr": float(msr),
        "msr_down": float(msr_d),
        "vmsr": float(vmsr),
        "vmsr_lam": float(lam if lam is not None else 0.0),
        "mean": float(np.nanmean(r)),
        "std": float(np.nanstd(r)),
        "std_down": float(np.nanstd(np.minimum(r, 0.0))),
    }
    return out


def grid_search_msr(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mult_grid: Iterable[float],
    lo_grid: Iterable[float],
    hi_grid: Iterable[float],
    eps: float = 1e-8,
    optimize_for: str = "msr",
    lam_grid: Optional[Iterable[float]] = None,
) -> Tuple[PostProcessParams, List[dict]]:
    """Grid-search post-process params to maximize a metric.

    optimize_for: "msr" or "msr_down"
    Returns: (best_params, all_results)
    """
    results: List[dict] = []
    best: Tuple[float, PostProcessParams] | None = None
    lam_list = list(lam_grid) if lam_grid is not None else [None]
    for m in mult_grid:
        for lo in lo_grid:
            for hi in hi_grid:
                if lo >= hi:
                    continue
                for lam in lam_list:
                    params = PostProcessParams(mult=float(m), lo=float(lo), hi=float(hi))
                    metrics = evaluate_msr_proxy(y_pred, y_true, params, eps=eps, lam=(lam if lam is not None else 0.0))
                    row = {"mult": float(m), "lo": float(lo), "hi": float(hi), **metrics}
                    results.append(row)
                    score_key = "vmsr" if optimize_for == "vmsr" else optimize_for
                    score = metrics.get(score_key, float("-inf"))
                    if best is None or score > best[0]:
                        best = (float(score), params)
    if best is None:
        # fallback to identity params
        best_params = PostProcessParams()
    else:
        best_params = best[1]
    return best_params, results
