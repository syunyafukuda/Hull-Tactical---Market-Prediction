# Two-Head Learning Implementation Specification

最終更新: 2025-12-15  
ステータス: 実装開始  
関連: [概要書](two_head_learning.md)

---

## 1. 概要

本ドキュメントは、`forward_returns` と `risk_free_rate` を同時予測する 2 ヘッド学習の**詳細な実装仕様**を定義します。

### 1.1 現状の課題

| 項目 | 現状 (1 ヘッド) | 課題 |
|------|-----------------|------|
| ターゲット | `market_forward_excess_returns` のみ | 情報ロス |
| Position 計算 | `position = clip(β + α * pred_excess)` | 理論的根拠が弱い |
| LB スコア | 3.318 | Local WF Sharpe と 30x 乖離 |

### 1.2 2 ヘッドのメリット

1. **評価式への整合**: `positions = (x - rf) / (fr - rf)` は Hull Sharpe の理論最適解に近い
2. **情報量増加**: 2 つの予測から position を導出 → 複合的なシグナル
3. **パラメータ削減**: α/β の 2 パラメータ → x の 1 パラメータ

---

## 2. 設計方針: コピー＆リネーム方式

既存ファイルを直接修正すると履歴追跡が困難になるため、以下の方針を採用:

1. **既存ファイルは変更しない**: `train_lgbm.py`, `predict_lgbm.py`, `signals.py` は現状維持
2. **コピー＆リネーム**: 既存ファイルをベースに新規ファイルを作成
3. **独立した実行パス**: 2ヘッド専用のスクリプト群として分離

### 2.1 コピー元 → コピー先 マッピング

| コピー元 (既存) | コピー先 (新規) | 目的 |
|-----------------|-----------------|------|
| `src/metrics/lgbm/train_lgbm.py` | `src/metrics/lgbm_two_head/train_lgbm_two_head.py` | 2ヘッドトレーニング |
| `src/metrics/lgbm/predict_lgbm.py` | `src/metrics/lgbm_two_head/predict_lgbm_two_head.py` | 2ヘッド推論 |
| `src/models/common/signals.py` | `src/models/common/signals_two_head.py` | 2ヘッド用シグナル関数 |

---

## 3. フォルダ構成

```
src/
├── metrics/
│   ├── lgbm/                              # [既存] 1ヘッド用（変更なし）
│   │   ├── __init__.py
│   │   ├── train_lgbm.py
│   │   └── predict_lgbm.py
│   │
│   └── lgbm_two_head/                     # [新規] 2ヘッド専用モジュール
│       ├── __init__.py                    # [新規]
│       ├── train_lgbm_two_head.py         # [新規] train_lgbm.py ベース
│       └── predict_lgbm_two_head.py       # [新規] predict_lgbm.py ベース
│
├── models/
│   └── common/
│       ├── signals.py                     # [既存] 変更なし
│       └── signals_two_head.py            # [新規] 2ヘッド用関数
│
scripts/
├── two_head/                              # [新規] 2ヘッド専用スクリプト
│   ├── tune_x_parameter.py                # [新規] x パラメータ最適化
│   └── compare_one_vs_two_head.py         # [新規] 1ヘッド vs 2ヘッド比較
│
configs/
└── evaluation/
    ├── walk_forward.yaml                  # [既存] 変更なし
    └── two_head.yaml                      # [新規] 2ヘッド設定
│
artifacts/
└── models/
    ├── lgbm-best-sharpe/                  # [既存] 1ヘッドモデル
    └── lgbm-two-head/                     # [新規] 2ヘッドモデル
        ├── forward_model.pkl              # forward_returns 予測モデル
        ├── rf_model.pkl                   # risk_free_rate 予測モデル
        ├── inference_bundle.pkl           # 統合推論バンドル
        ├── model_meta.json                # メタ情報 (best_x 含む)
        ├── oof_predictions.csv            # OOF予測 (forward/rf 両方)
        ├── x_search_results.csv           # x パラメータ探索結果
        └── submission.csv                 # 提出ファイル
│
results/
└── two_head/                              # [新規] 実験結果
    ├── x_search.csv                       # x 探索ログ
    └── comparison.csv                     # 1ヘッド vs 2ヘッド比較
│
notebooks/
└── submit/
    ├── LGBM-sharpe-clip.ipynb             # [既存] 1ヘッド提出
    └── LGBM-two-head.ipynb                # [新規] 2ヘッド提出
│
docs/
└── evaluation/
    ├── two_head_learning.md               # [既存] 概要書
    └── two_head_learning_spec.md          # [本ファイル] 仕様書
│
tests/
└── metrics/
    └── lgbm_two_head/                     # [新規] テスト
        ├── test_signals_two_head.py
        └── test_train_two_head.py
```

---

## 4. 作成ファイル一覧

### 4.1 コア実装 (src/)

| # | ファイル | ベース | 説明 |
|---|----------|--------|------|
| 1 | `src/metrics/lgbm_two_head/__init__.py` | - | モジュール初期化 |
| 2 | `src/metrics/lgbm_two_head/train_lgbm_two_head.py` | `train_lgbm.py` | 2ヘッド学習 |
| 3 | `src/metrics/lgbm_two_head/predict_lgbm_two_head.py` | `predict_lgbm.py` | 2ヘッド推論 |
| 4 | `src/models/common/signals_two_head.py` | `signals.py` (部分) | 2ヘッド用シグナル関数 |

### 4.2 スクリプト (scripts/)

| # | ファイル | 説明 |
|---|----------|------|
| 5 | `scripts/two_head/tune_x_parameter.py` | x パラメータ Grid Search |
| 6 | `scripts/two_head/compare_one_vs_two_head.py` | 比較評価スクリプト |

### 4.3 設定 (configs/)

| # | ファイル | 説明 |
|---|----------|------|
| 7 | `configs/evaluation/two_head.yaml` | 2ヘッド設定ファイル |

### 4.4 テスト (tests/)

| # | ファイル | 説明 |
|---|----------|------|
| 8 | `tests/metrics/lgbm_two_head/test_signals_two_head.py` | シグナル関数テスト |
| 9 | `tests/metrics/lgbm_two_head/test_train_two_head.py` | トレーニングテスト |

### 4.5 ノートブック (notebooks/)

| # | ファイル | ベース | 説明 |
|---|----------|--------|------|
| 10 | `notebooks/submit/LGBM-two-head.ipynb` | `LGBM-sharpe-clip.ipynb` | Kaggle提出用 |

---

## 5. 既存ファイルへの影響

**変更なし** (履歴保持のため):

| ファイル | 状態 |
|----------|------|
| `src/metrics/lgbm/train_lgbm.py` | 変更なし |
| `src/metrics/lgbm/predict_lgbm.py` | 変更なし |
| `src/models/common/signals.py` | 変更なし |
| `configs/evaluation/walk_forward.yaml` | 変更なし |

---

## 6. 詳細実装仕様

### 6.1 `src/models/common/signals_two_head.py` (新規作成)

**ベース**: `signals.py` から必要な dataclass と関数を参考に、2ヘッド専用として新規作成

```python
"""Two-Head Signal/Prediction to Position mapping utilities.

This module provides functions specifically for the two-head learning approach,
where forward_returns and risk_free_rate are predicted separately.

Based on Kaggle discussion/608349:
position = clip((x - rf_pred) / (forward_pred - rf_pred), clip_min, clip_max)
"""

from __future__ import annotations

from dataclasses import dataclass

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
    >>> map_positions_from_forward_rf(forward, rf, x=0.001)
    array([1.0, 0.538..., 0.368...])
    """
    forward_pred = np.asarray(forward_pred, dtype=float)
    rf_pred = np.asarray(rf_pred, dtype=float)
    
    # Denominator: (forward - rf)
    denom = forward_pred - rf_pred
    
    # Guard against division by zero
    denom_safe = np.where(
        np.abs(denom) < epsilon,
        np.sign(denom) * epsilon,
        denom
    )
    # Handle exact zero case
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
```

---

### 6.2 `src/metrics/lgbm_two_head/train_lgbm_two_head.py` (新規作成)

**ベース**: `src/metrics/lgbm/train_lgbm.py` をコピーして大幅改修

```python
#!/usr/bin/env python
"""Two-Head LightGBM Training for Hull Competition.

Trains two separate models:
1. forward_model: Predicts forward_returns
2. rf_model: Predicts risk_free_rate

Usage:
    python -m src.metrics.lgbm_two_head.train_lgbm_two_head \
        --cv-mode walk_forward \
        --out-dir artifacts/models/lgbm-two-head
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.pipeline import Pipeline

# 既存のユーティリティをインポート (1ヘッドから流用)
from src.metrics.lgbm.train_lgbm import (
    load_raw_train,
    load_preprocess_config,
    build_preprocessor,
    apply_feature_tier_selection,
    setup_walk_forward_cv,
)

# 2ヘッド専用シグナル関数
from src.models.common.signals_two_head import (
    map_positions_from_forward_rf,
    TwoHeadPositionConfig,
)


# =============================================================================
# Constants
# =============================================================================

TARGET_FORWARD = "forward_returns"
TARGET_RF = "risk_free_rate"
TARGET_EXCESS = "market_forward_excess_returns"

DEFAULT_LGBM_PARAMS = {
    "learning_rate": 0.05,
    "n_estimators": 600,
    "num_leaves": 63,
    "min_data_in_leaf": 32,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


# =============================================================================
# Two-Head Training Logic
# =============================================================================

def train_two_head_fold(
    X_train: pd.DataFrame,
    y_forward_train: pd.Series,
    y_rf_train: pd.Series,
    X_val: pd.DataFrame,
    preprocessor: Pipeline,
    lgbm_params: dict,
) -> tuple[Pipeline, Pipeline, np.ndarray, np.ndarray]:
    """Train forward and rf models for a single fold.
    
    Returns
    -------
    tuple
        (forward_pipeline, rf_pipeline, forward_pred, rf_pred)
    """
    # Clone preprocessor for each model
    forward_preprocessor = clone(preprocessor)
    rf_preprocessor = clone(preprocessor)
    
    # Build pipelines
    forward_model = LGBMRegressor(**lgbm_params)
    rf_model = LGBMRegressor(**lgbm_params)
    
    forward_pipeline = Pipeline([
        ("preprocessor", forward_preprocessor),
        ("model", forward_model),
    ])
    rf_pipeline = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("model", rf_model),
    ])
    
    # Fit
    forward_pipeline.fit(X_train, y_forward_train)
    rf_pipeline.fit(X_train, y_rf_train)
    
    # Predict on validation
    forward_pred = forward_pipeline.predict(X_val)
    rf_pred = rf_pipeline.predict(X_val)
    
    return forward_pipeline, rf_pipeline, forward_pred, rf_pred


def compute_hull_sharpe_two_head(
    positions: np.ndarray,
    forward_true: np.ndarray,
    rf_true: np.ndarray,
    annualization: float = 252.0,
) -> dict:
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
    """
    # Excess returns
    excess_true = forward_true - rf_true
    
    # Strategy returns
    strategy_returns = positions * excess_true
    
    # Statistics
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns, ddof=1)
    
    # Sharpe
    if std_return > 1e-10:
        sharpe = (mean_return / std_return) * np.sqrt(annualization)
    else:
        sharpe = 0.0
    
    # Vol ratio
    market_std = np.std(excess_true, ddof=1)
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
        "sharpe_raw": sharpe,
        "vol_ratio": vol_ratio,
        "vol_penalty": vol_penalty,
        "hull_sharpe": hull_sharpe,
        "mean_return": mean_return,
        "std_return": std_return,
    }


def optimize_x_parameter(
    forward_oof: np.ndarray,
    rf_oof: np.ndarray,
    forward_true: np.ndarray,
    rf_true: np.ndarray,
    x_grid: np.ndarray | None = None,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> tuple[float, float, pd.DataFrame]:
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
        (best_x, best_sharpe, results_df)
    """
    if x_grid is None:
        x_grid = np.linspace(-0.002, 0.002, 41)
    
    results = []
    for x in x_grid:
        positions = map_positions_from_forward_rf(
            forward_oof, rf_oof, x=x, clip_min=clip_min, clip_max=clip_max
        )
        metrics = compute_hull_sharpe_two_head(
            positions, forward_true, rf_true
        )
        results.append({
            "x": x,
            "hull_sharpe": metrics["hull_sharpe"],
            "sharpe_raw": metrics["sharpe_raw"],
            "vol_ratio": metrics["vol_ratio"],
            "vol_penalty": metrics["vol_penalty"],
            "position_mean": np.mean(positions),
            "position_std": np.std(positions),
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df["hull_sharpe"].idxmax()
    best_x = results_df.loc[best_idx, "x"]
    best_sharpe = results_df.loc[best_idx, "hull_sharpe"]
    
    return best_x, best_sharpe, results_df


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Two-Head LightGBM for Hull Competition"
    )
    parser.add_argument(
        "--config-path", type=str,
        default="configs/feature_generation/feature_generation.yaml",
    )
    parser.add_argument(
        "--preprocess-config", type=str,
        default="configs/preprocess/preprocess.yaml",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="artifacts/models/lgbm-two-head",
    )
    parser.add_argument(
        "--cv-mode", type=str, default="walk_forward",
        choices=["kfold", "walk_forward"],
    )
    parser.add_argument(
        "--feature-tier", type=str, default="tier3",
        choices=["tier1", "tier2", "tier3", "none"],
    )
    parser.add_argument(
        "--wf-train-window", type=int, default=6000,
    )
    parser.add_argument(
        "--wf-val-window", type=int, default=1000,
    )
    parser.add_argument(
        "--wf-step", type=int, default=500,
    )
    parser.add_argument(
        "--x-min", type=float, default=-0.002,
        help="Minimum x for grid search",
    )
    parser.add_argument(
        "--x-max", type=float, default=0.002,
        help="Maximum x for grid search",
    )
    parser.add_argument(
        "--x-steps", type=int, default=41,
        help="Number of x values to search",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[info] Loading data...")
    train_df = load_raw_train()
    
    # Validate columns
    for col in [TARGET_FORWARD, TARGET_RF, TARGET_EXCESS]:
        if col not in train_df.columns:
            raise KeyError(f"Required column '{col}' not found")
    
    # Feature preparation
    print("[info] Preparing features...")
    preprocess_cfg = load_preprocess_config(args.preprocess_config)
    preprocessor = build_preprocessor(preprocess_cfg)
    
    # Feature selection
    drop_cols = ["date_id", "row_id", TARGET_FORWARD, TARGET_RF, TARGET_EXCESS, "is_scored"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[feature_cols].copy()
    
    # Apply tier selection
    X, excluded_features = apply_feature_tier_selection(X, args.feature_tier)
    print(f"[info] Features: {X.shape[1]} (excluded {len(excluded_features)})")
    
    # Targets
    y_forward = train_df[TARGET_FORWARD].copy()
    y_rf = train_df[TARGET_RF].copy()
    y_excess = train_df[TARGET_EXCESS].copy()
    
    # Setup CV
    print(f"[info] Setting up {args.cv_mode} CV...")
    if args.cv_mode == "walk_forward":
        folds = setup_walk_forward_cv(
            len(train_df),
            train_window=args.wf_train_window,
            val_window=args.wf_val_window,
            step=args.wf_step,
        )
    else:
        # KFold fallback
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=4, shuffle=False)
        folds = list(kf.split(X))
    
    print(f"[info] {len(folds)} folds")
    
    # OOF arrays
    forward_oof = np.full(len(X), np.nan)
    rf_oof = np.full(len(X), np.nan)
    
    fold_logs = []
    
    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n[fold {fold_idx + 1}] train={len(train_idx)}, val={len(val_idx)}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_forward_train, y_forward_val = y_forward.iloc[train_idx], y_forward.iloc[val_idx]
        y_rf_train, y_rf_val = y_rf.iloc[train_idx], y_rf.iloc[val_idx]
        
        # Train
        forward_pipe, rf_pipe, forward_pred, rf_pred = train_two_head_fold(
            X_train, y_forward_train, y_rf_train, X_val,
            preprocessor, DEFAULT_LGBM_PARAMS,
        )
        
        # Store OOF
        forward_oof[val_idx] = forward_pred
        rf_oof[val_idx] = rf_pred
        
        # RMSE
        forward_rmse = np.sqrt(np.mean((forward_pred - y_forward_val) ** 2))
        rf_rmse = np.sqrt(np.mean((rf_pred - y_rf_val) ** 2))
        
        print(f"  forward_rmse={forward_rmse:.6f}, rf_rmse={rf_rmse:.6f}")
        
        fold_logs.append({
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "forward_rmse": forward_rmse,
            "rf_rmse": rf_rmse,
        })
    
    # Remove NaN rows (from walk-forward initial gap)
    valid_mask = ~np.isnan(forward_oof)
    forward_oof_valid = forward_oof[valid_mask]
    rf_oof_valid = rf_oof[valid_mask]
    forward_true_valid = y_forward.values[valid_mask]
    rf_true_valid = y_rf.values[valid_mask]
    
    print(f"\n[info] Valid OOF rows: {np.sum(valid_mask)}")
    
    # Optimize x parameter
    print("\n[info] Optimizing x parameter...")
    x_grid = np.linspace(args.x_min, args.x_max, args.x_steps)
    best_x, best_sharpe, x_results = optimize_x_parameter(
        forward_oof_valid, rf_oof_valid,
        forward_true_valid, rf_true_valid,
        x_grid=x_grid,
    )
    
    print(f"[info] Best x = {best_x:.6f}, Hull Sharpe = {best_sharpe:.4f}")
    
    # Compare with do-nothing baseline
    positions_donothing = np.full_like(forward_oof_valid, 0.806)
    donothing_metrics = compute_hull_sharpe_two_head(
        positions_donothing, forward_true_valid, rf_true_valid
    )
    print(f"[info] Do-nothing Hull Sharpe = {donothing_metrics['hull_sharpe']:.4f}")
    
    # Final positions with best x
    best_positions = map_positions_from_forward_rf(
        forward_oof_valid, rf_oof_valid, x=best_x
    )
    best_metrics = compute_hull_sharpe_two_head(
        best_positions, forward_true_valid, rf_true_valid
    )
    
    print("\n[info] Final Metrics (best x):")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save artifacts
    print("\n[info] Saving artifacts...")
    
    # OOF predictions
    oof_df = pd.DataFrame({
        "forward_oof": forward_oof,
        "rf_oof": rf_oof,
        "forward_true": y_forward.values,
        "rf_true": y_rf.values,
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    
    # Fold logs
    pd.DataFrame(fold_logs).to_csv(out_dir / "cv_fold_logs.csv", index=False)
    
    # x search results
    x_results.to_csv(out_dir / "x_search_results.csv", index=False)
    
    # Model meta
    meta = {
        "model_type": "lightgbm_two_head",
        "feature_tier": args.feature_tier,
        "n_features": X.shape[1],
        "cv_mode": args.cv_mode,
        "n_folds": len(folds),
        "best_x": best_x,
        "best_hull_sharpe": best_sharpe,
        "donothing_hull_sharpe": donothing_metrics["hull_sharpe"],
        "improvement_over_donothing": best_sharpe - donothing_metrics["hull_sharpe"],
        "position_mapping": {
            "x": best_x,
            "clip_min": 0.0,
            "clip_max": 2.0,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": DEFAULT_LGBM_PARAMS,
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Note: Full model bundles would be saved here for inference
    # For now, we focus on OOF evaluation
    
    print(f"\n[ok] Artifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

### 6.3 `configs/evaluation/two_head.yaml` (新規作成)

```yaml
# Two-Head Learning Configuration
# ================================

two_head:
  enabled: true
  
  targets:
    forward: "forward_returns"
    rf: "risk_free_rate"
  
  position_mapping:
    # x parameter: to be optimized via grid search
    x: 0.0
    clip_min: 0.0
    clip_max: 2.0
    epsilon: 1.0e-8
  
  x_optimization:
    # Grid search range for x
    min: -0.002
    max: 0.002
    steps: 41
  
  walk_forward:
    train_window: 6000
    val_window: 1000
    step: 500
    n_splits: 4

model:
  type: lightgbm
  params:
    learning_rate: 0.05
    n_estimators: 600
    num_leaves: 63
    min_data_in_leaf: 32
    feature_fraction: 0.9
    bagging_fraction: 0.9
    bagging_freq: 1
    random_state: 42
    n_jobs: -1
    verbosity: -1
```

---

### 6.4 `scripts/two_head/tune_x_parameter.py` (新規作成)

```python
#!/usr/bin/env python
"""Grid search for optimal x parameter in two-head position mapping.

Usage:
    python scripts/two_head/tune_x_parameter.py \
        --oof-path artifacts/models/lgbm-two-head/oof_predictions.csv \
        --out-dir results/two_head
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 2ヘッド専用シグナル関数
from src.models.common.signals_two_head import map_positions_from_forward_rf


def compute_hull_sharpe(
    positions: np.ndarray,
    forward_true: np.ndarray,
    rf_true: np.ndarray,
    annualization: float = 252.0,
) -> dict:
    """Compute Hull Sharpe score."""
    excess_true = forward_true - rf_true
    strategy_returns = positions * excess_true
    
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns, ddof=1)
    
    if std_return > 1e-10:
        sharpe = (mean_return / std_return) * np.sqrt(annualization)
    else:
        sharpe = 0.0
    
    market_std = np.std(excess_true, ddof=1)
    if market_std > 1e-10:
        vol_ratio = std_return / market_std
    else:
        vol_ratio = 1.0
    
    if vol_ratio > 1.2:
        vol_penalty = (vol_ratio - 1.2) * 100
    else:
        vol_penalty = 0.0
    
    hull_sharpe = sharpe - vol_penalty
    
    return {
        "sharpe_raw": sharpe,
        "vol_ratio": vol_ratio,
        "vol_penalty": vol_penalty,
        "hull_sharpe": hull_sharpe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/two_head")
    parser.add_argument("--x-min", type=float, default=-0.002)
    parser.add_argument("--x-max", type=float, default=0.002)
    parser.add_argument("--x-steps", type=int, default=101)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OOF predictions
    oof_df = pd.read_csv(args.oof_path)
    
    forward_oof = oof_df["forward_oof"].values
    rf_oof = oof_df["rf_oof"].values
    forward_true = oof_df["forward_true"].values
    rf_true = oof_df["rf_true"].values
    
    # Filter valid rows
    valid_mask = ~np.isnan(forward_oof)
    forward_oof = forward_oof[valid_mask]
    rf_oof = rf_oof[valid_mask]
    forward_true = forward_true[valid_mask]
    rf_true = rf_true[valid_mask]
    
    print(f"[info] Valid rows: {len(forward_oof)}")
    
    # Grid search
    x_grid = np.linspace(args.x_min, args.x_max, args.x_steps)
    results = []
    
    for x in x_grid:
        positions = map_positions_from_forward_rf(
            forward_oof, rf_oof, x=x
        )
        metrics = compute_hull_sharpe(positions, forward_true, rf_true)
        results.append({
            "x": x,
            **metrics,
            "position_mean": np.mean(positions),
            "position_std": np.std(positions),
            "position_min": np.min(positions),
            "position_max": np.max(positions),
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "x_search.csv", index=False)
    
    # Best result
    best_idx = results_df["hull_sharpe"].idxmax()
    best = results_df.loc[best_idx]
    
    print("\n[info] Best configuration:")
    print(f"  x = {best['x']:.6f}")
    print(f"  Hull Sharpe = {best['hull_sharpe']:.4f}")
    print(f"  Vol Ratio = {best['vol_ratio']:.4f}")
    print(f"  Position mean = {best['position_mean']:.4f}")
    print(f"  Position std = {best['position_std']:.4f}")
    
    # Do-nothing comparison
    donothing_positions = np.full_like(forward_oof, 0.806)
    donothing_metrics = compute_hull_sharpe(donothing_positions, forward_true, rf_true)
    print(f"\n[info] Do-nothing baseline:")
    print(f"  Hull Sharpe = {donothing_metrics['hull_sharpe']:.4f}")
    print(f"\n[info] Improvement: {best['hull_sharpe'] - donothing_metrics['hull_sharpe']:.4f}")
    
    print(f"\n[ok] Results saved to {out_dir / 'x_search.csv'}")


if __name__ == "__main__":
    main()
```

---

### 6.5 `src/metrics/lgbm_two_head/predict_lgbm_two_head.py` (新規作成)

**ベース**: `src/metrics/lgbm/predict_lgbm.py` をコピーして2ヘッド専用に改修

```python
#!/usr/bin/env python
"""Two-Head LightGBM Prediction for Hull Competition.

Loads trained forward_model and rf_model, generates positions using
the optimized x parameter, and creates submission.csv.

Usage:
    python -m src.metrics.lgbm_two_head.predict_lgbm_two_head \
        --artifacts-dir artifacts/models/lgbm-two-head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

# 2ヘッド専用シグナル関数
from src.models.common.signals_two_head import map_positions_from_forward_rf


def run_two_head_inference(
    test_df: pd.DataFrame,
    forward_model_path: Path,
    rf_model_path: Path,
    x: float,
    feature_cols: list[str],
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> np.ndarray:
    """Run two-head inference to generate positions.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data.
    forward_model_path : Path
        Path to forward_returns model bundle.
    rf_model_path : Path
        Path to risk_free_rate model bundle.
    x : float
        Optimized x parameter from grid search.
    feature_cols : list[str]
        Feature columns.
    clip_min : float
        Minimum position.
    clip_max : float
        Maximum position.
        
    Returns
    -------
    np.ndarray
        Position values.
    """
    forward_model = joblib.load(forward_model_path)
    rf_model = joblib.load(rf_model_path)
    
    X_test = test_df[feature_cols].copy()
    
    forward_pred = forward_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    positions = map_positions_from_forward_rf(
        forward_pred, rf_pred,
        x=x,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    
    return positions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Two-Head inference for Hull Competition"
    )
    parser.add_argument(
        "--artifacts-dir", type=str,
        default="artifacts/models/lgbm-two-head",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="data/raw",
    )
    parser.add_argument(
        "--out-csv", type=str, default=None,
    )
    args = parser.parse_args(argv)
    
    art_dir = Path(args.artifacts_dir)
    
    # Load model_meta.json for x parameter
    with open(art_dir / "model_meta.json") as f:
        meta = json.load(f)
    
    x = meta["position_mapping"]["x"]
    clip_min = meta["position_mapping"].get("clip_min", 0.0)
    clip_max = meta["position_mapping"].get("clip_max", 2.0)
    
    print(f"[info] x={x}, clip=[{clip_min}, {clip_max}]")
    
    # Load test data
    test_path = Path(args.data_dir) / "test.csv"
    test_df = pd.read_csv(test_path)
    
    # Get feature columns from meta
    feature_cols = meta.get("feature_columns", [])
    
    # Run inference
    positions = run_two_head_inference(
        test_df,
        forward_model_path=art_dir / "forward_model.pkl",
        rf_model_path=art_dir / "rf_model.pkl",
        x=x,
        feature_cols=feature_cols,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    
    # Build submission
    out_csv = args.out_csv or str(art_dir / "submission.csv")
    submission = pd.DataFrame({
        "date_id": test_df["date_id"],
        "prediction": positions.astype(np.float32),
    })
    
    # Filter to scored rows
    if "is_scored" in test_df.columns:
        scored_mask = test_df["is_scored"].astype(bool)
        submission = submission[scored_mask]
    
    submission.to_csv(out_csv, index=False)
    print(f"[ok] Saved {out_csv} ({len(submission)} rows)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

## 7. 実装手順

### Phase 1: 基盤構築 (Day 1)

| # | タスク | 詳細 |
|---|--------|------|
| 1 | `signals_two_head.py` 作成 | `TwoHeadPositionConfig`, `map_positions_from_forward_rf` |
| 2 | `__init__.py` 作成 | `src/metrics/lgbm_two_head/__init__.py` |
| 3 | Config 作成 | `configs/evaluation/two_head.yaml` |
| 4 | ユニットテスト | `test_signals_two_head.py` 作成 |

### Phase 2: トレーニング (Day 2)

| # | タスク | 詳細 |
|---|--------|------|
| 5 | `train_lgbm_two_head.py` 作成 | `train_lgbm.py` をコピー＆改修 |
| 6 | Walk-Forward CV 実行 | 4-fold で OOF 生成 |
| 7 | x 最適化 | `tune_x_parameter.py` で Grid search |

### Phase 3: 評価・提出 (Day 3)

| # | タスク | 詳細 |
|---|--------|------|
| 8 | `predict_lgbm_two_head.py` 作成 | 2ヘッド推論スクリプト |
| 9 | 1 ヘッドとの比較 | Hull Sharpe, vol_ratio を比較 |
| 10 | Submission 生成 | テストデータで推論 |
| 11 | ドキュメント更新 | 結果を `submissions.md` に追記 |

---

## 8. 評価基準

### 8.1 成功条件

| 指標 | 閾値 | 説明 |
|------|------|------|
| Hull Sharpe (mean) | ≥ 1 ヘッド | 既存手法と同等以上 |
| Hull Sharpe (min) | > -0.2 | 最悪 fold でも大きな損失なし |
| Vol Ratio | < 1.2 | ペナルティ回避 |
| vs Do-nothing | > 0 | ベースラインを上回る |

### 8.2 比較実験

| 設定 | 説明 |
|------|------|
| 1 ヘッド (baseline) | `α=0.05, β=1.0` |
| 2 ヘッド (proposed) | `x` 最適化 |
| Do-nothing | `β=0.806, α=0` |

---

## 9. リスク管理

| リスク | 対策 |
|--------|------|
| 分母ゼロ | `epsilon=1e-8` でガード |
| x スケール不適合 | `forward_returns` の統計を確認してグリッド調整 |
| 過学習 | Walk-Forward CV で検証 |
| モデル相関 | 特徴量サブセット or 正則化強化 |

---

## 10. 参照

- [概要書](two_head_learning.md)
- [Kaggle Discussion 608349](https://www.kaggle.com/competitions/hull-tactical-asset-allocation-challenge/discussion/608349)
- [Kaggle Discussion 611071](https://www.kaggle.com/competitions/hull-tactical-asset-allocation-challenge/discussion/611071)
- [既存 signals.py](../../src/models/common/signals.py) - 変更なし（参考のみ）
- [既存 train_lgbm.py](../../src/metrics/lgbm/train_lgbm.py) - コピー元（変更なし）
- [既存 predict_lgbm.py](../../src/metrics/lgbm/predict_lgbm.py) - コピー元（変更なし）
