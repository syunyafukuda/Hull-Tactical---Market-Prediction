# Prediction Clipping Implementation Specification

最終更新: 2025-12-14  
ステータス: 実装準備完了

本ドキュメントは `docs/evaluation/prediction_clipping.md` の設計を具体的なコード実装に落とし込んだ詳細仕様書です。

---

## 1. 概要

### 目的

LGBM が出力する `pred_excess`（予測超過リターン）を、Hull Competition が要求する `position ∈ [0, 2]` に変換する。

### 変換式

```
position = clip(beta + alpha * pred_excess, clip_min, clip_max)
```

- **alpha**: 予測のスケーリング係数（0 = 予測無視、1 = 生予測使用）
- **beta**: オフセット（市場中立 = 1.0、do-nothing最適 ≈ 0.806）
- **clip_min/max**: ポジションの上下限（通常 0.0〜2.0）
- **winsor_pct**: Winsorization 比率（外れ値除去）

### Do-nothing ベースライン再現

```
alpha = 0, beta = 0.806 → position = 0.806（固定） → WF Sharpe ≈ 0.469
```

---

## 2. ファイル構成

```
src/models/common/
├── signals.py          # ★ map_predictions_to_positions 追加
└── __init__.py

scripts/
├── tune_position_mapping.py   # ★ 新規: α/β グリッドサーチ CLI

configs/evaluation/
└── walk_forward.yaml          # ★ position_mapping セクション追加

artifacts/models/<run>/
├── position_mapping.json      # 適用した α/β/clip 設定と Sharpe
└── oof_positions.csv          # クリッピング後の OOF ポジション
```

---

## 3. 実装詳細

### 3.1 signals.py への追加

```python
# 追加位置: signals.py 末尾

@dataclass
class AlphaBetaPositionConfig:
    """Alpha-Beta position mapping configuration.
    
    Based on Kaggle discussion/611071 (do-nothing baseline).
    
    position = clip(beta + alpha * pred_excess, clip_min, clip_max)
    """
    alpha: float = 0.25       # Prediction scale (0 = ignore prediction)
    beta: float = 1.0         # Offset (1.0 = market neutral)
    clip_min: float = 0.0     # Min position (100% cash)
    clip_max: float = 2.0     # Max position (200% market)
    winsor_pct: float | None = None  # Winsorization percentile (e.g., 0.01)


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
    
    Parameters
    ----------
    pred_excess : np.ndarray
        Predicted excess returns from LGBM.
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


def map_positions_from_config(
    pred_excess: np.ndarray,
    config: AlphaBetaPositionConfig | None = None,
) -> np.ndarray:
    """Map predictions to positions using config object."""
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
```

### 3.2 walk_forward.yaml への追加

```yaml
# configs/evaluation/walk_forward.yaml

# 既存設定...

position_mapping:
  # Alpha-Beta mapping (Kaggle discussion/611071)
  alpha: 0.25           # Prediction scale (0 = do-nothing)
  beta: 1.0             # Offset (market neutral)
  clip_min: 0.0         # Min position
  clip_max: 2.0         # Max position
  winsor_pct: 0.01      # Winsorize at 1st/99th percentile
  
  # Do-nothing reproduction settings (for comparison)
  # alpha: 0.0
  # beta: 0.806
```

### 3.3 tune_position_mapping.py CLI

```python
#!/usr/bin/env python3
"""Alpha/Beta grid search for position mapping optimization.

Usage:
    python scripts/tune_position_mapping.py \
        --oof-path artifacts/models/lgbm-sharpe-wf-opt/oof_predictions.csv \
        --returns-path data/processed/forward_returns.parquet \
        --output results/position_sweep/alpha_beta_search.csv
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.common.signals import map_predictions_to_positions
from src.metrics.lgbm.hull_sharpe import hull_sharpe_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-path", required=True)
    parser.add_argument("--returns-path", required=True)
    parser.add_argument("--output", default="results/position_sweep/alpha_beta_search.csv")
    parser.add_argument("--alpha-grid", type=float, nargs="+", 
                        default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--beta-grid", type=float, nargs="+",
                        default=[0.6, 0.8, 1.0, 1.2])
    parser.add_argument("--clip-min-grid", type=float, nargs="+",
                        default=[0.0, 0.2, 0.4])
    parser.add_argument("--clip-max-grid", type=float, nargs="+",
                        default=[1.6, 1.8, 2.0])
    args = parser.parse_args()
    
    # Load data
    oof_df = pd.read_csv(args.oof_path)
    returns_df = pd.read_parquet(args.returns_path)
    
    pred_excess = oof_df["prediction"].values
    forward_returns = returns_df["forward_returns"].values[:len(pred_excess)]
    
    results = []
    for alpha, beta, clip_min, clip_max in itertools.product(
        args.alpha_grid, args.beta_grid, args.clip_min_grid, args.clip_max_grid
    ):
        positions = map_predictions_to_positions(
            pred_excess, alpha=alpha, beta=beta,
            clip_min=clip_min, clip_max=clip_max
        )
        
        sharpe = hull_sharpe_score(
            prediction=positions,
            forward_returns=forward_returns
        )
        
        results.append({
            "alpha": alpha,
            "beta": beta,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "sharpe": sharpe,
            "pos_mean": positions.mean(),
            "pos_std": positions.std(),
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    
    print(f"Best config: alpha={results_df.iloc[0]['alpha']}, "
          f"beta={results_df.iloc[0]['beta']}, "
          f"sharpe={results_df.iloc[0]['sharpe']:.4f}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 4. パラメータ探索範囲

| Parameter | Search Grid | Notes |
|-----------|-------------|-------|
| alpha | 0, 0.1, 0.25, 0.5, 1.0 | 0 = do-nothing |
| beta | 0.6, 0.8, 1.0, 1.2 | 0.806 = do-nothing optimal |
| clip_min | 0.0, 0.2, 0.4 | 保守的なレバレッジ下限 |
| clip_max | 1.6, 1.8, 2.0 | 保守的なレバレッジ上限 |
| winsor_pct | None, 0.01, 0.05 | 外れ値除去 |

---

## 5. 評価指標

1. **Primary**: Walk-Forward Mean Sharpe
2. **Secondary**: WF Min Sharpe > 0（全foldで正のSharpe）
3. **Comparison**: Do-nothing baseline (0.469)

### 採用基準

```
WF Mean Sharpe ≥ 0.5 AND WF Min Sharpe > 0
```

---

## 6. 実装チェックリスト

- [ ] `signals.py` に `AlphaBetaPositionConfig` / `map_predictions_to_positions` 追加
- [ ] `walk_forward.yaml` に `position_mapping` セクション追加
- [ ] `scripts/tune_position_mapping.py` 作成
- [ ] Do-nothing 再現テスト: alpha=0, beta=0.806 → Sharpe ≈ 0.469
- [ ] グリッドサーチ実行、最適 α/β を特定
- [ ] 最適設定で WF CV 再実行、Sharpe > 0.5 を確認
- [ ] `position_mapping.json` に設定値と Sharpe を記録

---

## 7. 参照

- [prediction_clipping.md](prediction_clipping.md) - 設計概要
- [optimized_settings.md](optimized_settings.md) - 最適化結果と方針
- Kaggle discussion/611071 - Do-nothing baseline 分析
