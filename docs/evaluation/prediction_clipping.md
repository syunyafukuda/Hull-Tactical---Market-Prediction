# Prediction → Position Clipping Strategy

最終更新: 2025-12-14  
作成者: Codex CLI

本ドキュメントは、`pred_excess → positions` 変換（クリッピング／スケーリング）を RMSE 学習ラインに組み込む際の実装ガイドです。GitHub Copilot 実装者が外部 URL を閲覧できない前提で、Kaggle ディスカッションの要点と実装手順をすべてここに記載します。

---

## 1. 背景とモチベーション

- Hull Tactical コンペでは、**`excess_returns` をどう予測するか**に加えて、**予測値を `[0, 2]` のポジションに写像するルール**が Sharpe（LB）に直結する。
- 「The REAL "do-nothing" baseline is 0.469（Kaggle discussion/611071）」では、Public LB のピークが 0.469 付近にある理由として以下が説明されている:

  ```
  def map_positions(pred_excess: np.ndarray, alpha: float, beta: float) -> np.ndarray:
      return np.clip(beta + alpha * pred_excess, MIN_INVESTMENT, MAX_INVESTMENT)
  ```

  - `pred_excess` がノイズなら `alpha=0` が最良（=予測を無視）
  - `beta` を 0〜2 の範囲でスキャンすると `beta=0.806` で LB≈0.469（= do-nothing 基準）
  - わずかでも有効な予測があれば「0.5〜0.6+ が cutting-edge」

- つまり **RMSE を改善する前に map_positions をチューニング**するだけで LB を押し上げられる可能性が高い。

---

## 2. 実装タスク概要

| # | 項目 | 内容 |
|---|------|------|
| 1 | 変換関数の共通化 | `src/models/common/signals.py` を新設し、`map_predictions_to_positions(pred, alpha, beta, clip=True, clip_min=0.0, clip_max=2.0)` を定義。 |
| 2 | 設定項目の追加 | `configs/evaluation/walk_forward.yaml` など共通設定に `position_mapping` セクション（`alpha`, `beta`, `clip_min`, `clip_max`, `winsor_pct` 等）を追加。 |
| 3 | 推論パイプラインへの組み込み | `src/models/lgbm/train_lgbm.py` の OOF / 推論部で、`y_pred` をそのまま保存するルートとは別に **ポジション列**を生成して artifact 出力する。 |
| 4 | スキャン用 CLI | `scripts/tune_position_mapping.py`（仮）で α/β グリッドを回して擬似 LB（train[-180:]）の Sharpe を算出。`hull_sharpe_score` を利用。 |
| 5 | 設定管理 | クリッピング設定を YAML 1 箇所に集約し、RMSE 学習時は `alpha != 0`、do-nothing 比較時は `alpha=0` 等を切り替えやすくする。 |
| 6 | ログ/可視化 | `artifacts/models/<run>/position_mapping.json` に α/β/clip/winsor の実際の値と Sharpe/vol_ratio を記録。 |

---

## 3. 変換ロジック詳細

```python
from __future__ import annotations
import numpy as np

def map_predictions_to_positions(
    pred_excess: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
    winsor_pct: float | None = None,
) -> np.ndarray:
    """Affine transform + clipping for predicted excess returns.

    position = beta + alpha * pred_excess
    optional winsorization => clip_min/max
    """
    pred = np.asarray(pred_excess, dtype=float).copy()
    if winsor_pct:
        lower = np.quantile(pred, winsor_pct)
        upper = np.quantile(pred, 1 - winsor_pct)
        pred = np.clip(pred, lower, upper)
    position = beta + alpha * pred
    return np.clip(position, clip_min, clip_max)
```

**推奨初期値**:
- `alpha = 0.25`（LGBM の振幅を抑えボラティリティを減らす）
- `beta = 1.0`（市場中立）
- `clip_min = 0.4`, `clip_max = 1.6`（極端なレバレッジを排除）
- `winsor_pct = 0.01`（予測の上下 1% を丸める）

`alpha=0, beta≈0.8` で do-nothing を再現可能。上記パラメータを `walk_forward.yaml` などに記述し、CLI オプション `--position-config` で上書きできるようにする。

---

## 4. LGBM トレーニングへの具体的な埋め込み

1. **OOF/テストの後処理** (`train_lgbm.py` 内):
   ```python
   from src.models.common.signals import map_predictions_to_positions

   position_cfg = load_position_config(args.position_config)
   oof_positions = map_predictions_to_positions(
       oof_predictions,
       alpha=position_cfg["alpha"],
       beta=position_cfg["beta"],
       clip_min=position_cfg["clip_min"],
       clip_max=position_cfg["clip_max"],
       winsor_pct=position_cfg.get("winsor_pct"),
   )
   ```
   - `oof_positions` を Sharpe 評価用に保存 (`artifacts/.../oof_positions.csv`)。
   - 推論フェーズでも同じ関数を使用し、`submission.csv` の `prediction` に直接反映。

2. **Sharpe 評価**:
   - `hull_sharpe_score(prediction=oof_positions, forward_returns, risk_free_rate)` を記録。
   - クリッピング前後の `alpha`, `beta`, `winsor_pct` を `hull_sharpe_summary.json` に追記。

3. **Config 例 (`configs/evaluation/walk_forward.yaml`)**:
   ```yaml
   position_mapping:
     alpha: 0.25
     beta: 1.0
     clip_min: 0.4
     clip_max: 1.6
     winsor_pct: 0.01
   ```

---

## 5. α/β チューニングのサンプルフロー

1. `train.tail(180)` を使用して擬似 LB を作成。
2. α ∈ {0, 0.1, 0.25, 0.5}, β ∈ {0.6, 0.8, 1.0, 1.2} をグリッドサーチ。
3. 各組み合わせで
   - OOF RMSE（参考）
   - 擬似 Sharpe（`hull_sharpe_score`）
   - vol_ratio / penalty
   を CSV (`results/position_sweep/alpha_beta_search.csv`) に出力。
4. best α/β を `configs/evaluation/walk_forward.yaml` に反映。

---

## 6. 実装チェックリスト

- [ ] `src/models/common/signals.py` 追加（単体テストあり）
- [ ] `configs/evaluation/walk_forward.yaml` に `position_mapping` 追加
- [ ] `train_lgbm.py` でクリッピング処理を呼び出し、OOF/推論両方に適用
- [ ] `artifacts/.../position_mapping.json` に設定値＋Sharpeを保存
- [ ] α/β スキャンスクリプト（任意）を `scripts/` に追加
- [ ] README または `docs/ensemble/README.md` に、ポジション写像の重要性と設定方法を追記

---

## 7. 参考：ディスカッション引用（全文をここに転記）

> **Title: The REAL "do-nothing" baseline is 0.469 (Kaggle discussion/611071)**  
> 公開リーダーボードのヒストグラムには 3 つの極大が存在する。  
> 1. ~17.5: `train[-180:]` を最適化したリーク解  
> 2. ~10: 別のリーク系 NB  
> 3. ~0.5 (0.469 がピーク): リークなしの「do-nothing」戦略  
>  
> `excess_returns` の予測がノイズなら `alpha=0` とし、`beta` を 0〜2 でスキャンすると 0.806 で 0.4688。  
> わずかでも予測精度があれば 0.5〜0.6+ が最前線。  
> コード断片（原文そのまま）:
> ```python
> def map_positions(
>     pred_excess: np.ndarray,
>     alpha: float,
>     beta: float
> ) -> np.ndarray:
> 
>     return np.clip(beta + alpha * pred_excess, MIN_INVESTMENT, MAX_INVESTMENT)
> ```

> **Title: Public leaderboard: 17 is possible (Kaggle discussion/608349)**  
> `train[-180:]` の情報を使って `scipy.optimize.minimize` で `prediction` を直接最適化すると 17.39 台が出る。  
> 実際の未見データでは意味がないが、**「最適ポジションは `[0,2]` に clip された allocation」** である点が強調されている。  
> また、`positions = ((x - risk_free_rates) / (forward_returns - risk_free_rates)).clamp(0.0, 2.0)` の 1 次元最適化も議論されている。

これらの記述に基づき、ポジション変換を調整するバージョン管理を進めてください。

