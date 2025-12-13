# Step 3: LGBM + XGBoost + CatBoost（重み付き平均）

最終更新: 2025-12-13

## 概要

LightGBM、XGBoost、CatBoostの3モデルを重み付き平均でアンサンブル。
CatBoostは予測Stdが極端に小さいため、重みを調整。

## 実装ステータス

**Status**: ✅ OOF評価完了 → LB検証待ち（Step 1/2確認後）

## OOF評価結果（2025-12-13実施）

| 重みパターン | OOF RMSE | vs LGBM | 予測Std |
|--------------|----------|---------|---------|
| 60:30:10 | 0.011797 | -3.01% | 0.004319 |
| **50:35:15** | **0.011701** | **-3.80%** | 0.004035 |
| 55:40:05 | 0.011859 | -2.51% | 0.004498 |

**→ 50:35:15 が最適！CatBoost 15%で追加効果あり**

### CatBoost相関（低い = 多様性が高い）

- LGBM vs CatBoost: **0.0890**
- XGBoost vs CatBoost: **0.1006**

## 背景・根拠

### モデル特性

| モデル | OOF RMSE | LB Score | 対LGBM相関 | 予測Std | 重み |
|--------|----------|----------|------------|---------|------|
| LGBM | 0.012164 | 0.681 | - | 0.005246 | **50%** |
| XGBoost | 0.012062 | 0.622 | 0.684 | 0.004999 | **35%** |
| CatBoost | 0.011095 | 0.602 | 0.089 | **0.000495** | **15%** |

### 重み設定の根拠

1. **LGBM 50%**: LB最良（0.681）、最も信頼性が高い
2. **XGBoost 35%**: アンサンブル価値が高い（相関0.684）
3. **CatBoost 15%**: 多様性は非常に高い（相関0.089）

### CatBoostのリスク

- 予測Std 0.000495 = LGBMの **9%** しかない
- 予測値がほぼ定数（1.0付近に集中）
- 混ぜすぎると全体が平坦化する恐れ

---

## 入力

### 使用するartifacts

| モデル | ファイル |
|--------|----------|
| LGBM | `artifacts/models/lgbm/oof_predictions.csv` |
| XGBoost | `artifacts/models/xgboost/oof_predictions.csv` |
| CatBoost | `artifacts/models/catboost/oof_predictions.csv` |

---

## 処理ロジック

### アルゴリズム

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 1. OOF予測を読み込み
lgbm_oof = pd.read_csv("artifacts/models/lgbm/oof_predictions.csv")
xgb_oof = pd.read_csv("artifacts/models/xgboost/oof_predictions.csv")
cat_oof = pd.read_csv("artifacts/models/catboost/oof_predictions.csv")

# 2. 重み付き平均（60:30:10）
weights = {"lgbm": 0.6, "xgboost": 0.3, "catboost": 0.1}

ensemble_pred = (
    weights["lgbm"] * lgbm_oof["prediction"] +
    weights["xgboost"] * xgb_oof["prediction"] +
    weights["catboost"] * cat_oof["prediction"]
)

# 3. RMSE計算
oof_rmse = np.sqrt(mean_squared_error(lgbm_oof["actual"], ensemble_pred))

# 4. Step 1/2との比較
print(f"Step 3 OOF RMSE: {oof_rmse:.6f}")
```

### 代替重みパターン（検討用）

| パターン | LGBM | XGB | CatBoost | 目的 |
|----------|------|-----|----------|------|
| **A (推奨)** | 60% | 30% | 10% | CatBoost最小限 |
| B | 50% | 35% | 15% | CatBoost少し多め |
| C | 55% | 40% | 5% | CatBoost極小 |

---

## 出力

### 成果物

| ファイル | 内容 |
|----------|------|
| `artifacts/ensemble/step3_lgbm_xgb_cat/oof_predictions.csv` | OOF予測 |
| `artifacts/ensemble/step3_lgbm_xgb_cat/submission.csv` | Kaggle提出用 |
| `artifacts/ensemble/step3_lgbm_xgb_cat/metrics.json` | 評価指標 |
| `artifacts/ensemble/step3_lgbm_xgb_cat/weight_analysis.csv` | 重みパターン比較 |

### metrics.json形式

```json
{
  "method": "weighted_average",
  "weights": {"lgbm": 0.6, "xgboost": 0.3, "catboost": 0.1},
  "oof_rmse": 0.01185,
  "oof_rmse_vs_lgbm": -0.026,
  "oof_rmse_vs_step1": -0.007,
  "prediction_std": 0.00498
}
```

---

## 成功基準

### OOF基準

| 指標 | 基準 | 根拠 |
|------|------|------|
| OOF RMSE | < Step 1/2のベスト | CatBoost追加で改善 |
| 予測Std | > 0.004 | 平坦化していないこと |

### LB基準（OOF改善時のみ検証）

| 指標 | 基準 | 根拠 |
|------|------|------|
| LB Score | > Step 1/2のベスト | CatBoost追加が有効 |

### 判断フロー

```
OOF RMSE が Step 1/2 より改善？
├─ Yes → LB検証へ
│        └─ LB改善？ → Step 4/5 へ進む
│                  → CatBoost追加は見送り
└─ No → CatBoost追加は見送り、Step 1/2で確定
```

---

## 実行手順

### ローカル評価

```bash
python -m src.ensemble.blend \
    --config configs/ensemble/step3_lgbm_xgb_cat.yaml \
    --out-dir artifacts/ensemble/step3_lgbm_xgb_cat
```

### 重みパターン比較（オプション）

```bash
python -m src.ensemble.weight_search \
    --models lgbm xgboost catboost \
    --out-dir artifacts/ensemble/step3_lgbm_xgb_cat
```

---

## Kaggle NB実装イメージ

```python
import pickle
import numpy as np
import pandas as pd

# 重み設定
WEIGHTS = {"lgbm": 0.6, "xgboost": 0.3, "catboost": 0.1}

def predict(test: pd.DataFrame) -> float:
    features = prepare_features(test)
    
    # 各モデルで予測
    lgbm_pred = lgbm_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]
    cat_pred = catboost_model.predict(features)[0]
    
    # 重み付き平均
    ensemble_pred = (
        WEIGHTS["lgbm"] * lgbm_pred +
        WEIGHTS["xgboost"] * xgb_pred +
        WEIGHTS["catboost"] * cat_pred
    )
    
    # シグナル変換
    signal = np.clip(ensemble_pred * 1.0 + 1.0, 0.9, 1.1)
    return float(signal)
```

---

## リスクと対策

### リスク1: 予測の平坦化

**症状**: アンサンブル後の予測Stdが著しく低下
**対策**: CatBoostの重みを5%以下に下げる or 除外

### リスク2: OOF↔LB乖離

**症状**: OOFでは改善だがLBでは悪化
**対策**: CatBoostを除外してStep 1/2に戻る

---

## Step 4/5への分岐条件

**Step 4/5に進む条件:**
- OOF RMSE が Step 1/2 より改善
- LB Score が Step 1/2 より改善
- 予測Std が 0.004 以上を維持

**Step 4/5をスキップする条件:**
- OOF RMSE が悪化
- LB Score が悪化
- 予測Stdが大幅に低下

---

## 参考リンク

- [アンサンブル概要](README.md)
- [Step 1: 単純平均](step1_lgbm_xgb_avg.md)
- [Step 2: Rank Average](step2_lgbm_xgb_rank.md)
- [CatBoost実装](../models/catboost.md)
