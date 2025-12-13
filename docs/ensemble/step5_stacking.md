# Step 5: Stacking（メタ学習）

最終更新: 2025-12-13

## 概要

LGBM、XGBoost、CatBoostのOOF予測を入力として、メタモデル（Ridge）で最適な組み合わせを学習。
Step 3 が有効だった場合のみ実行。

## 実装ステータス

**Status**: ❌ 不採用（過学習確認）

## OOF評価結果（2025-12-13実施）

| 指標 | 値 |
|------|-----|
| OOF RMSE | 0.010998 |
| vs LGBM | -9.59% |
| 予測Std | ⚠️ **0.000151** |

### メタモデル係数

```
Intercept: 0.000069
lgbm:      0.010143
xgboost:   0.007620
catboost:  0.001520
```

**→ 予測Std = 0.000151 は過学習の明確な兆候。不採用。**

## 不採用理由

1. **予測Stdが極端に小さい**: 0.000151（LGBM比 2.9%）
2. **予測がほぼ定数**: メタモデルが平均化しすぎ
3. **過学習リスク大**: TimeSeriesSplitでも改善せず

---

## 背景・根拠（参考）

### Stackingのメリット

1. **自動重み学習**: 手動チューニング不要
2. **非線形組み合わせ**: 単純平均より柔軟
3. **CVベースで過学習抑制**: OOFを使用することでリークを防止

### 手動アンサンブルとの比較

| 観点 | 手動（Step 1-4） | Stacking |
|------|-----------------|----------|
| 重み決定 | 経験則・グリッドサーチ | データから学習 |
| 柔軟性 | 線形のみ | メタモデル次第 |
| 過学習リスク | 低い | 中程度 |
| 解釈性 | 高い | 中程度 |

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

### アーキテクチャ

```
Level 0 (Base Models)
┌─────────┬──────────┬──────────┐
│ LightGBM│ XGBoost  │ CatBoost │
└────┬────┴────┬─────┴────┬─────┘
     │         │          │
     ▼         ▼          ▼
   OOF予測   OOF予測    OOF予測
     │         │          │
     └────┬────┴────┬─────┘
          │         │
          ▼         ▼
Level 1 (Meta Model)
┌─────────────────────────────┐
│      Ridge Regression       │
│   (alpha=1.0, normalize)    │
└──────────────┬──────────────┘
               │
               ▼
         最終予測
```

### メタモデル学習

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# 1. OOF予測を読み込み
lgbm_oof = pd.read_csv("artifacts/models/lgbm/oof_predictions.csv")
xgb_oof = pd.read_csv("artifacts/models/xgboost/oof_predictions.csv")
cat_oof = pd.read_csv("artifacts/models/catboost/oof_predictions.csv")

# 2. スタッキング用特徴量を作成
X_stack = pd.DataFrame({
    "lgbm": lgbm_oof["prediction"],
    "xgboost": xgb_oof["prediction"],
    "catboost": cat_oof["prediction"]
})
y = lgbm_oof["actual"]

# 3. メタモデル学習（時系列CV）
tscv = TimeSeriesSplit(n_splits=5)
meta_oof_pred = np.zeros(len(y))

for train_idx, val_idx in tscv.split(X_stack):
    X_train, X_val = X_stack.iloc[train_idx], X_stack.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_train, y_train)
    
    meta_oof_pred[val_idx] = meta_model.predict(X_val)

# 4. 最終モデル学習
final_meta_model = Ridge(alpha=1.0)
final_meta_model.fit(X_stack, y)

# 5. RMSE計算
oof_rmse = np.sqrt(mean_squared_error(y, meta_oof_pred))
print(f"Step 5 OOF RMSE: {oof_rmse:.6f}")

# 6. 学習された重み確認
print(f"Meta weights: {dict(zip(X_stack.columns, final_meta_model.coef_))}")
```

### メタモデル候補

| モデル | 特徴 | 推奨 |
|--------|------|------|
| Ridge | 正則化あり、係数が安定 | ✅ |
| LinearRegression | 正則化なし、過学習リスク | ⚠️ |
| ElasticNet | L1+L2正則化、スパース解 | 🔬 |
| XGBoost (shallow) | 非線形、過学習リスク高 | ❌ |

---

## 出力

### 成果物

| ファイル | 内容 |
|----------|------|
| `artifacts/ensemble/step5_stacking/oof_predictions.csv` | メタモデルのOOF予測 |
| `artifacts/ensemble/step5_stacking/submission.csv` | Kaggle提出用 |
| `artifacts/ensemble/step5_stacking/metrics.json` | 評価指標 |
| `artifacts/ensemble/step5_stacking/meta_model.pkl` | 学習済みメタモデル |
| `artifacts/ensemble/step5_stacking/meta_weights.json` | 学習された重み |

### metrics.json形式

```json
{
  "method": "stacking",
  "meta_model": "Ridge",
  "meta_alpha": 1.0,
  "base_models": ["lgbm", "xgboost", "catboost"],
  "oof_rmse": 0.01175,
  "oof_rmse_vs_lgbm": -0.035,
  "oof_rmse_vs_step3": -0.010,
  "meta_weights": {
    "lgbm": 0.55,
    "xgboost": 0.38,
    "catboost": 0.07
  }
}
```

### meta_weights.json形式

```json
{
  "intercept": 0.0012,
  "coefficients": {
    "lgbm": 0.55,
    "xgboost": 0.38,
    "catboost": 0.07
  },
  "interpretation": "LGBMが最重要、CatBoostは微量貢献"
}
```

---

## 成功基準

### OOF基準

| 指標 | 基準 | 根拠 |
|------|------|------|
| OOF RMSE | < Step 3/4 のベスト | メタ学習で改善 |
| meta_weight 合計 | ≈ 1.0 | 正規化されている |

### LB基準（OOF改善時のみ検証）

| 指標 | 基準 | 根拠 |
|------|------|------|
| LB Score | > Step 3/4 のベスト | Stackingが有効 |

### 過学習チェック

| 指標 | 警告基準 |
|------|----------|
| Train RMSE vs OOF RMSE | 差が 10% 以上 |
| CV fold間のRMSE分散 | 分散が大きすぎる |

---

## 実行手順

### ローカル評価

```bash
python -m src.ensemble.stacking \
    --config configs/ensemble/step5_stacking.yaml \
    --out-dir artifacts/ensemble/step5_stacking
```

### メタモデルチューニング（オプション）

```bash
python -m src.ensemble.stacking_tune \
    --alpha-range 0.01 10 \
    --out-dir artifacts/ensemble/step5_stacking
```

---

## Kaggle NB実装

```python
import pickle
import numpy as np
import pandas as pd

# メタモデルと各ベースモデルを読み込み
with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

def predict(test: pd.DataFrame) -> float:
    features = prepare_features(test)
    
    # Level 0: 各ベースモデルで予測
    lgbm_pred = lgbm_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]
    cat_pred = catboost_model.predict(features)[0]
    
    # Level 1: メタモデルで統合
    stack_features = np.array([[lgbm_pred, xgb_pred, cat_pred]])
    ensemble_pred = meta_model.predict(stack_features)[0]
    
    # シグナル変換
    signal = np.clip(ensemble_pred * 1.0 + 1.0, 0.9, 1.1)
    return float(signal)
```

### 必要なartifacts

| ファイル | 内容 |
|----------|------|
| `lgbm_model.pkl` | LightGBMモデル |
| `xgb_model.pkl` | XGBoostモデル |
| `catboost_model.pkl` | CatBoostモデル |
| `meta_model.pkl` | Ridge メタモデル |

---

## リスクと対策

### リスク1: メタモデルの過学習

**症状**: Train RMSEは良いがOOF RMSEが悪い
**対策**: 
- Ridge の alpha を大きくする
- TimeSeriesSplit を使用（リークを防止）
- CV数を増やす

### リスク2: 負の係数

**症状**: メタモデルが負の係数を学習
**対策**:
- `positive=True` オプション（sklearn 0.24+）
- 係数が負の場合は手動でクリップ

### リスク3: CatBoostへの過度な依存

**症状**: CatBoostの係数が異常に大きい
**対策**:
- スケーリング（StandardScaler）を適用
- CatBoostを除外して再学習

```python
# 正の係数のみを強制（過学習対策）
from sklearn.linear_model import Ridge

class PositiveRidge(Ridge):
    def fit(self, X, y):
        super().fit(X, y)
        self.coef_ = np.maximum(self.coef_, 0)  # 負の係数をゼロにクリップ
        return self
```

---

## Step 3/4 との比較

| 観点 | Step 3（手動重み） | Step 4（Rank Avg） | Step 5（Stacking） |
|------|-------------------|-------------------|-------------------|
| 重み決定 | 手動 | 等重み | データ駆動 |
| 柔軟性 | 低い | 中程度 | 高い |
| 過学習リスク | 最低 | 低い | 中程度 |
| 実装複雑度 | 低い | 中程度 | 高い |
| 解釈性 | 高い | 高い | 中程度 |

---

## アンサンブル選択の最終判断

Step 1〜5 の結果をまとめて、最終的に採用するアンサンブル手法を決定:

```
Step 1: LGBM + XGB 50:50
├─ LB改善 → Step 2, 3 を検証
└─ LB悪化 → LGBMソロ確定

Step 2: Rank Average
├─ Step 1 より LB改善 → Step 2 採用候補
└─ Step 1 より LB悪化 → Step 1 優先

Step 3: +CatBoost 60:30:10
├─ LB改善 → Step 4, 5 を検証
└─ LB悪化 → CatBoost不採用、Step 1/2 のベスト採用

Step 4: 3-Model Rank Average
├─ Step 3 より LB改善 → Step 4 採用候補
└─ Step 3 より LB悪化 → Step 3 優先

Step 5: Stacking
├─ 全Stepより LB改善 → Step 5 採用
└─ 他Stepより LB悪化 → 他のベスト採用
```

---

## 参考リンク

- [アンサンブル概要](README.md)
- [Step 3: 3-Model 重み付き平均](step3_lgbm_xgb_cat.md)
- [Step 4: 3-Model Rank Average](step4_3model_rank.md)
- [sklearn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
