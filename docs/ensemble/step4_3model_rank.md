# Step 4: 3-Model Rank Average

最終更新: 2025-12-13

## 概要

LGBM、XGBoost、CatBoostの3モデルをRank Averageでアンサンブル。
Step 3（重み付き平均）が有効だった場合のみ実行。

## 実装ステータス

**Status**: ⚠️ OOF評価完了 → 予測Std低下のため要注意

## OOF評価結果（2025-12-13実施）

| 指標 | 値 |
|------|-----|
| **OOF RMSE** | **0.011457** |
| vs LGBM | **-5.82%** |
| vs Step 3b | **-2.08%** |
| 予測Std | ⚠️ **0.003120** |

**→ OOF最良だが予測Stdが低下（LGBM比 59%）、慎重に検討**

## 前提条件

**このステップに進む条件:**
- Step 3 で CatBoost追加が有効であることが確認済み
  - OOF RMSE: Step 1/2 より改善
  - LB Score: Step 1/2 より改善

---

## 背景・根拠

### Rank Averageの利点（3モデル版）

1. **スケール差の吸収**: CatBoostの極端に狭い予測レンジを正規化
2. **外れ値の影響軽減**: 異常予測の影響を順位で抑制
3. **等重み化**: 各モデルが同等に貢献

### CatBoostでの期待効果

| 課題 | Rank Averageでの解決 |
|------|---------------------|
| 予測Std 0.000495 と極端に小さい | 順位変換で 0〜1 に正規化される |
| 予測レンジが 0.999〜1.001 と狭い | 他モデルと同スケールで比較可能 |
| 重み調整が必要 | 自動的に等重みで寄与 |

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

# 2. パーセンタイル順位に変換
lgbm_rank = lgbm_oof["prediction"].rank(pct=True)
xgb_rank = xgb_oof["prediction"].rank(pct=True)
cat_rank = cat_oof["prediction"].rank(pct=True)

# 3. 順位の平均
ensemble_rank = (lgbm_rank + xgb_rank + cat_rank) / 3

# 4. 実スケールに復元（パーセンタイルマッピング）
# 参照分布としてLGBMを使用
lgbm_sorted = lgbm_oof["prediction"].sort_values().values
ensemble_pred = np.interp(ensemble_rank, np.linspace(0, 1, len(lgbm_sorted)), lgbm_sorted)

# 5. RMSE計算
oof_rmse = np.sqrt(mean_squared_error(lgbm_oof["actual"], ensemble_pred))
print(f"Step 4 OOF RMSE: {oof_rmse:.6f}")
```

### スケール復元オプション

| 方法 | 説明 | 推奨 |
|------|------|------|
| LGBMパーセンタイル | LGBMの分布にマッピング | ✅ |
| 3モデル平均パーセンタイル | 3モデルの平均分布にマッピング | ⚠️ |
| アンサンブル用分布 | OOFで学習した分布 | 🔬 |

---

## 出力

### 成果物

| ファイル | 内容 |
|----------|------|
| `artifacts/ensemble/step4_3model_rank/oof_predictions.csv` | OOF予測 |
| `artifacts/ensemble/step4_3model_rank/submission.csv` | Kaggle提出用 |
| `artifacts/ensemble/step4_3model_rank/metrics.json` | 評価指標 |
| `artifacts/ensemble/step4_3model_rank/reference_distribution.npy` | スケール復元用分布 |

### metrics.json形式

```json
{
  "method": "rank_average_3model",
  "models": ["lgbm", "xgboost", "catboost"],
  "scale_restore": "lgbm_percentile",
  "oof_rmse": 0.01182,
  "oof_rmse_vs_lgbm": -0.029,
  "oof_rmse_vs_step3": -0.003,
  "prediction_std": 0.00510
}
```

---

## 成功基準

### OOF基準

| 指標 | 基準 | 根拠 |
|------|------|------|
| OOF RMSE | < Step 3 | Rank化で更に改善 |
| 予測Std | ≈ LGBM のStd | スケール復元が適切 |

### LB基準（OOF改善時のみ検証）

| 指標 | 基準 | 根拠 |
|------|------|------|
| LB Score | > Step 3 | Rank Average が有効 |

---

## 実行手順

### ローカル評価

```bash
python -m src.ensemble.rank_blend \
    --config configs/ensemble/step4_3model_rank.yaml \
    --out-dir artifacts/ensemble/step4_3model_rank
```

---

## Kaggle NB実装

### 課題

Rank Averageを推論時に適用するには、**参照分布**が必要。

### 実装方法

```python
import pickle
import numpy as np
import pandas as pd

# 事前にOOFから計算した参照分布を保存
# artifacts/ensemble/step4_3model_rank/reference_distribution.npy
reference_dist = np.load("reference_distribution.npy")

# 推論時のRankを近似する方法
class RankAverageEnsemble:
    def __init__(self, lgbm_model, xgb_model, cat_model, reference_dist):
        self.lgbm = lgbm_model
        self.xgb = xgb_model
        self.cat = cat_model
        self.ref_dist = reference_dist
        
        # 各モデルのOOF予測の累積分布を事前計算
        self.lgbm_cdf = self._build_cdf(lgbm_oof_preds)
        self.xgb_cdf = self._build_cdf(xgb_oof_preds)
        self.cat_cdf = self._build_cdf(cat_oof_preds)
    
    def _build_cdf(self, values):
        sorted_vals = np.sort(values)
        return sorted_vals
    
    def _get_percentile(self, value, cdf):
        """値をパーセンタイルに変換"""
        idx = np.searchsorted(cdf, value)
        return idx / len(cdf)
    
    def predict(self, features):
        # 各モデルの予測
        lgbm_pred = self.lgbm.predict(features)[0]
        xgb_pred = self.xgb.predict(features)[0]
        cat_pred = self.cat.predict(features)[0]
        
        # パーセンタイルに変換
        lgbm_pct = self._get_percentile(lgbm_pred, self.lgbm_cdf)
        xgb_pct = self._get_percentile(xgb_pred, self.xgb_cdf)
        cat_pct = self._get_percentile(cat_pred, self.cat_cdf)
        
        # 平均パーセンタイル
        avg_pct = (lgbm_pct + xgb_pct + cat_pct) / 3
        
        # 実スケールに復元
        idx = int(avg_pct * len(self.ref_dist))
        idx = min(max(idx, 0), len(self.ref_dist) - 1)
        return self.ref_dist[idx]
```

### 必要なartifacts

| ファイル | 内容 |
|----------|------|
| `lgbm_model.pkl` | LightGBMモデル |
| `xgb_model.pkl` | XGBoostモデル |
| `catboost_model.pkl` | CatBoostモデル |
| `lgbm_cdf.npy` | LGBM OOF予測のソート済み配列 |
| `xgb_cdf.npy` | XGBoost OOF予測のソート済み配列 |
| `cat_cdf.npy` | CatBoost OOF予測のソート済み配列 |
| `reference_distribution.npy` | スケール復元用参照分布 |

---

## Step 3 vs Step 4 比較

| 観点 | Step 3（重み付き平均） | Step 4（Rank Average） |
|------|----------------------|----------------------|
| 実装複雑度 | 低い | 高い |
| CatBoost対応 | 重みで調整 | 自動正規化 |
| 推論時の要件 | なし | 参照分布が必要 |
| スケール | 保持 | 復元が必要 |

---

## リスクと対策

### リスク1: CatBoostの順位情報が無意味

**症状**: CatBoostの予測がほぼ定数のため、順位変換しても情報量がない
**対策**: Step 3 の結果を見て、CatBoostの有効性を事前確認

### リスク2: 参照分布の不適切さ

**症状**: スケール復元後の予測が不自然
**対策**: 複数の参照分布オプションを比較

---

## 参考リンク

- [アンサンブル概要](README.md)
- [Step 2: 2-Model Rank Average](step2_lgbm_xgb_rank.md)
- [Step 3: 3-Model 重み付き平均](step3_lgbm_xgb_cat.md)
