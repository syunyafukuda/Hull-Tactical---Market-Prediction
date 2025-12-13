# Step 2: LGBM + XGBoost Rank Average

最終更新: 2025-12-13

## 概要

LightGBMとXGBoostの予測値を**ランク変換**してから平均する。
スケール差や外れ値の影響を吸収し、より安定したアンサンブルを目指す。

## 実装ステータス

**Status**: ✅ OOF評価完了 → LB検証待ち

## OOF評価結果（2025-12-13実施）

| 指標 | 値 |
|------|-----|
| **OOF RMSE** | **0.011876** |
| vs LGBM | **-2.36%** |
| vs Step 1 | **-0.47%** |
| 予測Std | 0.004507 |

**→ Step 1より改善！Rank化が有効**

## 背景・根拠

### Rank Averageの利点

1. **スケール差の吸収**: 各モデルの予測値レンジが異なっても公平に統合
2. **外れ値の影響軽減**: 極端な予測値がランク化により緩和される
3. **単調性の保持**: ランク化は順序を保つため、予測の方向性は維持

### 単純平均との違い

| 手法 | メリット | デメリット |
|------|----------|------------|
| 単純平均 | シンプル、解釈容易 | スケール差に敏感 |
| Rank Average | スケール不変、外れ値に強い | 情報量がやや減少 |

---

## 入力

### 使用するartifacts

| モデル | ファイル | 用途 |
|--------|----------|------|
| LGBM | `artifacts/models/lgbm/oof_predictions.csv` | OOF評価用 |
| XGBoost | `artifacts/models/xgboost/oof_predictions.csv` | OOF評価用 |

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

# 2. ランク変換（パーセンタイルランク: 0〜1）
lgbm_rank = lgbm_oof["prediction"].rank(pct=True)
xgb_rank = xgb_oof["prediction"].rank(pct=True)

# 3. ランクの平均
ensemble_rank = (lgbm_rank + xgb_rank) / 2

# 4. 元のスケールに戻す（オプション）
# 方法A: 単純にランクのまま使用
# 方法B: 元の予測値の分布にマッピング
all_preds = pd.concat([lgbm_oof["prediction"], xgb_oof["prediction"]])
ensemble_pred = ensemble_rank.map(
    lambda r: np.percentile(all_preds, r * 100)
)

# 5. RMSE計算
oof_rmse = np.sqrt(mean_squared_error(lgbm_oof["actual"], ensemble_pred))
```

### スケール復元の選択肢

| 方法 | 説明 | 使用場面 |
|------|------|----------|
| **ランクのまま** | 0〜1の値で提出 | 順位のみ重要な評価指標 |
| **パーセンタイルマッピング** | 元の分布にマッピング | RMSEなど絶対値が重要 |
| **線形スケーリング** | min-maxで元のレンジに | シンプルな復元 |

### 推奨: パーセンタイルマッピング

```python
def rank_to_value(rank_series, reference_values):
    """ランクを元の値分布にマッピング"""
    sorted_ref = np.sort(reference_values)
    n = len(sorted_ref)
    indices = (rank_series * (n - 1)).astype(int)
    return sorted_ref[indices]
```

---

## 出力

### 成果物

| ファイル | 内容 |
|----------|------|
| `artifacts/ensemble/step2_lgbm_xgb_rank/oof_predictions.csv` | アンサンブルOOF予測 |
| `artifacts/ensemble/step2_lgbm_xgb_rank/submission.csv` | Kaggle提出用 |
| `artifacts/ensemble/step2_lgbm_xgb_rank/metrics.json` | 評価指標 |

### oof_predictions.csv形式

```csv
index,actual,lgbm_pred,xgb_pred,lgbm_rank,xgb_rank,ensemble_rank,ensemble_pred
0,-0.003038,0.00123,0.00145,0.456,0.478,0.467,0.00134
```

### metrics.json形式

```json
{
  "method": "rank_average",
  "scale_restoration": "percentile_mapping",
  "oof_rmse": 0.01190,
  "oof_rmse_vs_lgbm": -0.022,
  "oof_rmse_vs_step1": -0.003
}
```

---

## 成功基準

| 指標 | 基準 | 根拠 |
|------|------|------|
| OOF RMSE | < 0.011932 | Step 1（単純平均）より改善 |
| **LB Score** | **> Step 1 LB** | 単純平均を超える |

---

## 実行手順

### ローカル評価

```bash
python -m src.ensemble.rank_average \
    --config configs/ensemble/step2_lgbm_xgb_rank.yaml \
    --out-dir artifacts/ensemble/step2_lgbm_xgb_rank
```

### Kaggle提出

1. `notebooks/submit/ensemble_step2.ipynb` を作成
2. テスト予測に対してもランク平均を適用

---

## Kaggle NB実装イメージ

```python
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# 過去の予測分布（参照用）
reference_preds = np.load("/kaggle/input/ensemble-data/reference_preds.npy")

def predict(test: pd.DataFrame) -> float:
    features = prepare_features(test)
    
    # 各モデルで予測
    lgbm_pred = lgbm_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]
    
    # 参照分布でのランク位置を推定
    lgbm_rank = (reference_preds["lgbm"] < lgbm_pred).mean()
    xgb_rank = (reference_preds["xgb"] < xgb_pred).mean()
    
    # ランク平均
    ensemble_rank = (lgbm_rank + xgb_rank) / 2
    
    # パーセンタイルで値に戻す
    ensemble_pred = np.percentile(reference_preds["all"], ensemble_rank * 100)
    
    # シグナル変換
    signal = np.clip(ensemble_pred * 1.0 + 1.0, 0.9, 1.1)
    return float(signal)
```

### 注意点

- テスト時には「参照分布」が必要（OOF予測の分布を保存しておく）
- 1サンプルずつの予測では正確なランク化が困難
- 代替案: 過去の予測分布でのパーセンタイル位置を使用

---

## Step 1との比較ポイント

| 観点 | Step 1 | Step 2 |
|------|--------|--------|
| 計算方法 | 単純平均 | ランク平均 |
| 外れ値影響 | 受けやすい | 抑制される |
| 実装複雑度 | 低 | 中 |
| 解釈性 | 高 | 中 |

---

## 次のステップ

- Step 1とStep 2のLB結果を比較
- より良い方をベースにStep 3（CatBoost追加）へ進む

---

## 参考リンク

- [アンサンブル概要](README.md)
- [Step 1: 単純平均](step1_lgbm_xgb_avg.md)
