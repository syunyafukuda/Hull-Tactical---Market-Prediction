# Step 1: LGBM + XGBoost 50:50 単純平均

最終更新: 2025-12-13

## 概要

LightGBMとXGBoostの予測を50:50で単純平均し、アンサンブル効果を検証する。

## 実装ステータス

**Status**: ✅ OOF評価完了 → LB検証待ち

## OOF評価結果（2025-12-13実施）

| 指標 | 値 |
|------|-----|
| **OOF RMSE** | **0.011932** |
| vs LGBM | **-1.91%** |
| 予測Std | 0.004700 |
| 共通サンプル数 | 7,490 |

## 背景・根拠

### OOFでの既存分析結果

| 構成 | OOF RMSE | LGBM比 |
|------|----------|--------|
| LGBM単体 | 0.012164 | - |
| XGBoost単体 | 0.012062 | -0.84% |
| **50% LGBM + 50% XGB** | **0.011932** | **-1.91%** |

### モデル間の相関

- LGBM vs XGBoost 予測相関: **0.684**
- 相関が低いほどアンサンブル効果が高い

---

## 入力

### 使用するartifacts

| モデル | ファイル | 用途 |
|--------|----------|------|
| LGBM | `artifacts/models/lgbm/oof_predictions.csv` | OOF評価用 |
| LGBM | `artifacts/models/lgbm/inference_bundle.pkl` | テスト予測用（※） |
| XGBoost | `artifacts/models/xgboost/oof_predictions.csv` | OOF評価用 |
| XGBoost | `artifacts/models/xgboost/inference_bundle.pkl` | テスト予測用 |

> ※ LGBMのinference_bundleがない場合は、submission.csvから直接読み込む

### OOF予測ファイルの形式

```csv
index,actual,prediction
0,-0.0030384793599786,0.00123...
1,-0.0091140456193133,0.00456...
```

---

## 処理ロジック

### アルゴリズム

```python
import pandas as pd
import numpy as np

# 1. OOF予測を読み込み
lgbm_oof = pd.read_csv("artifacts/models/lgbm/oof_predictions.csv")
xgb_oof = pd.read_csv("artifacts/models/xgboost/oof_predictions.csv")

# 2. 単純平均（50:50）
ensemble_pred = (lgbm_oof["prediction"] + xgb_oof["prediction"]) / 2

# 3. RMSE計算
from sklearn.metrics import mean_squared_error
oof_rmse = np.sqrt(mean_squared_error(lgbm_oof["actual"], ensemble_pred))

# 4. テスト予測も同様に平均
# lgbm_test + xgb_test の平均
```

### シグナル変換

Kaggle提出用にシグナル形式に変換:

```python
# pred: excess returns → signal: [0.9, 1.1]
signal_pred = np.clip(ensemble_pred * 1.0 + 1.0, 0.9, 1.1)
```

---

## 出力

### 成果物

| ファイル | 内容 |
|----------|------|
| `artifacts/ensemble/step1_lgbm_xgb_avg/oof_predictions.csv` | アンサンブルOOF予測 |
| `artifacts/ensemble/step1_lgbm_xgb_avg/submission.csv` | Kaggle提出用 |
| `artifacts/ensemble/step1_lgbm_xgb_avg/metrics.json` | 評価指標 |

### oof_predictions.csv形式

```csv
index,actual,lgbm_pred,xgb_pred,ensemble_pred
0,-0.003038,0.00123,0.00145,0.00134
```

### metrics.json形式

```json
{
  "method": "weighted_average",
  "weights": {"lgbm": 0.5, "xgboost": 0.5},
  "oof_rmse": 0.011932,
  "oof_rmse_vs_lgbm": -0.0191,
  "lgbm_xgb_correlation": 0.684
}
```

---

## 成功基準

| 指標 | 基準 | 根拠 |
|------|------|------|
| OOF RMSE | < 0.012164 | LGBM単体より改善 |
| **LB Score** | **> 0.681** | LGBMベースライン超え |

---

## 実行手順

### ローカル評価

```bash
python -m src.ensemble.blend \
    --config configs/ensemble/step1_lgbm_xgb_avg.yaml \
    --out-dir artifacts/ensemble/step1_lgbm_xgb_avg
```

### Kaggle提出

1. `notebooks/submit/ensemble_step1.ipynb` を作成
2. LGBM/XGBoostのinference_bundleをDatasetとしてアップロード
3. NBで予測を平均してsubmit

---

## Kaggle NB実装イメージ

```python
import pickle
import numpy as np
import pandas as pd

# モデル読み込み
with open("/kaggle/input/lgbm-bundle/inference_bundle.pkl", "rb") as f:
    lgbm_bundle = pickle.load(f)
with open("/kaggle/input/xgb-bundle/inference_bundle.pkl", "rb") as f:
    xgb_bundle = pickle.load(f)

def predict(test: pd.DataFrame) -> float:
    # 特徴量生成
    features = prepare_features(test)
    
    # 各モデルで予測
    lgbm_pred = lgbm_bundle["models"][0].predict(features)
    xgb_pred = xgb_bundle["models"][0].predict(features)
    
    # 50:50平均
    ensemble_pred = (lgbm_pred + xgb_pred) / 2
    
    # シグナル変換
    signal = np.clip(ensemble_pred * 1.0 + 1.0, 0.9, 1.1)
    return float(signal[0])
```

---

## 次のステップ

- LB提出後、結果を `docs/submissions.md` に記録
- LB > 0.681 なら Step 2 へ進む
- LB ≤ 0.681 なら原因分析（OOF↔LB乖離の可能性）

---

## 参考リンク

- [アンサンブル概要](README.md)
- [LGBM実装](../models/lgbm.md)
- [XGBoost実装](../models/xgboost.md)
