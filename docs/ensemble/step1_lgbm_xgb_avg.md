# Step 1: LGBM + XGBoost 50:50 単純平均

最終更新: 2025-12-14

## 概要

LightGBMとXGBoostの予測を50:50で単純平均し、アンサンブル効果を検証する。

## 実装ステータス

**Status**: ❌ **非採用** - LB検証の結果、LGBM単体より大幅に劣化

---

## LB検証結果（2025-12-14実施）

| 指標 | 期待 (OOF) | 実際 (LB) | 判定 |
|------|-----------|----------|------|
| **LGBM 単体** | 0.012164 | **0.681** | baseline |
| **XGBoost 単体** | 0.012062 | **0.622** | -8.7% |
| **50:50 Ensemble** | **0.011932** | **0.615** | **-9.7%** ❌ |

### 判定結果

| 項目 | 内容 |
|------|------|
| **Decision** | ❌ **非採用** |
| **理由** | XGBoost の LB 劣化がアンサンブルに波及 |
| **推奨** | LGBM 単体 (0.681) を継続使用 |

### 原因分析

1. **XGBoost の OOF↔LB 乖離が主因**
   - XGBoost 単体: OOF では LGBM より良い (0.012062 vs 0.012164)
   - XGBoost 単体 LB: 0.622 (LGBM の 0.681 より大幅に劣る)
   - **XGBoost が将来データで過学習している**

2. **アンサンブルが XGBoost の悪影響を受けた**
   - 50:50 混合により、XGBoost の劣化がそのまま反映
   - 期待値: 0.681 × 0.5 + 0.622 × 0.5 ≈ 0.65
   - 実際は 0.615 でさらに悪い → **非線形な劣化効果**

3. **市場レジーム変化への脆弱性**
   - XGBoost のハイパーパラメータ (max_depth=10, min_child_weight=5) が訓練データに過適合
   - テスト期間の市場環境が異なり、XGBoost の予測が外れている

---

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
| LGBM | `artifacts/models/lgbm-artifacts/oof_predictions.csv` | OOF評価用 |
| LGBM | `artifacts/models/lgbm-artifacts/inference_bundle.pkl` | テスト予測用 |
| XGBoost | `artifacts/models/xgboost-artifacts/oof_predictions.csv` | OOF評価用 |
| XGBoost | `artifacts/models/xgboost-artifacts/inference_bundle.pkl` | テスト予測用 |
| XGBoost | `artifacts/models/xgboost-artifacts/xgboost-*.whl` | XGBoost wheel |

> Kaggle用に `lgbm-artifacts` と `xgboost-artifacts` という名前でディレクトリをリネーム済み

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

## Kaggle NB実装方針（確定）

### アプローチ: 2データセットインプット

```
Kaggle NB
├── Input Dataset 1: lgbm-su8-artifacts
│   ├── inference_bundle.pkl (LGBM)
│   ├── feature_list.json
│   ├── model_meta.json
│   └── scikit_learn-*.whl
│
├── Input Dataset 2: xgboost-su8-artifacts
│   ├── inference_bundle.pkl (XGBoost)
│   ├── feature_list.json
│   ├── model_meta.json
│   ├── scikit_learn-*.whl
│   └── xgboost-*.whl
│
└── NB Code:
    # 両方のパイプラインを初期化
    lgbm_pred = lgbm_pipeline.predict(X)
    xgb_pred = xgb_pipeline.predict(X)
    ensemble_pred = (lgbm_pred + xgb_pred) / 2
```

### 選択理由

| 観点 | 評価 |
|------|------|
| 既存artifacts流用 | ✅ そのまま使える |
| 実装の複雑さ | ⚠️ 2つのパイプライン初期化が必要 |
| メモリ使用量 | ⚠️ 2モデル分（約4.5GB） |
| 検証速度 | ✅ すぐにLB検証可能 |

### 必要なartifacts（2025-12-13確認済み）

| モデル | 必要ファイル | 現状 | サイズ |
|--------|-------------|------|--------|
| LGBM | inference_bundle.pkl | ✅ 生成済み | 2.2GB |
| LGBM | feature_list.json | ✅ 生成済み | 3KB |
| LGBM | model_meta.json | ✅ 生成済み | 0.5KB |
| XGBoost | inference_bundle.pkl | ✅ 生成済み | 2.3GB |
| XGBoost | feature_list.json | ✅ 生成済み | 3KB |
| XGBoost | model_meta.json | ✅ 生成済み | 0.7KB |
| XGBoost | xgboost-*.whl | ✅ 生成済み | 116MB |

---

## Kaggle NB実装イメージ

```python
import pickle

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
