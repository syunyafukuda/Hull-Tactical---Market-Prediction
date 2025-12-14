# Ensemble Phase

最終更新: 2025-12-14

## 概要

Model Selection Phase完了後、**アンサンブルフェーズ**に移行。
採用候補モデル（LGBM, XGBoost, CatBoost）を組み合わせ、単体モデルを超える予測性能を目指す。

## ⚠️ フェーズ終了判定

**2025-12-14: アンサンブルフェーズを終了し、LGBM単体を維持**

| Step | 手法 | OOF RMSE | LB Score | 判定 |
|------|------|----------|----------|------|
| ベースライン | LGBM単体 | 0.012164 | **0.681** | ✅ 維持 |
| Step 1 | 50:50 単純平均 | 0.011932 | 0.615 | ❌ -9.7% |
| Step 2 | Rank Average | 0.011876 | 0.616 | ❌ -9.5% |
| Step 3 | 3モデル加重平均 | - | - | ❌ 中止 |
| Step 4 | 3モデル Rank Avg | - | - | ❌ 中止 |
| Step 5 | Stacking | - | - | ❌ 中止 |

**根本原因**: XGBoost/CatBoostの**OOF↔LB乖離**
- OOFでは両モデルともLGBMより優秀
- LBではXGBoost -8.7%、CatBoost -11.6%と大幅劣化
- どの混合手法でもこの劣化が伝播し、アンサンブル全体を悪化させる

---

## 現状ベースライン

| 項目 | 値 |
|------|-----|
| 採用特徴セット | FS_compact（116列） |
| ベースモデル | LightGBM |
| OOF RMSE | 0.012164 |
| LB Score | 0.681 |

## アンサンブル候補モデル

| モデル | OOF RMSE | LB Score | 対LGBM相関 | 予測Std | 採用判断 |
|--------|----------|----------|------------|---------|----------|
| **LightGBM** | 0.012164 | **0.681** | - | 0.005246 | ✅ 必須 |
| **XGBoost** | 0.012062 | 0.622 | **0.684** | 0.004999 | ✅ 必須 |
| CatBoost | 0.011095 | 0.602 | 0.35 | 0.000495 | ⚠️ 要検討 |

### CatBoostに関する注意

- 対LGBM相関0.35と低く、理論上は多様性が高い
- **しかし予測Stdが0.000495とLGBMの9%しかない**
- 混ぜると全体予測を平坦化するリスク大
- 試す場合は重みを10%程度に抑制

---

## フェーズ構成（全Step非採用）

```
Ensemble Phase ❌ 終了（2025-12-14）
├── Step 1: LGBM + XGBoost（50:50 単純平均）  ❌ LB 0.615 - 非採用
│   └── 基本的なアンサンブル効果を確認
├── Step 2: LGBM + XGBoost（Rank Average）     ❌ LB 0.616 - 非採用
│   └── スケール差を吸収した平均
├── Step 3: LGBM + XGBoost + CatBoost（60:30:10） ❌ 中止
│   └── CatBoost少量追加の効果を検証
├── Step 4: LGBM + XGBoost + CatBoost（Rank Average） ❌ 中止
│   └── 3モデルのRank Average
└── Step 5: LGBM + XGBoost + CatBoost（Stacking）    ❌ 中止
    └── Meta-Learnerによる重み学習

結論: LGBM単体（LB 0.681）を維持
```

---

## Step 1: LGBM + XGBoost（50:50 単純平均）

### 目的

LGBM + XGBoost 50:50 単純平均のLB効果を確認

### 根拠

OOFでの既存分析結果:

| 構成 | OOF RMSE | LGBM比 |
|------|----------|--------|
| LGBM単体 | 0.012164 | - |
| XGBoost単体 | 0.012062 | -0.84% |
| **50% LGBM + 50% XGB** | **0.011932** | **-1.91%** |

### 実装

```python
pred = (lgbm_pred + xgb_pred) / 2
```

### 成功基準

- LB Score > 0.681（LGBMベースライン超え）

### 成果物

- `artifacts/ensemble/step1_lgbm_xgb_avg/submission.csv`
- `artifacts/ensemble/step1_lgbm_xgb_avg/oof_predictions.csv`

---

## Step 2: LGBM + XGBoost（Rank Average）

### 目的

予測値をランク変換してから平均することで、スケール差を吸収

### 実装

```python
lgbm_rank = lgbm_pred.rank(pct=True)
xgb_rank = xgb_pred.rank(pct=True)
pred_rank = (lgbm_rank + xgb_rank) / 2
# 必要に応じて元のスケールに変換
```

### 期待効果

- 外れ値の影響軽減
- 異なるスケールのモデルを公平に統合

### 成果物

- `artifacts/ensemble/step2_lgbm_xgb_rank/submission.csv`
- `artifacts/ensemble/step2_lgbm_xgb_rank/oof_predictions.csv`

---

## Step 3: LGBM + XGBoost + CatBoost（60:30:10）

### 目的

CatBoostを少量（10%）追加した場合の効果を検証

### 重み設定

| モデル | 重み | 理由 |
|--------|------|------|
| LGBM | 60% | LB最良（0.681） |
| XGBoost | 30% | アンサンブル価値高い（相関0.684） |
| CatBoost | 10% | 予測Stdが極端に小さいため少量に抑制 |

### 実装

```python
pred = 0.6 * lgbm_pred + 0.3 * xgb_pred + 0.1 * catboost_pred
```

### リスク

- CatBoostの予測Stdが極端に小さい（0.000495）
- アンサンブル全体が平坦化する恐れ
- **OOFで効果を確認してからLB検証**

### 成果物

- `artifacts/ensemble/step3_lgbm_xgb_cat/submission.csv`
- `artifacts/ensemble/step3_lgbm_xgb_cat/oof_predictions.csv`

---

## Step 4: LGBM + XGBoost + CatBoost（Rank Average）

### 前提条件

**Step 3でCatBoost追加が有効だった場合のみ実施**

### 目的

3モデルのRank Averageで多様性を最大化

### 実装

```python
lgbm_rank = lgbm_pred.rank(pct=True)
xgb_rank = xgb_pred.rank(pct=True)
cat_rank = catboost_pred.rank(pct=True)
pred_rank = (lgbm_rank + xgb_rank + cat_rank) / 3
```

### 成果物

- `artifacts/ensemble/step4_3model_rank/submission.csv`
- `artifacts/ensemble/step4_3model_rank/oof_predictions.csv`

---

## Step 5: LGBM + XGBoost + CatBoost（Stacking）

### 前提条件

**Step 3でCatBoost追加が有効だった場合のみ実施**

### 目的

Meta-Learnerで各モデルの最適重みを学習

### 実装

```python
# OOF予測を特徴量にしてMeta-Learnerを学習
meta_features = np.column_stack([lgbm_oof, xgb_oof, catboost_oof])
meta_model = Ridge(alpha=1.0).fit(meta_features, y)

# テスト予測
test_meta = np.column_stack([lgbm_test, xgb_test, catboost_test])
pred = meta_model.predict(test_meta)
```

### 注意

- **過学習リスクが高い**（金融データでは特に危険）
- CV設計を慎重に（OOF予測をそのまま使うとリーク）

### 成果物

- `artifacts/ensemble/step5_stacking/submission.csv`
- `artifacts/ensemble/step5_stacking/oof_predictions.csv`
- `artifacts/ensemble/step5_stacking/meta_model.pkl`

---

## ファイル構成

```
src/ensemble/
├── __init__.py
├── blend.py              # 重み付き平均、単純平均
├── rank_average.py       # Rank Average
└── stacking.py           # Stacking

configs/ensemble/
├── step1_lgbm_xgb_avg.yaml
├── step2_lgbm_xgb_rank.yaml
├── step3_lgbm_xgb_cat.yaml
├── step4_3model_rank.yaml
└── step5_stacking.yaml

artifacts/ensemble/
├── step1_lgbm_xgb_avg/
│   ├── submission.csv
│   └── oof_predictions.csv
├── step2_lgbm_xgb_rank/
│   ├── submission.csv
│   └── oof_predictions.csv
├── step3_lgbm_xgb_cat/
│   ├── submission.csv
│   └── oof_predictions.csv
├── step4_3model_rank/
│   ├── submission.csv
│   └── oof_predictions.csv
└── step5_stacking/
    ├── submission.csv
    ├── oof_predictions.csv
    └── meta_model.pkl

notebooks/submit/
├── ensemble_step1.ipynb
├── ensemble_step2.ipynb
├── ensemble_step3.ipynb
├── ensemble_step4.ipynb
└── ensemble_step5.ipynb

tests/ensemble/
├── test_blend.py
├── test_rank_average.py
└── test_stacking.py
```

---

## 実行コマンド（予定）

### Step 1

```bash
python -m src.ensemble.blend \
    --config configs/ensemble/step1_lgbm_xgb_avg.yaml \
    --out-dir artifacts/ensemble/step1_lgbm_xgb_avg
```

### Step 2

```bash
python -m src.ensemble.rank_average \
    --config configs/ensemble/step2_lgbm_xgb_rank.yaml \
    --out-dir artifacts/ensemble/step2_lgbm_xgb_rank
```

---

## OOF評価サマリー（2025-12-13実施）

| Step | 手法 | OOF RMSE | vs LGBM | 予測Std | 状態 |
|------|------|----------|---------|---------|------|
| baseline | LGBM単体 | 0.012164 | - | 0.005246 | ✅ |
| **Step 1** | LGBM+XGB 50:50 | 0.011932 | **-1.91%** | 0.004700 | ✅ LB検証へ |
| **Step 2** | LGBM+XGB Rank | 0.011876 | **-2.36%** | 0.004507 | ✅ LB検証へ |
| Step 3a | 60:30:10 | 0.011797 | -3.01% | 0.004319 | 📊 |
| Step 3b | 50:35:15 | 0.011701 | **-3.80%** | 0.004035 | 📊 ベスト重み |
| Step 4 | 3-Model Rank | 0.011457 | **-5.82%** | ⚠️ 0.003120 | 📊 Std低い |
| Step 5 | Stacking | 0.010998 | -9.59% | ⚠️ 0.000151 | ❌ 過学習 |

### 重要な発見

1. **Step 1/2は安全**: 予測Stdも維持され、OOF改善 → LB検証推奨
2. **CatBoost追加は効果あり**: Step 3b (50:35:15) が重み平均ベスト
3. **Step 4は要注意**: OOF最良だがStdが低下（0.003120）
4. **Step 5は不採用**: 予測Std=0.000151は過学習の明確な兆候

### 推奨戦略

```
まずStep 1/2のLB検証 → 良ければStep 3bのLB検証 → Step 4は慎重に
```

---

## 優先度と見積もり（OOF結果を踏まえて更新）

| Step | 内容 | 優先度 | 見積もり | 依存 |
|------|------|--------|----------|------|
| **Step 1** | LGBM+XGB 50:50 | ★★★ | 1時間 | - |
| **Step 2** | LGBM+XGB Rank Average | ★★★ | 1時間 | - |
| **Step 3b** | 50:35:15 | ★★☆ | 1時間 | Step 1/2確認後 |
| Step 4 | 3モデル Rank Average | ★☆☆ | 2時間 | ⚠️ Stdに注意 |
| ~~Step 5~~ | ~~Stacking~~ | - | - | ❌ 不採用 |

---

## 参考リンク

- [モデル選定結果](../models/README.md)
- [LB提出履歴](../submissions.md)
- [XGBoost仕様書](../models/xgboost.md)
- [CatBoost仕様書](../models/catboost.md)
