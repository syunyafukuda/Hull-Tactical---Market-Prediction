# Ensemble Phase

最終更新: 2025-12-13

## 概要

Model Selection Phase完了後、**アンサンブルフェーズ**に移行。
採用候補モデル（LGBM, XGBoost）を組み合わせ、単体モデルを超える予測性能を目指す。

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
- 試す場合は重みを5-15%程度に抑制

---

## フェーズ構成

```
Ensemble Phase
├── Phase 1: 基本検証（最優先）
│   └── LGBM + XGBoost 50:50 単純平均 → LB検証
├── Phase 2: 重み最適化
│   └── Grid Search で最適重みを探索
├── Phase 3: CatBoost追加検討
│   └── 少量（5-15%）追加の効果を検証
└── Phase 4: 高度なアンサンブル（必要に応じて）
    ├── Rank Average
    └── Stacking（慎重に）
```

---

## Phase 1: 基本検証（最優先）

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
# 単純平均
pred = (lgbm_pred + xgb_pred) / 2
```

### 成功基準

- LB Score > 0.681（LGBMベースライン超え）

### 成果物

- `artifacts/ensemble/phase1/submission.csv`
- `artifacts/ensemble/phase1/oof_predictions.csv`

---

## Phase 2: 重み最適化

### 目的

LGBM/XGBoostの最適重みを探索

### 手法

**Grid Search**:
```python
weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
for w_lgbm, w_xgb in weights:
    pred = w_lgbm * lgbm_pred + w_xgb * xgb_pred
    oof_rmse = compute_rmse(y, pred)
```

### 既存分析結果

| LGBM重み | XGB重み | OOF RMSE |
|----------|---------|----------|
| 50% | 50% | **0.011932** ← 最良 |
| 60% | 40% | 0.011950 |
| 70% | 30% | 0.011982 |

→ 50:50付近が最適の可能性高い

### 成果物

- `artifacts/ensemble/phase2/weight_search.csv`
- `artifacts/ensemble/phase2/submission.csv`

---

## Phase 3: CatBoost追加検討

### 目的

CatBoostを少量追加した場合の効果を検証

### 試行パターン

| パターン | LGBM | XGB | CatBoost |
|----------|------|-----|----------|
| A | 45% | 45% | 10% |
| B | 50% | 40% | 10% |
| C | 40% | 45% | 15% |

### リスク

- CatBoostの予測Stdが極端に小さい（0.000495）
- アンサンブル全体が平坦化する恐れ
- **OOFで効果を確認してからLB検証**

### 成果物

- `artifacts/ensemble/phase3/catboost_analysis.csv`
- `artifacts/ensemble/phase3/submission.csv`

---

## Phase 4: 高度なアンサンブル（必要に応じて）

### 4.1 Rank Average

```python
# 予測値をランク変換してから平均
lgbm_rank = lgbm_pred.rank(pct=True)
xgb_rank = xgb_pred.rank(pct=True)
pred = (lgbm_rank + xgb_rank) / 2
```

**用途**: スケールの違いを吸収、外れ値の影響軽減

### 4.2 Stacking

```python
# OOF予測を特徴量にしてMeta-Learnerを学習
meta_features = np.column_stack([lgbm_oof, xgb_oof])
meta_model = Ridge(alpha=1.0).fit(meta_features, y)
```

**注意**: 過学習リスクが高い。金融データでは特に慎重に。

---

## ファイル構成

```
src/ensemble/
├── __init__.py
├── blend.py              # 重み付き平均、単純平均
├── weight_search.py      # Grid Search / 最適化
├── stacking.py           # Stacking（Phase 4用）
└── evaluate.py           # OOF評価ユーティリティ

configs/ensemble/
├── phase1.yaml           # Phase 1 設定
├── phase2.yaml           # Phase 2 設定
└── phase3.yaml           # Phase 3 設定

artifacts/ensemble/
├── phase1/
│   ├── submission.csv
│   └── oof_predictions.csv
├── phase2/
│   ├── weight_search.csv
│   └── submission.csv
└── phase3/
    ├── catboost_analysis.csv
    └── submission.csv

notebooks/submit/
└── ensemble_lgbm_xgb.ipynb  # Kaggle提出用

tests/ensemble/
├── test_blend.py
└── test_weight_search.py
```

---

## 実行コマンド（予定）

### Phase 1

```bash
# OOF予測のブレンド
python -m src.ensemble.blend \
    --models lgbm xgboost \
    --weights 0.5 0.5 \
    --out-dir artifacts/ensemble/phase1
```

### Phase 2

```bash
# 重み探索
python -m src.ensemble.weight_search \
    --models lgbm xgboost \
    --out-dir artifacts/ensemble/phase2
```

---

## 優先度と見積もり

| Phase | 内容 | 優先度 | 見積もり |
|-------|------|--------|----------|
| **Phase 1** | LGBM+XGB 50:50 LB検証 | ★★★ | 1-2時間 |
| Phase 2 | 重み最適化 | ★★☆ | 2-3時間 |
| Phase 3 | CatBoost追加 | ★☆☆ | 2-3時間 |
| Phase 4 | 高度なアンサンブル | ★☆☆ | 半日以上 |

---

## 参考リンク

- [モデル選定結果](../models/README.md)
- [LB提出履歴](../submissions.md)
- [XGBoost仕様書](../models/xgboost.md)
- [CatBoost仕様書](../models/catboost.md)
