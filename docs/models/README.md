# Model Selection Phase

## 概要

Feature Selection Phase（Phase 0-3）完了後、**モデル多様化フェーズ**に移行。
FS_compact（116列）を固定した上で、複数のモデルタイプを同一CV設定で比較し、
アンサンブルに向けた候補モデルを選定する。

## 現状ベースライン

| 項目 | 値 |
|------|-----|
| 採用特徴セット | FS_compact（116列） |
| ベースモデル | LightGBM |
| OOF RMSE | 0.012164 |
| LB Score | 0.681 |
| ブランチ | `dev` |

---

## 戦略背景

### なぜモデル多様化か？

Feature Selection Phase の結論として：

1. **特徴量削減は成功**: 577列 → 116列（-80%）で LB 維持
2. **これ以上の削減はリスキー**: FS_topK（50列）で LB が 0.589 に崩壊
3. **特徴量側の改善余地は小さい**: SU1 寄与 2.3%、予測相関 0.895

よって、**特徴量は FS_compact（116列）で固定**し、以下の軸でパフォーマンス向上を狙う：

- **モデル多様化**: 異なるアルゴリズムで別視点を導入
- **ハイパラチューニング**: 各モデルの最適化
- **アンサンブル**: 予測相関が低いモデルを組み合わせ

> **重要**: すべてのモデルは FS_compact（Feature Selection Phase で確定した116列）を入力として使用する。
> 特徴量セットの変更は行わない（Feature Selection Phase での結論と整合を保つ）。

---

## フェーズ構成

```
Model Selection Phase
├── Step 1: モデル候補の実装・評価
│   ├── XGBoost
│   ├── CatBoost
│   └── Ridge
├── Step 2: ハイパラチューニング
│   └── 各モデルで Optuna or Grid Search
├── Step 3: アンサンブル構築
│   ├── 単純平均
│   ├── 重み付き平均（OOF RMSE ベース）
│   └── Stacking（必要に応じて）
└── Step 4: LB 検証・最終選定
```

---

## Step 1: モデル候補

### 評価対象モデル

#### 勾配ブースティング系

| 順番 | モデル | 理由 | 初期ハイパラ |
|------|--------|------|-------------|
| 1 | **XGBoost** | LGBM と同系統だが実装差異あり。アンサンブル効果期待 | `max_depth=6, lr=0.05, n_est=600` |
| 2 | **CatBoost** | 順序付きブースティング。過学習に強い | `depth=6, lr=0.05, iterations=600` |

#### バギング系ツリー（新規追加）

| 順番 | モデル | 理由 | 初期ハイパラ |
|------|--------|------|-------------|
| 3 | **ExtraTrees** | 分割点もランダム。過学習しにくく GBDT と予測の揺れ方が異なる | `n_est=500, max_depth=15, min_samples_leaf=5` |
| 4 | **RandomForest** | バギング+ランダムサブスペース。GBDT とはバイアスが異なる | `n_est=500, max_depth=15, min_samples_leaf=5` |

#### 線形モデル系

| 順番 | モデル | 理由 | 初期ハイパラ |
|------|--------|------|-------------|
| 5 | **Ridge** | L2正則化線形。多様性確保・ベースライン確認 | `alpha=1.0`（CV で調整） |
| 6 | **Lasso** | L1正則化でスパース性。効いてない特徴を 0 に | `alpha=0.001`（CV で調整） |
| 7 | **ElasticNet** | L1+L2 ハイブリッド。スパース性と安定性のバランス | `alpha=0.001, l1_ratio=0.5` |

### 評価基準

**主指標**:
1. **OOF RMSE**: 予測精度の一次評価（**選定フェーズの最重要指標**）

**補助指標**:
2. **予測相関**: LGBM との相関（低いほどアンサンブル価値あり）
3. **OOF MSR**: トレード観点での補助的監視指標
4. **LB Score**: 最終判断（主力候補のみ提出）

### 成功条件

- OOF RMSE が LGBM ベースライン（0.01216）と同等以上
- 予測相関が 0.95 未満（アンサンブル効果の見込み）
- LB Score が 0.68 以上を維持

### 多様性の期待値

| モデル種別 | 対LGBM予測相関（期待） | アンサンブル価値 |
|------------|----------------------|------------------|
| XGBoost | 0.95-0.98 | 中程度（同系統だが実装差） |
| CatBoost | 0.92-0.96 | 中〜高（Ordered Boosting差） |
| ExtraTrees | 0.85-0.92 | 高（バギング系で揺れ方が異なる） |
| RandomForest | 0.85-0.92 | 高（バギング系） |
| Ridge | 0.70-0.85 | 非常に高（線形 vs 非線形） |
| Lasso | 0.70-0.85 | 非常に高（線形・スパース） |
| ElasticNet | 0.70-0.85 | 非常に高（線形・ハイブリッド） |

---

## Step 2: ハイパラチューニング

### 方針

- **控えめなチューニング**: 過学習リスクを避けるため探索範囲を限定
- **OOF と LB の整合性重視**: OOF 最良でも LB 崩壊（FS_topK 教訓）を回避

### チューニング対象

#### LGBM

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `num_leaves` | [31, 63, 127] | 63 |
| `max_depth` | [6, 8, 10, -1] | -1 |
| `min_data_in_leaf` | [20, 32, 50] | 32 |
| `feature_fraction` | [0.7, 0.8, 0.9] | 0.9 |
| `learning_rate` | [0.03, 0.05, 0.1] | 0.05 |
| `reg_alpha` | [0, 0.1, 0.5] | 0 |
| `reg_lambda` | [0, 0.1, 0.5] | 0 |

#### XGBoost

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `max_depth` | [4, 6, 8] | 6 |
| `learning_rate` | [0.03, 0.05, 0.1] | 0.05 |
| `subsample` | [0.7, 0.8, 0.9] | 0.8 |
| `colsample_bytree` | [0.7, 0.8, 0.9] | 0.8 |
| `reg_alpha` | [0, 0.1, 0.5] | 0 |
| `reg_lambda` | [0, 1, 5] | 1 |

#### CatBoost

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `depth` | [4, 6, 8] | 6 |
| `learning_rate` | [0.03, 0.05, 0.1] | 0.05 |
| `l2_leaf_reg` | [1, 3, 5, 10] | 3 |
| `random_strength` | [0, 1, 2] | 1 |

#### Ridge

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `alpha` | `np.logspace(-3, 3, 20)` | 1.0 |

#### Lasso

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `alpha` | `np.logspace(-5, -1, 20)` | 0.001 |

#### ElasticNet

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `alpha` | `np.logspace(-5, -1, 20)` | 0.001 |
| `l1_ratio` | [0.1, 0.3, 0.5, 0.7, 0.9] | 0.5 |

#### ExtraTrees / RandomForest

| パラメータ | 探索範囲 | デフォルト |
|-----------|---------|-----------|
| `n_estimators` | [300, 500, 700] | 500 |
| `max_depth` | [10, 15, 20, None] | 15 |
| `min_samples_leaf` | [3, 5, 10] | 5 |
| `max_features` | [0.5, 0.7, 1.0, "sqrt"] | 0.7 |

---

## Step 3: アンサンブル戦略

### 基本方針

1. **単純平均**: まず試す最もシンプルな方法
2. **OOF RMSE ベース重み付け**: `weight_i = 1/rmse_i / sum(1/rmse_j)`
3. **Stacking**: 必要に応じて（過学習リスクあり）

### アンサンブル候補の組み合わせ

| 組み合わせ | 期待効果 |
|-----------|---------|
| LGBM + XGBoost | 同系統だが実装差異による多様性 |
| LGBM + CatBoost | ブースティング手法の違い |
| LGBM + Ridge | 非線形 + 線形の組み合わせ |
| LGBM + XGBoost + CatBoost | GBDT 3本アンサンブル |
| LGBM + ExtraTrees | GBDT + バギング |
| LGBM + Ridge | 非線形 + 線形 |
| LGBM + ExtraTrees + Ridge | 3系統混合 |
| Full Ensemble (7本) | 全モデル平均 |

### 予測相関の目安

| 相関係数 | 解釈 |
|---------|------|
| > 0.98 | ほぼ同一予測。アンサンブル効果薄い |
| 0.95 - 0.98 | 微差。効果限定的 |
| 0.90 - 0.95 | 適度な多様性。効果期待 |
| < 0.90 | 高い多様性。大きな効果期待 |

---

## ディレクトリ構造

```
src/models/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── cv_utils.py          # CV分割・評価ロジック
│   └── feature_loader.py    # FS_compact読み込み
├── lgbm/                     # ✅ 実装済み
│   ├── __init__.py
│   └── train_lgbm.py
├── xgboost/                  # 勾配ブースティング
│   ├── __init__.py
│   └── train_xgb.py
├── catboost/                 # 勾配ブースティング
│   ├── __init__.py
│   └── train_catboost.py
├── extratrees/               # バギング系ツリー（新規）
│   ├── __init__.py
│   └── train_extratrees.py
├── randomforest/             # バギング系ツリー（新規）
│   ├── __init__.py
│   └── train_randomforest.py
├── ridge/                    # 線形モデル（L2）
│   ├── __init__.py
│   └── train_ridge.py
├── lasso/                    # 線形モデル（L1）（新規）
│   ├── __init__.py
│   └── train_lasso.py
└── elasticnet/               # 線形モデル（L1+L2）（新規）
    ├── __init__.py
    └── train_elasticnet.py

configs/models/
├── lgbm.yaml
├── xgboost.yaml
├── catboost.yaml
├── extratrees.yaml           # 新規
├── randomforest.yaml          # 新規
├── ridge.yaml
├── lasso.yaml                 # 新規
└── elasticnet.yaml            # 新規

artifacts/models/
├── lgbm/                      # ✅ 成果物あり
│   ├── inference_bundle.pkl
│   ├── oof_predictions.csv
│   └── model_meta.json
├── xgboost/
├── catboost/
├── extratrees/                # 新規
├── randomforest/              # 新規
├── ridge/
├── lasso/                     # 新規
└── elasticnet/                # 新規

docs/models/
├── README.md                 # 本ファイル（戦略概要）
├── xgboost.md                # XGBoost実装仕様書
├── catboost.md               # CatBoost実装仕様書
├── extratrees.md             # ExtraTrees実装仕様書（新規）
├── randomforest.md           # RandomForest実装仕様書（新規）
├── ridge.md                  # Ridge実装仕様書
├── lasso.md                  # Lasso実装仕様書（新規）
├── elasticnet.md             # ElasticNet実装仕様書（新規）
└── model_comparison.md       # 比較結果レポート（後で作成）
```

---

## 進捗トラッキング

### Step 1: モデル候補の実装・評価

#### 勾配ブースティング系

| モデル | 仕様書 | 実装 | OOF評価 | LB提出 | 状態 |
|--------|--------|------|---------|--------|------|
| LGBM (現行) | - | ✅ | 0.01216 | 0.681 | ベースライン |
| XGBoost | [xgboost.md](xgboost.md) | ⬜ | - | - | 未着手 |
| CatBoost | [catboost.md](catboost.md) | ⬜ | - | - | 未着手 |

#### バギング系ツリー

| モデル | 仕様書 | 実装 | OOF評価 | LB提出 | 状態 |
|--------|--------|------|---------|--------|------|
| ExtraTrees | [extratrees.md](extratrees.md) | ⬜ | - | - | 未着手 |
| RandomForest | [randomforest.md](randomforest.md) | ⬜ | - | - | 未着手 |

#### 線形モデル系

| モデル | 仕様書 | 実装 | OOF評価 | LB提出 | 状態 |
|--------|--------|------|---------|--------|------|
| Ridge | [ridge.md](ridge.md) | ⬜ | - | - | 未着手 |
| Lasso | [lasso.md](lasso.md) | ⬜ | - | - | 未着手 |
| ElasticNet | [elasticnet.md](elasticnet.md) | ⬜ | - | - | 未着手 |

### Step 2: ハイパラチューニング

| モデル | 状態 | ベスト OOF |
|--------|------|-----------|
| LGBM | ⬜ | - |
| XGBoost | ⬜ | - |
| CatBoost | ⬜ | - |
| ExtraTrees | ⬜ | - |
| RandomForest | ⬜ | - |
| Ridge | ⬜ | - |
| Lasso | ⬜ | - |
| ElasticNet | ⬜ | - |

### Step 3: アンサンブル

| 組み合わせ | OOF RMSE | LB Score | 状態 |
|-----------|----------|----------|------|
| - | - | - | 未着手 |

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| OOF↔LB 乖離（過学習） | FS_topK 教訓を活かし、OOF 最良でも慎重に LB 確認 |
| 計算コスト増大 | FS_compact（116列）固定により 1 モデルあたりのコストを抑制 |
| アンサンブルで悪化 | 単体 LB 確認後にアンサンブル。悪いモデルは除外 |
| ハイパラ探索で過学習 | 探索範囲を限定。Early stopping を徹底 |

---

## 次のアクション

1. **XGBoost 実装**: `src/models/xgboost/train_xgb.py`
2. **XGBoost OOF 評価**: 同一 CV で LGBM と比較
3. **予測相関計算**: LGBM との相関を確認
4. **LB 提出**: 有望であれば Kaggle 提出

---

## 参考リンク

- [Feature Selection README](../feature_selection/README.md)
- [Phase 3 Report](../feature_selection/phase3_report.md)
- [Submissions Log](../submissions.md)
