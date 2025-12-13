# Model Selection Phase

## 概要

Feature Selection Phase（Phase 0-3）完了後、**モデル多様化フェーズ**に移行。
FS_compact（116列）を固定した上で、複数のモデルタイプを同一CV設定で比較し、
アンサンブルに向けた候補モデルを選定する。

## モデル実績サマリー

| モデル | OOF RMSE | LB Score | 対 LGBM 予測相関 | アンサンブル価値 | ステータス |
|--------|----------|----------|-----------------|------------------|----------|
| **LightGBM** | 0.012164 | **0.681** | - | ベースライン | ✅ 完了 |
| **XGBoost** | **0.012062** | 0.622 | **0.684** | ✅ **高い** | ✅ 完了 |
| ExtraTrees | 0.011440 | 0.500 | - | ❌ なし | ❌ 非採用 |
| ElasticNet | 0.011091 | 0.461 | - | ❌ なし | ❌ 非採用 |
| CatBoost | - | - | - | TBD | ⬜ 未着手 |
| Ridge | - | - | - | ❌ | ⬜ **試行不要** |
| Lasso | - | - | - | ❌ | ⬜ **試行不要** |
| RandomForest | - | - | - | ❌ | ⬜ **試行不要** |

### ⚠️ 重要な教訓（2025-12-13）

**線形モデルは不適合**:
- ElasticNet（L1+L2）がLB 0.461（ベースライン以下）
- 116特徴量中わずか2個だけ非ゼロ係数 → 実質的に定数予測
- Ridge/Lassoも同様に失敗する見込み → **試行不要**

**バギング系ツリーも不適合**:
- ExtraTreesがLB 0.500（ベースライン同等 = 情報なし）
- RandomForestも同様の失敗が予想される → **試行不要**

### XGBoost アンサンブル分析

XGBoostは単体LBが0.622とLGBMより劣るが、**予測相関が0.684と低くアンサンブル価値が高い**。

| 構成 | OOF RMSE | LGBM比 |
|------|----------|--------|
| LGBM単体 | 0.012164 | - |
| XGBoost単体 | 0.012062 | -0.84% |
| 50% LGBM + 50% XGB | **0.011932** | **-1.91%** |
| 60% LGBM + 40% XGB | 0.011950 | -1.76% |
| 70% LGBM + 30% XGB | 0.011982 | -1.50% |

**結論**: 50:50アンサンブルでOOF RMSEが1.91%改善。LB検証推奨。

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

### 多様性の期待値と実績

| モデル種別 | 対LGBM予測相関（期待） | 実績 | アンサンブル価値 |
|------------|----------------------|------|------------------|
| XGBoost | 0.95-0.98 | **0.684** ✅ | ★★★ 非常に高い（期待以上） |
| CatBoost | 0.92-0.96 | TBD | 中〜高（Ordered Boosting差） |
| ExtraTrees | 0.85-0.92 | **LB 0.500** ❌ | ❌ なし（ベースライン同等） |
| RandomForest | 0.85-0.92 | 試行不要 | ❌ なし（ExtraTrees類似） |
| Ridge | 0.70-0.85 | 試行不要 | ❌ なし（線形不適合） |
| Lasso | 0.70-0.85 | 試行不要 | ❌ なし（線形不適合） |
| ElasticNet | 0.70-0.85 | **LB 0.461** ❌ | ❌ なし（ベースライン以下） |

> **XGBoostの教訓**: 同系統の勾配ブースティングでも予測相関0.684と低く、
> アンサンブル価値が期待以上に高かった。単体LBが劣ってもアンサンブルには有効。
>
> **ExtraTrees/ElasticNetの教訓（2025-12-13）**: 
> - OOF RMSEが良くてもLBで崩壊するケースがある
> - 線形モデルは本問題に不適合（非ゼロ係数が極端に少ない）
> - バギング系ツリーも同様に不適合

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
│   ├── inference_bundle.pkl   # モデル + 前処理パイプライン
│   ├── oof_predictions.csv    # OOF予測値
│   ├── cv_fold_logs.csv       # フォールドごとのメトリクス
│   ├── model_meta.json        # メタデータ（id_col, target_col, oof_best_params含む）
│   ├── feature_list.json      # 特徴量リスト（Kaggle NB用）
│   └── submission.csv         # テスト予測（signal形式）
├── xgboost/                   # ✅ 成果物あり
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

## 成果物出力仕様（Kaggle NB用）

全モデルで共通の成果物フォーマットを使用し、Kaggle Notebookでの推論に必要な情報を出力する。

### model_meta.json スキーマ

```json
{
  "model_type": "xgboost",
  "feature_tier": "tier3",
  "n_features": 116,
  "oof_rmse": 0.011091,
  "oof_msr": 0.039917,
  "n_splits": 5,
  "gap": 0,
  "hyperparameters": { ... },
  "created_at": "2025-12-12T19:32:40.937939+00:00",
  "id_col": "date_id",
  "target_col": "market_forward_excess_returns",
  "oof_best_params": {
    "mult": 1.0,
    "lo": 0.9,
    "hi": 1.1
  }
}
```

**必須フィールド（Kaggle NB用）**:
- `id_col`: ID列名（Kaggle NBでprediction取得時に使用）
- `target_col`: 目的変数列名（参照用）
- `oof_best_params`: signal変換パラメータ
  - `mult`: 予測値に掛ける係数（デフォルト1.0）
  - `lo`, `hi`: クリッピング範囲（デフォルト0.9〜1.1）

### feature_list.json スキーマ

```json
{
  "version": "xgboost-tier3-v1",
  "created_at": "2025-12-12T19:32:40.944155+00:00",
  "pipeline_input_columns": ["D1", "D2", ...],
  "su1_generated_columns": ["run_na/E1", "run_obs/E1", ...],
  "su5_generated_columns": [],
  "model_input_columns": ["D1", "D2", ..., "run_na/E1", ...],
  "total_feature_count": 116,
  "source_commit": "a55f8e8...",
  "source_branch": "feat/model-xgboost"
}
```

**フィールド説明**:
- `pipeline_input_columns`: パイプラインへの入力特徴量（生データの94列）
- `su1_generated_columns`: SU1で生成された特徴量（run_na/*, run_obs/*, avg_run_na/*）
- `su5_generated_columns`: SU5で生成された特徴量（現在は空）
- `model_input_columns`: tier除外後の最終特徴量リスト
- `total_feature_count`: 最終特徴量数

### submission.csv フォーマット

```csv
date_id,prediction
8980,1.00021046416077
8981,0.99721418204717
...
```

**Signal変換**:
```python
# 生予測値をsignal形式に変換
signal_pred = np.clip(raw_pred * mult + 1.0, lo, hi)
```

- `is_scored == True` の行のみを含める
- 列名は `prediction` （target_col名ではない）
- 値は ~1.0 を中心とした signal 形式（0.9〜1.1の範囲）

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| OOF↔LB 乖離（過学習） | FS_topK 教訓を活かし、OOF 最良でも慎重に LB 確認 |
| 計算コスト増大 | FS_compact（116列）固定により 1 モデルあたりのコストを抑制 |
| アンサンブルで悪化 | 単体 LB 確認後にアンサンブル。悪いモデルは除外 |
| ハイパラ探索で過学習 | 探索範囲を限定。Early stopping を徹底 |

---

## 共通教訓（Lessons Learned）

### 予測分散の診断（pred/actual ratio）

**診断指標**: `pred.std() / actual.std()`

モデルが実際のターゲット変動をどれだけ捉えているかを示す重要な指標。

| 値 | 解釈 | 対処 |
|-----|------|------|
| < 10% | **過少学習**: モデルがほぼ定数を予測 | 正則化を緩める、学習を増やす |
| 30-70% | **正常範囲**: LGBMは約50% | - |
| > 90% | **過学習の兆候**: ノイズまで学習 | 正則化を強める |

```python
# 診断コード（OOF評価後に必ず実行）
ratio = oof['prediction'].std() / oof['actual'].std()
print(f"pred/actual ratio: {ratio:.1%}")
if ratio < 0.1:
    print("⚠️ WARNING: Model may be underfitting")
```

### XGBoostハイパーパラメータの教訓

XGBoost実装時に発見された問題（LB 0.542）:

| パラメータ | 問題のあった値 | 修正後 | 影響 |
|-----------|--------------|--------|------|
| `min_child_weight` | 32 | **1** | 過度な正則化を緩和 |
| `reg_lambda` | 1.0 | **0.001** | L2正則化を大幅に緩和 |
| `max_depth` | 6 | **10** | より深い木を許容 |
| `early_stopping_rounds` | 50 | **0（無効）** | 途中停止を防止 |
| `n_estimators` | 600 | **2000** | 十分な学習を確保 |

**結果**: pred/actual ratio が 4% → 47% に改善

### Kaggle提出時のバージョン互換性

勾配ブースティング系モデル（XGBoost, CatBoost, LightGBM）はバージョン間でモデルフォーマットが異なる場合がある。

**対策**:
1. 学習に使用したwheelファイルをartifactsに同梱
2. ノートブックでバージョンチェックを実装
3. 不一致時は明確なエラーメッセージを出力

```python
# バージョンチェック例
if not xgb.__version__.startswith("3."):
    raise RuntimeError(f"XGBoost version mismatch: expected 3.x, got {xgb.__version__}")
```

### ローカル推論スクリプト vs Kaggleノートブック

各モデルには**2種類の推論コード**を用意することを推奨：

| ファイル | 用途 | 依存関係 |
|----------|------|----------|
| `src/models/<model>/predict_<model>.py` | ローカル再推論 | `import` で既存モジュール使用 |
| `notebooks/submit/<model>.ipynb` | Kaggle提出 | 依存クラスをインライン埋込（~3000行） |

**使い分け**:
- **ローカル推論スクリプト**: 再学習なしで `submission.csv` を再生成（デバッグ・検証用）
- **Kaggleノートブック**: 実際のコンペ提出。Kaggle環境で必要なクラスがすべて揃っている

**両方で推論ロジックは同一に保つ**:
```python
# 共通の推論ロジック
prediction = pipeline.predict(X_test)
signal = to_signal(prediction, postprocess_params)  # np.clip(pred * mult + 1.0, lo, hi)
```

**環境差による微小な予測差**:
- Python 3.11 vs 3.12、NumPy バージョン差で ~0.02% の差異が発生しうる
- 重要なのは **OOF RMSEが一致する** こと（絶対値差は許容）

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
