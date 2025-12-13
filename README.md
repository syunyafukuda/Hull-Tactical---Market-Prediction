# Hull-Tactical---Market-Prediction

<https://www.kaggle.com/competitions/hull-tactical-market-prediction>

このリポジトリは [Hull Tactical Market Prediction コンペティション](https://www.kaggle.com/competitions/hull-tactical-market-prediction) への参加用プロジェクトです。  
GitHub Codespaces を開発環境とし、パッケージ管理は **[uv](https://github.com/astral-sh/uv)** を利用しています。

---

## 現在の状態

### ベストスコア

| 項目 | 値 |
|------|-----|
| **LB Score** | **0.681** |
| 特徴セット | FS_compact (116列) |
| ベースモデル | LightGBM |
| OOF RMSE | 0.012164 |
| ブランチ | `dev` |

### フェーズ進捗

```
[完了] 特徴量生成 (SU1-SU11)
    └─ SU1 + SU5 を採用 (577列 → LB 0.681)
        ↓
[完了] 特徴量選定 (Phase 0-3)
    └─ 577列 → 116列 に削減 (LB 0.681 維持)
        ↓
[進行中] モデル選定フェーズ  ← 現在ここ
    └─ 8モデルの比較・アンサンブル構築
```

---

## プロジェクト構成

```text
├─ .devcontainer/                 # Codespaces 用開発コンテナ設定
├─ src/
│  ├─ hull_tactical/              # 共通ユーティリティ
│  ├─ feature_generation/         # 特徴量生成モジュール
│  │  ├─ su1/                     # SU1 (欠損構造一次特徴) ✅ 採用
│  │  ├─ su5/                     # SU5 (共欠損構造特徴) ✅ 採用
│  │  └─ su2-su11/                # その他SU (非採用)
│  ├─ feature_selection/          # 特徴量選定モジュール
│  ├─ models/                     # モデル実装 (8種類)
│  │  ├─ common/                  # CV・特徴量ロード共通モジュール
│  │  ├─ lgbm/                    # LightGBM ✅ ベースライン
│  │  ├─ xgboost/                 # XGBoost ✅ アンサンブル候補
│  │  ├─ catboost/                # CatBoost ✅ (非採用)
│  │  ├─ extratrees/              # ExtraTrees ✅ (非採用)
│  │  ├─ randomforest/            # RandomForest ✅ (非採用)
│  │  ├─ ridge/                   # Ridge ✅ (非採用)
│  │  ├─ lasso/                   # Lasso ✅ (非採用)
│  │  └─ elasticnet/              # ElasticNet ✅ (非採用)
│  ├─ ensemble/                   # アンサンブル実装 ← NEW
│  └─ preprocess/                 # 特徴量グループ別欠損補完
├─ scripts/                       # CLI・ユーティリティ
├─ configs/
│  ├─ preprocess.yaml             # 前処理設定
│  ├─ feature_generation.yaml     # 特徴量生成設定
│  ├─ feature_selection/          # 特徴量選定設定 (tier0-3)
│  ├─ models/                     # モデル設定 (YAML)
│  └─ ensemble/                   # アンサンブル設定 ← NEW
├─ artifacts/                     # 学習済み成果物 (Git管理外)
│  ├─ tier0-3/                    # 特徴量選定の中間成果物
│  ├─ models/                     # モデル別成果物
│  └─ ensemble/                   # アンサンブル成果物 ← NEW
├─ docs/
│  ├─ feature_generation/         # SU1-SU11 仕様書
│  ├─ feature_selection/          # 特徴量選定レポート
│  ├─ ensemble/                   # アンサンブル仕様書 ← NEW
│  └─ models/                     # モデル選定仕様書
├─ notebooks/                     # Kaggle提出用ノートブック
├─ tests/                         # Pytest テスト
├─ data/                          # Kaggleデータ (Git管理外)
├─ pyproject.toml                 # uv 依存設定
└─ README.md                      # 本ドキュメント
```

---

## 開発環境

- **エディタ**: GitHub Codespaces (4-core / 16GB RAM)  
- **パッケージ管理**: [uv](https://github.com/astral-sh/uv)  
- **Python**: 3.11  

### クイックスタート

```bash
# 依存同期
uv sync

# データ取得
./scripts/fetch_data.sh

# 品質チェック (CI相当)
./scripts/check_quality.sh

# LGBM 学習
python -m src.models.lgbm.train_lgbm
```

### 重要な制約

- **numpy==1.26.4 固定**: Kaggle環境との互換性のため変更禁止
- **sklearn互換性**: Kaggle Dataset に wheel を同梱し `--no-index` でインストール

---

## 特徴量生成フェーズ (完了)

### 採用ライン

| ライン | LB Score | 特徴量数 | 状態 |
|--------|----------|---------|------|
| **SU1 + SU5** | **0.681** | 577 | ✅ 採用 |
| SU1 のみ | 0.674 | 462 | ベースライン |

### 非採用ライン

| ライン | LB Score | 非採用理由 |
|--------|----------|-----------|
| SU2 | 0.597 | 過学習 (特徴量爆発) |
| SU3 | 0.461 | コンセプト不適合 |
| SU7 | 0.476 | OOF改善もLB大幅悪化 |
| SU8 | 0.624 | OOF/LB両方で悪化 |
| SU9 | 0.679 | OOF改善もLB微悪化 |
| SU10 | 0.597 | 外部データ時間ミスマッチ |
| SU11 | 0.464 | shrinkage問題 (-31.9%) |

詳細: `docs/feature_generation/README.md`

---

## 特徴量選定フェーズ (完了)

### 結果サマリー

| Phase | 手法 | 残列数 | OOF RMSE | LB Score |
|-------|------|--------|----------|----------|
| Phase 0 | ベースライン凍結 | 577 | 0.012134 | 0.681 |
| Phase 1 | 統計フィルタ | 160 | 0.012168 | 0.681 |
| Phase 2 | モデル重要度 | 120 | 0.012172 | 0.681 |
| **Phase 3** | 相関クラスタリング | **116** | 0.012164 | **0.681** |

### 採用Feature Set

| セット名 | 列数 | LB Score | 状態 |
|---------|------|----------|------|
| **FS_compact** | 116 | 0.681 | ✅ 採用 |
| FS_full | 577 | 0.681 | 非採用 (冗長) |
| FS_topK | 50 | 0.589 | 非採用 (過学習) |

**結論**: 577列 → 116列 (-80%) で LB スコアを維持。FS_compact を最終特徴セットとして固定。

詳細: `docs/feature_selection/README.md`

---

## モデル選定フェーズ (進行中)

### 戦略

FS_compact (116列) を固定し、8種類のモデルを同一CV設定で比較。
アンサンブルに向けた候補モデルを選定する。

### モデル実績サマリー

| カテゴリ | モデル | 実装 | OOF RMSE | LB Score | 対LGBM相関 | ステータス |
|----------|--------|------|----------|----------|------------|----------|
| **GBDT** | LightGBM | ✅ | 0.012164 | **0.681** | - | ✅ ベースライン |
| GBDT | XGBoost | ✅ | 0.012062 | 0.622 | 0.684 | ✅ アンサンブル候補 |
| GBDT | CatBoost | ✅ | 0.011095 | 0.602 | 0.35 | ❌ 非採用 |
| **バギング** | ExtraTrees | ✅ | 0.011440 | 0.500 | - | ❌ 非採用（LB検証済） |
| バギング | RandomForest | ✅ | （未実行） | （未実行） | - | ❌ 非採用（※2） |
| **線形** | Ridge | ✅ | （未実行） | （未実行） | - | ❌ 非採用（※1） |
| 線形 | Lasso | ✅ | （未実行） | （未実行） | - | ❌ 非採用（※1） |
| 線形 | ElasticNet | ✅ | 0.011091 | 0.461 | - | ❌ 非採用（LB検証済） |

> **※1**: ElasticNet（L1+L2）の失敗から、Ridge（L2のみ）/Lasso（L1のみ）も同様に失敗すると判断し、LB検証なしで非採用。
> **※2**: ExtraTreesの失敗から、同系統のRandomForestも同様に失敗すると判断し、LB検証なしで非採用。

### 重要な教訓（2025-12-13）

**線形モデルは不適合**:
- ElasticNet（L1+L2）がLB 0.461（ベースライン以下）
- 116特徴量中わずか2個だけ非ゼロ係数 → 実質的に定数予測
- **Ridge/LassoはLB未検証だが、ElasticNetの結果から同様に失敗すると判断し非採用**

**バギング系ツリーも不適合**:
- ExtraTreesがLB 0.500（ベースライン同等 = 情報なし）
- **RandomForestはLB未検証だが、ExtraTreesの結果から同様に失敗すると判断し非採用**

### アンサンブル分析

| 構成 | OOF RMSE | LGBM比 |
|------|----------|--------|
| LGBM単体 | 0.012164 | - |
| XGBoost単体 | 0.012062 | -0.84% |
| 50% LGBM + 50% XGB | **0.011932** | **-1.91%** |

**結論**: LGBM + XGBoost 50:50アンサンブルでOOF RMSEが1.91%改善。

詳細: `docs/models/README.md`

---

## アンサンブルフェーズ (進行中)

### 戦略

採用候補モデル（LGBM, XGBoost, CatBoost）を組み合わせ、単体モデルを超える予測性能を目指す。

### ステップ構成

| Step | 内容 | 優先度 | 依存 | ステータス |
|------|------|--------|------|----------|
| **Step 1** | LGBM+XGB 50:50 単純平均 | ★★★ | - | ⬜ 未着手 |
| **Step 2** | LGBM+XGB Rank Average | ★★★ | - | ⬜ 未着手 |
| Step 3 | +CatBoost 10% (60:30:10) | ★★☆ | - | ⬜ 未着手 |
| Step 4 | 3モデル Rank Average | ★☆☆ | Step3有効時 | ⬜ 未着手 |
| Step 5 | Stacking | ★☆☆ | Step3有効時 | ⬜ 未着手 |

詳細: `docs/ensemble/README.md`

---

## データ運用

### 取得手順

```bash
# 自動取得スクリプト
./scripts/fetch_data.sh

# または手動
kaggle competitions download -c hull-tactical-market-prediction -p data/raw
unzip -o data/raw/hull-tactical-market-prediction.zip -d data/raw
```

### Git管理ポリシー

```gitignore
data/       # 競技データ
artifacts/  # 学習済み成果物
```

データと成果物はコミットしません。再現は「スクリプトで毎回ダウンロード」で行います。

---

## 品質チェック

プッシュ前に必ず実行:

```bash
./scripts/check_quality.sh
```

実行内容:
- **Ruff**: Lint / フォーマット
- **Pyright**: 型チェック
- **Pytest**: ユニットテスト

---

## コミットメッセージ規約

[Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に準拠:

```text
<type>(<scope>): <subject>
```

| type | 用途 |
|------|------|
| feat | 新機能 |
| fix | バグ修正 |
| docs | ドキュメント |
| refactor | リファクタリング |
| test | テスト |
| chore | 設定・依存更新 |

---

## ブランチ戦略

| ブランチ | 用途 |
|---------|------|
| `main` | 安定ブランチ |
| `dev` | 統合用ブランチ |
| `feat/*` | 機能開発 |
| `exp/*` | モデル実験 |

小さく早くマージし、不要になったら削除します。

---

## Kaggle提出

### 推論API形式

```python
def predict(test: pl.DataFrame) -> float:
    # 予測ロジック
    return prediction_value

DefaultInferenceServer(predict)
```

### 注意点

- 戻り値はスカラ float（配列やDataFrameは不可）
- scikit-learn 互換性のため wheel を Dataset に同梱
- `meta["feature_columns"]` を基準に列順・欠損補完を再現

詳細: `docs/submissions.md`

---

## ドキュメント構成

| パス | 内容 |
|------|------|
| `docs/feature_generation/` | SU1-SU11 仕様書・判断根拠 |
| `docs/feature_selection/` | Phase 0-3 レポート |
| `docs/models/` | モデル選定戦略・実装仕様書 |
| `docs/submissions.md` | 提出履歴・スコア推移 |
| `docs/preprocessing.md` | 前処理パイプライン |

---

## 参考リンク

- [Kaggle Competition Page](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [uv Documentation](https://github.com/astral-sh/uv)
- [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/)

