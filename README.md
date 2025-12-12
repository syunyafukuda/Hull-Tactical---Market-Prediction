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
│  │  ├─ xgboost/                 # XGBoost (実装予定)
│  │  ├─ catboost/                # CatBoost (実装予定)
│  │  ├─ extratrees/              # ExtraTrees (実装予定)
│  │  ├─ randomforest/            # RandomForest (実装予定)
│  │  ├─ ridge/                   # Ridge (実装予定)
│  │  ├─ lasso/                   # Lasso (実装予定)
│  │  └─ elasticnet/              # ElasticNet (実装予定)
│  └─ preprocess/                 # 特徴量グループ別欠損補完
├─ scripts/                       # CLI・ユーティリティ
├─ configs/
│  ├─ preprocess.yaml             # 前処理設定
│  ├─ feature_generation.yaml     # 特徴量生成設定
│  ├─ feature_selection/          # 特徴量選定設定 (tier0-3)
│  └─ models/                     # モデル設定 (YAML)
├─ artifacts/                     # 学習済み成果物 (Git管理外)
│  ├─ tier0-3/                    # 特徴量選定の中間成果物
│  └─ models/                     # モデル別成果物
├─ docs/
│  ├─ feature_generation/         # SU1-SU11 仕様書
│  ├─ feature_selection/          # 特徴量選定レポート
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

### モデル候補

| カテゴリ | モデル | 仕様書 | 実装 | OOF RMSE | 予測相関 (期待) |
|----------|--------|--------|------|----------|-----------------|
| **勾配ブースティング** | LGBM | - | ✅ | 0.01216 | - (ベースライン) |
| 勾配ブースティング | XGBoost | ✅ | ⬜ | - | 0.95-0.98 |
| 勾配ブースティング | CatBoost | ✅ | ⬜ | - | 0.92-0.96 |
| **バギング系ツリー** | ExtraTrees | ✅ | ⬜ | - | 0.85-0.92 |
| バギング系ツリー | RandomForest | ✅ | ⬜ | - | 0.85-0.92 |
| **線形モデル** | Ridge | ✅ | ⬜ | - | 0.70-0.85 |
| 線形モデル | Lasso | ✅ | ⬜ | - | 0.70-0.85 |
| 線形モデル | ElasticNet | ✅ | ⬜ | - | 0.70-0.85 |

### 成功基準

- OOF RMSE ≤ 0.013 (線形モデルは ≤ 0.015)
- 予測相関 (vs LGBM) < 0.95 (アンサンブル効果の見込み)
- LB Score ≥ 0.68

### 次のステップ

1. XGBoost / CatBoost 実装・OOF評価
2. バギング系 (ExtraTrees / RandomForest) 実装
3. 線形モデル (Ridge / Lasso / ElasticNet) 実装
4. 予測相関分析・アンサンブル構築

詳細: `docs/models/README.md`

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

