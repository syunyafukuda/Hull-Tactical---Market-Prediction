# Hull-Tactical---Market-Prediction
https://www.kaggle.com/competitions/hull-tactical-market-prediction

このリポジトリは [Hull Tactical Market Prediction コンペティション](https://www.kaggle.com/competitions/hull-tactical-market-prediction) への参加用プロジェクトです。  
GitHub Codespaces を開発環境とし、パッケージ管理は **[uv](https://github.com/astral-sh/uv)** を利用しています。

---

## プロジェクト構成

```text
├─ .devcontainer/ # Codespaces 開発環境の定義ファイル
│ └─ devcontainer.json
├─ src/ # ライブラリコード（再利用可能な処理をモジュール化）
├─ scripts/ # 学習・推論などの実行スクリプト（Submit単位でディレクトリを切る）
│  ├─ simple_baseline/
│  │   ├─ train_simple.py   # シンプル提出用の学習
│  │   └─ predict_simple.py # シンプル提出用の推論/提出生成
├─ docs/ # 知見やドキュメント類
├─ notebooks/ # 実験やEDA用のJupyterノートブック
├─ configs/ # HydraやYAML形式の設定ファイル
├─ tests/ # pytestによるテストコード
├─ pyproject.toml # uvによる依存関係の定義
├─ uv.lock # uvによる依存関係ロックファイル（再現性保証）
└─ README.md # 本ファイル
```

---

## 開発環境

- **エディタ**: GitHub Codespaces (4-core / 16GB RAM)  
- **パッケージ管理**: [uv](https://github.com/astral-sh/uv)  
- **Python**: 3.11  

### よく使うコマンド

依存関係を追加:

```bash
uv add パッケージ名
```

環境を再現:

```bash
uv sync
```

スクリプト実行（例: simple_baseline 提出ライン）:

```bash
# 依存を同期（初回/更新時）
uv sync

# データ取得（未取得の場合）
./scripts/fetch_data.sh

# 学習（成果物は artifacts/simple_baseline/ に保存）
uv run python scripts/simple_baseline/train_simple.py --data-dir data/raw

# 推論・提出生成（submission.parquet と submission.csv を artifacts/simple_baseline/ に出力）
uv run python scripts/simple_baseline/predict_simple.py --data-dir data/raw
```

品質チェック（CI相当）:

```bash
./scripts/check_quality.sh
```

Jupyter起動:

```bash
uv run jupyter notebook
```

### Kaggle API

Kaggle API を利用してデータ取得や提出を自動化できます。
事前に ~/.kaggle/kaggle.json を配置済みであることが前提です。

例: コンペ一覧表示

```bash
kaggle competitions list
```

---

## 品質チェック（CI相当）

ローカルでCIと同等のチェックを実行するためのスクリプトを用意しています。プッシュ前に必ず実行してください。

- スクリプト: `scripts/check_quality.sh`
- 実行内容:
  - Ruff: Lint/フォーマットチェック（`uv run ruff check .`）
  - Pyright: 型チェック（`uv run pyright`）
  - Pytest: ユニットテスト＋カバレッジ（`uv run pytest --cov=src --cov-report=term-missing`）

実行方法:

```bash
chmod +x scripts/check_quality.sh  # 初回のみ
./scripts/check_quality.sh
```

すべてのチェックが通らない場合は、コミットやプッシュ前に修正してください。

---

## コミットメッセージ規約

このプロジェクトでは、Conventional Commits に準拠したコミットメッセージを使用します。

公式: [Conventional Commits v1.0.0 日本語版](https://www.conventionalcommits.org/ja/v1.0.0/)

フォーマット:

```text
<type>(<scope>): <subject>
```

- type: 変更の種類
  - feat: 新機能
  - fix: バグ修正
  - docs: ドキュメント
  - style: コードスタイル変更（空白やフォーマットなど）
  - refactor: リファクタリング
  - test: テスト追加・修正
  - chore: ビルド・設定・依存関係更新
- scope: 影響範囲（任意、例: api, db, ui）
- subject: 簡潔な説明（小文字始まり、末尾ピリオドなし）

例:

```text
feat(api): add prediction endpoint
fix(db): handle null values in loader
docs(readme): add commit message guideline
style: run ruff formatting
refactor: extract feature engineering module
test(scripts): add unit tests for trainer
chore: bump dependencies and update lockfile
```

---

## データ運用ポリシー（EDA）

結論: 競技データは Kaggle API で取得し、一切 Git に入れません。`data/` 配下でローカル管理し、再現は「スクリプトで毎回ダウンロード」。Codespaces への手動アップロードは非推奨です。評価用ファイルも同様の扱いです。

### 最短手順（手動）

```bash
# 1) ディレクトリ設計
mkdir -p data/raw data/interim data/processed data/external artifacts

# 2) 競技データをKaggle APIで取得
kaggle competitions download -c hull-tactical-market-prediction -p data/raw

# 3) 展開
unzip -o data/raw/hull-tactical-market-prediction.zip -d data/raw
```

以後、常にこの手順で再現できます。アップデートが出ても同様に取り直せます。

### 自動化スクリプト（Git管理）

- `scripts/fetch_data.sh`

```bash
./scripts/fetch_data.sh
```

初回のみ実行権限付与:

```bash
chmod +x scripts/fetch_data.sh
```

ダウンロード先: `data/raw/`

### .gitignore

データと成果物はコミットしません（ノートとコードのみをPRに載せる）。

```gitignore
data/
artifacts/
```

補足: CI とローカルの品質チェックには誤コミット防止のガードがあり、`data/` や `artifacts/` 配下のファイルが Git で追跡されているとジョブが失敗します。

### notebooks からの参照例

```python
import pandas as pd

# 例: 実データのファイル名に合わせて置換してください
train = pd.read_parquet("data/raw/train.parquet")
test = pd.read_parquet("data/raw/test.parquet")
train.info(); train.head()
```

### 提出単位の運用方針（scripts / artifacts）

提出（Submit）単位で「コード」と「成果物」をディレクトリ分割します。

- コード: `scripts/<submit_name>/...`
  - 例: `scripts/simple_baseline/train_simple.py`, `scripts/simple_baseline/predict_simple.py`
- 成果物: `artifacts/<submit_name>/...`
  - 例: `artifacts/simple_baseline/model_simple.pkl`, `artifacts/simple_baseline/model_meta.json`, `artifacts/simple_baseline/cv_simple.json`

この方針により、複数の提出ライン（ベースライン/改良版/アブレーションなど）を並行管理できます。コードと成果物の対応が明確になり、切り替えが容易です。Git には引き続き成果物は含めません。

#### simple_baseline の注意点

- 依存: LightGBM を使用します。`uv sync` で自動導入されますが、環境によっては `uv add lightgbm` が必要な場合があります。
- 特徴量: 学習時に `forward_returns` / `risk_free_rate` / `market_forward_excess_returns` から shift(1) で `lagged_*` を生成し、元列は除外します。train/test の共通列のみを採用します。
- 進捗/ログ: 学習中に LightGBM の評価ログと進捗（iteration/%）を表示し、学習後はある程度のステップごとに RMSE を追試算してログ出力します。
- 再現性: 重要度が高い場合は `--seed` を LightGBM 側パラメータに追加する等で固定可能です（現状は `random_state=42` を指定）。
- 出力: 提出ファイルは `artifacts/simple_baseline/submission.parquet` と `artifacts/simple_baseline/submission.csv` に保存されます（CSVはデフォルト有効、`--no-csv` で抑止）。

### 評価用ファイルの扱い

競技ページの評価用（推論対象）も同じzipに含まれます。APIで取得した中身を `data/raw/` にそのまま保持します。ブランチには入れません。管理は「取得スクリプト＋READMEに手順記載」で十分です。

提出ノートブックでは `/kaggle/input/hull-tactical-market-prediction/...` を参照します。生データは自作 Dataset 化せず、規約上の再配布を避けます。

### ブランチ運用

- `eda/*` ブランチではノート・コードのみをコミット。`data/` はローカル専用。
- 有用な関数や前処理は `src/` へ昇格し、`dev` → `main` にマージ。
- 提出用の前処理・推論は学習済み成果物を Kaggle Private Dataset（自作）に載せ、入力は公式データを使用。

### Makefile（任意）

```makefile
.PHONY: data
data:
  ./scripts/fetch_data.sh
```

実行は `make data`。

---

## ブランチ戦略（最小実務形）

- main: 安定ブランチ。学習・推論が常に動く状態を維持します。
- dev: 統合用ブランチ。複数作業の一時的な受け皿。
- `eda/<topic>`: EDA用の短命ブランチ（例: `eda/target-dist`）。
- `feat/<feature>` / `exp/<model>`: 特徴量追加やモデル実験用の短命ブランチ。

ブランチは原則維持しません。長生きブランチは劣化（conflict・差分肥大・責務曖昧化）の温床になります。小さく早くマージし、不要になったら削除します。
