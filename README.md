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
├─ scripts/ # 学習・推論などの実行スクリプト (train.py, predict.py)
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

スクリプト実行:

```bash
uv run python scripts/train.py
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
