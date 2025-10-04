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
