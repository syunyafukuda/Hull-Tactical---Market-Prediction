# Hull-Tactical---Market-Prediction

<https://www.kaggle.com/competitions/hull-tactical-market-prediction>

このリポジトリは [Hull Tactical Market Prediction コンペティション](https://www.kaggle.com/competitions/hull-tactical-market-prediction) への参加用プロジェクトです。  
GitHub Codespaces を開発環境とし、パッケージ管理は **[uv](https://github.com/astral-sh/uv)** を利用しています。

---

## プロジェクト構成

```text
├─ .devcontainer/        # Codespaces 用のコンテナ設定。開発環境を再現するための定義ファイルが入っています。
├─ src/                  # プロジェクトのコアロジック。共通ユーティリティと前処理モジュールをここに集約します。
│  ├─ hull_tactical/     # Notebook やスクリプトから使い回すユーティリティ関数。
│  └─ preprocess/        # 特徴量グループごとの欠損補完・学習・推論スクリプトをまとめたモジュール。
│      ├─ M_group/       # M 系特徴量向けの補完ロジック・学習 CLI ・推論 CLI ・スイープツール。
│      └─ D_group/       # D 系特徴量向けの補完ロジック・学習 CLI ・推論 CLI ・スイープツール。
├─ scripts/              # 提出ラインやユーティリティを実行するエントリポイント。データ取得・品質チェック・提出用パイプラインを配置。
├─ results/              # アブレーションやスイープ結果など解析アウトプットを保存（Git 管理対象はメタ情報のみ）。
│  └─ ablation/          # 特徴量グループごとのポリシー比較結果（CSV/JSON）を格納。
├─ docs/                 # 提出ログや前処理メモ、EDA ノートなどプロジェクトドキュメント。
├─ notebooks/            # 探索的データ分析や実験用の Jupyter Notebook。
├─ tests/                # Pytest ベースの自動テスト。前処理モジュールやパイプラインの振る舞いを検証します。
├─ configs/              # 必要に応じて利用する設定ファイル（Hydra/YAML 等を想定）。
├─ data/                 # Kaggle から取得した生データ・中間データをローカルで保持（Git には含めない）。
├─ artifacts/            # 学習済みモデルや提出ファイルの書き出し先（Git には含めない）。
├─ main.py               # ワークスペース内で試験的に実行するメインスクリプト（雛形）。
├─ pyproject.toml        # uv による依存関係・ツール設定を管理。
├─ uv.lock               # 依存関係のロックファイル。環境再現用。
└─ README.md             # 本ドキュメント。
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

スクリプト実行（例: simple_baseline / MSR-proxy 提出ライン）:

```bash
# 依存を同期（初回/更新時）
uv sync

# データ取得（未取得の場合）
./scripts/fetch_data.sh

# 学習（成果物は artifacts/simple_baseline/ に保存）
uv run python scripts/simple_baseline/train_simple.py --data-dir data/raw

# 推論・提出生成（submission.parquet と submission.csv を artifacts/simple_baseline/ に出力）
uv run python scripts/simple_baseline/predict_simple.py --data-dir data/raw

# MSR-proxy ライン（MSR/vMSR プロキシ最適化）
# 学習（成果物は artifacts/MSR-proxy/ に保存）
uv run python scripts/MSR-proxy/train_msr_proxy.py --data-dir data/raw --out-dir artifacts/MSR-proxy

# 推論・提出生成（submission.parquet / submission.csv を出力）
uv run python scripts/MSR-proxy/predict_msr_proxy.py --data-dir data/raw --artifacts-dir artifacts/MSR-proxy
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

---

## ローカルから Kaggle ノートブックで Submit する再現手順

以下は simple_baseline を例に、ローカル学習→Kaggle Notebook 提出までの、再現可能な最短手順です。

1. ローカルで学習・成果物作成

- 依存同期: `uv sync`
- データ取得: `./scripts/fetch_data.sh`
- 学習実行: `uv run python scripts/simple_baseline/train_simple.py --data-dir data/raw`
  - 成果物が `artifacts/simple_baseline/` に出力されます:
    - `model_simple.pkl`
    - `model_meta.json`

1. Kaggle Private Dataset を作成（成果物と互換 Wheel を格納）

- 上記 2 つの成果物を 1 つの Dataset にまとめます（例: 名前を「simple-baseline」）。
- 互換性のため、scikit-learn の Wheel を同 Dataset に置くとオフラインでも確実にインストールできます（例: `scikit_learn-1.7.2-...whl`）。
  - Notebook 冒頭で `!pip install --no-index /kaggle/input/simple-baseline/<wheel>.whl` を実行。
  - 必要に応じて `lightgbm`, `joblib`, `pyarrow`, `pandas`, `numpy`, `polars` を `!pip install -q ...` で補います。

1. Notebook 実装（推論 API 形式）

- Dataset を Notebook にアタッチし、下記を実装します。
  - 成果物読込: `pipe = joblib.load('/kaggle/input/simple-baseline/model_simple.pkl')`
  - メタ読込: `meta = json.load(open('/kaggle/input/simple-baseline/model_meta.json'))`
  - 特徴量整形: 学習時の `meta["feature_columns"]` を基準に「列順」「欠損補完（数値=NaN, カテゴリ='missing'）」を再現
  - 評価 API: `predict(test: pl.DataFrame) -> float` を定義し、`DefaultInferenceServer(predict)` で起動
    - 採点はこの関数の戻り値（単一 float）で行われます
  - 任意のローカル検証セル: `submission.parquet` / `submission.csv` を書き出して形式を確認（採点時には使われません）

1. Submit（インターネット OFF 推奨）

- Notebook を実行・Complete し、そのまま Submit します。
- `predict()` の戻り値仕様を満たしていれば、採点環境で自動評価されます。

注意点（落とし穴）

- 必須: `predict(test: pl.DataFrame) -> float` と `DefaultInferenceServer` のセル。
- 戻り値はスカラ float。配列や DataFrame を返すとエラーになります。
- 特徴量整形はローカル推論と同一に。`meta["feature_columns"]` を基準に列を揃え、欠損補完の型（数値/カテゴリ）を一致させます。
- scikit-learn などライブラリのバージョン差異に注意。Wheel を Dataset に同梱し `--no-index` でその版を入れると安全です。
- `submission.parquet`/`csv` の生成は任意（Notebook では API 呼び出しがスコアリングの本体）。ただし、形式検証には有用です。

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

---

## テスト方針（Submitラインを増やす場合）

結論: 各Submitラインで `scripts/<experiment>/` を増やしてOK。ただしテストは「共通インターフェース」を対象に統一します。

- 共通インターフェース（例）
  - 学習: `train_model(data_dir: str | None, target_col: str, id_col: str, sample_rows: int | None = 500) -> Pipeline`
  - 推論: `generate_submission(model, test_df: pd.DataFrame, id_col: str = "date_id", pred_col: str = "prediction", meta: dict | None = None) -> pd.DataFrame`
  - これらを各ラインの `train_simple.py` / `predict_simple.py` に実装（軽量版でOK）

- 最小テスト例
  - `tests/test_pipeline_simple_baseline.py` は simple_baseline の軽量APIを使って、
    - ダミーデータで学習が走るか
    - 提出形式（id, prediction列、行数一致）が満たされるか
    を検証します。

- 追加ラインの流れ
  1) `scripts/<new_exp>/train_simple.py` と `predict_simple.py` を作成
  2) 上記の共通APIを実装
  3) tests に `<new_exp>` 用の軽量テストを1本追加（もしくは共通テストに追加）
  4) `./scripts/check_quality.sh` でCI相当チェック

- 実行時間の工夫
  - 重い学習はCIで回さず、`@pytest.mark.slow` で分離
  - 学習APIに `sample_rows` を用意し、CIでは小規模でのみ動作確認

---

### Kaggle Notebook 依存固定（MSR-proxy 提出ラインの補足）

MSR-proxy の成果物は、scikit-learn のバージョン依存で `joblib.load` 振る舞いが変わる場合があります。Notebook では Private Dataset に 1.7.2 の wheel を同梱し、冒頭で次のように導入してください。

- scikit-learn 1.7.2 を `--no-index --no-deps --force-reinstall` でインストール
- その後に `lightgbm==4.6.0`, `joblib`, `pyarrow`, `polars`, `pandas` をインストール

詳細は `docs/submissions.md` の 2025-10-12 エントリに Notebook のサンプル断片を記載しています。
