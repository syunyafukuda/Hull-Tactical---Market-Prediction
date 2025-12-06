# Hull-Tactical---Market-Prediction

<https://www.kaggle.com/competitions/hull-tactical-market-prediction>

このリポジトリは [Hull Tactical Market Prediction コンペティション](https://www.kaggle.com/competitions/hull-tactical-market-prediction) への参加用プロジェクトです。  
GitHub Codespaces を開発環境とし、パッケージ管理は **[uv](https://github.com/astral-sh/uv)** を利用しています。

---

## プロジェクト構成

```text
├─ .devcontainer/                 # Codespaces 用開発コンテナ設定。
├─ src/                           # コアロジック（共通ユーティリティ + 前処理パイプライン）。
│  ├─ hull_tactical/              # Notebook や提出ラインから再利用する共通ユーティリティ。
│  ├─ feature_generation/         # 特徴量生成モジュール（SU1〜SU9）。
│  │  ├─ su1/                     # SU1（欠損構造一次特徴） - **採用** (LB 0.674)
│  │  ├─ su2/                     # SU2（二次欠損特徴スイープ） - **非採用** (LB 0.597, 過学習で却下)
│  │  ├─ su3/                     # SU3（遷移・再出現・時間バイアス） - **非採用** (LB 0.461, コンセプト不適合)
│  │  ├─ su4/                     # SU4（補完トレース） - **実装済みだが削除決定**（寄与ほぼゼロ）
│  │  ├─ su5/                     # SU5（共欠損構造特徴） - **現行ベストライン** (LB 0.681)
│  │  ├─ su6/                     # SU6（欠損軸PCA圧縮候補） - **保留中 / 未実装**
│  │  ├─ su7/                     # SU7（モメンタム・リバーサル特徴） - **非採用** (OOF改善もLB大幅悪化)
│  │  ├─ su8/                     # SU8（ボラティリティ・レジーム特徴） - **非採用** (LB 0.624, OOF/LB両方で悪化)
│  │  ├─ su9/                     # SU9（カレンダー・季節性特徴） - **非採用** (LB 0.679, OOF改善もLB微悪化)
│  │  └─ su10/                    # SU10（外部レジーム特徴: SPY） - **非採用** (LB 0.597, -12.3%大幅悪化)
│  └─ preprocess/                 # 特徴量グループ別に欠損補完・学習・推論を実装。
│      ├─ E_group/                # E 系特徴量向けパイプライン（train/predict/sweep）。
│      ├─ I_group/                # I 系特徴量向けパイプライン。
│      ├─ M_group/                # M 系特徴量向けパイプライン。
│      ├─ P_group/                # P 系特徴量向けパイプライン。
│      ├─ S_group/                # S 系特徴量向けパイプライン（2025-10-26 missforest 採択）。
│      └─ V_group/                # V 系特徴量向けパイプライン（2025-11-01 ffill_bfill/holiday_bridge を検証済み、現行ラインでは未採用）。
├─ scripts/                       # 提出ラインやユーティリティの CLI。`simple_baseline/`, `MSR-proxy/`, S/M/E/I/P グループ共通の補助スクリプトを収録。
├─ results/
│  └─ ablation/                   # グループ別スイープ結果（CSV/JSON）。`E_group/`, `I_group/`, `M_group/`, `P_group/`, `S_group/` の比較ログが入る。
├─ artifacts/                     # 学習済み成果物の書き出し先。`Preprocessing_<group>/` 以下にモデル・メタデータ・submission を置く（Git 管理外）。
├─ docs/                          # `preprocessing.md`, `submissions.md` などの運用ドキュメント。
├─ configs/
│  └─ preprocess.yaml             # 各グループの採択ポリシー・ハイパーパラメータ設定を集約。
├─ notebooks/                     # Kaggle Private Notebook と同期する検証ノート・EDA ノート。
├─ tests/                         # Pytest による単体/統合テスト。`tests/preprocess/<group>/` で各ポリシーを検証。
├─ data/                          # Kaggle 公式データの配置場所（Git 管理外）。`raw/`, `interim/`, `processed/` などを区分。
├─ main.py                        # ワークスペース用エントリポイント（雛形）。
├─ pyproject.toml                 # uv による依存・各種ツール設定。
├─ uv.lock                        # 依存関係のロックファイル。
└─ README.md                      # 本ドキュメント。
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

### アーティファクトの一時配布（HTTP サーバ）

スイープ後の `model_pre_*.pkl` など容量の大きい成果物を Codespaces からローカルへダウンロードしたい場合、Python の簡易 HTTP サーバを利用すると再現性が高いです。

1. 共有したい成果物ディレクトリに移動

  ```bash
  cd /workspaces/Hull-Tactical---Market-Prediction/artifacts/Preprocessing_S/missforest
  ```

2. 一時 HTTP サーバを起動（他のポリシーでもディレクトリを切り替えるだけで可）

  ```bash
  python -m http.server 8000
  ```

  起動するとターミナルに `Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...` が表示されます。Codespaces の場合は Port Forwarding で 8000 番を開放し、ブラウザから `https://<codespace-url>-8000.app.github.dev/` にアクセスするとファイル一覧が表示され、クリックでダウンロードできます。

3. 転送が完了したら `Ctrl+C` でサーバを終了してください。必要に応じて `python -m http.server <port>` でポート番号を変えることも可能です。

この手順を README に記載しておくことで、今後も同じ操作で成果物の取得が再現できます。

---

### numpy バージョン固定ポリシー

- 依存管理は `uv` で統一し、`numpy==1.26.4` を必須バージョンとします。
- 理由: Kaggle（Python 3.11, numpy 1.26.4）と整合させ、`joblib` が生成する `MT19937` BitGenerator を安全にロードするため。異なる numpy 版で学習した pickle は Kaggle 上でロードに失敗します。
- 運用: 依存を追加・更新する際は `pyproject.toml` の numpy 行を変更せず、`uv lock --python 3.11` → `uv sync --python 3.11` でロックファイルを再生成します。
- 手元で numpy バージョンを変更した場合は、SU 系成果物・前処理バンドルを全て再生成する必要があります。原則として numpy のバージョン変更は禁止です。

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

- D 系特徴量（列プレフィクス `D`）は train/test ともに欠損が確認されておらず、追加の補完処理は不要です。パイプラインでは監視ログのみで十分です。

#### SU5 (共欠損構造特徴) の注意点

- 依存: SU1をベースとし、共欠損ペアの特徴量（co_miss_now, co_miss_rollrate, co_miss_deg）を追加します。
- 成果物: `inference_bundle.pkl` (2.1GB)、`model_meta.json`、`feature_list.json`
- Kaggle Dataset: sklearn 1.7.2 wheel を同梱し、`--no-index` で導入します。
- 推論: Notebookは自動でartifactを検出します（`/kaggle/input/<dataset>/inference_bundle.pkl`）。
- 互換性: numpy 1.26.4 で学習・推論を統一（MT19937 BitGenerator互換性）。

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

## 現在のベストスコアと SU ライン採否状況

| ライン / Submit 単位 | LB Score (Public) | 特徴量数 | OOF RMSE | 状態 |
|----------------------|-------------------|---------|----------|------|
| **SU5 (Policy1)**    | **0.681**         | 567 (94+368+105) | 0.012139 | **✅ 採用（現行ベスト）** |
| SU1                  | 0.674             | 462 (94+368)     | 0.012120 | ベースライン（採用） |
| SU7 (case_c/case_d)  | 0.476 / 0.469     | +SU7 96〜144列   | ≈0.01205 | ❌ 非採用（OOF改善も LB 大幅悪化） |
| SU8                  | 0.624             | 567+11 (SU5+SU8) | 0.012230 | ❌ 非採用（OOFでもLBでも悪化） |
| SU2                  | 0.597             | 1397 (94+368+935)| 0.012230 | ❌ 非採用（過学習） |
| SU3                  | 0.461             | 444 (368+76)     | 0.011418 | ❌ 非採用（コンセプト不適合） |
| Preprocessing P (mice) | 0.625           | -                | -        | 採用 |
| Preprocessing I (ridge_stack) | 0.623   | -                | -        | 採用 |
| Preprocessing M      | 0.629             | -                | -        | 採用 |
| MSR-proxy            | 0.610             | -                | 0.012410 | 参考ライン |
| simple_baseline      | 0.554             | -                | -        | 初期ベースライン |

- **現在のベストスコア**: SU5 Policy1 (LB 0.681) - SU1 から +0.007 改善。
- SU7 は OOF RMSE では SU1+SU5 より改善しているものの、Public LB では 0.47 台まで大きく悪化したため、
  広義の過学習 / レジームミスマッチと判断し **非採用** としています。
- SU8 は OOF の時点で SU5 ベースラインより悪化しており、Public LB でも 0.681 → 0.624 と
  明確に劣化したため **非採用** としています。ボラティリティ/レジーム軸の特徴は本コンペでは有効でありませんでした。

より詳細なスコア推移や判断理由は `docs/submissions.md` と
各 SU の仕様書（例: `docs/feature_generation/SU7.md`, `docs/feature_generation/SU8.md`）を参照してください。

---

### SU5 Implementation (2025-11-23) ✅ 正式採用

**採用構成**: Policy1 (top_k=10, windows=[5])

- LB Score: **0.681** (従来最高: SU1の0.674から+0.007改善)
- 特徴量: 共欠損構造特徴105列（co_miss_now, co_miss_rollrate, co_miss_deg）
- OOF RMSE: 0.012139
- Artifacts: `artifacts/SU5/policy1_top10_w5/`

詳細: `docs/submissions.md`

---

### SU8 Implementation (2025-12-04) ❌ 非採用

**目的**: ボラティリティ・レジーム特徴（ewmstd, vol_ratio, vol_level, vol/trend regime tags, ret_vol_adj）を付加し、市場モードをモデルに渡す。

**パイプライン位置**: 生データ → SU1 → SU5 → GroupImputers → **SU8** → 前処理 → LGBM

- LB Score: **0.624** (SU5ベースライン0.681から **-0.057ポイント悪化**)
- 特徴量: 11列
  - ボラティリティ指標 (4列): `ewmstd_short`, `ewmstd_long`, `vol_ratio`, `vol_level`
  - ボラレジームタグ (3列): `vol_regime_low`, `vol_regime_mid`, `vol_regime_high`
  - トレンドレジームタグ (3列): `trend_regime_down`, `trend_regime_flat`, `trend_regime_up`
  - ボラ調整リターン (1列): `ret_vol_adj`
- OOF RMSE: 0.012230 (SU5: 0.012139から+0.00009悪化)
- Artifacts: `artifacts/SU8/`
- 実装: `src/feature_generation/su8/` (feature_su8.py, train_su8.py, predict_su8.py)

**非採用理由**:
- OOFの時点でSU5ベースラインよりわずかに悪化していた
- Public LBでも明確に悪化（0.681 → 0.624）
- ボラティリティ/レジーム軸の特徴がPublic評価期間で有効でなかった

**コンフィグ**: `configs/feature_generation.yaml` の `su8.enabled: false` に設定済み。コード・アーティファクトは将来の別レジーム検証用に保持。

詳細: `docs/feature_generation/SU8.md`, `docs/submissions.md`

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

