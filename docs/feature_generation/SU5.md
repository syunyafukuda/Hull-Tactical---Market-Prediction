# SU5: Co-Missing Structure Feature Generation

## 概要

SU5 (Structure Unit 5) は、データ中の**共欠損（co-missing）構造**を特徴量として抽出するモジュールです。「どの列ペアが一緒に欠けがちか」「どの列が共欠損ネットワークのハブか」を過去情報のみから算出し、欠損パターンの横方向の関係性を学習モデルに提供します。

## 目的

- SU1で生成された欠損インジケータ（`m/<col>`）を入力として、列間の共欠損パターンを分析
- Top-k ペア選択により特徴量爆発を防止
- SU2/SU3の教訓を活かし、解釈可能で制御可能な特徴量設計

## 機能

SU5は以下の3種類の特徴量を生成します:

### カテゴリA: 単日共欠損フラグ (`co_miss_now`)

```
co_miss_now/<col_a>__<col_b>: uint8
```

- 現在の行で列Aと列Bが同時に欠損しているかをフラグで表現
- 値: 0 (どちらかが観測) または 1 (両方とも欠損)

### カテゴリB: ローリング共欠損率 (`co_miss_rollrate_W`)

```
co_miss_rollrate_5/<col_a>__<col_b>: float32
co_miss_rollrate_20/<col_a>__<col_b>: float32
```

- 過去Wウィンドウ内での共欠損発生率
- ウィンドウサイズは設定ファイルで指定（デフォルト: [5, 20]）
- fold境界でのリセット対応（`reset_each_fold: true`）

### カテゴリC: 列ごとの共欠損度 (`co_miss_deg`)

```
co_miss_deg/<col>: int16
```

- その列が Top-k ペアに何回登場するか（共欠損ネットワークでの次数）
- 値が大きいほど、他の多くの列と共欠損しやすい

## 設定

設定ファイル: `configs/feature_generation.yaml`

```yaml
su5:
  enabled: false  # 特徴量生成の有効/無効
  base_features: su1  # 依存する特徴量セット
  id_column: date_id  # 時系列ID列
  output_prefix: su5  # 出力プレフィクス

  # ペア選択
  top_k_pairs: 10  # 上位K個の共欠損ペアを選択
  top_k_pairs_per_group: null  # グループ単位のK（未実装）

  # ローリング共欠損率
  windows: [5, 20]  # ウィンドウサイズのリスト

  # fold 境界リセット
  reset_each_fold: true  # CV fold境界でローリング統計をリセット

  # データ型
  dtype:
    flag: uint8  # フラグ型
    int: int16   # 整数型
    float: float32  # 浮動小数点型

  metadata:
    artifacts_dir: artifacts/SU5
    depends_on: SU1
    expected_usage: "Co-missing structure features (top-K pairs, rolling co-miss rate, degree)"
    numpy_version: 1.26.4
    status: development
    updated_at: 2025-11-22
```

## 使用方法

### 基本的な使い方

```python
from feature_generation.su5 import SU5Config, SU5FeatureGenerator, load_su5_config

# 設定ファイルから読み込み
config = load_su5_config("configs/feature_generation.yaml")

# または直接設定
config = SU5Config(
    enabled=True,
    top_k_pairs=10,
    windows=[5, 20],
    reset_each_fold=True
)

# 特徴量生成器の作成
generator = SU5FeatureGenerator(config)

# 学習データでfitしてtop-kペアを選択
generator.fit(train_df)  # train_df には m/<col> 列が必要

# 特徴量を生成
train_features = generator.transform(train_df, fold_indices=fold_ids)
test_features = generator.transform(test_df)
```

### sklearn パイプラインでの使用

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

pipeline = Pipeline([
    ('su5', SU5FeatureGenerator(config)),
    ('scaler', StandardScaler()),
    ('model', LGBMRegressor())
])

pipeline.fit(train_df, y_train)
predictions = pipeline.predict(test_df)
```

## 特徴量選択アルゴリズム

### Top-K ペア選択

1. 学習データで全ペアの共欠損回数を集計
2. スコア降順でソート
3. 上位K個のペアを選択して `self.top_pairs_` に保存

```python
# 疑似コード
scores = {}
for col_a, col_b in all_pairs:
    co_miss_count = sum((m[col_a] == 1) & (m[col_b] == 1))
    scores[(col_a, col_b)] = co_miss_count

top_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

## 未来参照の防止

- 共欠損スコアの計算: 同時刻の `m/<col>` を使用（同時刻は許可）
- ローリング統計: 過去ウィンドウのみ使用
- fold境界リセット: `reset_each_fold=True` で前のfoldの情報を引き継がない

## パフォーマンス

### 特徴量数

- Stage 1 (ミニマル版): 50〜80列
  - `top_k_pairs=10`, `windows=[5, 20]` の場合:
    - co_miss_now: 10列
    - co_miss_rollrate: 10 × 2 = 20列
    - co_miss_deg: 最大20列（ペアに登場する列数）
    - 合計: 約50列

- Stage 2 (拡張版): 最大+100列
  - `top_k_pairs=20`, `windows=[5, 10, 20]` の場合: 約100列

### 計算量

- Fit時: O(n_cols² × n_rows) - 全ペアの共欠損スコア計算
- Transform時: O(k × n_rows × w) - k個のペアについてウィンドウサイズwのローリング計算

## 制約事項

1. **SU1依存**: SU1で生成された `m/<col>` 列が必須
2. **メモリ使用量**: 全ペアのスコア計算時に n_cols² のメモリが必要
3. **型制約**: 生成される特徴量は uint8, int16, float32 に固定

## テスト

テストスイート: `tests/feature_generation/test_su5.py`

### テストケース

1. **test_su5_config_loading**: YAML設定の読み込み確認
2. **test_su5_all_observed_columns**: 全て観測の場合（共欠損なし）
3. **test_su5_all_nan_columns**: 全て欠損の場合（最大共欠損）
4. **test_su5_single_co_miss_pair**: 特定ペアのみ共欠損
5. **test_su5_fold_reset**: fold境界でのリセット動作
6. **test_su5_output_shape**: 出力列数の確認
7. **test_su5_dtype**: データ型の確認
8. **test_su5_no_missing_columns**: `m/<col>` 列がない場合
9. **test_su5_config_loading_missing_file**: 設定ファイル不在時のエラー

### テスト実行

```bash
# 全テスト実行
python -m pytest tests/feature_generation/test_su5.py -v

# カバレッジ付き
python -m pytest tests/feature_generation/test_su5.py --cov=src/feature_generation/su5

# 単体テスト実行
python -m pytest tests/feature_generation/test_su5.py::test_su5_fold_reset -v
```

## 実装の詳細

### クラス構造

```
SU5Config
  ├─ enabled: bool
  ├─ top_k_pairs: int
  ├─ windows: list[int]
  ├─ reset_each_fold: bool
  └─ dtype_*: str

SU5FeatureGenerator(BaseEstimator, TransformerMixin)
  ├─ fit(X, y=None)
  │   └─ 全ペアの共欠損スコアを計算してtop-k選択
  ├─ transform(X, fold_indices=None)
  │   ├─ カテゴリA: co_miss_now 生成
  │   ├─ カテゴリB: co_miss_rollrate 生成
  │   └─ カテゴリC: co_miss_deg 生成
  └─ _compute_comiss_scores(m_df)
      └─ ペアごとの共欠損回数を集計
```

### 内部メソッド

- `_compute_comiss_scores()`: 全ペアの共欠損スコア計算
- `_select_top_k_pairs()`: スコア降順でtop-k選択
- `_compute_rolling_comiss_rate()`: ローリング共欠損率計算
- `_rolling_with_fold_reset()`: fold境界対応のローリング計算
- `_compute_fold_boundaries()`: fold境界の特定
- `_build_feature_names()`: 生成される特徴量名のリスト構築

## 既知の制限

1. **大規模データへの対応**: 列数が1000を超えるとfit時のメモリ使用量が増大
   - 対策: グループ単位のペア選択（`top_k_pairs_per_group`）の実装を検討

2. **グループ対応**: 現在は全体から top-k を選択するのみ
   - 将来的に M/E/I/P/S グループごとの top-k も検討

3. **動的ウィンドウ**: 現在は固定ウィンドウのみ
   - 将来的に expanding window や exponential weighted average も検討

## 開発履歴

### 2025-11-22: Initial Implementation (v0.1.0)

- 基本的な SU5 モジュールの実装
- Top-k ペア選択機能
- 3種類の特徴量生成（co_miss_now, co_miss_rollrate, co_miss_deg）
- fold境界リセット対応
- 包括的なテストスイート (9 test cases, 99% coverage)
- 型チェック完全対応 (pyright 0 errors)

## 参考文献

- SU1 仕様書: 欠損構造一次特徴（基底特徴）
- SU3 仕様書: 時系列欠損パターン（最終的には非採用）
- プロジェクトロードマップ: `docs/feature_generation/README.md`

## 今後の拡張予定

1. **Stage 2 拡張**
   - グループ単位のペア選択
   - より多様なウィンドウサイズ
   - 3次以上の共欠損パターン（トリプレット等）

2. **性能最適化**
   - Numba/Cython による高速化
   - メモリ効率の改善
   - 並列計算対応

3. **分析ツール**
   - 共欠損ネットワークの可視化
   - ペア重要度のランキング
   - グループ間の共欠損パターン分析
