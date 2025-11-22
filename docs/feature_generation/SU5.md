# 欠損構造特徴 SU5 仕様（共欠損・co-miss 構造）

最終更新: 2025-11-22

## 実装ステータス（計画）

**Status**: 📝 設計完了・実装これから（このドキュメント）

- 📝 `feature_su5.py`: Core feature generation logic をこれから実装
- 📝 `SU5Config`: 設定用 dataclass を定義予定
- 📝 `SU5FeatureGenerator`: sklearn-compatible transformer
- 📝 `SU5FeatureAugmenter`: SU1 パイプラインへの統合クラス
- 📝 Unit tests: `tests/feature_generation/test_su5.py` を新規追加
- 📝 Quality checks: Ruff + Pyright + Pytest を SU1/SU3 と同水準で通す
- 🔒 Configuration: `configs/feature_generation.yaml` に `su5` セクション追加（初期状態では `enabled: false`）

この仕様書は、SU3（遷移・再出現）実装のパターンを踏襲しつつ、「共欠損（co-miss）構造」を系統的に扱うための設計ドキュメントです。実装者が迷わずコードを書けること、後から見ても設計意図がわかることを目的とします。

---

## 1. SU5 の目的と位置づけ

### 1.1 SU シリーズの中での SU5

- **SU1**: 単列レベルの欠損構造（`m/<col>`, `gap_ffill/<col>`, `run_na/<col>` など）
- **SU2**: 欠損率のローリング統計・履歴率（→ 過学習により非採用）
- **SU3**: 欠損の遷移・再出現パターン（→ OOF/LB 劣化により完全放棄）
- **SU4**: 代入影響トレース（imputation trace, 未実装）
- **SU5**: **共欠損（co-miss）構造** — 「どの列とどの列が一緒に欠けがちか」を特徴量化

SU5 は、SU1 が扱う「単列ごとの欠損構造」を横方向（列間の関係）に拡張するユニットです。欠損が強く関連している列ペア・列グループを明示的に捉え、

- 「この 2 列はいつも一緒に欠ける」
- 「この列は特定グループと共欠損クラスタを形成している」

といった構造をモデルに渡すことを目的とします。

### 1.2 コンペ全体の文脈での狙い

- これまでの前処理フェーズで、M/E/I/P/S/V 各グループに対して「単列ごとの欠損補完」方針は固まっている。
- SU1 により「どこで・どれだけ欠けているか」はある程度捉えられているが、「**どの列どうしが共通の欠損パターンを持っているか**」はまだ特徴として明示されていない。
- 市場の regime・カレンダー・イベント要因により、一部の指標群がまとめて落ちる（観測されない）日があると仮定すると、その **共欠損クラスタ** 自体が有用なシグナルになりうる。
- SU5 では、この共欠損構造を
	- 列ペア単位のフラグ
	- 群内 aggregate
	- PCA などの圧縮の前段
として設計する。

**評価方針**:

- あくまで SU1 ベースライン（LB 0.674）を守るのが最優先。
- SU3 の反省から「列数の爆発」「時系列過適合」を避けるため、**強い列数キャップと top-k 選択**を前提とする。
- OOF MSR で +0.5σ 程度の改善、あるいは LB で明確な改善が見える場合のみ採用候補とする。

---

## 2. 設計方針

### 2.1 入出力と依存関係

- **入力**
	- SU1 で生成された欠損フラグ: `m/<col>`
	- 必要に応じて group 情報（列名プレフィクス `M`, `E`, `I`, `P`, `S`, `V`）

- **出力（SU5 特徴の例）**
	- 列ペア単位の共欠損フラグ・共欠損率
		- `co_miss_now/<a>__<b>`: 当日 a と b が同時に NaN かどうか (0/1)
		- `co_miss_rollrate_W/<a>__<b>`: 過去 W 日での同時 NaN 比
	- 群内・全体の共欠損強度
		- `co_miss_deg/<col>`: その列が「よく一緒に欠ける相手列」を何本持っているか
		- `co_miss_cluster_strength/<grp>`: グループ単位の共欠損強度

- **依存関係**
	- SU1 の `m/<col>` を前提とする（SU1 なしの SU5 は実装しない）。
	- 前処理（M/E/I/P/S グループ imputer）やモデル本体には依存しない。

### 2.2 列数制御と top-k 方針

共欠損ペアは組み合わせ数が爆発しやすいため、**列数を強く制御する**:

- まず学習期間における「共欠損度」を集計し、上位ペアのみを採用：
	- 指標例: `co_miss_rate(a,b) = P(m_a==1 & m_b==1) / P(m_a==1 or m_b==1)` のような Jaccard 風のスコア
	- あるいは単純に `P(m_a==1 & m_b==1)`（同時 NaN 比）
- 上位 K ペアのみを SU5 特徴として採用する:
	- `top_k_pairs`（グローバル）または `top_k_pairs_per_group` パラメータで制御
	- 目安: K=10〜20 程度（初期は K=10 から）

### 2.3 リーク防止と CV 一貫性

- 共欠損指標は **行方向** の同時 NaN に基づくため、時刻 $t$ の特徴は同時刻 $t$ の `m` のみを参照し、未来情報は使わない。
- ローリング共欠損率 `co_miss_rollrate_W` を計算する場合は、過去のみを対象とした `rolling(window=W, min_periods=W)` で実装する（`shift(1)` は不要: 当日分は含めてよいが未来は見ない）。
- SU2/SU3 と異なり、fold ごとに状態を持つ必要は限定的だが、再現性のために `fold_indices` を受け取り、
	- ローリング統計を fold 境界でリセットするかどうかをオプション化する（`reset_each_fold`）。

---

## 3. 生成する特徴の詳細仕様

### カテゴリ A: 列ペア単位の共欠損フラグ

#### A-1. 単日共欠損フラグ `co_miss_now/<a>__<b>`

- 対象: 選択された列ペア `<a>, <b>`（プレフィクス・グループを跨いでもよいが、初期は同一グループ内を優先）
- 定義:
	- `co_miss_now/<a>__<b>[t] = 1` if `m/<a>[t] == 1 and m/<b>[t] == 1`
	- それ以外は 0
- 型: `uint8`
- 範囲: {0, 1}
- 意図: 「その日、この 2 列が同時に欠けているか」を直接表す最小単位の共欠損情報。

#### A-2. 群内共欠損カウント `co_miss_cnt_day/<grp>`（オプション）

- 対象: 各グループ `<grp> ∈ {M,E,I,P,S,V}`
- 定義例:
	- `co_miss_cnt_day/<grp>[t] =` その日の `<grp>` 内で `m==1` の列本数
- 型: `int16`
- 範囲: 0〜列数
- 意図: 「そのグループ全体がどれくらい死んでいる日か」を粗く測る。

**特徴量数（目安）**:

- ペアフラグ: `top_k_pairs` 個
- 群内カウント: 最大 6 列

---

### カテゴリ B: ローリング共欠損率 `co_miss_rollrate_W/<a>__<b>`

#### B-1. 過去 W 日での同時 NaN 比

- 対象: カテゴリ A で選択した列ペア `<a>, <b>`
- 定義:
	- 各ペアに対して、`co_miss_now/<a>__<b>` のローリング平均を計算:
	- `co_miss_rollrate_W/<a>__<b>[t] = mean(co_miss_now/<a>__<b>[t-W+1..t])`
- パラメータ:
	- `windows: List[int] = [5, 20]` など（初期は 1〜2 種類に絞る）
	- `min_periods = W`（窓不足時は NaN）
- 型: `float32`
- 範囲: [0, 1]
- 意図: 「このペアが最近どれくらい頻繁に一緒に欠けているか」を平滑化して捉える。

**特徴量数（目安）**:

- `top_k_pairs * len(windows)`
	- 例: `top_k_pairs=10`, `windows=[5,20]` → 20 列

---

### カテゴリ C: 列ごとの共欠損“次数” `co_miss_deg/<col>`

#### C-1. 共欠損次数（degree）

- 対象: SU1 で扱う各列 `<col>`
- 定義例（学習期間で計算）:
	- `co_miss_deg/<col> =` その列が「共欠損上位 K ペア」に何回現れるか
	- あるいは、`co_miss_rate(a,b)` を合計・平均した値
- 型: `float32` または `int16`
- 意図: 「この列はどれくらい他列と共通運命を持っているか」を 1 本のスカラーで表す。

#### C-2. グループ単位の共欠損強度 `co_miss_cluster_strength/<grp>`（オプション）

- 対象: 各グループ `<grp>`
- 定義例:
	- 該当グループ内ペアの `co_miss_rate` を平均 or 最大したもの
- 型: `float32`
- 意図: 「このグループは構造的な共欠損をどれくらい抱えているか」を粗く把握。

**特徴量数（目安）**:

- `co_miss_deg/<col>`: 列数（最大 90〜100）
- `co_miss_cluster_strength/<grp>`: 最大 6 列

必要に応じて、上位 K 列（`deg` が大きい列のみ）に絞る。

---

## 4. 列ペア選択（top-k）のアルゴリズム

### 4.1 共欠損スコアの定義

学習期間（train 全体、あるいは各 fold の train 部分）において、全列ペア `(a,b)` に対し共欠損スコアを計算する。

候補となるスコア例:

1. **同時 NaN 比**
	 - `score(a,b) = P(m_a==1 & m_b==1)`
2. **Jaccard 類似度風スコア**
	 - `score(a,b) = P(m_a==1 & m_b==1) / P(m_a==1 or m_b==1)`
3. **条件付き確率の対称化**
	 - `score(a,b) = 0.5 * (P(m_b==1 | m_a==1) + P(m_a==1 | m_b==1))`

初期実装では 1. の単純な同時 NaN 比を使い、必要であれば 2. を検討する。

### 4.2 計算コストと制約

- 列数 n=O(100) の場合、全ペアは n(n-1)/2 ≒ 5,000 個であり、train 全期間でのカウント計算は十分現実的。
- 各ペアに対し、
	- `count_both_na`, `count_either_na` 程度をカウントすればスコアを計算できる。

### 4.3 top-k 選択

- グローバルに上位 `top_k_pairs` を選ぶか、グループ単位に上位 `top_k_pairs_per_group` を選ぶ実装を用意する。
- 初期案:
	- `top_k_pairs_global = 10`
	- or `top_k_pairs_per_group = 5`（×グループ数）
- 設定は `SU5Config` に持たせ、`configs/feature_generation.yaml` から注入する。

---

## 5. クラス設計（`feature_su5.py`）

### 5.1 SU5Config

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SU5Config:
		"""SU5（共欠損構造）特徴生成の設定"""

		id_column: str = "date_id"
		output_prefix: str = "su5"

		# ペア選択
		top_k_pairs: int = 10
		top_k_pairs_per_group: Optional[int] = None  # どちらか一方を使用

		# ローリング共欠損率
		windows: List[int] = (5, 20)

		# fold 境界でローリング統計をリセットするか
		reset_each_fold: bool = True

		# 型
		dtype_flag: str = "uint8"
		dtype_int: str = "int16"
		dtype_float: str = "float32"
```

### 5.2 SU5FeatureGenerator

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SU5FeatureGenerator(BaseEstimator, TransformerMixin):
		"""SU5 共欠損特徴量生成器

		入力: SU1 で生成された `m/<col>` 列を含む DataFrame
		出力: co-miss フラグ・ローリング共欠損率・degree などの特徴
		"""

		def __init__(self, config: SU5Config):
				self.config = config

		def fit(self, X: pd.DataFrame, y=None) -> "SU5FeatureGenerator":
				# 1. SU1 の m 列抽出
				self.m_columns_ = [c for c in X.columns if c.startswith("m/")]

				# 2. 列プレフィクスからグループを抽出（M/E/I/P/S/V）
				self.groups_ = self._extract_groups(self.m_columns_)

				# 3. 学習期間で共欠損スコアを集計し、top-k ペアを決定
				self.top_pairs_ = self._select_top_k_pairs(X)

				# 4. feature_names_ を組み立て（transform 出力列の順序を固定）
				self.feature_names_ = self._build_feature_names()

				return self

		def transform(self, X: pd.DataFrame, fold_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
				n = len(X)
				features = {}

				# fold 境界決定（SU3 と同じロジックを再利用）
				boundaries = self._compute_fold_boundaries(n, fold_indices)

				# A. 単日共欠損フラグ
				co_now = self._compute_co_miss_now(X, boundaries)
				features.update(co_now)

				# B. ローリング共欠損率
				co_roll = self._compute_co_miss_rollrate(features, boundaries)
				features.update(co_roll)

				# C. degree / cluster 強度（必要なら）
				#   → fit 時に precompute しておき、全行同じ値を返すか、列ごとに系列を作る

				return pd.DataFrame(features, index=X.index)

		# 以降、_extract_groups, _select_top_k_pairs, _compute_fold_boundaries,
		# _compute_co_miss_now, _compute_co_miss_rollrate などのヘルパを実装
```

### 5.3 SU5FeatureAugmenter

SU1 と同様に、「生データ → SU1 → SU5 → 前処理 → モデル」を 1 本の `Pipeline` に載せるためのラッパーを用意する:

```python
class SU5FeatureAugmenter(BaseEstimator, TransformerMixin):
		"""生データに対して SU1+SU5 特徴を付与する augmenter"""

		def __init__(self, su1_config: SU1Config, su5_config: SU5Config, fill_value: float = 0.0):
				self.su1_config = su1_config
				self.su5_config = su5_config
				self.fill_value = fill_value

		def fit(self, X: pd.DataFrame, y=None):
				# SU1 → SU5 の順で fit
				self.su1_ = SU1FeatureGenerator(self.su1_config).fit(X, y)
				X_su1 = self.su1_.transform(X)
				self.su5_ = SU5FeatureGenerator(self.su5_config).fit(X_su1, y)
				return self

		def transform(self, X: pd.DataFrame, fold_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
				X_su1 = self.su1_.transform(X, fold_indices=fold_indices)
				X_su5 = self.su5_.transform(X_su1, fold_indices=fold_indices)
				return pd.concat([X, X_su1, X_su5], axis=1)
```

---

## 6. スイープ戦略と採用基準

### 6.1 スイープパラメータ案

```yaml
su5_sweep:
	top_k_pairs: [5, 10, 20]
	windows: [[5], [5, 20]]
	reset_each_fold: [true, false]
```

### 6.2 評価指標

- OOF RMSE
- OOF MSR（Sharpe-like 指標）
- 特徴量数（増加分）
- 学習時間

### 6.3 採用基準（たたき台）

- 必須条件:
	- SU1 ベースライン比で OOF MSR が **非劣化**（-0.001 以内）
	- 特徴量数の増加が SU5 分で +50 列以内
	- 学習時間が SU1 比で +50% 以内
- 推奨条件:
	- OOF MSR が SU1 比で +0.001 以上
	- LB スコアが SU1 比で非劣化（-0.002 以内）

採用する場合は `configs/feature_generation.yaml` の `su5.enabled` を `true` にし、`docs/submissions.md` に OOF/LB の差分・列数・学習時間を記録する。

---

## 7. テスト方針（`tests/feature_generation/test_su5.py`）

最低限、以下のケースをユニットテストでカバーする:

1. **全観測列**
	 - 入力がすべて `m/<col> == 0` の場合、`co_miss_now`・`co_miss_rollrate` がすべて 0 or NaN になること。
2. **全 NaN 列**
	 - 入力がすべて `m/<col> == 1` の場合、共欠損スコアが最大になり、top-k ペア選択が期待どおりになること。
3. **単一ペアのみ共欠損**
	 - 2 列だけが完全に同じ NaN パターンを持つ場合、このペアが top-1 で選ばれること。
4. **fold 境界の扱い**
	 - `reset_each_fold=True` の場合、fold 境界を跨がないようにローリング共欠損率が計算されること。
5. **出力形状**
	 - 設定した `top_k_pairs`, `windows` に応じて、出力列数が期待値と一致すること。

---

## 8. トラブルシューティングの想定

### 問題1: 列数が多すぎる

- 症状: SU5 を有効化しただけで総特徴量数が 800 列など、SU2 と同じ失敗パターンに近づく。
- 対策:
	- `top_k_pairs` を 5〜10 まで絞る。
	- ローリング窓数を 1 種類に減らす（例: `windows: [5]`）。
	- degree 系特徴（C カテゴリ）をオフにする。

### 問題2: OOF MSR が悪化 / LB 劣化

- 症状: SU1 比で Sharpe 系指標が明確に悪化、LB も落ちる。
- 対策:
	- まず SU5 特徴を完全オフに戻す（SU1 ラインを維持）。
	- 特徴重要度や SHAP を確認し、寄与の低いペアだけを削除。
	- `top_k_pairs` を小さくし、最も強い共欠損ペアのみ残す実験から再開する。

### 問題3: 計算時間が長い

- 症状: fit 時の共欠損スコア計算や transform 時のローリング計算がボトルネックになる。
- 対策:
	- スコア計算は 1 回だけ行い、結果を pickle でキャッシュするオプションを検討。
	- ローリング窓数を減らす。
	- どうしても重い場合は、SU5 自体を見送り SU1 ライン維持に戻る。

---

## 9. 関連ドキュメント

- `docs/feature_generation/README.md` — 特徴量生成ロードマップ（SU シリーズ全体の位置づけ）
- `docs/feature_generation/SU1.md` — 欠損構造一次特徴（ベースライン）
- `docs/feature_generation/SU2.md` — SU2 の設計と非採用の経緯
- `docs/feature_generation/SU3.md` — SU3 の設計と完全放棄の記録
- `docs/submissions.md` — Submit ラインごとの OOF/LB 実績と意思決定

SU5 は「欠損構造の横方向（共欠損）の網羅」を担うユニットとして設計しており、SU1 の成功と SU2/SU3 の失敗から学んだ制約（列数キャップ・top-k・シンプルさ）を前提に実験を進める。

