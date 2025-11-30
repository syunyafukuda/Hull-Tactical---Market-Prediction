# ボラティリティ・レジーム特徴 SU8 仕様

最終更新: 2025-11-30

## 1. SU8 の目的と位置づけ

- 目的: **局所ボラティリティとマーケットレジーム（低/中/高ボラ、トレンド/レンジなど）**を特徴量化し、「今どんな相場モードか」をモデルに渡す。
- 軸: SU7 のモメンタム軸と補完的な「ボラ・レジーム軸」の SU。
- パイプライン上の位置:
  - 生データ → SU1 → SU5 → GroupImputers（M/E/I/P/S）
  → **SU8** → 前処理（スケーラー＋OneHot）→ LightGBM
  - SU8 は **欠損補完後の代表インデックス系列**を入力にボラ・レジームを定義する。

---

## 2. 代表列の選定

### 2.1 目的

- レジーム判定用の「代表インデックス」を 1〜2 列に絞り、列数と実装複雑性を抑える。

### 2.2 代表列の例と具体的ルール

- `core_index_1`: 市場全体を代表するインデックス（例: 全銘柄の平均的価格・指数）。
- `core_index_2`: ターゲットに最も近いリターン列 or ベンチマーク価格列（必要なら）。

実装上は、次のルールで代表列を固定する:

- M/E/P グループに属するインデックス列のうち、ターゲットに対する FI（feature importance）が高いものから 1 本だけ選び、
- その列名を `configs/feature_generation.yaml` の `su8.core_index_col` に明示的に記載する。

SU8 のボラ計算・レジーム判定は、**基本的に `core_index_1` に対して実施**し、`core_index_2` は将来拡張用とする。

---

## 3. 生成する特徴量

### 3.1 ボラティリティ指標（`core_index_1`）

- 特徴量:
  - `ewmstd_short` = EWMA 標準偏差（halflife=5）
  - `ewmstd_long`  = EWMA 標準偏差（halflife=20）
  - `vol_ratio`    = `ewmstd_short / (ewmstd_long + eps)`
  - `vol_level`    = `ewmstd_long`
- 列数: **4 列**。
- 実装メモ:
  - pandas の `ewm` は `adjust=False` を用い、オンライン更新に近い挙動とする。
  - `min_periods` は短期/長期ともにウィンドウ長（例: 5, 20）とし、それ未満の期間は NaN のまま許容する（train 初期のみで使用頻度が低いため）。
  - 各 fold の train 全期間で時系列順に EWMA を計算し、val/test も同じ式を適用（追加 fit なし）。

### 3.2 ボラレジームタグ（low/mid/high）

- train 区間の `vol_level` から:
  - `q_low  = quantile(vol_level_train, 0.33)`
  - `q_high = quantile(vol_level_train, 0.66)`
- val/test では train で得た閾値をそのまま使用し、
  - `vol_regime_low  = (vol_level <= q_low)`
  - `vol_regime_mid  = (q_low < vol_level <= q_high)`
  - `vol_regime_high = (vol_level > q_high)`
- 列数: **3 列**（one-hot）。

### 3.3 トレンドレジームタグ（up/down/flat）

- 指標:
  - `ma_short` = EWMA or rolling_mean (window=5)
  - `ma_long`  = EWMA or rolling_mean (window=20)
  - `trend_indicator = ma_short - ma_long`

- 分類ルール（正規実装）:
  - train 区間の `trend_indicator` から quantile を用いてしきい値を決める:
    - `τ_down = quantile(trend_indicator_train, 0.33)`
    - `τ_up   = quantile(trend_indicator_train, 0.66)`
  - val/test では train で得た閾値をそのまま使用し、
    - `trend_regime_down = (trend_indicator <= τ_down)`
    - `trend_regime_up   = (trend_indicator >= τ_up)`
    - `trend_regime_flat = otherwise`
  - 閾値の fit は **各 fold の train 区間のみ** で行い、val/test には持ち越す。

- 列数: **3 列**。

### 3.4 ボラ調整リターン

- 対象列: 「ターゲットに最も近いリターン列」1 本のみ。
- 特徴量:
  - `denom = 1 + max(ewmstd_short, eps)` （eps=1e-4 程度）
  - `ret_vol_adj` = `ret / denom`
  - 必要に応じて、`ret_vol_adj` を p99 で winsorize する（SU4/SU5 で用いている winsor ロジックに合わせる）。
- 列数: **1 列**。

### 3.5 列数レンジ（初期）

- ewmstd & vol_ratio/level: 4 列
- vol_regime: 3 列
- trend_regime: 3 列
- ret_vol_adj: 1 列
- 合計: **11 列**（将来拡張を含めても 20〜30 列に収まる設計）。

---

## 4. 共通ルールとリーク防止

- quantile・閾値・分布推定は **各 CV fold の train 区間のみ** で fit し、val/test には持ち越す。
- EWMA・rolling_mean は「時刻 t までの過去のみ」を参照する。
- レジームタグは one-hot で ColumnTransformer に渡し、スケーリング対象には含めない。
- regime 系カラム（`vol_regime_*`, `trend_regime_*`）は `category` または `bool` dtype で出力し、
  `configs/feature_generation.yaml` の `su8.categorical_cols` と `model_meta.json` の `categorical_features` に明示的に含める。

---

## 5. クラス構成（想定）

- `src/feature_generation/su8/feature_su8.py`
  - `SU8Config`: 代表列名、EWMA パラメータ、quantile 設定など。
  - `SU8FeatureGenerator`: ボラ・レジーム特徴生成クラス。
  - `SU8FeatureAugmenter`: 既存特徴行列に SU8 特徴を付与する augmenter。
- `src/feature_generation/su8/train_su8.py`
  - SU1+SU5+GroupImputers+SU8+前処理+LGBM のパイプライン学習・OOF 評価。
- `src/feature_generation/su8/predict_su8.py`
  - 上記パイプラインの inference bundle による推論。
- `tests/feature_generation/test_su8.py`
  - quantile fit / 閾値の fold 分離・タグのクラスバランスなどのテスト。

### 5.1 fit / transform の責務

- `SU8FeatureGenerator.fit(X_train)`
  - `core_index_1` 列の存在チェック。
  - train データから `vol_level` と `trend_indicator` の分布を集計し、
    - `q_low`, `q_high`（ボラレジーム用）
    - `τ_down`, `τ_up`（トレンドレジーム用）
    を算出して `self.quantiles_` などに保持する。

- `SU8FeatureGenerator.transform(X_all_sorted)`
  - 時刻順に EWMA/rolling を計算してボラ系列を生成。
  - `self.quantiles_` を用いて各行に vol_regime / trend_regime タグを付与。

- CV 側（`train_su8.py`）では、fold ごとに `clone(SU8FeatureGenerator).fit(X_train) → transform(X_train+X_val)` という、SU1/SU5 と同様のパターンを採用する。

---

## 6. PoC とロールバック方針

- 導入順: SU9 → SU7 → **SU8（レジームタグ）**。
- 評価: SU1+SU5 ベースラインと比較し、OOF MSR/RMSE を指標とする。

- LB 提出・post-process の運用ルール:
  - SU8 では **固定構成 1 パターンのみ** を基本とし、LB 提出も 1 ラインのみとする。
  - post-process は `mult=1.0, lo=0.9, hi=1.1` など、安全側に寄せた「完全保守」設定で固定し、OOF 最適化は行わない。

- ロールバック条件:
  - OOF 改善が **fold 間分散を考慮して +0.5σ 以上** かつ MSR 非劣化を満たさない限り、SU8 を恒常採用しない。
  - 上記条件を満たさない場合、SU8 をオフに戻し、LB への再提出も行わない。

## 7. スイープ方針（必要な場合のみ）

- 初期 PoC では、SU8 は **halflife / 窓長を固定した 1 パターンのみ** で十分とする（例: 短期=5, 長期=20）。
- それでも追加検証が必要な場合のみ、限定的なスイープを許容する:
  - `halflife_short ∈ {5, 10}`
  - `halflife_long  ∈ {20, 60}`
  - 合計 2×2 = 4 パターンまでを上限とする。
- これ以上のスイープは、SU7 で経験した「OOF 最適化 → LB 崩壊」のリスクを高めるため、仕様レベルで禁止する。
