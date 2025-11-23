# ボラティリティ・レジーム特徴 SU8 仕様

最終更新: 2025-11-23

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

### 2.2 代表列の例

- `core_index_1`: 市場全体を代表するインデックス（例: 全銘柄の平均的価格・指数）。
- `core_index_2`: ターゲットに最も近いリターン列 or ベンチマーク価格列（必要なら）。

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

- 分類ルール（初期実装）:
  - 閾値 τ = 0 とし、
    - `trend_regime_up   = (trend_indicator > 0)`
    - `trend_regime_down = (trend_indicator < 0)`
    - `trend_regime_flat = (trend_indicator == 0)`

- 将来拡張（fallback）:
  - train 区間で `trend_regime_up/down/flat` のクラス分布が極端に偏る場合、
    - train の `trend_indicator` 分布の quantile（例: 0.33/0.66）から `τ_down`, `τ_up` を決め、
    - `trend_indicator <= τ_down` を down, `>= τ_up` を up、それ以外を flat とする案も許容。
  - 閾値の fit は **各 fold の train 区間のみ** で行い、val/test には持ち越す。

- 列数: **3 列**。

### 3.4 ボラ調整リターン

- 対象列: 「ターゲットに最も近いリターン列」1 本のみ。
- 特徴量:
  - `ret_vol_adj` = `ret / (1 + ewmstd_short)`
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

---

## 6. PoC とロールバック方針

- 導入順: SU9 → SU7 → **SU8（レジームタグ）**。
- 評価: SU1+SU5 ベースラインと比較し、OOF MSR/RMSE を指標とする。
- ロールバック条件:
  - OOF 改善が **fold 間分散を考慮して +0.3〜0.5σ 未満** の場合は SU8 をオフに戻す。
  - LB 提出は SU8 について最大 1 回までとし、OOF で明確に悪い設定は提出しない。
