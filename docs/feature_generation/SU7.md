# モメンタム・リバーサル特徴 SU7 仕様

最終更新: 2025-11-23

## 1. SU7 の目的と位置づけ

- 目的: **価格・リターン系列のモメンタム/リバーサル構造**を特徴量化し、短期〜中期の値動きパターンを明示的にモデルへ渡す。
- 軸: 欠損構造（SU1〜SU5）とは独立した「価格変化軸」の SU。
- パイプライン上の位置:
  - 生データ → SU1 → SU5 → GroupImputers（M/E/I/P/S）
  → **SU7** → 前処理（スケーラー＋OneHot）→ LightGBM
  - SU7 は **SU1/SU5 + GroupImputers による欠損補完後の特徴行列を入力**とし、その上に派生特徴を作る（生データには直接作用しない）。

## 2. 想定列数と対象カラム

### 2.1 想定列数レンジ

- 目標: **+80〜120 列程度**（1 SU あたりの上限 150 列の範囲内）
- 実効レンジ（詳細仕様ベース）: **約 40〜80 列** を目安とし、対象列数 B を 6〜8 列に抑える。

### 2.2 対象カラムの選定ルール

- 対象: 既存 SU1+SU5 ベースラインで **feature_importance が高い価格/リターン系列**。
- 手順（PoC 初期案）:
  1. SU5 ベースラインの LightGBM から、価格・指数・ターゲット近傍リターンなどを候補セットとして抽出。
  2. その中から **手動で 6〜8 列** を選び、`su7_base_cols` として固定。
- 将来拡張案:
  - FI 閾値や上位 N 列を自動抽出するロジックを導入し、`su7_base_cols` を自動選択する。

### 2.3 ベース系列の定義

- SU7 では `su7_base_cols` は **「リターン（または差分）系列 r_t」** として扱う。
- diff / roll_ret / RSI / sign など **すべて同じ r_t を前提に計算**し、一部だけ価格ベースになることを避ける。

---

## 3. 生成する特徴量

以降、`B = len(su7_base_cols)` とし、PoC では **B=6**（上限 8）を想定する。

### 3.1 diff / lag（k を 3 本に絞る）

- k セット: `K = {1, 5, 20}`
- 特徴量:
  - `diff_k/<col>` = `x[t] - x[t-k]` （k ∈ K）
  - `lag_k/<col>` = `x[t-k]` （k ∈ K）
- 列数:
  - diff: `3B`、lag: `3B` → 合計 **6B 列**。
- 実装メモ:
  - 入力は GroupImputers 通過後の `X`（欠損補完済み）。
  - 先頭 k 日分（`t-k` が存在しない部分）は NaN のまま（後段前処理で処理）。

### 3.2 ローリング・モメンタム（W を 2 本に絞る）

- 窓: `W = {5, 20}`
- 特徴量（リターン系列 `r_t` 前提）:
  - `roll_ret_W/<col>` = `∑_{i=t-W+1}^{t} r_i` （単純和）
  - `roll_mean_diff_W/<col>` = `mean(diff_1[t-W+1:t])`（`diff_1` は `r_t` の 1 ラグ差分）
- 列数:
  - 各列あたり 4 本（2 指標 × 2 窓） → **4B 列**。
- 実装メモ:
  - `rolling(window=W, min_periods=W).sum()/mean()` を用い、`min_periods=W` 未満は NaN。

### 3.3 RSI ライク指標（W=5 のみ）

- 窓: W=5 固定。
- 手順（すべて `r_t` ベース）:
  1. `gains = max(r_t, 0)`
  2. `losses = max(-r_t, 0)`
  3. `ema_gain = ema(gains, halflife=5)`
  4. `ema_loss = ema(losses, halflife=5)`
  5. `rs = ema_gain / (ema_loss + eps)`
  6. `rsi_5 = rs / (1 + rs)`
- 列数:
  - 各ベース列に 1 本 → **B 列**。

### 3.4 方向フラグ（最小限）

- 役割重複を避けるため、direction flag は 1 種類のみ定義。
- 特徴量:
  - `sign_r_t/<col>` = sign(`r_t`) ∈ {−1, 0, 1}（int8）
- 列数:
  - **B 列**。

### 3.5 列数レンジの試算（例: B=6）

- diff/lag: `6B = 36`
- roll_ret/roll_mean_diff: `4B = 24`
- rsi_5: `B = 6`
- sign_r_t: `B = 6`
- 合計: **72 列**（目標 40〜80 列レンジ内）。

---

## 4. 共通ルールとリーク防止

- 統計量・EMA 等の fit は **各 CV fold の train 区間のみ** で行い、val/test には持ち越す。
- rolling / EMA は「時刻 t までの過去のみ」を参照し、未来情報は一切使わない。
- 型は `float32`（連続値）/`int8`（sign フラグ）を基本とし、列数とメモリを抑える。

---

## 5. クラス構成（想定）

- `src/feature_generation/su7/feature_su7.py`
  - `SU7Config`: 対象列リスト、K・W・halflife などの設定。
  - `SU7FeatureGenerator`: sklearn Transformer 互換の特徴生成クラス。
  - `SU7FeatureAugmenter`: SU1+SU5 後の特徴行列に SU7 を付与する augmenter。
- `src/feature_generation/su7/train_su7.py`
  - SU1+SU5+GroupImputers+SU7+前処理+LGBM のパイプライン学習・OOF 評価。
- `src/feature_generation/su7/predict_su7.py`
  - 上記パイプラインの inference bundle による推論。
- `tests/feature_generation/test_su7.py`
  - diff/roll/RSI/sign の境界ケースとリーク防止のテスト。

---

## 6. PoC とロールバック方針

- 導入順: SU9 → **SU7（本仕様のコア部分）** → SU8。
- 評価: SU1+SU5 ベースラインと比較し、OOF MSR/RMSE を指標とする。
- ロールバック条件:
  - OOF 改善が **fold 間分散を考慮して +0.3〜0.5σ 未満** の場合は SU7 をオフに戻す。
  - LB 提出は SU7 について最大 1 回までとし、OOF で明確に悪い設定は提出しない。
