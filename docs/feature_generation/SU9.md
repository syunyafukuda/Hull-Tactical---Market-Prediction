# カレンダー・季節性特徴 SU9 仕様

最終更新: 2025-12-05

**Status: 非採用（rejected）** - OOF改善するもLB悪化のため

## 0. 最終結果サマリ

| 指標 | SU5 baseline | SU9 best (month+holiday) | 差分 |
|------|--------------|--------------------------|------|
| OOF RMSE | 0.012088 | 0.012041 | -0.39% (改善) |
| LB score | 0.681 | **0.679** | **-0.002 (悪化)** |

**判定**: OOF改善がLB改善に繋がらず、過学習パターンと判断。SU9は非採用。

---

## 1. SU9 の目的と位置づけ

- 目的: **曜日・月・四半期・祝日・月末/期末などのカレンダー構造**を特徴量化し、カレンダー起因の季節性・アノマリーをモデルに渡す。
- 軸: 欠損軸・モメンタム軸・ボラ軸とは独立した「時間構造軸」の SU。
- パイプライン上の位置:
  - 生データ → SU1 → SU5 → GroupImputers（M/E/I/P/S）
  → **SU9** → 前処理（スケーラー＋OneHot）→ LightGBM
  - SU9 は「決定可能な日付情報」を元にした特徴であり、リークリスクはない。

---

## 2. 想定列数レンジ

- 目標: **+30〜60 列程度**。
- 初期仕様ベースの実効レンジ: **約 30〜35 列** を想定。

---

## 3. 生成する特徴量

### 3.1 曜日 (DOW)

- 入力: `dow`（0〜6）。
- 特徴量:
  - one-hot: `dow_0`〜`dow_6`（7 列）。

### 3.2 月内位置 (DOM)

- 入力: day-of-month (1〜31)。
- ビン分割:
  - `dom_early`: day 1〜10 のとき 1
  - `dom_mid`:   day 11〜20 のとき 1
  - `dom_late`:  day 21〜月末 のとき 1
- 列数: **3 列**。

### 3.3 月・四半期

- 初期実装では、**month_one_hot のみに寄せ、qtr は持たない** 方針とする。

- 月 (1〜12):
  - `month` を one-hot 12 列で表現: `month_1`〜`month_12`。

- 将来拡張（必要になった場合）:
  - 四半期だけを別軸で使いたいケースでは、
    - `qtr` を整数カテゴリ 1 列 or one-hot 4 列として追加する。
  - ただし、`month_one_hot + qtr` のような強い冗長構造は避ける。

### 3.4 月末・期末フラグ

- フラグ:
  - `is_month_start`, `is_month_end`
  - `is_qtr_start`,  `is_qtr_end`
- 列数: **4 列**。

### 3.5 祝日・ブリッジ関連

- 既存の前処理で使用している `holiday_bridge` 定義と **完全に整合** させる。

- 特徴量（いずれも「営業日ベース」で定義）:
  - `is_holiday`: 公式マーケットカレンダーに基づく休場フラグ（当日の営業日単位）。
  - `is_holiday_eve`: **翌営業日** が `is_holiday` のとき 1（カレンダー日ではなく営業日シーケンスで 1 日前）。
  - `is_holiday_next`: **前営業日** が `is_holiday` のとき 1（同様に営業日シーケンス基準）。
  - `is_bridge_day`: 既存 `holiday_bridge` と同じロジックで算出（前後の **営業日** の休場状況に基づく）。

- 実装方針:
  - 可能であれば `preprocess` 側の `holiday_bridge` 実装を共通ユーティリティとして切り出し、SU9 からもそれを利用する。

### 3.6 年内ポジション

- 特徴量:
  - `yday_norm`        = `day_of_year / 365.0`
  - `days_to_year_end` = `(365 - day_of_year) / 365.0`
- 列数: **2 列**。

### 3.7 イベント距離（将来オプション）

- 初期実装では **含めない**。
- 外部イベントカレンダーが整備された段階で、
  - 特定イベントまでの日数 `days_to_event_k` などを SU9 拡張として追加する。

### 3.8 列数レンジの試算（初期）

- dow one-hot: 7 列
- dom 3 ビン: 3 列
- month one-hot: 12 列
- 月末・期末フラグ: 4 列
- holiday 系: 4 列
- 年内ポジション: 2 列
- 合計: **32 列**（目標レンジ内）。

---

## 4. 共通ルールとリーク防止

- すべて「将来も確定している決定可能な日付情報」のみを使用し、未来情報は含まない。
- `holiday_bridge` 等のロジックは train/test で共通のテーブル・実装から derive する。
- カテゴリ系は one-hot 等で ColumnTransformer に渡し、連続値は標準的なスケーリングに乗せる。

---

## 5. クラス構成（想定）

- `src/feature_generation/su9/feature_su9.py`
  - `SU9Config`: 祝日カレンダー、列名、エンコーディングオプションなど。
  - `SU9FeatureGenerator`: カレンダー・季節性特徴生成クラス。
  - `SU9FeatureAugmenter`: 既存特徴行列に SU9 特徴を付与する augmenter。
- `src/feature_generation/su9/train_su9.py`
  - SU1+SU5+GroupImputers+SU9+前処理+LGBM のパイプライン学習・OOF 評価。
- `src/feature_generation/su9/predict_su9.py`
  - 上記パイプラインの inference bundle による推論。
- `tests/feature_generation/test_su9.py`
  - カレンダー境界・祝日/bridge ロジック・リーク防止のテスト。

---

## 6. PoC とロールバック方針

- 導入順の最初のステップとして **SU9 を先に試す**（モメンタム・ボラよりも安全かつ汎用性が高いため）。
- 評価: SU1+SU5 ベースラインと比較し、OOF MSR/RMSE を指標とする。
- ロールバック条件:
  - OOF 改善が **fold 間分散を考慮して +0.3〜0.5σ 未満** の場合は SU9 をオフに戻す。
  - LB 提出は SU9 について最大 1 回までとし、OOF で明確に悪い設定は提出しない。

---

## 7. スイープ処理

### 7.1 目的

SU9 の 32 特徴量のうち、どのサブセットが最も有効かを探索するため、
6 つの特徴グループフラグ（`include_dow`, `include_dom`, `include_month`,
`include_month_flags`, `include_holiday`, `include_year_position`）を
グリッドサーチで評価する。

### 7.2 スイープ対象パラメータ

| フラグ | 列数 | 特徴名 |
|--------|------|--------|
| `include_dow` | 7 | dow_0〜dow_6 |
| `include_dom` | 3 | dom_early, dom_mid, dom_late |
| `include_month` | 12 | month_1〜month_12 |
| `include_month_flags` | 4 | is_month_start, is_month_end, is_qtr_start, is_qtr_end |
| `include_holiday` | 4 | is_holiday, is_holiday_eve, is_holiday_next, is_bridge_day |
| `include_year_position` | 2 | yday_norm, days_to_year_end |

**組み合わせ数**: 2^6 = 64 通り（全て False を除くと 63 通り）

### 7.3 スイープスクリプトの使用方法

```bash
# 全 64 パターンを評価（約 1-2 時間）
uv run python src/feature_generation/su9/sweep_oof.py \
  --data-dir data/raw \
  --out-dir results/ablation/SU9 \
  --n-splits 5 \
  --include-dow-grid true false \
  --include-dom-grid true false \
  --include-month-grid true false \
  --include-month-flags-grid true false \
  --include-holiday-grid true false \
  --include-year-position-grid true false

# 全 False を除外して 63 パターンを評価
uv run python src/feature_generation/su9/sweep_oof.py \
  --data-dir data/raw \
  --out-dir results/ablation/SU9 \
  --n-splits 5 \
  --skip-all-false \
  --include-dow-grid true false \
  --include-dom-grid true false \
  --include-month-grid true false \
  --include-month-flags-grid true false \
  --include-holiday-grid true false \
  --include-year-position-grid true false

# 特定のフラグのみスイープ（例: 曜日と月のみ）
uv run python src/feature_generation/su9/sweep_oof.py \
  --data-dir data/raw \
  --out-dir results/ablation/SU9 \
  --n-splits 5 \
  --include-dow-grid true false \
  --include-dom-grid true \
  --include-month-grid true false \
  --include-month-flags-grid true \
  --include-holiday-grid true \
  --include-year-position-grid true
```

### 7.4 出力ファイル

```
results/ablation/SU9/
├── sweep_<timestamp>.json      # 全試行の詳細結果（JSON 形式）
└── sweep_summary.csv           # 全試行のサマリ（CSV 形式）
```

#### sweep_summary.csv の列

| 列名 | 説明 |
|------|------|
| `include_dow` | 曜日 one-hot の有無 |
| `include_dom` | 月内位置ビンの有無 |
| `include_month` | 月 one-hot の有無 |
| `include_month_flags` | 月末・期末フラグの有無 |
| `include_holiday` | 祝日・ブリッジの有無 |
| `include_year_position` | 年内ポジションの有無 |
| `oof_rmse` | OOF RMSE（主指標） |
| `oof_msr` | OOF MSR（副指標） |
| `oof_msr_down` | OOF MSR_down（副指標） |
| `feature_count_su1` | SU1 特徴量数 |
| `feature_count_su5` | SU5 特徴量数 |
| `feature_count_su9` | SU9 特徴量数 |
| `feature_count_total` | 総特徴量数 |
| `training_time_sec` | 学習時間（秒） |
| `folds_used` | 使用した fold 数 |

### 7.5 評価指標

- **主指標**: OOF RMSE（低いほど良い）
- **副指標**: OOF MSR / OOF MSR_down（高いほど良い）

### 7.6 期待される成果

1. **最適な特徴サブセットの特定**
   - 例: `include_dow=True, include_month=False, ...` など
   
2. **OOF RMSE の改善**
   - 目標: SU5 単体（0.012088）以下

3. **LB スコアの改善**
   - 目標: 0.681 以上（歴代 1 位更新）

### 7.7 スイープ結果

**実行日時**: 2025-12-05  
**総パターン数**: 63（全 false を除く）  
**総実行時間**: 約173分

#### ベスト構成

| フラグ | 値 | 列数 |
|--------|-----|------|
| `include_dow` | False | 0 |
| `include_dom` | False | 0 |
| `include_month` | **True** | 12 |
| `include_month_flags` | False | 0 |
| `include_holiday` | **True** | 4 |
| `include_year_position` | False | 0 |

**SU9 特徴量合計**: 16列（month 12 + holiday 4）

#### OOF 評価結果

| 指標 | ベースライン (SU5) | SU9 best | 変化 |
|------|-------------------|----------|------|
| RMSE | 0.012088 | 0.012041 | **-0.39%** (改善) |

#### Kaggle LB 結果

| ライン | LB score | 備考 |
|--------|----------|------|
| SU1+SU5 baseline | **0.681** | 現行ベスト |
| SU9 (month+holiday) | **0.679** | -0.002 悪化 |

#### 考察

1. **OOF改善がLB改善に繋がらなかった**
   - OOFでは0.39%改善したが、LBでは0.002ポイント悪化
   - カレンダー特徴が訓練期間の局所パターンを学習した可能性

2. **過学習パターンの兆候**
   - 月・祝日という「決定可能な」特徴でも、訓練期間のノイズを拾いうる
   - 特に12列の月one-hotが過学習リスク要因

3. **他の特徴グループとの比較**
   - SU7 (モメンタム): OOF改善 → LB大幅悪化 (0.476)
   - SU8 (ボラティリティ): OOF悪化 → LB悪化 (0.624)
   - SU9 (カレンダー): OOF改善 → LB微悪化 (0.679)
   - パターンとしてはSU7に近い「過学習型」

#### 最終判断

**SU9は非採用**。理由:
- LBスコアがベースラインより悪化
- OOF改善はCV分割内でのみ有効だった可能性
- カレンダー特徴は本コンペでは有効でない

現行ベストラインは引き続き **SU1+SU5 (LB 0.681)** を維持する。

---

## 8. 開発履歴（feat/su9 ブランチ）

### 8.1 主要コミット

1. **初期実装 (PR#59)**: feature_su9.py, train_su9.py, predict_su9.py
2. **スイープ実装 (PR#61)**: sweep_oof.py, テスト追加
3. **バグ修正**: eval_set不整合の解消

### 8.2 実装ファイル

```
src/feature_generation/su9/
├── __init__.py
├── feature_su9.py    # SU9Config, SU9FeatureGenerator, SU9FeatureAugmenter
├── train_su9.py      # 学習パイプライン (1271行)
├── predict_su9.py    # 推論パイプライン (678行)
└── sweep_oof.py      # スイープスクリプト (506行)

tests/feature_generation/
├── test_su9.py       # 単体テスト (598行)
└── test_su9_sweep.py # スイープテスト (330行)
```

### 8.3 学んだ教訓

1. **カレンダー特徴でも過学習は起こりうる**
   - 「決定可能な情報」でも訓練データの局所パターンを学習するリスクあり

2. **OOF改善 ≠ LB改善**
   - SU7/SU8/SU9 すべてでこのパターンが確認された
   - CV分割の範囲外でのロバスト性検証が重要

3. **ベースラインの安定性を優先**
   - SU5 (LB 0.681) が依然として最も安定したライン
   - 追加特徴による複雑化は必ずしも改善に繋がらない
