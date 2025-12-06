# SU1/SU5 ブラッシュアップ仕様書

最終更新: 2025-12-06

---

## 0. ステータス

| 項目 | 状態 |
|------|------|
| 実装状況 | ✅ **実装完了** |
| OOF評価 | ✅ **完了** (RMSE: 0.012134, -0.04%改善) |
| LB評価 | ✅ **完了** (LB: 0.681, ベースライン同等) |
| ベースライン | SU1+SU5 (LB 0.681) |
| 目標 | 既存ラインを壊さず +9〜10 列の微調整で改善を狙う |
| 追加特徴量 | SU1: +5列、SU5: +5列 (合計 +10列 → 577列) |
| テスト状況 | ✅ 全ユニットテストパス (17件) |
| 最終判断 | ⏳ **採用保留** - 後続PCA/特徴量選定で再評価 |

---

## 1. 背景と目的

### 1.1 これまでの教訓

SU7〜SU11 の実験を通じて、以下が明確になりました：

| SU | 試みた内容 | 結果 | 教訓 |
|----|-----------|------|------|
| SU2 | rolling/EWMA 大量追加 (935列) | LB 0.597 (-12.4%) | 特徴量爆発 → 過学習 |
| SU3 | 遷移・再出現パターン (76列) | LB 0.461 (-31.6%) | コンセプト不適合 |
| SU7 | モメンタム・RSI 系 | LB 0.476 (-30.1%) | OOF改善もLB崩壊 |
| SU8 | ボラティリティ・レジーム | LB 0.624 (-8.4%) | OOF/LB両方で悪化 |
| SU9 | カレンダー・季節性 | LB 0.679 (-0.3%) | OOF改善もLB微悪化 |
| SU10 | 外部レジーム (SPY) | LB 0.597 (-12.3%) | 時間的ミスマッチ |
| SU11 | Level-2 Stacking | LB 0.464 (-31.9%) | shrinkage問題 |

**結論**: 「新しいSUを増やす」方向は失敗続き。既存のSU1/SU5の中身を**ほんの少しだけ厚くする**方向に転換。

### 1.2 方針の核心

```
❌ やらないこと
  - 列数を +50 以上増やす拡張
  - 多window (5,10,20,60) × 多集約 (mean,std,max,min) の rolling 系
  - 外部レジームや date_id と強く絡めた高次特徴
  - Level-2 stacking

✅ やること
  - SU1 を時間方向にほんの少しだけ拡張 (+3〜5列)
  - SU5 の共欠損構造を粗く圧縮したメタ列を追加 (+3〜5列)
  - 合計 +10列以下に厳密に抑える
```

---

## 2. SU1 拡張仕様

### 2.1 現在の SU1 特徴（確認）

| カテゴリ | 特徴名パターン | 列数目安 |
|---------|---------------|---------|
| 欠損フラグ | `m/<col>` | 94列 |
| 群集約 | `m_cnt/<grp>`, `m_rate/<grp>` | 12列 |
| 直近距離 | `gap_ffill/<col>` | 94列 |
| 連続長 | `run_na/<col>`, `run_obs/<col>` | 188列 |
| 群平均 | `avg_gapff/<grp>`, `avg_run_na/<grp>` | 12列 |

合計: 約 **368列**（SU1のみ）

### 2.2 追加特徴（SU1拡張）

**目標**: 時間方向にほんの少しだけ伸ばす（+3〜5列）

#### 2.2.1 直近K日の欠損頻度（全体レベル）

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `miss_count_last_5d` | 直近5日間の全列の欠損数合計 | int16 | `sum(m/<col>[t-4:t+1])` の全列合計 |
| `miss_ratio_last_5d` | 直近5日間の欠損率 | float32 | `miss_count_last_5d / (5 * n_cols)` |

**実装詳細**:
```python
# 行ごとに「その日の欠損数」を計算
daily_miss_count = df[[f'm/{col}' for col in cols]].sum(axis=1)

# 直近5日のローリング合計
miss_count_last_5d = daily_miss_count.rolling(5, min_periods=5).sum()

# 比率
miss_ratio_last_5d = miss_count_last_5d / (5 * n_cols)
```

#### 2.2.2 欠損ストリークのフラグ

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `is_long_missing_streak` | 直近で「3日以上連続欠損」の列が存在するか | uint8 | `max(run_na/<col>) >= 3` |
| `long_streak_col_count` | 「3日以上連続欠損」の列数 | int16 | `sum(run_na/<col> >= 3)` |

**実装詳細**:
```python
run_na_cols = [f'run_na/{col}' for col in cols]
is_long_missing_streak = (df[run_na_cols].max(axis=1) >= 3).astype('uint8')
long_streak_col_count = (df[run_na_cols] >= 3).sum(axis=1).astype('int16')
```

#### 2.2.3 欠損レジーム変化フラグ

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `miss_regime_change` | 直近5日で欠損し始めたが、それ以前30日はほぼ欠損なしの列が存在するか | uint8 | regime change 検知 |

**実装詳細**:
```python
# 各列について
# - 直近5日の欠損率 > 0.5 (最近欠損が多い)
# - それ以前30日の欠損率 < 0.1 (以前はほぼ欠損なし)
recent_miss_rate = df[f'm/{col}'].rolling(5).mean()
past_miss_rate = df[f'm/{col}'].shift(5).rolling(30).mean()

is_regime_change = (recent_miss_rate > 0.5) & (past_miss_rate < 0.1)
miss_regime_change = is_regime_change.any(axis=1).astype('uint8')
```

### 2.3 SU1拡張の列数まとめ

| 特徴名 | 列数 |
|--------|-----|
| `miss_count_last_5d` | 1 |
| `miss_ratio_last_5d` | 1 |
| `is_long_missing_streak` | 1 |
| `long_streak_col_count` | 1 |
| `miss_regime_change` | 1 |
| **合計** | **5列** |

---

## 3. SU5 拡張仕様

### 3.1 現在の SU5 特徴（確認）

| カテゴリ | 特徴名パターン | 列数目安 |
|---------|---------------|---------|
| 共欠損フラグ | `co_miss_now/<a>__<b>` | 10列 (top-10ペア) |
| ローリング共欠損率 | `co_miss_rollrate_5/<a>__<b>` | 10列 |
| 共欠損次数 | `co_miss_deg/<col>` | 85列 |

合計: 約 **105列**（SU5のみ）

### 3.2 追加特徴（SU5拡張）

**目標**: 共欠損構造を粗く圧縮したメタ列を追加（+3〜5列）

#### 3.2.1 欠損パターンのクラスタID

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `miss_pattern_cluster` | 行ごとの欠損パターン（0/1ベクトル）をk-meansでクラスタリングしたID | int8 | k=4〜8 |

**実装詳細**:
```python
from sklearn.cluster import KMeans

# 欠損パターン行列 (n_samples, n_cols)
miss_matrix = df[[f'm/{col}' for col in cols]].values

# 学習時: k-meansをfit
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(miss_matrix[train_idx])

# 全データにtransform
miss_pattern_cluster = kmeans.predict(miss_matrix)
```

**注意点**:
- CV fold ごとに train 部分で fit し、val 部分で predict
- test 時は全 train で fit した kmeans を使用
- k は小さく抑える（4〜8）

#### 3.2.2 共欠損密度スカラー

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `co_miss_density` | その行で top-k 共欠損ペアのうち、同時欠損しているペアの割合 | float32 | 0〜1 |

**実装詳細**:
```python
# top-k 共欠損ペアの「同時欠損フラグ」を取得
co_miss_flags = df[[f'co_miss_now/{pair}' for pair in top_k_pairs]]

# 同時欠損しているペアの割合
co_miss_density = co_miss_flags.mean(axis=1).astype('float32')
```

#### 3.2.3 共欠損次数サマリ

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `co_miss_deg_sum` | その行で欠損している列の共欠損次数の合計 | float32 | 共欠損の「中心性」 |
| `co_miss_deg_mean` | その行で欠損している列の共欠損次数の平均 | float32 | 正規化版 |

**実装詳細**:
```python
# 各列の共欠損次数（学習時に計算済み）
deg_values = {col: co_miss_deg[col] for col in cols}

# その行で欠損している列のdegを集計
def calc_deg_stats(row):
    missing_cols = [col for col in cols if row[f'm/{col}'] == 1]
    if not missing_cols:
        return 0.0, 0.0
    degs = [deg_values[col] for col in missing_cols]
    return sum(degs), sum(degs) / len(degs)

co_miss_deg_sum, co_miss_deg_mean = zip(*df.apply(calc_deg_stats, axis=1))
```

#### 3.2.4 欠損列グラフの簡易中心性（オプション）

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `miss_graph_centrality` | その行で欠損している列が top-k ペアに含まれる回数 | int8 | グラフ的な中心性 |

**実装詳細**:
```python
# top-k ペアに含まれる列の集合
pair_cols = set()
for pair in top_k_pairs:
    a, b = pair.split('__')
    pair_cols.add(a)
    pair_cols.add(b)

# その行で欠損している列がペアに含まれるか
def calc_centrality(row):
    missing_cols = [col for col in cols if row[f'm/{col}'] == 1]
    return sum(1 for col in missing_cols if col in pair_cols)

miss_graph_centrality = df.apply(calc_centrality, axis=1).astype('int8')
```

### 3.3 SU5拡張の列数まとめ

| 特徴名 | 列数 |
|--------|-----|
| `miss_pattern_cluster` | 1 |
| `co_miss_density` | 1 |
| `co_miss_deg_sum` | 1 |
| `co_miss_deg_mean` | 1 |
| `miss_graph_centrality` (オプション) | 1 |
| **合計** | **4〜5列** |

---

## 4. 合計特徴量数

| カテゴリ | 現在の列数 | 追加列数 | 合計 |
|---------|-----------|---------|------|
| 生データ | 94 | 0 | 94 |
| SU1 | 368 | **+5** | 373 |
| SU5 | 105 | **+4〜5** | 109〜110 |
| **合計** | **567** | **+9〜10** | **576〜577** |

**列数増加率**: +1.6〜1.8%（非常に控えめ）

---

## 5. 実装計画

### 5.1 ファイル構成

```
src/feature_generation/
  su1/
    feature_su1.py        # 既存 + 拡張特徴を追加
    train_su1.py          # 変更なし
    predict_su1.py        # 変更なし
  su5/
    feature_su5.py        # 既存 + 拡張特徴を追加
    train_su5.py          # k-means の fit/transform を追加
    predict_su5.py        # k-means の transform を追加

configs/
  feature_generation.yaml  # 拡張パラメータを追加
```

### 5.2 設定パラメータ（追加）

```yaml
su1:
  # 既存設定...
  
  # === SU1 拡張 ===
  brushup:
    enabled: true
    miss_count_window: 5        # 直近K日の欠損頻度
    streak_threshold: 3         # 欠損ストリーク閾値
    regime_change:
      recent_window: 5          # 直近ウィンドウ
      past_window: 30           # 過去ウィンドウ
      recent_threshold: 0.5     # 最近欠損率の閾値
      past_threshold: 0.1       # 過去欠損率の閾値

su5:
  # 既存設定...
  
  # === SU5 拡張 ===
  brushup:
    enabled: true
    cluster:
      n_clusters: 6             # k-means のクラスタ数
      random_state: 42
    include_density: true       # co_miss_density を追加
    include_deg_stats: true     # deg_sum, deg_mean を追加
    include_centrality: true    # miss_graph_centrality を追加
```

### 5.3 スイープ構成（軽め）

```yaml
brushup_sweep:
  su1_brushup_enabled: [true, false]
  su5_brushup_enabled: [true, false]
  su5_n_clusters: [4, 6, 8]
```

**スイープ数**: 2 × 2 × 3 = **12パターン**（非常に軽量）

---

## 6. 評価基準

### 6.1 採用条件

| 指標 | 条件 |
|------|------|
| OOF RMSE | ベースライン (0.012139) と同等以下 |
| OOF MSR | ベースライン比 +0.5σ 以上の改善 |
| LB Score | ベースライン (0.681) 以上 |

### 6.2 非採用条件（即座にロールバック）

- OOF RMSE が +2% 以上悪化
- LB Score が -0.01 以上悪化
- 特徴量数が +20 列を超える

### 6.3 評価フロー

```
1. SU1拡張のみでOOF評価
   - 改善 → 2へ
   - 悪化 → SU1拡張はOFF

2. SU5拡張のみでOOF評価
   - 改善 → 3へ
   - 悪化 → SU5拡張はOFF

3. SU1拡張 + SU5拡張 併用でOOF評価
   - 改善 → 4へ
   - 悪化 → 単独で良い方のみ採用

4. ベスト構成でKaggle LB提出
   - 改善 → 採用確定
   - 悪化 → ロールバック
```

---

## 7. リスクと制約

### 7.1 リスク

| リスク | 対策 |
|--------|------|
| k-means の CV リーク | fold ごとに fit/transform を分離 |
| 列数増加による過学習 | +10列以下に厳守 |
| OOF/LB 乖離 | 軽いスイープに留め、大量パターンは試さない |

### 7.2 明示的な「やらないこと」リスト

1. **多 window rolling**: window を 5 のみに限定、10/20/60 は使わない
2. **多集約**: mean のみ使用、std/max/min は使わない
3. **列ごとの拡張**: 全列に対して個別に特徴を追加しない（集約列のみ）
4. **外部データ連携**: date_id や SPY レジームとの組み合わせはしない
5. **Level-2 stacking**: 予測値の再利用はしない

---

## 8. 実装状況

### Sprint 1: SU1拡張実装 ✅

- [x] `feature_su1.py` に拡張特徴を追加
  - [x] `miss_count_last_5d` - 直近5日間の全列欠損数合計
  - [x] `miss_ratio_last_5d` - 直近5日間の欠損率
  - [x] `is_long_missing_streak` - 3日以上連続欠損フラグ
  - [x] `long_streak_col_count` - 3日以上連続欠損の列数
  - [x] `miss_regime_change` - 欠損レジーム変化フラグ
- [x] ユニットテストを追加 (`tests/feature_generation/test_su1.py`)
- [x] 設定を `configs/feature_generation.yaml` に追加
- [ ] OOF 評価を実行

### Sprint 2: SU5拡張実装 ✅

- [x] `feature_su5.py` に拡張特徴を追加
  - [x] `miss_pattern_cluster` - k-means クラスタID
  - [x] `co_miss_density` - top-k 共欠損ペアの同時欠損割合
  - [x] `co_miss_deg_sum` - 欠損列の共欠損次数合計
  - [x] `co_miss_deg_mean` - 欠損列の共欠損次数平均
  - [x] `miss_graph_centrality` - 欠損列が top-k ペアに含まれる回数
- [x] k-means の CV 対応を実装（fit時にtrainで学習）
- [x] ユニットテストを追加 (`tests/feature_generation/test_su5.py`)
- [x] 設定を `configs/feature_generation.yaml` に追加
- [ ] OOF 評価を実行

### Sprint 3: 統合・スイープ・LB提出 ✅

- [x] `train_su5.py` は brushup 対応済み（設定読み込みで自動適用）
- [x] `predict_su5.py` は brushup 対応済み（パイプライン内に組み込み）
- [x] 併用構成の OOF 評価
  - **結果**: OOF RMSE 0.012134（ベースライン 0.012139 から -0.04% 改善）
- [x] ベスト構成の Kaggle LB 提出
  - **結果**: LB 0.681（ベースラインと同等）
- [x] 採否判断・ドキュメント更新
  - **判断**: 採用保留（後続PCA/特徴量選定で再評価）

---

## 10. 評価結果サマリー

### OOF 評価結果

| 指標 | SU5 (ベースライン) | Brushup | 変化 |
|------|---------------------|---------|------|
| OOF RMSE | 0.012139 | **0.012134** | **-0.04%** ✅ |
| OOF MSR | 0.0199 | 0.01993 | ≈ 同等 |
| 特徴量数 | 567列 | 577列 | +10列 |

### LB 評価結果

| 指標 | SU5 (ベースライン) | Brushup | 変化 |
|------|---------------------|---------|------|
| **LB score** | 0.681 | **0.681** | **±0** |

### 結論

- **OOF**: 微改善（-0.04%）
- **LB**: ベースライン維持（悪化なし）
- **判断**: 変更を取り込み、後続のPCA/特徴量選定で再評価

### 学んだ教訓

1. **列数を抑えた微調整は安全**: +10列でLB悪化なし（SU7〜SU11のような大幅悪化なし）
2. **OOF微改善でもLB維持**: 過学習リスクを最小化できた
3. **「攻め」より「守り」**: 既存ラインを壊さない慎重なアプローチが重要

---

## 9. 参考リンク

- [SU1 仕様書](./SU1.md) - 現在の SU1 仕様
- [SU5 仕様書](./SU5.md) - 現在の SU5 仕様
- [submissions.md](../submissions.md) - 提出履歴
- [feature_generation.yaml](../../configs/feature_generation.yaml) - 設定ファイル
