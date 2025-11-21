# 欠損構造特徴 SU2 仕様（SU1を基盤とした二次派生） - **非採用**

> **⚠️ 重要**: SU2は2025-11-21時点で**非採用**と決定。LBスコア大幅悪化により採用見送り。
> - SU1 LB: 0.674 (Public) → **SU2 LB: 0.597 (Public)** (-0.077ポイント)
> - 原因: 935個の二次特徴量による過学習・汎化性能低下
> - 決定: SU1をベースラインとして継続採用

---

## 非採用の経緯と分析 (2025-11-21)

### 性能比較

| モデル | OOF RMSE | OOF MSR | LB Score (Public) | 特徴量数 |
|--------|----------|---------|-------------------|----------|
| SU1    | 0.01212  | 0.01821 | **0.674** | 462 (94+368) |
| SU2    | 0.01223  | 0.02319 | **0.597** | 1397 (94+368+935) |

### 問題点の特定

1. **過学習**: CV上ではSU1と同等の性能だが、LBでは大幅劣化
   - OOFとLBのギャップが異常に大きい (-0.077ポイント)
   - 935個の二次特徴量が訓練データに過適合

2. **特徴量爆発**: 94→462(SU1)→1397(SU2) と3倍に増加
   - rolling/EWMA/transition/normalization特徴をすべて生成
   - 特徴量選択なし・正則化不足

3. **時系列分割への過適合**: fold_indicesによるCV最適化が未来データで無効
   - 学習時の折境界リセットが、連続した未来データでは機能しない
   - CV上の良好な性能が実運用で再現されない

4. **市場体制変化への脆弱性**: rolling/EWMA特徴が過去パターンに依存しすぎ
   - 過去の統計量が未来の予測に有効でない（非定常性）
   - SU1の「欠損構造そのもの」はロバストだが、SU2の「統計的派生」は不安定

### 技術的検証

- パイプライン実装: 正常動作確認済み（ローカル推論で1397列正しく生成）
- 推論コード: Kaggleノートブックも正常動作（エラーなし）
- 問題はロジックではなく**設計思想**にあると判断

### 今後の方針

- **SU1をベースライン継続**: LB 0.674で十分有効
- **SU2は完全見送り**: 特徴量数削減では根本解決にならないと判断
- **改善方向**: SU1ベースで特徴量選択・正則化強化を検討
  - LightGBMの`feature_fraction`調整
  - `reg_alpha`, `reg_lambda`の増加
  - SHAP/permutation importanceによる特徴量削減

---

## 目的（参考: 当初設計）

- SU1で生成した一次信号（欠損フラグ・距離・連続長）を時間方向に集計・平滑化し、**局所的な欠損レジーム**（急増・慢性化・回復）を検知する。
- 予測器の分散を不必要に増やさず、**リークなし**で時系列の状態変化を捉える。

## スコープ

- 入力: 生データではなく **SU1出力**（`m/<col>`, `gap_ffill/<col>`, `run_na/<col>`, `run_obs/<col>`）
- 対象グループ: D/M/E/I/P/S/V の全列
- 出力: ローリング統計・指数平滑・遷移統計・正規化派生
- 依存: SU1が先に生成済みであること（fold境界のリセット規約を継承）

## 運用ルール（SU2）

1. 未来参照禁止。すべて **過去方向のみ** のロール・遅延で構成。
2. 折境界で**状態をリセット**。学習各foldは独立した初期状態。
3. クリップは上限60を継承（距離・連続長系）。
4. データ型は省メモリを優先。flag=uint8, small ints=int16, 小数=float32。
5. 列名は SU1命名に `su2/` プレフィックスを追加し一意化。

## 生成する特徴（定義）

### 1. ローリング統計（過去のみ）

- 対象信号: `m/<col>`, `run_na/<col>`, `run_obs/<col>`
- デフォルト窓候補: `W ∈ {5, 10, 20, 60}`
- 現行RMSEベスト設定（2025-11-07 時点）: 使用窓 = `[5]`
- 出力例:
  - `su2/roll_mean[W]/m/<col>`: 過去Wでの平均欠損率
  - `su2/roll_std[W]/m/<col>`: 欠損フラグの標準偏差
  - `su2/roll_zscore[W]/m/<col>`: (現在値 - 過去W平均) / 過去W不偏標準偏差（分母<epsは0）
  - `su2/roll_max[W]/run_na/<col>` / `run_obs/<col>`: 最大連続長

### 2. 指数平滑（EWMA/EWSTD）

- 対象信号: `m/<col>` と `gap_ffill/<col>`
- 平滑係数 候補例: `α ∈ {0.1, 0.3, 0.5}` → 最終は `[0.1]` に縮約
- 出力例:
  - `su2/ewma[α]/m/<col>`
  - `su2/ewstd[α]/m/<col>`（Welford型逐次更新）
  - `su2/ewma[α]/gap/<col>`

### 3. 遷移・レジーム統計

- `su2/flip_rate[W]/m/<col>`: 過去Wでの 0↔1 遷移回数 / W
- `su2/burst_score[W]/<col>`: `max_roll(run_na)/(max_roll(run_na)+max_roll(run_obs)+1e-6)`
- `su2/recovery_lag/<col>`: 直近 run_na>0 から run_obs>0 への変化後経過日数（clip≤60）

### 4. 正規化・スケーリング

- `su2/minmax[W]/gap/<col>`: 過去Wに対するmin-max正規化（分母<epsは0）
- `su2/rank[W]/gap/<col>`: 過去W内ランク / W（0-1）

## 実装手順（パイプライン）

1. 入力取得: SU1完成テーブルと`date_id`を受け取る。
2. 折境界インデックス生成: CV foldごとに連続チャンク。
3. 逐次処理: 各チャンクで時系列順に一回走査。
4. クリップと型ダウン: clip≤60, dtype最適化。
5. 定数列除去: 全欠損・定数化列をドロップ。
6. メタ出力: `model_meta.json` に SU2列の由来とパラメータを書き出し。

## スイープ可能パラメータ一覧（実装観点）

| 区分 | キー | 役割 | 例 | 対応 | 備考 |
|------|------|------|----|------|------|
| Rolling | rolling_windows | 時間窓 | [5,10] | 実装済 | 最終は[5]固定 |
| Rolling | include_metrics | 統計種 | [mean,std,zscore] | 実装済 | 最終は mean,std |
| Rolling | include_current | 現在値含む | bool | 実装済 | false 固定 |
| EWMA | ewma_alpha | 平滑係数 | [0.1,0.3] | 実装済 | 最終は[0.1] |
| EWMA | signals | 入力信号集合 | [m,gap_ffill] | 実装済 | 拡張余地あり |
| EWMA | include_std | 分散付与 | bool | 実装済 | true 固定 |
| EWMA | reset_each_fold | Fold初期化 | bool | 実装済 | true 固定 |
| Transitions | windows | 遷移窓 | [5,10,20,60] | 実装済 | 部分集合アブレーション可 |
| Transitions | recovery_clip | 回復ラグ上限 | 60 | 実装済 | 30/90検証候補 |
| Normalization | mode | both / minmax / rank | both | 実装済 | 最終 both 固定 |
| Normalization | windows | 正規化窓 | [5,10,20,60] | 実装済 | 短縮アブレーション可 |
| Normalization | epsilon | 数値安定 | 1e-6 | 実装済 | 微調整影響小 |
| Sources | input_sources | 入力基底集合 | [m,gap_ffill,run_na,run_obs] | 実装済 | 削減で速度向上可 |
| Global | clip_max | 長さ系clip上限 | 30/60 | 実装済 | 30採択 |
| Global | drop_constant_columns | 定数列drop | bool | 実装済 | true 推奨 |
| Global | dtype.* | 型指定 | float32等 | 実装済 | メモリ最適化 |

## 現行RMSEベスト構成サマリ

```yaml
最終決定 (2025-11-07 Stage2, config_id=3):
rolling_windows: [5]
ewma_alpha: [0.1]
include_transitions: true (windows=[5,10,20,60])
include_normalization: true (mode=both, windows=[5,10,20,60], epsilon=1e-6)
metrics: mean,std
ewma.signals: m,gap_ffill (include_std=true, reset_each_fold=true)
input_sources: m,gap_ffill,run_na,run_obs
recovery_clip: 60
clip_max: 30
```

指標 (5-fold OOF):

- RMSE: 0.012153
- MSR: 0.022968
- coverage: 0.8331

【更新 2025-11-07 再学習 OOF 指標】

- RMSE: 0.012300
- MSR: 0.017477
- coverage: 0.8331
- source: artifacts/SU2/model_meta.json（oof_best_metrics / oof_coverage）

同値クラス（スコア完全同値）: config_id=6, 9, 12

- これらは transition/normalization 窓の組み合わせ簡略差のみで指標同一。
- 再現性と網羅性を重視し、最も包括的な窓集合を含む config_id=3 を代表採択。

## 追加のアブレーション候補

- 窓簡略化: normalization_windows を [5,10] / [5,10,20] へ短縮し計算量を削減可能か検証。
- recovery_clip 調整: 60 → 30/90 で鋭敏度 vs 安定性バランスを評価。
- signals 拡張: EWMA 入力へ run_na/run_obs を追加し marginal gain を確認（改善なければ現状維持）。


<!-- 重複していた旧仕様セクションを整理済み：不要な再掲を削除 -->

### 4. 正規化・スケーリング

- `su2/minmax[W]/gap/<col>`: 過去Wに対するmin-max正規化（分母<epsは0）
- `su2/rank[W]/gap/<col>`: 過去W内ランク / W（0-1）

## 実装手順（パイプライン）

1. **入力取得**: SU1完成テーブルと`date_id`を受け取る。
2. **折境界インデックス生成**: CV foldごとに連続チャンクを作成。
3. **逐次処理**: 各チャンクで時系列順に一次パス。
  - ローリング統計: `deque` もしくは滑り窓累積和。現在tの出力は **t−1まで** を元に計算。
  - EWMA/EWSTD: 再帰更新。
  - 遷移数: `m[t] != m[t-1]` をカウント。
  - 回復ラグ: run_na→run_obs の境界検出でカウンタリセット。
4. **型とクリップ**: clip≤60、型ダウンクラス（uint8,int16,float32）。
5. **カラム選別**: 全欠損・定数化の列はドロップ。
6. **メタ出力**: `model_meta.json` に SU2列の由来とパラメータを書き出す。

## スイープ可能パラメータ一覧（実装観点）
| 区分 | キー | 役割 | 例 | 対応 | 備考 |
|------|------|------|----|------|------|
| Rolling | rolling_windows | 時間窓 | [5,10] | 実装済 | 最終は[5]固定 |
| Rolling | include_metrics | 統計種 | [mean,std,zscore] | 実装済 | 最終は mean,std |
| Rolling | include_current | 現在値含む | bool | 実装済 | false 固定 |
| EWMA | ewma_alpha | 平滑係数 | [0.1,0.3] | 実装済 | 最終は[0.1] |
| EWMA | signals | 入力信号集合 | [m,gap_ffill] | 実装済 | 拡張余地あり |
| EWMA | include_std | 分散付与 | bool | 実装済 | true 固定 |
| EWMA | reset_each_fold | Fold初期化 | bool | 実装済 | true 固定 |
| Transitions | windows | 遷移窓 | [5,10,20,60] | 実装済 | 部分集合アブレーション可 |
| Transitions | recovery_clip | 回復ラグ上限 | 60 | 実装済 | 30/90検証候補 |
| Normalization | mode | both / minmax / rank | both | 実装済 | 最終 both 固定 |
| Normalization | windows | 正規化窓 | [5,10,20,60] | 実装済 | 短縮アブレーション可 |
| Normalization | epsilon | 数値安定 | 1e-6 | 実装済 | 微調整影響小 |
| Sources | input_sources | 入力基底集合 | [m,gap_ffill,run_na,run_obs] | 実装済 | 削減で速度向上可 |
| Global | clip_max | 長さ系clip上限 | 30/60 | 実装済 | 30採択 |
| Global | drop_constant_columns | 定数列drop | bool | 実装済 | true 推奨 |
| Global | dtype.* | 型指定 | float32等 | 実装済 | メモリ最適化 |

※ △ は今回コード拡張でスイープ対象に追加予定の項目です。

## 現行RMSEベスト構成サマリ
```yaml
最終決定 (2025-11-07 Stage2, config_id=3):
rolling_windows: [5]
ewma_alpha: [0.1]
include_transitions: true (windows=[5,10,20,60])
include_normalization: true (mode=both, windows=[5,10,20,60], epsilon=1e-6)
metrics: mean,std
ewma.signals: m,gap_ffill (include_std=true, reset_each_fold=true)
input_sources: m,gap_ffill,run_na,run_obs
recovery_clip: 60
clip_max: 30
```

指標 (5-fold OOF):
- RMSE: 0.012153
- MSR: 0.022968
- coverage: 0.8331

同値クラス（スコア完全同値）: config_id=6, 9, 12
- これらは transition/normalization の窓の組み合わせが一部簡略化されているが、指標は同一。
- 再現性と汎用性を優先して「双方とも [5,10,20,60] を含む」config_id=3 を代表として採択。

## 追加のアブレーション候補
- 窓の簡略化: normalization_windows を [5,10] / [5,10,20] に短縮し計算量を削減（RMSE差分が無いか検証）。
- recovery_clip の微調整: 60 → 30/90 で鋭敏度と外れ抑制のバランスを評価。
- signals 拡張: EWMA 入力へ run_na/run_obs を加える（寄与が小さい場合は現状維持）。