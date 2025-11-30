# モメンタム・リバーサル特徴 SU7 仕様

最終更新: 2025-11-30

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

- SU7 では `su7_base_cols` は **「1 日リターン系列 r_t」** を前提とする。
  - 例: `ret_1d_spx`, `ret_1d_sector1` など、既存ベースラインで FI が高い日次リターン列。
- SU7 内で新たに価格からリターンを計算し直すことはせず、
  `su7_base_cols` をそのまま r_t とみなして diff / roll_ret / RSI / sign を計算する。
  - これにより、「一度だけ return 化してそこからすべて派生」というコンセプトを明示する。

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
- 数値安定性:
  - `eps` は `1e-8` 程度の小さな正数を推奨し、`ema_loss ≈ 0` のときの発散を抑える。
  - 必要に応じて `rs = clip(rs, 0, rs_max)`（例: `rs_max = 100`）などのクリップを入れ、NaN / Inf の発生を防ぐ。
- 列数:
  - 各ベース列に 1 本 → **B 列**。

### 3.4 方向フラグ（最小限）

- 役割重複を避けるため、direction flag は 1 種類のみ定義。
- 特徴量:
  - `sign_r_t/<col>` = sign(`r_t`) ∈ {−1, 0, 1}（int8）
- 列数:
  - **B 列**。
 - 前処理との整合:
   - `sign_r_t` は **順序を持つ 3 値の数値特徴**とみなし、他の数値特徴と同様にスケーリング対象に含める（OneHot には載せない）。

### 3.5 列数レンジの試算（例: B=6）

- diff/lag: `6B = 36`
- roll_ret/roll_mean_diff: `4B = 24`
- rsi_5: `B = 6`
- sign_r_t: `B = 6`
- 合計: **72 列**（目標 40〜80 列レンジ内）。

---

## 4. 共通ルールとリーク防止

- 想定ターゲット:
  - 本コンペのターゲットは「翌営業日の excess return（t+1）」を想定する。
  - よって **t 行の SU7 特徴はすべて時刻 t までの r_t のみから構成し、t+1 以降は一切参照しない。**
- 実装上の扱い:
  - SU7 の lag/diff/rolling/RSI はすべて **決まった数式をそのまま適用する deterministic transform** であり、学習データからパラメータを推定しない。
  - そのため、`SU7FeatureGenerator.fit` は `su7_base_cols` の存在確認と列順固定（`self.base_cols_`）のみを行い、fold ごとの統計量 fit は不要とする。
  - `transform` は date_id で sort 済みの全期間 DataFrame に対して `shift` / `rolling` / `ewm` を適用し、pandas の過去方向のみ参照する性質を前提にリークを防ぐ。
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

- SU7/SU8/SU9 の最終的な実行順序は `docs/feature_generation/README.md` 側に定義し、
  SU7.md からはそちらを参照する（本仕様では順序を固定しない）。
- 評価: SU1+SU5 ベースラインと比較し、OOF MSR/RMSE を指標とする。
- ロールバック条件:
  - OOF 改善が **fold 間分散を考慮して +0.3〜0.5σ 未満** の場合は SU7 をオフに戻す。
  - LB 提出は SU7 について最大 1 回までとし、OOF で明確に悪い設定は提出しない。

---

## 7. SU7 スイープ機能

### 7.1 スイープの目的

SU7 の列数・変換セットのバリエーションを複数パターン試し、
SU1+SU5 ベースラインと比較することで、最適な SU7 構成を探る。

評価指標:
- **RMSE を第一優先**
- 補助指標: MSR / vMSR / MSR_down

### 7.2 スイープ設定ファイル

スイープバリアントは `configs/su7_sweep.yaml` で定義する。

```yaml
variants:
  case_a:
    name: "top4_default_transforms"
    description: "Top 4 FI columns with default transform set"
    su7_base_cols:
      - M3
      - M4
      - S2
      - P5
    lags: [1, 5, 20]
    windows: [5, 20]
    halflife_rsi: 5
    use_rsi: true      # RSI 生成の ON/OFF
    use_sign: true     # sign フラグ生成の ON/OFF
    expected_feature_count: 48
```

### 7.3 変換フラグ

| フラグ | デフォルト | 説明 |
|--------|----------|------|
| `use_rsi` | `true` | RSI ライク指標を生成するか |
| `use_sign` | `true` | 方向フラグ sign(r_t) を生成するか |

これらのフラグを `false` に設定することで、特定の変換を無効化できる。

### 7.4 スイープ実行スクリプト

```bash
# 全バリアントを実行
python src/feature_generation/su7/run_su7_sweep.py

# 特定のバリアントのみ実行
python src/feature_generation/su7/run_su7_sweep.py --variants case_a case_b

# ドライラン（実行せずに設定を確認）
python src/feature_generation/su7/run_su7_sweep.py --dry-run
```

### 7.5 比較レポート

```bash
# テーブル形式で比較表示
python src/feature_generation/su7/compare_su7_variants.py

# CSV 出力
python src/feature_generation/su7/compare_su7_variants.py --output-format csv

# ベースラインを指定して差分計算
python src/feature_generation/su7/compare_su7_variants.py --baseline baseline
```

### 7.6 train_su7.py との統合

`train_su7.py` は以下のオプションでバリアントを指定可能:

```bash
python src/feature_generation/su7/train_su7.py \
  --su7-variant-name case_a \
  --su7-sweep-config configs/su7_sweep.yaml \
  --out-dir artifacts/SU7/case_a
```

## 8. スイープ結果サマリ（2025-11-29 時点・OOF ベース）

SU7 スイープは `configs/su7_sweep.yaml` で定義した `case_a`〜`case_f`, `baseline` を対象に、
SU1+SU5 ベースラインと同じ TimeSeriesSplit(5fold) 設定で実行した。

### 8.1 ベースライン（SU1+SU5）の指標

- OOF RMSE ≈ **0.012097**
- OOF MSR ≈ **0.013805**

### 8.2 主なバリアントの結果（OOF, best post-process）

- `case_a`（4 cols, RSI+sign）
  - RMSE: 0.012141
  - MSR: 0.014805
- `case_b`（6 cols, RSI+sign）
  - RMSE: 0.012135
  - MSR: 0.014235
- `case_c`（10 cols, RSI+sign）
  - `su7_base_cols` = `[M3, M4, M11, S2, S5, P5, P8, V9, M1, M2]`
  - RMSE: 0.012047
  - MSR: **0.019161**（全バリアント中 最大）
- `case_d`（12 cols, RSI+sign）
  - `su7_base_cols` = `[M3, M4, M11, S2, S5, P5, P8, V9, M1, M2, M5, M6]`
  - RMSE: **0.012045**（全バリアント中 最小）
  - MSR: 0.015722
- `case_e`（8 cols, sign のみ, RSI OFF）
  - RMSE: 0.012129
  - MSR: 0.009527
- `case_f`（8 cols, RSI+sign, rolling 5/10/20/60）
  - `su7_base_cols` = `[M3, M4, M11, S2, S5, P5, P8, V9]`
  - `windows` = `[5, 10, 20, 60]`
  - RMSE: 0.012106
  - MSR: 0.012968

### 8.3 採用ポリシー

- 評価ポリシーは「**RMSE を第一優先、MSR は RMSE 同等時のタイブレーク**」。
- 上記方針と結果に基づき、以下を採用した。

1. **本命（adopted）: case_d バリアント**
   - `su7_base_cols` = `[M3, M4, M11, S2, S5, P5, P8, V9, M1, M2, M5, M6]`（12 列）
   - `lags = [1, 5, 20]`, `windows = [5, 20]`, `use_rsi = true`, `use_sign = true`
   - baseline (SU1+SU5) に対して最良の RMSE を達成しつつ、MSR も十分良好。
   - `configs/feature_generation.yaml` の SU7 セクションは case_d に合わせて更新済み。

2. **サブ候補（fallback）: case_c バリアント**
   - `su7_base_cols` = `[M3, M4, M11, S2, S5, P5, P8, V9, M1, M2]`（10 列）
   - RMSE は case_d とほぼ同等（0.012047）、MSR は全バリアント中最大（0.019161）。
   - 列数を 10 に抑えたい・MSR をわずかに優先したい場合のバックアップ構成として温存。

3. その他の知見
   - RSI を切った sign-only 構成（case_e）は RMSE/MSR ともに悪化 → **RSI は有効**。
   - 4〜6 列構成（case_a, case_b）は RMSE/MSR ともに 10〜12 列構成に劣り、採用見送りとした。

今後 SU7 を変更する場合は、再度 `src/feature_generation/su7/run_su7_sweep.py` を用いて同様のスイープを行い、
本セクションの数値と `configs/feature_generation.yaml` の SU7 設定を同期させること。

---

## 9. Kaggle LB 結果と最終判断（2025-11-30 時点）

### 9.1 提出ラインと LB 結果

SU7 では、OOF で最も良好だった 2 つのバリアント（case_c, case_d）について、
SU5 ベースラインと同じ条件でフルパイプライン学習・成果物生成を行い、Kaggle Private Notebook から Public LB を取得した。

- ベースライン（SU1+SU5, SU7なし）
  - LB: **0.681** （Submission Unit 5, Policy1）

- SU7 case_c ライン
  - su7_base_cols: 10 列
  - OOF RMSE: ≈ **0.012047**（ベースライン 0.012097 よりわずかに良い）
  - Kaggle LB (Public): **0.476**

- SU7 case_d ライン（本命スイープ結果）
  - su7_base_cols: 12 列
  - OOF RMSE: ≈ **0.012045**（全バリアント中で最良）
  - Kaggle LB (Public): **0.469**

どちらの SU7 ラインも、OOF 指標上は SU1+SU5 ベースラインを僅かに上回っているにもかかわらず、
Public LB 上では **0.47 台まで大きく悪化** し、従来ベストスコア 0.681 を大きく下回る結果となった。

### 9.2 技術的検証と切り分け

LB 崩壊が実装バグや環境不整合によるものではないかを確認するため、以下を実施した。

1. **環境の固定と再学習**
   - `numpy==1.26.4`, `scikit-learn==1.7.2` を uv で固定し、train/predict の両方を同一環境で実行。
   - SU7 case_c / case_d の `inference_bundle.pkl`, `model_meta.json`, `feature_list.json`, `cv_fold_logs.csv`, `oof_predictions.csv` をすべて再生成。

2. **ポストプロセスの保守化**
   - 当初は OOF で最適化した post-process パラメータ（mult, lo, hi）を使用していたが、
     これが MSR 観点では攻め過ぎている可能性を疑い、`predict_su7.py` の `_resolve_postprocess_params` を
     「常に `mult=1.0, lo=0.9, hi=1.1` を返す**保守設定**」に変更。
   - この変更により、シグナル振れ幅を抑えた安全側の submit を case_c / case_d それぞれで再生成し、Kaggle に再提出。

3. **行数・フォーマットの確認**
   - `submission.csv` は `date_id,prediction` ヘッダ + 10 行（計 11 行）という Hull Tactical 固有仕様を満たしていることを確認。
   - SU5 ラインと同一フォーマットであり、「行数・形式起因のスコア異常」ではないことを確認。

4. **Kaggle 側 Notebook との互換性**
   - Notebook 用に `SU7Config`, `SU7FeatureGenerator`, `SU7FeatureAugmenter`, `SU7FullFeatureAugmenter` を埋め込み、
     `src.feature_generation.su7.feature_su7`, `train_su7` のモジュール登録を行い、`joblib.load` 時のクラス解決エラーを解消。
   - 旧版の「`__main__.SU7FullFeatureAugmenter` を指す pickle」をすべて置き換え、明示的モジュールパスを持つ pickle のみを使用。

これらの技術的対処の後も、SU7 case_c/case_d の LB は **0.47 台のまま大きく改善しなかった**。

### 9.3 原因分析: 広義の過学習 / レジームミスマッチ

上記の切り分けの結果、

- 実装バグ（特徴生成のリーク、fold_indices 不整合など）
- ライブラリバージョン不整合（MT19937 互換性問題）
- submission 行数・形式の誤り
- 過度な post-process によるシグナル暴走

といった「機械的な不具合」はほぼ排除できた。

それにもかかわらず、OOF では良好な SU7 case_c/case_d が Public LB 上で大きく崩れていることから、
最も説明力が高い仮説は以下となる。

- SU7 のモメンタム/リバーサル特徴は、**train+OOF 期間の市場レジームでは有効**だが、
  Public 評価期間のレジームとは相性が悪い。
- 特に MSR proxy を用いたシグナル最適化によって、ある種の「方向性に強く張る」構造になっており、
  Public 期間ではこれが逆方向に働いた可能性が高い。
- これはデータ分割の範囲内では検出しにくく、OOF では改善して見える一方で、
  実運用期間（Public 期間）では大きく負ける、**レジームミスマッチを含む広義の過学習**と解釈できる。

### 9.4 最終判断

- SU7 は OOF 指標上は魅力的に見えるものの、Public LB では致命的な悪化を招くことが分かった。
- 技術的なバグや環境の問題ではなく、「特徴の性質 × 評価期間の市場レジーム」の相性問題と判断。
- よって、**現行コンテストでは SU7 を採用しない** 方針とする。

結論:

- `configs/feature_generation.yaml` の `su7.enabled` を `false` とし、`status: not_adopted` へ変更。
- 本番ラインは引き続き **SU1+SU5（LB 0.681）** を使用する。
- SU7 のコード・スイープ設定・ドキュメントは将来の別レジーム/別コンペ向けの参考実装として残すが、
  本コンペにおける再提出は行わない。

