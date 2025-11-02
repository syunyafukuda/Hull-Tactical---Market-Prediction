# 特徴量生成ロードマップ

最終更新: 2025-11-01

これまでにすべての特徴量グループ（M/E/I/P/S/V）について欠損補完ポリシー戦略と Kaggle LB 検証を完了した。次のフェーズでは、新規特徴量の生成と拡充を体系的に進める。以下に網羅候補と実装ガードライン、実行優先度を整理する。

---

## 網羅リスト（生成候補）

### 直近観測距離・連続長
- `days_since_first_obs/<col>`, `days_since_last_obs/<col>`（未来は見ない）

### ローリング欠損率（過去のみ）
- 窓 `W∈{3,5,10,20,60}`: `roll_m_rate_W/<col>` = 過去W日でのNaN比
- 拡張: `exp_m_rate/<col>` = 0..tの累積NaN比（burn-in=20日）

### パターン変化フラグ
- `na_to_obs/<col>` = 1 if 前日NaNかつ当日観測
- `obs_to_na/<col>` = 1 if 前日観測かつ当日NaN
- `reappear_gap/<col>` = 直前の観測→観測間隔（NaNを跨いだ戻り間隔）

### 代入影響量（ポリシー追跡）
- `imp_used/<col>` = 代入実施フラグ（当日が補完で埋まったか）
- `imp_delta/<col>` = `x_imputed - x_raw`（観測日は0）
- `imp_absdelta/<col>` = `|imp_delta|`
- 代入種別ワンホット（例 `ffill`, `bfill`, `missforest`, `holiday_bridge`）。同日複数不可

### 共欠損構造（群内限定）
- `co_miss_cnt/<a>_<b>` = 当日 a と b 同時NaN のブール
- `co_miss_rolrate_W/<a>_<b>` = 過去W日での同時NaN比
- 次元圧縮案: 欠損フラグ行列の群内PCA成分 `miss_pca1..k`（fitはCV折内）

### 曜日・月次傾向（決定可能情報のみ）
- `dow_m_rate/<col>` = 過去同曜日のNaN比（expanding）
- `month_m_rate/<col>` = 過去同月のNaN比
- 祝日橋渡し×欠損の交差: `holiday_bridge * m/<col>`

### 欠損順位・相対位置
- 当日、群内での「欠けやすさ」順位: `rank_miss_prob_day/<grp>`
- 列内での観測再開の相対位置: `pos_since_reappear/<col>` = `gap_ffill` を `[0,1]` に正規化

### マスク付き統計の下準備用メタ
- `valid_share_W/<col>` = 過去W日での有効観測比
- `stable_window/<col>` = 最小Wで `valid_share_W≥τ` となるW（τ例0.8）
- 下流で「窓不足による不安定」を弾くフィルタ信号

### ヘッド/テイル保護とクリップ
- 先頭連続NaN時: `gap_ffill` は `W_max+1` でクリップ
- すべてNaN列は派生特徴を全てNaNまたは0に固定し除外候補タグ付け
- すべて観測列は `m/*` 系を0に統一

### 推奨パラメータ
- 窓: `W = {3,5,10,20,60}`。`min_periods = W`。
- クリップ: `gap_ffill`, `run_na ≤ 60`。`imp_delta` は ±p99でwinsorize。
- burn-in: 20日。burn-in未満は `exp_*` をNaNのまま。

# 特徴量生成ロードマップ（着手順）

最終更新: 2025-11-01

全グループ（M/E/I/P/S/V）の欠損補完ポリシーを確定し、Kaggle LB 検証まで完了した。次フェーズでは本ドキュメントの順序で特徴量生成 PoC を進める。各ステップは時系列リーク防止と CV 内 fit→transform を前提とし、V グループは当該前処理非変換（スケーリングのみ別検証）とする。

---

## 1. ガードライン（共通）

- 指標算出はすべて過去情報のみ。テスト 10 日分には学習期で fit した統計を持ち越す。
- パイプライン順序は「特徴生成 → 欠損補完（既定ポリシー）→ スケーリング」で固定。
- PoC は TimeSeriesSplit+gap による OOF MSR/Sharpe を必須とし、結果を `results/ablation/` に集約する。

---

## 2. 欠損構造特徴（全群）

- **2.1 欠損フラグ**: 列単位 `isna_col`、群単位 `miss_rate_group`、当日総欠損数 `miss_cnt_all`（`m_cnt/ALL`, `m_rate/ALL` を監視用に併記）。
- **2.2 直近観測距離**: 各列の `last_obs_gap`（ffill 距離、上限 `clip=60`）。
- **2.3 実装案**: `MissingnessFlags(cols_by_group, max_gap=60)`。
- **2.4 平均処理**: 全 NaN 列はグループ平均 (`avg_gapff/*`, `avg_run_na/*`) から除外し、監視用途の `m_cnt/ALL`, `m_rate/ALL` を併記。
- **2.5 テスト**: 全 NaN 列、先頭連続 NaN、島状 NaN、全観測列の 4 ケースで期待値一致。

---

## 3. ローリング統計（短・中・長）

- **3.1 窓設定**: \(W = \{3,5,10,20,60,120\}\)。
- **3.2 指標**: `mean`, `median`, `std`, `q10`, `q90`, `iqr`, `min`, `max`, `ema`（halflife=`{5,10,20,60}`）。
- **3.3 乖離系**: `x - roll_mean_W` と `x / (ε + roll_std_W)`。
- **3.4 実装案**: `RollingStats(windows=W, percentiles=(10,90), ema_halflife=(5,10,20,60))`。
- **3.5 ガード**: `min_periods=W` とし窓不足時は NaN 維持。
- **3.6 テスト**: 単調系列で閉形式に一致、境界インデックスで NaN を確認。

---

## 4. 拡張 Z・Robust-Z（expanding）

- **4.1 標準化**: 拡張平均・標準偏差で `z_exp=(x-μ_t)/σ_t`。初期除外 `burn_in=20`。
- **4.2 Robust-Z**: `(x-med_t)/(1.4826*MAD_t + ε)`。
- **4.3 実装案**: `ExpandingNormalizer(burn_in=20, robust=True)`。
- **4.4 テスト**: ホールドアウト分割で学習統計のみ用い test を正規化。

---

## 5. ボラティリティ特徴

- **5.1 局所変動**: `abs(Δx)` の `ema` / `mean`（窓はセクション 3 と同一）。二乗差 `(x - roll_mean)^2` の `roll_sum`。
- **5.2 σ 比**: `r = ewmstd_short / (ε + ewmstd_long)` （short ∈ {5,10}, long ∈ {20,60}）。
- **5.3 ボラ調整**: `x_adj = x / (1 + ewmstd_short)`。
- **5.4 実装案**: `VolFeatures(ewm_pairs=[(5,20),(10,60)], adjust=True)`。
- **5.5 テスト**: 定数列で 0、ホワイトノイズで推定 σ が理論値に近いか確認。

---

## 6. レジームタグ

- **6.1 ボラ区分**: `r` の分位 (0.33, 0.66) で低/中/高ボラの 3 値タグ。
- **6.2 トレンド代理**: 短期平滑 > 長期平滑 のブール（ゴールデンクロス指標）。
- **6.3 実装案**: `RegimeTagger(inputs=['r','ma_short','ma_long'], quantiles=(0.33,0.66))`。
- **6.4 テスト**: 人工系列でタグ切り替えが期待通りか検証。

---

## 7. レジーム × 基礎特徴の交差

- **7.1 交差方式**: `x * I(low_vol)`, `x * I(mid_vol)`, `x * I(high_vol)` のダミー乗算。
- **7.2 過学習抑制**: 対象は核となる特徴（各群で上位相関 10 列）に限定。`select_topK_by_variance/corr` で抽出。
- **7.3 実装案**: `RegimeInteractions(selected_cols, regime_cols)`。
- **7.4 テスト**: One-hot 漏れがなく、交差特徴の和が元の `x` に一致しないことを確認。

---

## 8. モメンタム / リバーサル簡易変換

- **8.1 差分**: `diff_k`（k∈{1,2,5,10,20}）、符号 `sign(diff_1)`。
- **8.2 RSI 風**: `rs = ema(gains)/ema(losses+ε)`、`rsi = rs/(1+rs)`。
- **8.3 クロス判定**: 短長期平均の `cross_up` / `cross_down` ブール。
- **8.4 実装案**: `MomentumLite(diffs=[1,2,5,10,20], rsi_halflife=5)`。
- **8.5 テスト**: 単調増加列で RSI→1、往復列で 0.5 付近となるか確認。

---

## 9. カレンダー特徴（決定可能情報のみ）

- **9.1 基本タグ**: `dow`, `dom`, `month`, `qtr`, `yday`。末日・四半期末・年末フラグ。
- **9.2 祝日前後**: 提供カレンダー基準のブリッジフラグ。
- **9.3 実装案**: `CalendarFeatures(holidays=provided_calendar)`。
- **9.4 テスト**: 日付境界の整合とリーク防止を確認。

---

## 10. 非線形基底の少数追加

- **10.1 基底**: `tanh(αx)`（α∈{0.5,1.0}), `log1p(|x|)*sign(x)`。
- **10.2 対象**: 拡張 Z 済みの中核列のみ。
- **10.3 実装案**: `NonlinearBasis(target_cols, kinds=['tanh','log1pabs'])`。
- **10.4 テスト**: 無限大や NaN が発生しないことを確認。

---

## 11. 群内 PCA と高相関合成（任意・後回し可）

- **11.1 高相関ペア**: |ρ|>0.95 の組み合わせで差分・比を 1–2 本のみ生成。
- **11.2 PCA**: 群内 PCA の第 1–3 成分（`whiten=False`、fit は CV 内）。
- **11.3 実装案**: `GroupPCA(groups, n_components=3)` と `PairwiseComposites(corr_thr=0.95)`。
- **11.4 テスト**: 分散保持率 ≥ 60% を確認。

---

## 12. クリーニングと選択

- **12.1 高相関除去**: |ρ|>0.98 で片方 drop（学習折内統計）。
- **12.2 VIF**: VIF>10 を除外候補として多重共線性を縮小。
- **12.3 PSI**: train 後期 vs test の PSI を算出し、PSI>0.3 を要監視。
- **12.4 実装案**: `FeatureSelect(corr=0.98, vif=10.0, psi_bins=10)`。
- **12.5 テスト**: 除去前後で学習安定性が向上するか確認。

---

## 13. スケーリング（CV 内 fit）

- **13.1 群別戦略**: `RobustScaler` を優先、次点 `StandardScaler`。外れ値の多い群は Robust。
- **13.2 適用順**: 生成 → 欠損補完 → スケーリング（既定ポリシーを踏襲）。
- **13.3 実装案**: `GroupScaler(strategy={'M':'robust','E':'robust','I':'standard','P':'robust','S':'standard'})`。
- **13.4 テスト**: train 統計で test を変換し、平均 0 付近・スケール安定を確認。

---

## 14. パイプライン統合と A/B

- **14.1 順序固定**: `sklearn.Pipeline` で `[Missingness → Rolling → ExpandingZ → Vol → Regime → Interactions → Momentum → Calendar → Nonlinear → Select → Scale]`。
- **14.2 評価**: TimeSeriesSplit+gap で OOF MSR を測定し、ベースライン比で平均上昇・分散減少を確認。
- **14.3 ログ**: 追加前後の OOF Sharpe、LB 差分、予測分布（平均/分散/尖度）を記録。
- **14.4 採用条件**: OOF MSR が +1σ 上振れ、かつ LB 非劣敗。劣敗時はロールバック。

---

## 15. 昇格と凍結

- **15.1 設定保存**: 勝ち設定を `configs/fe_*.yaml` に反映しタグ付け。
- **15.2 記録**: `docs/submissions.md` に OOF 指標・LB・差分要約・列数・学習時間を追記。
- **15.3 V グループ**: 引き続き非変換。スケーリングのみ許容するかは別 A/B で判断。

---

## 優先スプリント

- **Sprint 1**: セクション 2–7（欠損構造〜レジーム交差）
- **Sprint 2**: セクション 8–10（モメンタム・カレンダー・非線形）
- **Sprint 3**: セクション 11–13（PCA/合成・クリーニング・スケーリング）
- **Sprint 4**: セクション 14–15（統合 A/B・昇格）

各ステップ完了後は MSR/Sharpe 指標と Kaggle LB を両面確認し、効果が出ない場合は即座にロールバックする。

## SU シリーズの実装方針と構成

Sprint 1 以降で扱う SU（Submission Unit）は同一フォーマットで進める。以下は SU1 の例だが、SU2 以降も `su{n}` ディレクトリと `artifacts/su{n}` をペアで用意し、命名規約のみ置換する。

> 重要: SU 系の学習・推論成果物はすべて `numpy==1.26.4` で再生成する。Kaggle 推論環境とバージョンを揃えることで `joblib` が内部で利用する `MT19937` BitGenerator の互換性問題を回避できる。`pyproject.toml` の numpy ピンを変更しないまま `uv lock` / `uv sync` を実行し、異なる numpy 版で生成したバンドルは使用しない。

```
src/feature_generation/su1
	__init__.py
	feature_su1.py          # SU1生成器（sklearn Transformer互換）
	train_su1.py            # 既定前処理→SU1→学習→一体pkl出力
	predict_su1.py          # 一体pklによる推論
	sweep_oof.py            # OOFのみの軽スイープ（提出はしない）
configs/feature_generation.yaml            # SU1のパラメータ（clip, group mean など）
src/feature_generation/su1/test_su1.py
artifacts/su1/
	inference_bundle.pkl    # = 前処理+SU + スケール + モデル の Pipeline
	model_meta.json         # バージョン・seed・fold境界・前処理ハッシュ
	feature_list.json       # 生成カラム一覧
	cv_fold_logs.csv        # 各 fold の学習・評価ログ
	oof_grid_results.csv    # OOF スイープ結果
	submission.csv          # 提出ファイル（バックテスト用）
	submission.parquet      # 提出ファイル parquet 版
```

`feature_su1.py` で SU1 の欠損フラグ・距離・群集約を生成し、`configs/feature_generation.yaml` の設定値でパラメータを一元管理する。`train_su1.py` は既定前処理と SU1 を連結した `sklearn.Pipeline` を学習し、`artifacts/su1/inference_bundle.pkl` へ出力する。`predict_su1.py` はこのバンドルで推論を行う想定。`sweep_oof.py` では提出を伴わない軽量 OOF スイープを実施して設定を評価する。UT として `test_su1.py` を配置し、全 NaN 列・先頭欠損・島状欠損シナリオの再現テストを行う。

SU2 以降は同様に `feature_su2.py`, `config_su2.yaml`, `train_su2.py`…と命名を揃え、該当 SU が担当する特徴セットのみ差し替える。`artifacts/su{n}` も SU 番号ごとに分離し、バージョンアップ時は `model_meta.json` の `version` と `preprocess_hash` を更新する。OOF スイープは `sweep_oof.py` 内で `configs/preprocess.yaml` のフラグ切り替えを自動化し、`results/ablation/miss/SU{n}_yyyy-mm-dd.csv` へ書き出す運用を徹底する。

命名規約が統一されることで、CI/UT やアーティファクトの追跡、PR テンプレの再利用が容易になる。SU を跨いだ差分比較や提出ローテーションも、ディレクトリとファイル名の規則に従うだけで自動化スクリプトに乗せられる。

生データに対してまず SU1 を適用し、欠損の構造情報を固めてから、確定前処理（欠補や変換）をかける。

**理由**

- 前処理で欠補・補間を先に行うと、欠損構造が消えたり歪む。SU1 は **"生の欠損"** を数値化する設計。
- SU1 は未来参照なしで生成可。以後の欠補・スケールと独立。

**パイプライン構成**

- CLI から `configs/preprocess.yaml` を読み込み、そこで定義されたポリシーで M/E/I/P/S の各 `GroupImputer` を構築する。
- 学習・推論ともに処理順序は **生データ → SU1FeatureAugmenter → MGroupImputer → EGroupImputer → IGroupImputer → PGroupImputer → SGroupImputer → ColumnTransformer → LightGBM** で固定する。
- CV では SU1 を全期間一括で生成した後に時系列分割し、fold 内では確定前処理以降を clone して学習する（履歴の切断を防ぐため）。
- `train_su1.py` / `predict_su1.py` は上記パイプラインをバンドルとして保存・再利用するため、学習時と同じ確定前処理が推論にも確実に適用される。

## SU 単位ごとの運用メモ

- **SU1_単列コア**
	- 内容: `m/<col>`, `m_any_day`, `m_rate_day`, `m_cnt/ALL`, `m_rate/ALL`, `gap_ffill/<col>`, `run_na/<col>`, `run_obs/<col>`, 群集約 `m_cnt/m_rate/<grp>`, `avg_gapff/<grp>`, `avg_run_na/<grp>`
	- ブランチ: `feat/miss-core-su1`
	- 提出可否: `ΔMSR ≥ +0.5σ` かつ 分散過度↑なし
	- 備考: 全グループ（含 V）を対象に、先頭 NaN 保護と `clip≤60`、全 NaN 列除外ロジック、列名プレフィクス（英大文字+数字）前提を明示テスト
- **SU2_履歴率**
	- 内容: `roll_m_rate_W/<col>` `W∈{5,10,20,60}`, `exp_m_rate/<col>`, `valid_share_W/<col>`, `stable_window/<col>`
	- ブランチ: `feat/miss-hist-su2`（設定スイープは `sweep/miss-hist-win`）
	- 提出可否: SU1 適用を前提に OOF で `+0.5σ`
	- 備考: `burn-in=20` 未満は NaN 据置
- **SU3_遷移・再出現**
	- 内容: `na_to_obs/<col>`, `obs_to_na/<col>`, `reappear_gap/<col>`, `pos_since_reappear/<col>`, `rank_miss_prob_day/<grp>`
	- ブランチ: `feat/miss-transitions-su3`
	- 提出可否: `ΔMSR` が僅少でも予測分散が安定なら提出候補
	- 備考: すべて過去参照のみ
- **SU4_代入影響トレース**
	- 内容: `imp_used/<col>`, `imp_delta/<col>`, `imp_absdelta/<col>`, 代入種別 One-hot（`ffill/bfill/missforest/holiday_bridge`）, 交差 `holiday_bridge * m/<col>`
	- ブランチ: `feat/miss-imptrace-su4`
	- 提出可否: `ΔMSR ≥ +0.5σ` または外れ提出で LB ↑
	- 備考: `imp_delta` は ±p99 winsorize
- **SU5_共欠損・交差**
	- 内容: `co_miss_cnt/<a>_<b>`, `co_miss_rolrate_W/<a>_<b>`、群内上位 K ペアのみ（`K=10` 目安）
	- ブランチ: `feat/miss-comiss-k10-su5`
	- 提出可否: OOF で `+0.5σ` かつ PSI 悪化なし
	- 備考: ペア選定は学習期の相関 `|ρ|` 上位→fold 内固定
- **SU6_圧縮（必要時のみ）**
	- 内容: 群内欠損フラグ行列 PCA `miss_pca1..k`（`k≤3`, fold 内 fit）
	- ブランチ: `feat/miss-pca-k2-su6`
	- 提出可否: 特徴爆発時のみ適用し `ΔMSR` 非劣化を確認
	- 備考: コンポーネントは学習折のみに fit

### 実行順（軽→重）
- コア: 1,2,3,4,5,9,10
- 拡張: 6,7,8（共欠損と曜日・月傾向、順位系）
- 圧縮: 群内 miss-PCA（必要最低限の成分のみ）

### コスト最適化
- 2値フラグは `uint8`、距離は `int16` に収める。
- ペア特徴は高相関列上位Kのみ。`K=10` 想定。
- 列数が閾値超過時は `valid_share_W` で弱列を先に間引く。

### 提出運用ルール
- 1日最大2提出。各SUでOOFスイープ→最良構成のみ提出。
- LB劣化閾値: ベース比 `−0.002` 以下で即リバート。
- 横ばいは保留1回まで。改善なければ落とす。

### フラグ設計例（`configs/feature_generation.yaml`）
```
features:
	miss_core: true      # SU1
	miss_hist: false     # SU2
	miss_transitions: false  # SU3
	miss_imptrace: false     # SU4
	miss_comiss: false       # SU5
	miss_pca: false          # SU6
params:
	miss_hist:
		windows: [5,10,20,60]
		burn_in: 20
	miss_core:
		max_gap_clip: 60
	miss_imptrace:
		winsor_p: 0.99
	miss_comiss:
		top_k_pairs: 10
```

### PRテンプレ最小項目
- 目的とリークリスク確認チェック
- 追加特徴一覧と命名規約
- OOF表: MSR/Sharpe/分散/PSI
- 提出判断: 可 or 否（根拠）
- 再現条件: seed, fold境界, 有効フラグ
