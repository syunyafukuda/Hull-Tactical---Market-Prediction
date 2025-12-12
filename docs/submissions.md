# Submissions log

提出結果の履歴を管理します。原則として成果物（artifacts）はGitに含めないため、ここではメタ情報のみを記録します。

- Repo: Hull-Tactical---Market-Prediction
- Branch: feature/simple-baseline-sub

## 2025-10-05 simple_baseline

- Commit: 418c4cd
- Submit line: simple_baseline
- Kaggle Notebook: Private (Dataset: simple_baseline_hulltactical)
- LB score: 0.554
- Notes:
  - ローカルで `train_simple.py` → `predict_simple.py` にて整合性確認。
  - Kaggle Notebook では `joblib.load(model_simple.pkl)` と `model_meta.json` から `feature_columns` を再現。
  - `predict(test: pl.DataFrame) -> float` を `DefaultInferenceServer` と組み合わせて採点APIに対応。
  - scikit-learn は互換WheelをDatasetに同梱し `--no-index` で導入。

 
## 2025-10-12 msr_proxy (MSR-proxy)

- Commit: bba4fdc
- Submit line: MSR-proxy（MSR/vMSR プロキシ最適化ライン）
- Kaggle Notebook: Private（Dataset: msr-proxy-hulltactical）
- LB score: 0.610
- Notes:
  - 依存のピン止め（Notebook 冒頭）
    - scikit-learn 1.7.2 を Private Dataset の wheel から `--no-index --no-deps --force-reinstall` で導入
    - lightgbm==4.6.0, joblib, pyarrow, polars, pandas を pip で追加
  - 成果物
    - `model_msr_proxy.pkl`（ColumnTransformer+LGBM のパイプライン）
    - `model_meta.json`（feature_columns, numeric/categorical, OOF 指標, OOF 最適 post-process など）
  - 予測ロジック（Notebook 概要）
    - meta の `feature_columns` に列を合わせ、欠損は数値=NaN/カテゴリ="missing" で補完
    - `pipe.predict(X)` の生予測に対して post-process を適用
      - 既定は meta の OOF 最適: mult=1.5, lo=0.8, hi=1.0（clip 0..2）
      - signal = 1 + mult*yhat を上下で非対称スロープ変換→必要に応じクリップ
    - API モード: `predict(test: pl.DataFrame) -> float` を `DefaultInferenceServer` で提供
  - OOF 概要（ローカル学習ログより）
    - rmse≈0.01241, msr≈0.02307, coverage≈0.833
    - グリッド最良は一貫して mult≈1.5、hi=1.0 を選択（分散抑制）
  - 所感
    - 出力は 1±数ミリ（例: 0.996〜1.005）に収束するが、MSR 指標上は有効な微傾き
    - LightGBM の「No further splits…」警告は一部foldで観測。特徴量/ハイパラの掘り下げ余地あり

## 2025-10-16 m_policy=ridge_stack (Preprocessing M)

- Commit: 9db9dcb
- Submit line: preprocessing_m (M-group ridge_stack policy)
- Kaggle Notebook: Private（Dataset: preprocess-m-group-hull-tactical）
- LB score: 0.629
- Notes:
  - 新規カスタム推論ノートブック `src/preprocess/M_group/kaggle_preprocessing_m.ipynb` を Kaggle に移植。
  - scikit-learn 1.7.2 wheel（cp311 build）を private dataset に同梱し `--no-index --no-deps --force-reinstall` で導入。
  - `htmpre.m_group.MGroupImputer` / `preprocess.M_group.m_group` のモジュール名を shim し、`joblib.load(model_pre_m.pkl)` 前に登録。
  - メタデータ `model_meta.json` の `feature_columns` / `m_policy` / calendar column を全て検証してから推論。
  - 最終セルは `DefaultInferenceServer` で `predict(test: pl.DataFrame) -> float` を公開し、Private LB で 0.629 を確認。

## 2025-10-19 i_policy sweep (Preprocessing I)

- Kaggle Notebook: Private（Dataset: preprocess-i-group-hull-tactical）
- LB scores (Public):
  - ridge_stack — 0.623
  - knn_k — 0.623
  - missforest — 0.561
- Decision: ridge_stack を I 系特徴量の既定ポリシーとして継続採用（knn_k は同スコア、missforest は今回劣後）。

## 2025-10-25 p_policy sweep (Preprocessing P)

- Kaggle Notebooks (Private dataset: preprocess-p-group-hull-tactical)
- LB scores (Public):
  - kaggle_preprocessing_p_mice — 0.625
  - kaggle_preprocessing_p_ridge_stack — 0.623
  - kaggle_preprocessing_p_kalman_local_level — 0.557
  - kaggle_preprocessing_p_state_space_custom — 0.554
- Decision: mice を P 系特徴量の既定ポリシーとして採用（最良スコア 0.625）。

## 2025-10-26 s_policy sweep (Preprocessing S)

- Kaggle Notebooks (Private dataset: preprocess-s-group-hull-tactical)
- LB scores (Public):
  - kaggle_s_missforest — 0.616
  - kaggle_s_kalman_local_level — 0.616
  - kaggle_s_holiday_bridge — 0.557
- Decision: missforest を S 系特徴量の既定ポリシーとして採択（kalman_local_level は同率次点、holiday_bridge は Sharpe 低下のため不採用）。
- Notes:
  - 3 ポリシーとも `numpy==1.26.4`, `scikit-learn==1.7.2` で再学習し、MT19937 pickle 互換エラーを解消。
  - Kaggle Notebook では `sys.path.append('src')` を追加して `joblib.load` を実行することで `preprocess` モジュールの import エラーを回避。

## 2025-11-01 v_policy sweep (Preprocessing V)

- Kaggle Notebooks (Private dataset: preprocess-v-group-hull-tactical)
- LB scores (Public):
  - kaggle_preprocessing_v_ffill_bfill — 0.590
  - kaggle_preprocessing_v_holiday_bridge — 0.590
- Decision: V 系ポリシーは現行ラインを下回ったため未採用 (`configs/preprocess.yaml` では `enabled=false`)。
- Notes:
  - numpy 1.26.4 / pandas 2.2.2 へ互換化した上で再学習・再提出したが Sharpe 系改善に繋がらず。
  - アーティファクトは `artifacts/Preprocessing_V/` に保持し、後続改善のベースラインとして残す。

## 2025-11-04 su1 (Submission Unit 1)

- Kaggle Notebook: Private（Dataset: su1-missingness-core）
- LB score: 0.674 (Public)
- Notes:
  - `src/feature_generation/su1/train_su1.py` を uv 経由で再学習し、`artifacts/SU1/` に `inference_bundle.pkl`, `model_meta.json`, `feature_list.json`, `submission.csv` を再生成。
  - 学習・推論ともに `numpy==1.26.4` を固定。`model_meta.json` の `library_versions.numpy` も 1.26.4 へ更新済み。
  - Kaggle Notebook 側では追加の BitGenerator alias を挿入せず、`joblib.load` がそのまま通ることを確認。
  - Private Dataset には SU1 バンドルと scikit-learn 1.7.2 wheel を同梱し、Notebook 起動時に `pip install --no-index` で導入する。

## 2025-11-21 su2 (Submission Unit 2) - **非採用**

- Kaggle Notebook: Private（Dataset: su2-missingness-core）
- LB score: 0.597 (Public) ← **SU1の0.674から大幅悪化**
- Decision: **非採用** - SU1をベースラインとして継続採用
- Notes:
  - OOF性能
    - SU1: OOF RMSE=0.01212, MSR=0.01821 → LB 0.674
    - SU2: OOF RMSE=0.01223, MSR=0.02319 → LB 0.597
    - **CV上ではSU1と同等だが、LBでは-0.077ポイントの大幅劣化**
  - 特徴量構成
    - Pipeline input: 94列（生データ）
    - SU1特徴: 368列（欠損構造一次特徴）
    - SU2特徴: 935列（二次特徴: rolling/EWMA/transition/normalization）
    - 合計: 1397列 ← **特徴量爆発**
  - 問題分析
    - パイプライン実装は正常動作確認済み（ローカル推論でも1397列正しく生成）
    - **過学習**: 935個の二次特徴量が訓練データに過適合し汎化性能が低下
    - **時系列分割への過適合**: fold_indicesによるCV最適化が未来データで無効
    - **市場体制変化への脆弱性**: rolling/EWMA特徴が過去パターンに依存しすぎ
  - SU2設定（configs/feature_generation.yaml）
    - rolling_windows=[5], ewma_alpha=[0.1]
    - drop_constant_columns=true
    - 特徴量選択なし → 935列すべて使用
  - 結論
    - SU2の設計思想（二次特徴量の大量生成）が本コンペには不適合
    - 特徴量数94→462(SU1)→1397(SU2)の爆発的増加が主因
    - SU1の368列で十分有効であり、SU2による複雑化は不要
    - 今後の改善方向: SU1ベースで特徴量選択・正則化強化を検討

## 2025-11-22 su3_stage1 (Submission Unit 3 - Stage 1) - **非採用**

- Branch: `feat/miss-core-su3`
- Status: **非採用** - LBスコア改善なし、SU1をベースラインとして継続
- Kaggle Notebook: Private（Dataset: su3-missingness-core）
- LB score: **0.461 (Public)** ← **初回提出と同じ、修正の効果なし**
- Decision: **完全放棄** - Stage 2の開発は行わない
- ベスト構成 (OOF sweep結果):
  - reappear_top_k: 20
  - temporal_top_k: 10
  - holiday_top_k: 10
  - include_imputation_trace: False (Stage 1)
- OOF性能:
  - OOF RMSE: 0.011418
  - OOF MSR: 0.017162 ← **SU1の0.026376の65.1%（大幅劣化）**
  - 特徴量数: 444列 (SU1: 368列 + SU3: 76列)
- LB推移:
  - 初回提出（fold_indices不整合あり）: 0.461
  - 修正版（fold_indices不整合解消）: **0.461（変化なし）**
- 問題分析:
  - **致命的な性能劣化**: OOF MSRがSU1の65%しかない
  - **修正の無効性**: train/inference不整合の修正後もLBスコア不変
  - **根本的なコンセプトの失敗**: 欠損パターンの二次特徴（遷移/再出現/時間的傾向）が本コンペでは無効
  - **テストデータでの予測崩壊**: 予測値が全て同じ値（-0.0046前後）→signal変換後に全て1.0
- 修正履歴:
  1. **fold_indices不整合の修正**:
     - 問題: 最終学習時にfold_indicesを渡していなかった
     - 修正: 推論時と同じ挙動（fold_indicesなし）に統一
     - 結果: **LBスコア不変（0.461）** → 修正は無意味だった
  2. **トラブルシューティング記録**:
     - MSR=0問題: PostProcessParams(lo=1.0, hi=1.0)の設定ミス（解決済み）
     - 詳細: docs/feature_generation/troubleshooting/MSR_zero_issue.md
     - LB崩壊分析: docs/feature_generation/SU3_LB_COLLAPSE_ANALYSIS.md
- SU2との比較:
  - SU2: OOF良好 → LB 0.597（過学習）
  - SU3: OOF時点で劣化 → LB 0.461（コンセプトの失敗）
  - SU3の方がより根本的な問題を抱えている
- 教訓:
  - ❌ 欠損パターンの高次特徴化は本コンペでは効果なし
  - ❌ OOF MSRがベースラインの65%の時点で採用すべきでなかった
  - ✅ SU1（一次欠損特徴）が最適解、これ以上の複雑化は不要
  - ✅ Stage 2の開発は時間の無駄、SU1の改善に集中すべき
- 最終判断:
  - **SU3完全放棄**: Stage 2は開発しない
  - **SU1継続採用**: LB 0.674がベストスコア
  - **今後の方向性**: SU1のハイパーパラメータチューニング、アンサンブル検討
- 成果物:
  - スイープ結果: results/ablation/SU3/sweep_2025-11-22_110535.json
  - アーティファクト: artifacts/SU3/ (参考保存のみ)
  - 構成: configs/feature_generation.yaml (su3.enabled=false に戻す)
  - 今後SU3関連のコードは保守対象外

## 2025-11-23 su5 (Submission Unit 5) - **最高スコア達成🎉 採用決定**

- Branch: `feat/miss-core-su5`
- Kaggle Notebook: Private（Dataset: su5-missingness-core）
- **LB score: 0.681 (Public)** ← **これまでの最高スコア（従来: SU1の0.674）**
- Status: **正式採用** - Policy1を採用
- 採用構成（Policy1）:
  - top_k_pairs: 10
  - windows: [5]
  - reset_each_fold: True
  - OOF RMSE: 0.012139
  - OOF MSR: 0.024071
- 特徴量構成:
  - Pipeline input: 94列（生データ）
  - SU1特徴: 368列（欠損構造一次特徴）
  - SU5特徴: 105列（共欠損構造特徴）
    - co_miss_now: 10列（top-10ペアの共欠損フラグ）
    - co_miss_rollrate_5: 10列（5日ローリング共欠損率）
    - co_miss_deg: 85列（各列の共欠損次数、top-10ペアのみカウント）
  - 合計: 567列（SU1: 462列 → SU5: 567列、+105列）
- LBスコア比較:
  - SU1（ベースライン）: 0.674
  - SU5 Policy1: **0.681** (+0.007ポイント、+1.04%改善) ← **採用**
  - SU5 Policy2: 0.679 (+0.005ポイント、+0.74%改善)
- 採用理由:
  - Policy1がPolicy2を0.002ポイント上回る
  - 特徴量数が少ない（Policy1: 567列 vs Policy2: 562列）
  - windowsがシンプル（[5]のみ）で解釈性が高い
  - 従来最高スコアを更新し、安定した改善を確認
- 技術的ポイント:
  1. **共欠損ペアの選定**: top-k=10で特徴量爆発を抑制
  2. **時系列整合性**: reset_each_fold=Trueで各foldごとにペア再選択
  3. **ローリング統計**: window=5で短期共欠損パターンを捕捉
  4. **numpy互換性**: 1.26.4で学習・推論を統一してpickleエラー解消
- Notes:
  - artifacts/SU5/policy1_top10_w5/inference_bundle.pkl (2.1GB)
  - Kaggle Notebook: notebooks/submit/su5.ipynb (3749行)
  - Private Dataset: su5-missingness-core
  - スイープ結果: 12パターン検証、Policy1が最良

## 2025-11-23 su4 (Submission Unit 4) - **開発完了後、削除決定❌**

- Branch: `feat/miss-imptrace-su4`
- Status: **削除決定** - 特徴重要度分析・Ablation Studyにより予測性能への寄与がほぼゼロと判明
- Kaggle提出: **実施せず**（OOF評価で無効性が確認されたため）
- 開発状況:
  - 実装完了: src/feature_generation/su4/（1157行のtrain_su4.py含む）
  - OOF評価完了: RMSE=0.012141, MSR=0.023319
  - ハイパーパラメータスイープ完了: 18設定検証
  - 特徴重要度分析完了
  - Ablation Study完了

### 削除決定の根拠

**1. 特徴重要度分析結果**:
- 138個のSU4特徴のうち136個が重要度0
- わずか2個（`imp_method/missforest`, `imp_method/ridge_stack`）のみ重要度1
- SU4合計重要度: 2.00（全体の0.0%）
- 非SU4合計重要度: 37,198.00（全体の100.0%）
- SU4平均重要度: 0.01 vs 非SU4平均: 65.60
- **結論**: LightGBMがSU4特徴をほぼ使用していない

**2. ハイパーパラメータスイープ結果**:
- 18設定で完全に同一のRMSE/MSR
- 全設定でOOF RMSE=0.012141, MSR=0.023319
- **結論**: SU4パラメータが予測性能に影響していない

**3. Ablation Study結果**:
- SU4なし（SU1+SU5+GroupImputersのみ）: OOF RMSE **0.012284**
- SU4あり（baseline）: OOF RMSE **0.012141**
- 差分: **+0.000143** (+1.2%、誤差範囲内)
- **結論**: SU4削除の性能への影響はほぼゼロ

### 削除理由

**補完トレース情報がtarget予測に無相関**:
- 欠損値がどう補完されたかは市場リターン予測に寄与しない
- 既存特徴（元データ+SU1+SU5）で予測に必要な情報は網羅済み
- 138特徴追加のコストに見合う性能改善なし

### 削除によるメリット

- 138特徴分の計算コスト削減
- 訓練時間短縮（SU4生成・補完トレース計算の削減）
- メモリ使用量削減（2.1GB削減）
- コードベース簡略化・保守性向上

### 今後の方向性

**標準パイプライン**（SU4削除後）:
```
入力データ（94列）
  ↓
SU1FeatureAugmenter（+368列） → 462列
  ↓
SU5FeatureAugmenter（+105列） → 567列
  ↓
GroupImputers（M/E/I/P/S欠損値補完）
  ↓
Preprocessing（StandardScaler）
  ↓
LightGBMRegressor
```

**期待される性能**:
- OOF RMSE: 0.01228程度（SU4ありの0.01214から+0.00014）
- LB score: SU5の0.681と同等またはわずかに低下（推定0.680程度）

### 参照

- **削除決定詳細**: `docs/feature_generation/SU4-ablation-conclusion.md`
- **削除アクションプラン**: `docs/feature_generation/SU4-deletion-plan.md`
- **特徴重要度分析結果**: `results/ablation/SU4/feature_importance_analysis.csv`
- **スイープ結果**: `results/ablation/SU4/sweep_summary.csv`
- **Ablation実行スクリプト**: `src/feature_generation/su4/ablation_no_su4.py`

### 学んだ教訓

1. 特徴重要度分析を早期に実施すべき
2. Ablation Studyは新特徴追加時の基本
3. コンセプトの妥当性を実装前に検証すべき
4. LightGBMの特徴選択能力を信頼する

## 2025-11-30 su7 (Submission Unit 7) - **非採用（過学習/レジームミスマッチ）**

- Branch: `feat/su7`
- Kaggle Notebook: Private（Dataset: su7-momentum-reversal-core, notebooks/submit/su7.ipynb）
- Lines:
  - SU7 case_c ライン（10 base cols, OOF best RMSE ≈ 0.012047）
  - SU7 case_d ライン（12 base cols, OOF best RMSE ≈ 0.012045）
- LB scores (Public):
  - **SU1+SU5 baseline (no SU7)**: 0.681
  - **su7_case_c**: 0.476
  - **su7_case_d**: 0.469
- Decision: **非採用 / not_adopted**（SU1+SU5 ラインを継続採用）

### 概要

SU7 は「モメンタム・リバーサル特徴（diff/lag/rolling/RSI/sign）」を、
SU1+SU5+GroupImputers 後の特徴行列に対して付加する Submission Unit として設計した。

- スイープ設定: `configs/su7_sweep.yaml`（variants: case_a〜case_f, baseline）
- 評価指標: OOF RMSE を第一指標、MSR/vMSR を補助指標として grid search
- OOF 結果（抜粋）:
  - Baseline (SU1+SU5): RMSE ≈ 0.012097, MSR ≈ 0.013805
  - case_c (10 cols, RSI+sign): RMSE=0.012047, **MSR=0.019161**
  - case_d (12 cols, RSI+sign): **RMSE=0.012045**, MSR=0.015722

OOF 上は case_c/case_d ともにベースラインより良好であり、特に RMSE は明確に改善していた。
このため当初は case_d を本命、case_c をサブ候補とみなし、Kaggle での実運用検証に進んだ。

### 技術的検証と再学習

1. **環境統一とアーティファクト再生成**
   - 学習・推論ともに `numpy==1.26.4`, `scikit-learn==1.7.2` に固定（uv 経由）。
   - `artifacts/SU7/case_c` / `case_d` に対し、以下を再生成:
     - `inference_bundle.pkl`
     - `model_meta.json`
     - `feature_list.json`
     - `cv_fold_logs.csv`
     - `oof_predictions.csv`

2. **Notebook 互換性の確保**
   - `notebooks/submit/su5.ipynb` をベースに `su7.ipynb` を作成。
   - Notebook 内に `SU7Config`, `SU7FeatureGenerator`, `SU7FeatureAugmenter`, `SU7FullFeatureAugmenter` を埋め込み、
     `src.feature_generation.su7.feature_su7` / `train_su7` を `sys.modules` に動的登録。
   - 旧来の `__main__.SU7FullFeatureAugmenter` を参照する pickle を廃止し、
     明示モジュールパス付きの `inference_bundle.pkl` のみを使用するよう統一。

3. **ポストプロセスの保守化**
   - 当初は OOF grid search で得た post-process params（mult, lo, hi）をそのまま使用していたが、
     これが MSR 観点で攻め過ぎている可能性を考慮。
   - `predict_su7.py` の `_resolve_postprocess_params` を修正し、
     すべての実運用 submit で **固定パラメータ `mult=1.0, lo=0.9, hi=1.1`** を使用。
   - これにより、シグナル振幅を抑えた「保守版」 submission を case_c / case_d それぞれで再生成し、Kaggle へ再提出。

4. **submission 形式の検証**
   - `submission.csv` は `date_id,prediction` ヘッダ + 10 行（計 11 行）で、
     Hull Tactical の評価仕様に沿った特殊フォーマット（内部API評価用）であることを確認。
   - SU5 ラインとのフォーマット差異はなく、行数・カラム名起因のスコア異常ではないことを確認。

これらの技術的修正・再学習にもかかわらず、SU7 ラインの Public LB は

- case_c: 0.476
- case_d: 0.469

から実質的に改善せず、SU1+SU5 ベースライン（0.681）との差は依然として大きかった。

### 原因分析: 過学習 / レジームミスマッチ

切り分けの結果、以下の可能性はおおむね除外できた。

- 実装バグ（特徴生成のリーク、fold_indices 不整合 など）
- ライブラリバージョンの不整合（numpy / MT19937）
- submission 行数やヘッダの誤り
- 過剰な post-process によるシグナル暴走

それでもなお OOF では改善している SU7 case_c/case_d が、
Public LB では 0.47 台まで崩れることから、最も説明力が高い仮説は次の通り:

- SU7 のモメンタム/リバーサル特徴は **学習 + OOF 期間の市場レジームでは有効** だが、
  Public 評価期間のレジームでは逆方向に働いた。
- 特に MSR proxy によるシグナル最適化が、ある種の方向性に強く張る形になっており、
  Public 期間ではその賭けが大きく外れた。
- これは分割範囲内の OOF では検出しきれず、「OOF 改善 ↔ LB 大幅悪化」という
  **レジームミスマッチを含む広義の過学習** として現れたと考えられる。

### 最終判断とコンフィグ反映

- 現コンテストでは SU7 を **採用しない**（非採用 / not_adopted）。
- `configs/feature_generation.yaml`:
  - `su7.enabled: false`
  - `status: not_adopted`, `decision: not_used`
  - `kaggle_lb_su5_baseline: 0.681`, `kaggle_lb_su7_case_c: 0.476`, `kaggle_lb_su7_case_d: 0.469` をメタ情報として記録。
- 本番ラインは引き続き **SU1+SU5 (SU5 Policy1, LB 0.681)** を使用する。
- SU7 のコード・スイープ設定・Notebook は、将来の別コンペ/別市場レジームでの検証用に残し、
  本コンペにおける追加提出は行わない。

### 学んだ教訓（SU7）

1. **「OOF 改善」だけでは不十分**
   - 特にモメンタム系・シグナル最適化系の特徴は、市場レジーム依存性が強く、
     OOF が良くても Public 期間では逆効果になりうる。

2. **MSR proxy の最適化は慎重に**
   - MSR/vMSR を最大化する post-process は、「勝ちやすい局面に強く張る」構造になりやすい。
   - 市場レジームが OOF と異なる場合、その賭けが一方向に外れて大きな損失となるリスクがある。

3. **環境・実装起因の問題を切り分けたうえで、戦略レベルの問題を認める**
   - ライブラリバージョン、pickle 互換性、fold_indices、ポストプロセス設定など、
     機械的な要因を一通り潰した後に、それでも改善しない場合は「戦略そのものが Public 期間に合っていない」
     という可能性を受け入れる必要がある。

4. **ベースラインの堅牢さを優先する**
   - SU1+SU5 ラインは LB 0.681 と安定しており、追加のモメンタム軸での攻めが必ずしも必要ではない。
   - 不確実な改善よりも、安定したベースラインを維持することを優先する判断も重要。


## 2025-12-04 su8 (Submission Unit 8) - **非採用（効果なし / LB悪化）**

- Branch: `feat/su8`
- Kaggle Notebook: Private（Dataset: feature-generation-su8, notebooks/submit/su8.ipynb）
- LB score (Public):
  - **SU1+SU5 baseline**: 0.681
  - **su8_line**: **0.624**
- Decision: **非採用 / not_adopted**（SU1+SU5 ラインを継続採用）

### 概要

SU8 は「ボラティリティ・レジーム特徴（ewmstd_short/long, vol_ratio, vol_level, ボラ/トレンドレジームタグ, ret_vol_adj）」を、
SU1+SU5+GroupImputers 後の特徴行列に対して付加する Submission Unit として設計した。

- 実装: `src/feature_generation/su8/feature_su8.py`, `train_su8.py`, `predict_su8.py`
- パイプライン: 生データ → SU1 → SU5 → GroupImputers → **SU8** → 前処理 → LGBM
- OOF 結果（SU5 baseline との比較）:
  - SU1+SU5 baseline: OOF RMSE ≈ **0.012139**
  - SU8 追加ライン: OOF RMSE ≈ **0.012230**（≒ +0.00009 悪化）

OOF の時点でわずかに悪化していたものの、ボラティリティ軸の情報が Public 期間で効く可能性を確認するため、
SU8 付きラインを Kaggle に submit した。

### Kaggle 実行と post-process

- Kaggle Notebook 内では `inference_bundle.pkl`, `model_meta.json`, `feature_list.json` を Private Dataset からロード。
- ライブラリバージョンは numpy 1.26.4, scikit-learn 1.7.2（Wheel 同梱）に揃え、ローカルと互換を確保。
- post-process は `model_meta.json` の `oof_best_params`（`mult=1.5, lo=0.8, hi=1.1`）を使用する実装と、
  保守版の `mult=1.0, lo=0.9, hi=1.1` 実装の両方を検証。
  最終的な SU8 ラインの submit では、OOF 最適パラメータ（`mult=1.5, lo=0.8, hi=1.1`）を採用した。

ローカル側の `predict_su8.py` も CLI 引数経由で post-process パラメータを上書きできるようにし、

```bash
uv run python src/feature_generation/su8/predict_su8.py --use-oof-params
```

とすることで Kaggle と同じ `mult=1.5, lo=0.8, hi=1.1` ラインの `submission.csv` を再現可能にしている。

### Public LB 結果

- SU1+SU5 baseline（SU8 なし）: **0.681**
- SU8 ライン（SU1+SU5+SU8）: **0.624**

OOF ではもともとわずかに悪化しており、Public LB でも 0.681 → 0.624 と 0.057 ポイントの明確な劣化となった。
ボラティリティ・レジーム特徴が train+OOF 期間・Public 評価期間のいずれでも優位性を示さなかったため、
SU8 は本コンペでは **非採用** と判断した。

### 結論と今後の扱い

- `configs/feature_generation.yaml` の `su8.enabled` を `false` に設定し、`status: not_adopted` としてアーカイブ扱いとする。
- 本番ラインは引き続き **SU1+SU5（LB 0.681）** を採用し、SU8 はコード・アーティファクトともに「将来別レジーム用の参考実装」として残す。
  - ボラティリティ/レジーム軸の特徴は、今回の Hull Tactical コンペの公開評価期間に対しては有効でなかったが、
  別の市場環境や評価期間では再検証の余地があるため、実装は削除せずに保守最小限で維持する。


## 2025-12-05 su9 (Submission Unit 9) - **非採用（OOF改善 / LB悪化）**

- Branch: `feat/su9`
- Kaggle Notebook: Private（Dataset: feature-generation-su9）
- LB score (Public):
  - **SU1+SU5 baseline**: 0.681
  - **su9_line (month+holiday)**: **0.679**
- Decision: **非採用 / not_adopted**（SU1+SU5 ラインを継続採用）

### 概要

SU9 は「カレンダー・季節性特徴（曜日one-hot, 月one-hot, 祝日フラグ, 月末/期末フラグ, 年内ポジション）」を、
SU1+SU5+GroupImputers 後の特徴行列に対して付加する Submission Unit として設計した。

- 実装: `src/feature_generation/su9/feature_su9.py`, `train_su9.py`, `predict_su9.py`, `sweep_oof.py`
- パイプライン: 生データ → SU1 → SU5 → GroupImputers → **SU9** → 前処理 → LGBM
- スイープ設定: 6つの特徴グループフラグ（dow/dom/month/month_flags/holiday/year_position）を grid search
- スイープ規模: 63パターン（全false除外）、約173分

### スイープ結果

| 構成 | 列数 | OOF RMSE | 備考 |
|------|------|----------|------|
| 全て有効 | 32 | 0.012131 | 悪化 |
| month + holiday のみ | 16 | **0.012041** | ベスト |
| baseline (SU5) | 0 | 0.012088 | 比較基準 |

**ベスト構成**: `include_month=True, include_holiday=True`（他は全て False）
- OOF RMSE: 0.012041（baseline 0.012088 から **0.39%改善**）
- 特徴量: 月one-hot 12列 + 祝日フラグ 4列 = 16列

### Kaggle LB 結果

| ライン | LB score | 差分 |
|--------|----------|------|
| SU1+SU5 baseline | 0.681 | - |
| SU9 (month+holiday) | **0.679** | **-0.002 悪化** |

**OOF改善がLB改善に繋がらなかった**。

### 原因分析

1. **過学習パターン**
   - OOFでは0.39%改善したが、LBでは0.002ポイント悪化
   - カレンダー特徴（特に月one-hot 12列）が訓練期間の局所パターンを学習した可能性

2. **SU7/SU8との類似性**
   - SU7 (モメンタム): OOF改善 → LB 0.476（大幅悪化）
   - SU8 (ボラティリティ): OOF悪化 → LB 0.624（悪化）
   - SU9 (カレンダー): OOF改善 → LB 0.679（微悪化）
   - いずれもベースラインより悪化するパターン

3. **カレンダー特徴の限界**
   - 「決定可能な情報」でも過学習は起こりうる
   - 月ごとのアノマリーが訓練期間固有のものだった可能性

### 技術的詳細

- 環境統一: `numpy==1.26.4`, `scikit-learn==1.7.2`
- アーティファクト: `artifacts/SU9/best/`（inference_bundle.pkl, model_meta.json, submission.csv）
- 設定: `configs/feature_generation.yaml` の `su9.enabled=false`

### 最終判断

**SU9は非採用**。理由:
- LBスコアがベースラインより悪化（0.681 → 0.679）
- OOF改善はCV分割内でのみ有効だった
- カレンダー特徴は本コンペでは有効でない

現行ベストラインは引き続き **SU1+SU5 (LB 0.681)** を維持する。

### 学んだ教訓

1. **OOF改善 ≠ LB改善**: SU7/SU8/SU9 すべてでこのパターンが確認された
2. **カレンダー特徴でも過学習は起こりうる**: 「決定可能な情報」の安全神話は崩れた
3. **ベースラインの安定性を優先**: 追加特徴による複雑化は必ずしも改善に繋がらない


## 2025-12-06 SU10 (External Regime Dataset: SPY)

- Commit: feat/SU10 ブランチ
- Submit line: SU10 (SU1+SU5+外部レジーム)
- Kaggle Notebook: Private (Dataset: feature-generation-su10-hulltactical)
- **LB score: 0.597** ❌
- Decision: **非採用**

### 概要

外部データソース（S&P 500 / SPY historical data）からボラティリティ・トレンドレジーム特徴を生成し、SU1+SU5 に追加する試み。

- データソース: [S and P historical data for Hull Tactical competition](https://www.kaggle.com/datasets/ambrosm/s-and-p-historical-data-for-hull-tactical-competition/)
- 外部特徴量: 14列（ボラ指標4列、ボラレジーム3列、トレンド指標1列、トレンドレジーム3列、リターン系3列）
- date_id 範囲: 780〜8989（date_id 0-779 はデータなし）

### 結果

| 指標 | SU5 (ベースライン) | SU10 | 変化 |
|------|---------------------|------|------|
| **LB score** | 0.681 | 0.597 | **-12.3%** ❌ |
| OOF RMSE | 0.01214 | 0.01227 | +1.1% |

### 問題分析

1. **OOF と LB の乖離が大きい**
   - OOF では +1.1% の軽微な悪化
   - LB では **-12.3%** の大幅悪化
   - → 過学習の兆候、または外部データの時間的ミスマッチ

2. **外部レジームデータの問題点**
   - **Look-ahead bias**: SPY の過去データを date_id にマッピングする際、将来情報が混入した可能性
   - **時系列のズレ**: Kaggle の隠しテスト期間と外部データの期間がズレている可能性
   - **レジーム変化**: 学習期間のレジームパターンが推論期間で通用しない
   - **date_id 範囲制約**: 外部データは date_id 780-8989 のみ、隠しテストが 8990 以降なら NaN

### 技術的詳細

- 実装: `src/feature_generation/su10/feature_su10.py`, `train_su10.py`, `predict_su10.py`
- テスト: 14件全てパス
- アーティファクト: `artifacts/SU10/` (inference_bundle.pkl, model_meta.json, submission.csv)
- 外部データ: `data/su10/su10_external_regime.csv` (8,210行, 15列)

### 最終判断

**SU10は非採用**。理由:
- LBスコアが大幅悪化（0.681 → 0.597, -12.3%）
- OOF と LB の乖離 → 汎化性能に問題
- 外部データの時間的整合性に懸念
- リスク・リターンが見合わない

現行ベストラインは引き続き **SU1+SU5 (LB 0.681)** を維持する。

### 学んだ教訓

1. **外部データは諸刃の剣**: 追加情報源でもデータの時間的整合性が重要
2. **OOF と LB の乖離が大きい場合は危険信号**: SU10 は OOF +1.1% 悪化に対し LB -12.3% 悪化
3. **SU7/SU8/SU9/SU10 すべて LB 悪化**: ベースライン (SU1+SU5) が最適解である可能性が高い


## 2025-12-06 SU11 (Level-2 Stacking: Ridge on SU5 predictions) - **非採用❌**

- Branch: `dev`
- Submit line: SU11 (Level-2 Stacking)
- Kaggle Notebook: Private (Dataset: su11-stacking-hulltactical, notebooks/submit/su11.ipynb)
- **LB score: 0.464** ❌
- Decision: **非採用**

### 概要

Level-1 モデル（SU5）の OOF/test 予測値を入力とし、Level-2 Ridge 回帰で最終予測を行う 2 段階スタッキング。

- Level-1: SU1+SU5+GroupImputers+StandardScaler+LGBMRegressor（既存 SU5 ライン）
- Level-2: Ridge Regression（alpha=1.0 → 0.001 に調整も効果なし）
- 評価指標: OOF RMSE, LB score

### 結果

| 指標 | SU5 (Level-1) | SU11 (Level-2) | 変化 |
|------|---------------|----------------|------|
| **LB score** | 0.681 | 0.464 | **-31.9%** ❌ |
| OOF RMSE | 0.012139 | 0.010997 | -9.4% (改善) |
| 予測 std | 0.00531 | 0.000146 | **-97.3%** (縮小) |
| Signal range | [0.97, 1.06] | [0.9997, 1.0006] | 1/36 に縮小 |
| Betting range | ±0.80% | ±0.02% | 1/36 に縮小 |

### 問題の根本原因

**Level-2 Ridge が予測を過度に縮小（shrinkage）**:

```
Ridge coef = 0.0137  (極端に小さい)
→ y_pred_L2 ≈ 0.0137 × y_pred_L1 + 0.00007
→ 予測の分散が 97% 縮小
→ ベッティングレンジが 1/36 に
→ ほぼ「常に 1.0 を予測」する状態
```

これは Ridge が OOF RMSE を最小化するために「正しく」動作した結果ですが、
LB の評価関数（収益性関連指標）では分散の縮小が悪影響となりました。

### 検証した対策

| 対策 | Ridge coef | 結果 |
|------|------------|------|
| alpha=1.0（デフォルト） | 0.0137 | 予測 97% 縮小、LB 0.464 |
| alpha=0.001 | 0.0779 | 予測 86% 縮小、改善なし |
| alpha=0.00001 | 0.0783 | coef 収束、これ以上改善せず |
| identity モード | 1.0 | Level-1 そのまま通過 → SU5 と同等 |

### 技術的詳細

- 実装: `src/feature_generation/su11/feature_su11.py`, `train_su11.py`, `predict_su11.py`
- テスト: 全てパス
- アーティファクト: `artifacts/SU11/` (inference_bundle.pkl, model_meta.json, submission.csv)
- Notebook: `notebooks/submit/su11.ipynb`

### 最終判断

**SU11は非採用**。理由:
- LBスコアが大幅悪化（0.681 → 0.464, -31.9%）
- OOF RMSE 改善 (-9.4%) は「shrinkage」による見かけの改善
- 予測の分散が 97% 縮小し、ベッティングシグナルが消失
- 単一 Level-1 入力での Ridge stacking は本コンペでは無効

現行ベストラインは引き続き **SU1+SU5 (LB 0.681)** を維持する。

### 学んだ教訓

1. **OOF RMSE 改善 ≠ LB 改善**: RMSE 最適化と収益性最適化は異なる
2. **単一モデル stacking は効果なし**: 複数の多様な Level-1 モデルが必要
3. **shrinkage のリスク**: 正則化が予測の「賭け」を消失させる
4. **SU7〜SU11 すべて LB 悪化**: ベースライン (SU1+SU5) が最適解

### 今後の方向性

Level-2 stacking で真の改善を得るには:
- **複数の多様な Level-1 モデル**（SU5, SU7, SU8 等）のアンサンブル
- **異なる特徴量セット**を持つモデルの組み合わせ
- **shrinkage を避ける** blending 手法（単純平均、加重平均など）

現時点では SU1+SU5 ベースラインが最適解であり、これ以上の複雑化は推奨しない。


## 2025-12-06 SU1/SU5 Brushup (微調整特徴追加) - **採用保留（効果なし / LB維持）**

- Branch: `dev`
- Submit line: SU1+SU5+Brushup
- Kaggle Notebook: Private (Dataset: su5-missingness-core)
- **LB score: 0.681** (ベースラインと同等)
- Decision: **採用保留** - 変更を取り込むが、後続のPCA/特徴量選定で再評価

### 概要

既存のSU1+SU5ラインを「新しいSUを増やす」方向ではなく「既存を少しだけ厚くする」方向で微調整。

- SU1拡張: +5列（欠損頻度、ストリーク、レジーム変化）
- SU5拡張: +5列（k-meansクラスタ、共欠損密度、次数サマリ、中心性）
- 合計: 567列 → **577列** (+10列、+1.8%)

### 結果

| 指標 | SU5 (ベースライン) | Brushup | 変化 |
|------|---------------------|---------|------|
| **LB score** | 0.681 | 0.681 | **±0** |
| OOF RMSE | 0.012139 | 0.012134 | -0.04% (微改善) |
| 特徴量数 | 567列 | 577列 | +10列 |

### 追加された特徴

**SU1拡張 (+5列)**:
- `miss_count_last_5d` - 直近5日間の全列欠損数合計
- `miss_ratio_last_5d` - 直近5日間の欠損率
- `is_long_missing_streak` - 3日以上連続欠損フラグ
- `long_streak_col_count` - 3日以上連続欠損の列数
- `miss_regime_change` - 欠損レジーム変化フラグ

**SU5拡張 (+5列)**:
- `miss_pattern_cluster` - k-means クラスタID (k=6)
- `co_miss_density` - top-k 共欠損ペアの同時欠損割合
- `co_miss_deg_sum` - 欠損列の共欠損次数合計
- `co_miss_deg_mean` - 欠損列の共欠損次数平均
- `miss_graph_centrality` - 欠損列が top-k ペアに含まれる回数

### 技術的詳細

- 実装: `src/feature_generation/su1/feature_su1.py`, `su5/feature_su5.py`
- テスト: 全17ユニットテストパス
- 環境: numpy 1.26.4, scikit-learn 1.7.0
- アーティファクト: `artifacts/SU5_brushup/` (inference_bundle.pkl, model_meta.json, submission.csv)

### 判断理由

1. **LBスコア維持**: ベースラインと同等（0.681）で悪化なし
2. **OOF微改善**: 0.012139 → 0.012134 (-0.04%)
3. **列数抑制**: +10列に厳守し、過学習リスクを最小化
4. **後続施策との相性**: PCA/特徴量選定で有効に働く可能性

### 今後の方向性

- **Brushup変更を取り込み**、後続のPCA/特徴量選定で再評価
- LB悪化がなければ正式採用、悪化すれば戻す
- 現行ベストラインは引き続き **SU1+SU5 (LB 0.681)** を維持

### 学んだ教訓

1. **列数を抑えた微調整は安全**: +10列でLB悪化なし
2. **OOF微改善でもLB維持**: SU7〜SU11のような大幅悪化は発生せず
3. **「攻め」より「守り」**: 既存ラインを壊さない慎重なアプローチが重要

---

## 2025-12-07 Feature Selection Phase 1 (統計フィルタ) - **採用確定✅**

- Commit: e1f5aad (feat/select-tier0)
- Submit line: fesel (Feature Selection Pipeline)
- Kaggle Notebook: Private (notebooks/submit/fesel.ipynb)
- LB score: **0.681** (Tier0ベストと同一)

### 概要

577列の全特徴量から統計フィルタで417列を削除し、160列への削減を実現。
OOF評価・LB評価の両方で性能維持を確認し、Phase 1を採用確定。

### 評価結果

| 指標 | Tier0 (全特徴) | Tier1 (フィルタ後) | 差分 |
|------|----------------|-------------------|------|
| 特徴量数 | 577 | 160 | **-417 (-72.3%)** |
| OOF RMSE | 0.012134 | 0.012168 | +0.28% |
| OOF MSR | 0.019929 | 0.019201 | -3.7% |
| **LB Score** | 0.681 | **0.681** | **±0.000** |

### 削除基準

| フィルタ種別 | 閾値 | 削除数 |
|-------------|------|--------|
| 低分散 | variance < 1e-10 | 9列 |
| 高欠損 | missing_rate > 0.99 | 0列 |
| 高相関 | \|corr\| > 0.999 | 408列 |
| **合計** | - | **417列** |

### Notes

- **Phase 0**: Tier0ベースライン凍結（577列, OOF RMSE 0.012134）
- **Phase 1**: 統計フィルタ適用で160列に削減
  - 低分散・高欠損・高相関の3種フィルタを適用
  - 高相関フィルタで大半（408列）を削除
- **実装**: `src/feature_selection/` 配下に評価・フィルタ・推論スクリプトを追加
- **設定**: `configs/feature_selection/tier1/excluded.json` に417列の除外リストを保存

### 判定理由

1. **RMSE**: +0.000034（+0.28%）は許容範囲内（判定基準「+0.0001以内」を満たす）
2. **LB Score**: 0.681（Tier0と同一）- LBでの性能劣化なしを確認
3. **特徴量削減効果**: 577列→160列（-72.3%）は非常に大きい削減効果
4. **結論**: 削除した417列は予測に寄与していなかった（または冗長だった）ことをLBで確認

### 今後の方向性

- **Phase 2**: モデル重要度ベースのフィルタ（LGBM importance → Permutation）
  - 160列からさらに50〜100列への削減を目標
- **Phase 3**: 相関クラスタからの代表選出
- **Phase 4**: 最終LB検証 + 安定性確認

### 学んだ教訓

1. **72%の列は冗長だった**: 高相関フィルタで大半が削除可能
2. **OOF微増でもLB維持**: +0.28%のRMSE増加はLBに影響しない
3. **段階的削減が有効**: Phase分けで安全に特徴量を削減できる

---

## 2025-12-07 Feature Selection Phase 2 (モデル重要度) - **採用確定✅**

- Commit: bb38da2 (feat/select-tier2)
- Submit line: fesel (Feature Selection Pipeline)
- Kaggle Notebook: Private (notebooks/submit/fesel.ipynb)
- LB score: **0.681** (Tier0/Tier1と同一)

### 概要

Tier1（160列）から LGBM gain importance を分析し、低重要度の40列を削除して120列に削減。
OOF評価・LB評価の両方で性能維持を確認し、Phase 2を採用確定。

### 評価結果

| 指標 | Tier0 | Tier1 | Tier2 | Tier2 vs Tier1 |
|------|-------|-------|-------|----------------|
| 特徴量数 | 577 | 160 | **120** | **-40 (-25%)** |
| OOF RMSE | 0.012134 | 0.012168 | 0.012172 | +0.000004 (+0.03%) |
| OOF MSR | 0.019929 | 0.019201 | 0.020386 | +0.001185 |
| **LB Score** | 0.681 | 0.681 | **0.681** | **±0.000** |

### 削除基準

| カテゴリ | 説明 | 削除数 |
|---------|------|--------|
| Zero importance | mean_gain = 0（全foldで未使用） | 22列 |
| Low importance | 0 < mean_gain < Q25（下位25%） | 18列 |
| **合計** | - | **40列** |

### 削除された特徴量の特徴

- **40列すべてが SU5 Augmented 特徴量**（欠損パターン関連）
- 生特徴量（M1, E1 など）は1つも削除されていない
- SU5で生成された欠損パターン特徴量の有効性は限定的であることが判明

### Notes

- **Phase 2-1**: LGBM gain importanceの分析
  - `results/feature_selection/tier1/importance_summary.csv` から削除候補を抽出
  - Zero importance（22列）+ Low importance（18列）= 40列を候補に
- **Phase 2-2**: Permutation Importance
  - 18列のLow importance特徴量はすべてSU5 Augmented特徴量
  - 生データに存在しないためPermutationテストは不適用
  - 全40列を削除確定
- **設定**: `configs/feature_selection/tier2/excluded.json` に457列の除外リストを保存（Tier1の417列 + Phase2の40列）

### 判定理由

1. **RMSE**: +0.000004（+0.03%）は許容範囲内（判定基準「+0.0001以内」を満たす）
2. **LB Score**: 0.681（Tier0/Tier1と同一）- LBでの性能劣化なしを確認
3. **特徴量削減効果**: 160列→120列（-25%）、全体では577列→120列（**-79%**）
4. **結論**: 削除した40列（SU5 Augmented特徴量）は予測に寄与していなかったことをLBで確認

### 今後の方向性

- **Phase 3**: Permutation Importance による更なる特徴量削減
  - 120列からさらに50〜80列への削減を目標
  - 生特徴量を対象にPermutationテストを実施
- **Phase 4**: 最終LB検証 + 安定性確認

### 学んだ教訓

1. **SU5欠損パターン特徴量の寄与は限定的**: 40列すべてが低重要度
2. **Zero importance特徴量は安全に削除可能**: 22列は全く使われていなかった
3. **段階的削減の継続**: Phase 1→2と安全に79%削減を達成

## 2025-12-12 Phase 3 Feature Selection (FS_compact / FS_topK) - **FS_compact採用✅**

- Branch: `feat/select-tier3`
- Kaggle Notebook: Private（Dataset: tier3-feature-selection, tier_topK-feature-selection）
- LB scores (Public):
  - **FS_full (Tier2 baseline)**: 0.681（120列）
  - **FS_compact (Tier3)**: 0.681（116列）✅ **採用**
  - **FS_topK**: 0.589（50列）❌ **非採用**
- Decision: **FS_compact採用** - Tier3を最終特徴セットとして確定

### Phase 3 概要

Phase 2で確定したTier2（120列）に対し、相関クラスタリングによる冗長性削減を実施。

**処理フロー**:
```
Phase 3-1: 相関クラスタリング（|ρ| > 0.95）
    ↓
Phase 3-2: クラスタ代表選出（importance最大の列を残す）
    ↓
Phase 3-3: Tier3評価（OOF RMSE判定）
    ↓
Phase 3-4: Feature Set定義 + LB検証
```

### 相関クラスタリング結果

- **閾値**: |ρ| > 0.95
- **検出クラスタ数**: 4
- **削除された特徴量**: 4列
  - E3（E4と相関0.9542、E4のimportanceが高い）
  - E11（E12と相関0.9711、E12のimportanceが高い）
  - V4（V5と相関0.9652、V5のimportanceが高い）
  - V10（V9と相関0.9614、V9のimportanceが高い）

### Feature Set比較

| Feature Set | 特徴量数 | OOF RMSE | LB Score | 判定 |
|-------------|---------|----------|----------|------|
| FS_full (Tier2) | 120 | 0.012172 | 0.681 | baseline |
| **FS_compact** | 116 | 0.012164 | **0.681** | ✅ 採用 |
| FS_topK | 50 | 0.012023 | 0.589 | ❌ 過学習 |

### FS_compact採用理由

1. **LB維持**: 0.681（Tier2と同一）
2. **OOF微改善**: 0.012164（-0.000008）
3. **軽量化**: 120→116列（-4列、-3.3%）
4. **SU1特徴量を含む**: 26列のSU1生成特徴量が維持
5. **リスク最小**: 相関0.95以上の冗長列のみを削除

### FS_topK非採用理由（過学習分析）

**致命的な問題**: OOF最良（0.012023）にもかかわらずLBで-13.5%大幅悪化

| 指標 | FS_full | FS_topK | 差分 |
|------|---------|---------|------|
| OOF RMSE | 0.012172 | 0.012023 | -0.000149 ✅ |
| LB Score | 0.681 | 0.589 | **-0.092 (-13.5%)** ❌ |

**過学習の原因分析**:

1. **SU1特徴量の欠落**
   - FS_topKではSU1生成列（`run_na/`, `run_obs/`, `avg_run_na/`など）が**全て除外**
   - importance上位50列が全て生特徴量だったため
   - SU1特徴量は時間外れデータへの汎化に寄与していた可能性

2. **情報の過度な圧縮**
   - 50列では異なる市場レジームをカバーする多様性が不足
   - 学習データの特定パターンに過適合

3. **Importanceの罠**
   - importance高 = 学習データでの予測に強く寄与
   - しかし未来データで役立つ保証はない

**教訓**:
- ❌ OOF改善だけを見て特徴量を大幅削減してはならない
- ❌ importance上位だけに絞るのは危険
- ✅ 相関クラスタリングで「同じ情報を持つ冗長列」を削除するのは安全
- ✅ LBで必ず検証してから採用を決定

### 技術的詳細

**artifacts生成**:
- `artifacts/tier3/`: FS_compact用（inference_bundle.pkl, feature_list.json, model_meta.json）
- `artifacts/tier_topK/`: FS_topK用（参考保存のみ）

**モデル構成**:
- Pipeline: SU1 Augmenter + 前処理（GroupImputers + Scaler） + LGBMRegressor
- Post-process: mult=1.5, lo=0.8, hi=1.1

**特徴量構成（FS_compact）**:
- 生特徴量: 90列（D, E, I, M, P, S, V）
- SU1生成特徴量: 26列（run_na, run_obs, avg_run_na, miss_count_last_5d, miss_pattern_cluster）
- 合計: 116列

### 最終判定

| 項目 | 値 |
|------|-----|
| **最終採用Feature Set** | FS_compact (Tier3) |
| 特徴量数 | 116列 |
| OOF RMSE | 0.012164 |
| LB Score | 0.681 |
| 設定ファイル | configs/feature_selection/tier3/excluded.json |
| Feature Set定義 | configs/feature_selection/feature_sets.json |

### Phase 3完了サマリー

**特徴量選定の全体推移**:
```
Phase 0: 577列（全特徴量）
    ↓ Phase 1: 低分散・定数列削除
Phase 1: 160列（-72%）
    ↓ Phase 2: Zero/Low importance削除
Phase 2: 120列（-25%）
    ↓ Phase 3: 相関クラスタリング
Phase 3: 116列（-3%）← 最終
```

**累積削減率**: 577列 → 116列 = **-80%**

### 次フェーズへの引き継ぎ

Phase 3完了により、以下のフェーズに進む準備が整った：

1. **モデル選定**: LGBM以外のモデル（XGBoost, CatBoost, Ridge等）をFS_compactで評価
2. **ハイパラ最適化**: 確定した116列でハイパラチューニング
3. **アンサンブル**: 複数モデル・Feature Setの組み合わせ

### 参照

- **Phase 3仕様書**: docs/feature_selection/phase3_spec.md
- **相関クラスタ結果**: results/feature_selection/phase3/correlation_clusters.json
- **代表選出結果**: results/feature_selection/phase3/cluster_representatives.json
- **Feature Set定義**: configs/feature_selection/feature_sets.json
- **Tier3除外リスト**: configs/feature_selection/tier3/excluded.json