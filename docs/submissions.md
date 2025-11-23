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
