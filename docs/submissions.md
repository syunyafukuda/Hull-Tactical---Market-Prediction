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
