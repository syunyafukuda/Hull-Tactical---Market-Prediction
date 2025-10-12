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
