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
