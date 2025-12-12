# ExtraTrees モデル実装仕様書

最終更新: 2025-12-12

## 実装ステータス

**Status**: ⬜ **未着手**

### 実装予定
- ⬜ `src/models/extratrees/train_extratrees.py`: 学習スクリプト
- ⬜ `configs/models/extratrees.yaml`: YAML設定ファイル
- ⬜ Unit tests: `tests/models/test_extratrees.py`

### 成果物
- ⬜ `artifacts/models/extratrees/inference_bundle.pkl`
- ⬜ `artifacts/models/extratrees/oof_predictions.csv`
- ⬜ `artifacts/models/extratrees/cv_fold_logs.csv`
- ⬜ `artifacts/models/extratrees/model_meta.json`

---

## 1. 目的と位置づけ

### 1.1 モデル選定フェーズでの役割

- **目的**: バギング系ツリーモデルとして、GBDTとは異なる多様性を導入
- **期待効果**: 
  - 勾配ブースティングとは「揺れ方」が異なる予測
  - LGBMとの予測相関が0.85-0.92程度（高いアンサンブル価値）
  - 過学習しにくい性質
- **比較対象**: LGBM ベースライン（OOF RMSE: 0.012164, LB: 0.681）

### 1.2 ExtraTreesの特徴

- **Extremely Randomized Trees**: 分割点もランダムにサンプリング
- **バギング**: 各木が独立に学習（ブースティングのような逐次学習ではない）
- **高いバリアンス削減**: 多数の弱学習器の平均で安定した予測
- **高速**: 最適分割点を探索しないため、RandomForestより高速

### 1.3 前提条件

- **特徴セット**: FS_compact（116列）を固定
- **CV設定**: TimeSeriesSplit, n_splits=5, gap=0（LGBMと同一）
- **評価指標**: OOF RMSE, OOF MSR, 予測相関（vs LGBM）
- **スケーリング**: 不要（ツリー系モデル）

---

## 2. 技術仕様

### 2.1 入出力

| 項目 | 仕様 |
|------|------|
| 入力 | `data/raw/train.csv`, `data/raw/test.csv` |
| 特徴量生成 | SU1 + SU5 → tier3除外 → 116列 |
| 出力 | `artifacts/models/extratrees/` 配下に成果物 |

### 2.2 パイプライン構造

```
生データ (94列)
    ↓
[SU1FeatureAugmenter + SU5FeatureAugmenter]  # 既存モジュール再利用
    ↓
合計 577列
    ↓
[tier3 feature exclusion]  # configs/feature_selection/tier3/excluded.json
    ↓
116列 (FS_compact)
    ↓
[GroupImputers: M/E/I/P/S]  # 既存前処理再利用
    ↓
[SimpleImputer]  # 残余NaN処理
    ↓
[ExtraTreesRegressor]  # ★ スケーリング不要
```

### 2.3 初期ハイパーパラメータ

```python
extratrees_params = {
    "n_estimators": 500,         # 木の数（多いほど安定）
    "max_depth": 15,             # 木の深さ制限
    "min_samples_split": 10,     # 分割に必要な最小サンプル数
    "min_samples_leaf": 5,       # 葉の最小サンプル数
    "max_features": 0.7,         # 各分割で使用する特徴量割合
    "bootstrap": False,          # ExtraTreesのデフォルト
    "random_state": 42,
    "n_jobs": -1,
}
```

### 2.4 GBDTとの主な違い

| 項目 | ExtraTrees | LGBM/XGBoost |
|------|------------|--------------|
| 学習方式 | バギング（並列） | ブースティング（逐次） |
| 分割点選択 | ランダム | 最適点を探索 |
| 予測の揺れ | 穏やか | 急峻 |
| 過学習傾向 | 低い | 中程度 |
| スケーリング | 不要 | 不要 |

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/extratrees/
├── __init__.py           # モジュール初期化
└── train_extratrees.py   # メイン学習スクリプト

configs/models/
└── extratrees.yaml       # YAML設定ファイル

tests/models/
└── test_extratrees.py    # ユニットテスト
```

### 3.2 train_extratrees.py の実装要件

#### 3.2.1 必須機能

1. **引数パース**: `argparse`で以下を受け付ける
   - `--data-dir`: データディレクトリ（デフォルト: `data/raw`）
   - `--out-dir`: 出力ディレクトリ（デフォルト: `artifacts/models/extratrees`）
   - `--config-path`: feature_generation.yaml パス
   - `--preprocess-config`: preprocess.yaml パス
   - `--feature-tier`: 使用するtier（デフォルト: `tier3`）
   - `--n-splits`, `--gap`: CV設定
   - ExtraTreesハイパラ: `--n-estimators`, `--max-depth`, `--min-samples-leaf`, `--max-features`

2. **特徴量生成**: 既存モジュールを再利用
   ```python
   from src.feature_generation.su5.train_su5 import (
       load_su1_config,
       load_su5_config,
       load_preprocess_policies,
       SU5FeatureAugmenter,
       _prepare_features,
   )
   from src.models.common.feature_loader import get_excluded_features
   ```

3. **前処理**: スケーリング不要
   ```python
   from sklearn.impute import SimpleImputer
   # StandardScalerは不要
   ```

4. **モデル学習**:
   ```python
   from sklearn.ensemble import ExtraTreesRegressor
   ```

5. **成果物出力**:
   - `inference_bundle.pkl`: 全fold のモデル + 前処理パイプライン
   - `oof_predictions.csv`: OOF予測値
   - `cv_fold_logs.csv`: fold別メトリクス
   - `model_meta.json`: 設定・評価サマリ

### 3.3 コードスケルトン

```python
"""ExtraTrees training script with unified CV framework."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# 既存モジュールのインポート
from src.feature_generation.su5.train_su5 import (
    SU5FeatureAugmenter,
    _prepare_features,
    load_preprocess_policies,
    load_su1_config,
    load_su5_config,
)
from src.models.common.cv_utils import (
    CVConfig,
    aggregate_fold_results,
    compute_fold_metrics,
    create_cv_splits,
)
from src.models.common.feature_loader import get_excluded_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ExtraTrees model")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/models/extratrees")
    parser.add_argument(
        "--config-path", type=str, default="configs/feature_generation.yaml"
    )
    parser.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml")
    parser.add_argument("--feature-tier", type=str, default="tier3")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=0)
    # ExtraTrees specific
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=float, default=0.7)
    return parser.parse_args()


def build_extratrees_pipeline(
    n_estimators: int = 500,
    max_depth: int = 15,
    min_samples_leaf: int = 5,
    max_features: float = 0.7,
) -> Pipeline:
    """Build ExtraTrees pipeline (no scaling needed)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    pipeline_params: dict[str, Any],
) -> tuple[Pipeline, np.ndarray]:
    """Train a single fold."""
    pipeline = build_extratrees_pipeline(**pipeline_params)
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    return pipeline, val_pred


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み・特徴量生成（既存パターン）
    train_df = pd.read_csv(Path(args.data_dir) / "train.csv")
    test_df = pd.read_csv(Path(args.data_dir) / "test.csv")

    # SU1 + SU5 特徴量生成
    su1_cfg = load_su1_config(args.config_path)
    su5_cfg = load_su5_config(args.config_path)
    preprocess_policies = load_preprocess_policies(args.preprocess_config)
    augmenter = SU5FeatureAugmenter(su1_cfg, su5_cfg, preprocess_policies)

    train_aug = augmenter.fit_transform(train_df.copy())
    test_aug = augmenter.transform(test_df.copy())

    # tier除外
    excluded = get_excluded_features(args.feature_tier)
    feature_cols = [c for c in train_aug.columns if c not in excluded and c not in ["Date", "TARGET"]]
    
    X = train_aug[feature_cols]
    y = train_aug["TARGET"]
    X_test = test_aug[feature_cols]

    # CV設定
    cv_config = CVConfig(n_splits=args.n_splits, gap=args.gap)
    splits = create_cv_splits(X, cv_config)

    # パイプラインパラメータ
    pipeline_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
    }

    # CV学習
    fold_results = []
    oof_preds = np.zeros(len(X))
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline, val_pred = train_fold(X_train, y_train, X_val, pipeline_params)
        oof_preds[val_idx] = val_pred
        models.append(pipeline)

        metrics = compute_fold_metrics(y_val.values, val_pred, fold_idx)
        fold_results.append(metrics)
        print(f"Fold {fold_idx}: RMSE={metrics.rmse:.6f}")

    # 集計
    summary = aggregate_fold_results(fold_results)
    print(f"\nOOF RMSE: {summary['oof_rmse']:.6f}")
    print(f"OOF MSR: {summary['oof_msr']:.6f}")

    # 成果物保存
    # 1. inference_bundle.pkl
    test_preds = np.mean([m.predict(X_test) for m in models], axis=0)
    bundle = {
        "models": models,
        "augmenter": augmenter,
        "feature_cols": feature_cols,
        "excluded_features": excluded,
    }
    with open(out_dir / "inference_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # 2. oof_predictions.csv
    oof_df = pd.DataFrame({
        "Date": train_aug["Date"],
        "TARGET": y,
        "prediction": oof_preds,
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    # 3. cv_fold_logs.csv
    fold_logs = pd.DataFrame([
        {"fold": r.fold_idx, "rmse": r.rmse, "msr": r.msr, "n_samples": r.n_samples}
        for r in fold_results
    ])
    fold_logs.to_csv(out_dir / "cv_fold_logs.csv", index=False)

    # 4. model_meta.json
    meta = {
        "model_type": "ExtraTreesRegressor",
        "feature_tier": args.feature_tier,
        "n_features": len(feature_cols),
        "cv_config": {"n_splits": args.n_splits, "gap": args.gap},
        "hyperparameters": pipeline_params,
        "oof_rmse": summary["oof_rmse"],
        "oof_msr": summary["oof_msr"],
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 5. submission.csv
    submission = pd.DataFrame({
        "Date": test_aug["Date"],
        "TARGET": test_preds,
    })
    submission.to_csv(out_dir / "submission.csv", index=False)

    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
```

---

## 4. テスト要件

### 4.1 ユニットテスト

```python
# tests/models/test_extratrees.py

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor


class TestExtraTreesPipeline:
    """Test ExtraTrees pipeline components."""

    def test_extratrees_basic_fit(self, sample_train_data: pd.DataFrame) -> None:
        """ExtraTreesRegressor can fit and predict."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        model = ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert not np.isnan(preds).any()

    def test_extratrees_no_scaling_needed(self, sample_train_data: pd.DataFrame) -> None:
        """ExtraTrees produces same results regardless of scaling."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        # Without scaling
        model1 = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        # With scaling (should give same results for tree models)
        from sklearn.preprocessing import StandardScaler
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X),
            columns=X.columns
        )
        model2 = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model2.fit(X_scaled, y)
        pred2 = model2.predict(X_scaled)

        # Tree models are scale-invariant (results should be identical)
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_feature_importance_available(self, sample_train_data: pd.DataFrame) -> None:
        """ExtraTrees provides feature importances."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert np.isclose(importances.sum(), 1.0)
```

### 4.2 統合テスト

```python
def test_extratrees_cv_pipeline(sample_train_data: pd.DataFrame) -> None:
    """Test full CV pipeline for ExtraTrees."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    X = sample_train_data.drop(columns=["Date", "TARGET"])
    y = sample_train_data["TARGET"]

    tscv = TimeSeriesSplit(n_splits=3)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=10, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        oof_preds[val_idx] = pipeline.predict(X_val)

    # Check OOF predictions are valid
    assert not np.isnan(oof_preds).any()
    assert oof_preds.std() > 0  # Not all same value
```

---

## 5. 成功基準

### 5.1 定量基準

| 指標 | 閾値 | 理由 |
|------|------|------|
| OOF RMSE | ≤ 0.0130 | LGBM（0.01216）より多少劣っても許容 |
| OOF MSR | > 0 | 正のリターンを維持 |
| 予測相関（vs LGBM） | < 0.92 | アンサンブル多様性の確保 |

### 5.2 定性基準

- [ ] 学習が正常に完了（エラーなし）
- [ ] 成果物が全て生成される
- [ ] 品質チェック（ruff, pyright）をパス
- [ ] ユニットテストが全てパス

---

## 6. 実行コマンド

### 6.1 学習実行

```bash
# デフォルト設定
python -m src.models.extratrees.train_extratrees

# ハイパラ指定
python -m src.models.extratrees.train_extratrees \
    --n-estimators 700 \
    --max-depth 20 \
    --min-samples-leaf 3 \
    --max-features 0.8
```

### 6.2 テスト実行

```bash
pytest tests/models/test_extratrees.py -v
```

### 6.3 品質チェック

```bash
ruff check src/models/extratrees/
ruff format src/models/extratrees/
pyright src/models/extratrees/
```

---

## 7. 参考リンク

- [scikit-learn ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [Extremely Randomized Trees 論文](https://link.springer.com/article/10.1007/s10994-006-6226-1)
- [LGBM実装](../models/lgbm/train_lgbm.py)
- [CV共通モジュール](../../src/models/common/cv_utils.py)
