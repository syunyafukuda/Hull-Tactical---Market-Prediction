# ElasticNet モデル実装仕様書

最終更新: 2025-12-12

## 実装ステータス

**Status**: ⬜ **未着手**

### 実装予定
- ⬜ `src/models/elasticnet/train_elasticnet.py`: 学習スクリプト
- ⬜ `configs/models/elasticnet.yaml`: YAML設定ファイル
- ⬜ Unit tests: `tests/models/test_elasticnet.py`

### 成果物
- ⬜ `artifacts/models/elasticnet/inference_bundle.pkl`
- ⬜ `artifacts/models/elasticnet/oof_predictions.csv`
- ⬜ `artifacts/models/elasticnet/cv_fold_logs.csv`
- ⬜ `artifacts/models/elasticnet/model_meta.json`
- ⬜ `artifacts/models/elasticnet/coefficients.csv` （係数出力）
- ⬜ `artifacts/models/elasticnet/submission.csv`

---

## 1. 目的と位置づけ

### 1.1 モデル選定フェーズでの役割

- **目的**: L1+L2正則化のハイブリッド線形モデル
- **期待効果**: 
  - Lassoのスパース性とRidgeの安定性を両立
  - 相関の高い特徴量グループからも選択可能
  - 高いアンサンブル多様性（線形 vs 非線形）
- **比較対象**: 
  - LGBM ベースライン（OOF RMSE: 0.012164, LB: 0.681）
  - Ridge（L2のみ）、Lasso（L1のみ）との比較

### 1.2 ElasticNetの特徴

- **L1+L2正則化**: `alpha * (l1_ratio * |w| + (1-l1_ratio) * w²)`
- **スパース性**: Lasso由来。不要な特徴の係数が0になる
- **グループ効果**: 相関の高い特徴量をグループとして選択
- **スケーリング依存**: 特徴量のスケールに敏感（StandardScaler必須）

### 1.3 前提条件

- **特徴セット**: FS_compact（116列）を固定（Feature Selection Phase での結論と整合）
- **CV設定**: TimeSeriesSplit, n_splits=5, gap=0（LGBMと同一）
- **評価指標**:
  - **主指標**: OOF RMSE（選定フェーズの最重要指標）
  - **補助指標**: 予測相関（vs LGBM）、OOF MSR（トレード観点での監視）
- **スケーリング**: **必須**

---

## 2. 技術仕様

### 2.1 入出力

| 項目 | 仕様 |
|------|------|
| 入力 | `data/raw/train.csv`, `data/raw/test.csv` |
| 特徴量生成 | SU1 + SU5 → tier3除外 → 116列 |
| 出力 | `artifacts/models/elasticnet/` 配下に成果物 |
| 追加出力 | `coefficients.csv`: 特徴量係数一覧 |

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
[StandardScaler]  # ★ ElasticNet必須：特徴量スケーリング
    ↓
[ElasticNet]  # ★ L1+L2正則化線形モデル
```

### 2.3 初期ハイパーパラメータ

```python
elasticnet_params = {
    "alpha": 0.001,          # 全体の正則化強度
    "l1_ratio": 0.5,         # L1の割合（0=Ridge, 1=Lasso）
    "fit_intercept": True,   # 切片を学習
    "max_iter": 10000,       # 収束までの最大イテレーション
    "tol": 1e-4,             # 収束判定閾値
    "selection": "cyclic",   # 座標降下法の選択方式
    "random_state": 42,
}
```

### 2.4 パラメータの解釈

```
正則化ペナルティ = alpha * (l1_ratio * |w| + 0.5 * (1-l1_ratio) * w²)

- l1_ratio = 0.0 → 純粋なRidge（L2のみ）
- l1_ratio = 0.5 → L1とL2が半々
- l1_ratio = 1.0 → 純粋なLasso（L1のみ）
```

### 2.5 チューニング戦略

`ElasticNetCV`で自動選択:

```python
from sklearn.linear_model import ElasticNetCV

# ElasticNetCVを使用した自動パラメータ選択
alphas = np.logspace(-5, -1, 30)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

model = ElasticNetCV(
    alphas=alphas,
    l1_ratio=l1_ratios,
    cv=5,
    max_iter=10000,
    random_state=42,
    n_jobs=-1,
)
```

> **⚠️ 時系列CVに関する注意**
> 
> `ElasticNetCV(cv=5)` の内部CVはデフォルトでランダムなK-Fold分割を行います。
> 外側のCV（TimeSeriesSplit）がfold間のリークを防いでいるため致命的な問題ではありませんが、
> 厳密に時系列を守りたい場合は以下の選択肢があります:
> 1. `cv=TimeSeriesSplit(n_splits=5)` を渡す
> 2. `--auto-params` を使わず、外側CVでグリッドサーチを行う
>
> 本フェーズでは「まず固定パラメータで位置確認 → 必要に応じて自動選択で微調整」の方針を推奨します。

### 2.6 Ridge / Lasso / ElasticNet の比較

| 項目 | Ridge | Lasso | ElasticNet |
|------|-------|-------|------------|
| L1正則化 | ✗ | ✓ | ✓ |
| L2正則化 | ✓ | ✗ | ✓ |
| スパース性 | なし | 高い | 中程度 |
| 相関特徴量 | 全て残す | 1つだけ選択 | グループで選択 |
| 安定性 | 高い | 低い | 中程度 |
| パラメータ数 | 1 (alpha) | 1 (alpha) | 2 (alpha, l1_ratio) |

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/elasticnet/
├── __init__.py              # モジュール初期化
└── train_elasticnet.py      # メイン学習スクリプト

configs/models/
└── elasticnet.yaml          # YAML設定ファイル

tests/models/
└── test_elasticnet.py       # ユニットテスト
```

### 3.2 train_elasticnet.py の実装要件

#### 3.2.1 必須機能

1. **引数パース**: `argparse`で以下を受け付ける
   - `--data-dir`: データディレクトリ
   - `--out-dir`: 出力ディレクトリ
   - `--feature-tier`: 使用するtier
   - `--n-splits`, `--gap`: CV設定
   - ElasticNetハイパラ: `--alpha`, `--l1-ratio`, `--auto-params`

2. **前処理**: StandardScalerを追加
3. **係数出力**: 各foldの係数をCSVで保存
4. **選択されたパラメータの記録**: alpha, l1_ratioの両方

### 3.3 コードスケルトン

```python
"""ElasticNet training script with unified CV framework."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 既存モジュールのインポート
from src.feature_generation.su5.train_su5 import (
    SU5FeatureAugmenter,
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
    parser = argparse.ArgumentParser(description="Train ElasticNet model")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/models/elasticnet")
    parser.add_argument(
        "--config-path", type=str, default="configs/feature_generation.yaml"
    )
    parser.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml")
    parser.add_argument("--feature-tier", type=str, default="tier3")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=0)
    # ElasticNet specific
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--auto-params", action="store_true", help="Use ElasticNetCV for auto parameter selection")
    return parser.parse_args()


def build_elasticnet_pipeline(alpha: float = 0.001, l1_ratio: float = 0.5) -> Pipeline:
    """Build ElasticNet pipeline with StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=10000,
            tol=1e-4,
            random_state=42,
        )),
    ])


def build_elasticnetcv_pipeline() -> Pipeline:
    """Build ElasticNetCV pipeline for auto parameter selection."""
    alphas = np.logspace(-5, -1, 30)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=5,
            max_iter=10000,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    alpha: float | None = None,
    l1_ratio: float | None = None,
    auto_params: bool = False,
) -> tuple[Pipeline, np.ndarray, float, float]:
    """Train a single fold, return pipeline, predictions, and selected params."""
    if auto_params:
        pipeline = build_elasticnetcv_pipeline()
    else:
        pipeline = build_elasticnet_pipeline(
            alpha=alpha or 0.001,
            l1_ratio=l1_ratio or 0.5,
        )
    
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    
    # Get actual params used
    model = pipeline.named_steps["model"]
    used_alpha = model.alpha_ if hasattr(model, "alpha_") else model.alpha
    used_l1_ratio = model.l1_ratio_ if hasattr(model, "l1_ratio_") else model.l1_ratio
    
    return pipeline, val_pred, used_alpha, used_l1_ratio


def extract_coefficients(pipeline: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Extract coefficients from trained ElasticNet model."""
    model = pipeline.named_steps["model"]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    })
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    return coef_df


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

    # CV学習
    fold_results = []
    alphas_used = []
    l1_ratios_used = []
    oof_preds = np.zeros(len(X))
    models = []
    all_coefficients = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline, val_pred, used_alpha, used_l1_ratio = train_fold(
            X_train, y_train, X_val,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            auto_params=args.auto_params,
        )
        oof_preds[val_idx] = val_pred
        models.append(pipeline)
        alphas_used.append(used_alpha)
        l1_ratios_used.append(used_l1_ratio)

        # Extract coefficients
        coef_df = extract_coefficients(pipeline, feature_cols)
        coef_df["fold"] = fold_idx
        all_coefficients.append(coef_df)

        metrics = compute_fold_metrics(y_val.values, val_pred, fold_idx)
        fold_results.append(metrics)
        
        n_nonzero = np.sum(np.abs(pipeline.named_steps["model"].coef_) > 1e-10)
        print(f"Fold {fold_idx}: RMSE={metrics.rmse:.6f}, alpha={used_alpha:.6f}, l1_ratio={used_l1_ratio:.2f}, non-zero={n_nonzero}/{len(feature_cols)}")

    # 集計
    summary = aggregate_fold_results(fold_results)
    print(f"\nOOF RMSE: {summary['oof_rmse']:.6f}")
    print(f"OOF MSR: {summary['oof_msr']:.6f}")
    print(f"Mean alpha: {np.mean(alphas_used):.6f}")
    print(f"Mean l1_ratio: {np.mean(l1_ratios_used):.2f}")

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
        {
            "fold": r.fold_idx,
            "rmse": r.rmse,
            "msr": r.msr,
            "n_samples": r.n_samples,
            "alpha": alphas_used[i],
            "l1_ratio": l1_ratios_used[i],
        }
        for i, r in enumerate(fold_results)
    ])
    fold_logs.to_csv(out_dir / "cv_fold_logs.csv", index=False)

    # 4. coefficients.csv (全foldの係数)
    coef_all = pd.concat(all_coefficients, ignore_index=True)
    coef_all.to_csv(out_dir / "coefficients.csv", index=False)

    # 5. coefficients_summary.csv (fold平均)
    # 注意: これはStandardScaler後の係数。元スケールでの寄与を見る場合は
    # scaler.scale_ で割り戻す必要がある
    coef_summary = (
        coef_all.groupby("feature")["coefficient"]
        .agg(["mean", "std"])
        .reset_index()
    )
    coef_summary["abs_mean"] = np.abs(coef_summary["mean"])
    coef_summary = coef_summary.sort_values("abs_mean", ascending=False)
    coef_summary.to_csv(out_dir / "coefficients_summary.csv", index=False)

    # 6. model_meta.json
    meta = {
        "model_type": "ElasticNet",
        "feature_tier": args.feature_tier,
        "n_features": len(feature_cols),
        "cv_config": {"n_splits": args.n_splits, "gap": args.gap},
        "hyperparameters": {
            "alpha": float(np.mean(alphas_used)),
            "l1_ratio": float(np.mean(l1_ratios_used)),
            "auto_params": args.auto_params,
        },
        "oof_rmse": summary["oof_rmse"],
        "oof_msr": summary["oof_msr"],
        "n_nonzero_features_mean": int(np.mean([
            np.sum(np.abs(m.named_steps["model"].coef_) > 1e-10) for m in models
        ])),
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 7. submission.csv
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
# tests/models/test_elasticnet.py

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler


class TestElasticNetPipeline:
    """Test ElasticNet pipeline components."""

    def test_elasticnet_basic_fit(self, sample_train_data: pd.DataFrame) -> None:
        """ElasticNet can fit and predict."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)

        assert len(preds) == len(y)
        assert not np.isnan(preds).any()

    def test_elasticnet_l1_ratio_effect(self, sample_train_data: pd.DataFrame) -> None:
        """l1_ratio affects sparsity."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # High l1_ratio (more like Lasso) -> more sparsity
        model_sparse = ElasticNet(alpha=0.01, l1_ratio=0.9, random_state=42)
        model_sparse.fit(X_scaled, y)

        # Low l1_ratio (more like Ridge) -> less sparsity
        model_dense = ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=42)
        model_dense.fit(X_scaled, y)

        sparse_zeros = np.sum(np.abs(model_sparse.coef_) < 1e-10)
        dense_zeros = np.sum(np.abs(model_dense.coef_) < 1e-10)

        # More sparsity expected with higher l1_ratio
        assert sparse_zeros >= dense_zeros

    def test_elasticnet_between_ridge_lasso(self, sample_train_data: pd.DataFrame) -> None:
        """ElasticNet is between Ridge and Lasso in sparsity."""
        from sklearn.linear_model import Lasso, Ridge

        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = Ridge(alpha=0.1, random_state=42)
        lasso = Lasso(alpha=0.01, random_state=42)
        elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)

        ridge.fit(X_scaled, y)
        lasso.fit(X_scaled, y)
        elastic.fit(X_scaled, y)

        # Ridge should have no zeros, Lasso most zeros
        ridge_zeros = np.sum(np.abs(ridge.coef_) < 1e-10)
        lasso_zeros = np.sum(np.abs(lasso.coef_) < 1e-10)
        elastic_zeros = np.sum(np.abs(elastic.coef_) < 1e-10)

        # ElasticNet should be between (or close to) Ridge and Lasso
        assert elastic_zeros >= ridge_zeros

    def test_elasticnetcv_selects_params(self, sample_train_data: pd.DataFrame) -> None:
        """ElasticNetCV automatically selects alpha and l1_ratio."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        alphas = np.logspace(-5, -1, 10)
        l1_ratios = [0.1, 0.5, 0.9]
        
        model = ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=3,
            random_state=42,
        )
        model.fit(X_scaled, y)

        assert hasattr(model, "alpha_")
        assert hasattr(model, "l1_ratio_")
        assert model.l1_ratio_ in l1_ratios
```

---

## 5. 成功基準

### 5.1 定量基準

| 優先度 | 指標 | 閾値 | 理由 |
|--------|------|------|------|
| **主指標** | OOF RMSE | ≤ 0.015 | 線形モデルとして許容範囲 |
| 補助 | 予測相関（vs LGBM） | < 0.85 | 高いアンサンブル多様性 |
| 補助 | OOF MSR | > 0（監視のみ） | トレード観点での健全性確認 |
| 参考 | 非ゼロ係数数 | 記録 | スパース性の確認 |

> **📌 アンサンブル候補としての判断基準**
>
> 意思決定はRMSEで行い、MSRと相関は補助的な診断指標とします。
> RMSEでLGBMに勝つことは求めず、**相関が十分低ければ（< 0.85）アンサンブル要員として価値あり**と判断します。

### 5.2 定性基準

- [ ] 学習が正常に完了（エラーなし）
- [ ] 成果物が全て生成される
- [ ] 係数CSVが正しく出力される
- [ ] 品質チェック（ruff, pyright）をパス
- [ ] ユニットテストが全てパス

---

## 6. 実行コマンド

### 6.1 学習実行

```bash
# デフォルト設定（固定パラメータ）
python -m src.models.elasticnet.train_elasticnet

# 自動パラメータ選択（ElasticNetCV）
python -m src.models.elasticnet.train_elasticnet --auto-params

# パラメータ指定
python -m src.models.elasticnet.train_elasticnet --alpha 0.0001 --l1-ratio 0.7
```

### 6.2 テスト実行

```bash
pytest tests/models/test_elasticnet.py -v
```

### 6.3 品質チェック

```bash
ruff check src/models/elasticnet/
ruff format src/models/elasticnet/
pyright src/models/elasticnet/
```

---

## 7. 診断的活用

### 7.1 l1_ratio の解釈

- **l1_ratio ≈ 0.9**: Lasso寄り。高スパース性。特徴選択効果強い
- **l1_ratio ≈ 0.5**: バランス型。適度なスパース性
- **l1_ratio ≈ 0.1**: Ridge寄り。低スパース性。安定性重視

### 7.2 Lasso/Ridge/ElasticNet の使い分け

```
特徴量間の相関が低い → Lasso
特徴量間の相関が高い → ElasticNet (グループ選択効果)
全特徴量を使いたい   → Ridge
```

### 7.3 係数分析

```python
# 係数分析
import pandas as pd

coef_df = pd.read_csv("artifacts/models/elasticnet/coefficients_summary.csv")

# 非ゼロ係数のみ
nonzero = coef_df[coef_df["abs_mean"] > 1e-10]
print(f"Non-zero features: {len(nonzero)} / {len(coef_df)}")

# Lasso/ElasticNet/Ridge の非ゼロ数比較
print("ElasticNet non-zero:", len(nonzero))
```

---

## 8. 参考リンク

- [scikit-learn ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [scikit-learn ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
- [Ridge仕様書](ridge.md)
- [Lasso仕様書](lasso.md)
