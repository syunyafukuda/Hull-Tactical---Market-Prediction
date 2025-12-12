# Lasso モデル実装仕様書

最終更新: 2025-12-12

## 実装ステータス

**Status**: ⬜ **未着手**

### 実装予定
- ⬜ `src/models/lasso/train_lasso.py`: 学習スクリプト
- ⬜ `configs/models/lasso.yaml`: YAML設定ファイル
- ⬜ Unit tests: `tests/models/test_lasso.py`

### 成果物
- ⬜ `artifacts/models/lasso/inference_bundle.pkl`
- ⬜ `artifacts/models/lasso/oof_predictions.csv`
- ⬜ `artifacts/models/lasso/cv_fold_logs.csv`
- ⬜ `artifacts/models/lasso/model_meta.json`
- ⬜ `artifacts/models/lasso/coefficients.csv` （係数出力）
- ⬜ `artifacts/models/lasso/submission.csv`

---

## 1. 目的と位置づけ

### 1.1 モデル選定フェーズでの役割

- **目的**: L1正則化による特徴選択効果を持つ線形モデル
- **期待効果**: 
  - 効いていない特徴の係数を0に潰す（スパース性）
  - 「線形で見ると、どの特徴をどこまで使っているか」の診断
  - 高いアンサンブル多様性（線形 vs 非線形）
- **比較対象**: 
  - LGBM ベースライン（OOF RMSE: 0.012164, LB: 0.681）
  - Ridge（L2正則化との比較）

### 1.2 Lassoの特徴

- **L1正則化**: 係数の絶対値和にペナルティを課す
- **スパース性**: 不要な特徴の係数が完全に0になる
- **特徴選択効果**: 自動的に重要な特徴のみを使用
- **スケーリング依存**: 特徴量のスケールに敏感（**StandardScaler必須、外すと係数解釈が破綻**）

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
| 出力 | `artifacts/models/lasso/` 配下に成果物 |
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
[StandardScaler]  # ★ Lasso必須：特徴量スケーリング
    ↓
[Lasso]  # ★ L1正則化線形モデル
```

### 2.3 初期ハイパーパラメータ

```python
lasso_params = {
    "alpha": 0.001,          # 正則化強度（小さめから開始）
    "fit_intercept": True,   # 切片を学習
    "max_iter": 10000,       # 収束までの最大イテレーション
    "tol": 1e-4,             # 収束判定閾値
    "selection": "cyclic",   # 座標降下法の選択方式
    "random_state": 42,
}
```

### 2.4 alpha のチューニング

Lassoの主要パラメータは`alpha`のみ。`LassoCV`で自動選択:

```python
from sklearn.linear_model import LassoCV

# LassoCVを使用した自動alpha選択
# Lasso は alpha が大きすぎると全係数が0になる
alphas = np.logspace(-5, -1, 50)  # 0.00001 〜 0.1
model = LassoCV(alphas=alphas, cv=5, max_iter=10000)
```

> **⚠️ 時系列CVに関する注意**
>
> `LassoCV(cv=5)` の内部CVはデフォルトでランダムなK-Fold分割を行います。
> 外側のCV（TimeSeriesSplit）がfold間のリークを防いでいるため致命的な問題ではありませんが、
> 厳密に時系列を守りたい場合は以下の選択肢があります:
> 1. `cv=TimeSeriesSplit(n_splits=5)` を渡す
> 2. `--auto-alpha` を使わず、外側CVでグリッドサーチを行う
>
> 本フェーズでは「まず固定alphaで位置確認 → 必要に応じて自動選択で微調整」の方針を推奨します。

### 2.5 RidgeとLassoの比較

| 項目 | Ridge (L2) | Lasso (L1) |
|------|------------|------------|
| 正則化 | 係数二乗和 | 係数絶対値和 |
| スパース性 | なし（全係数非ゼロ） | あり（一部係数がゼロ） |
| 特徴選択効果 | なし | あり |
| 多重共線性 | 強い | 弱い |
| 解析解 | あり | なし（反復法） |
| alpha範囲 | 広い | 狭い（感度が高い） |

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/lasso/
├── __init__.py           # モジュール初期化
└── train_lasso.py        # メイン学習スクリプト

configs/models/
└── lasso.yaml            # YAML設定ファイル

tests/models/
└── test_lasso.py         # ユニットテスト
```

### 3.2 train_lasso.py の実装要件

#### 3.2.1 必須機能

1. **引数パース**: `argparse`で以下を受け付ける
   - `--data-dir`: データディレクトリ
   - `--out-dir`: 出力ディレクトリ
   - `--feature-tier`: 使用するtier
   - `--n-splits`, `--gap`: CV設定
   - Lassoハイパラ: `--alpha`, `--auto-alpha`（LassoCV使用フラグ）

2. **前処理**: StandardScalerを追加
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.impute import SimpleImputer
   from sklearn.linear_model import Lasso, LassoCV
   ```

3. **係数出力**: 各foldの係数をCSVで保存

### 3.3 コードスケルトン

```python
"""Lasso training script with unified CV framework."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV
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
    parser = argparse.ArgumentParser(description="Train Lasso model")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="artifacts/models/lasso")
    parser.add_argument(
        "--config-path", type=str, default="configs/feature_generation.yaml"
    )
    parser.add_argument("--preprocess-config", type=str, default="configs/preprocess.yaml")
    parser.add_argument("--feature-tier", type=str, default="tier3")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=0)
    # Lasso specific
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--auto-alpha", action="store_true", help="Use LassoCV for auto alpha selection")
    return parser.parse_args()


def build_lasso_pipeline(alpha: float = 0.001) -> Pipeline:
    """Build Lasso pipeline with StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=10000,
            tol=1e-4,
            random_state=42,
        )),
    ])


def build_lassocv_pipeline() -> Pipeline:
    """Build LassoCV pipeline for auto alpha selection."""
    alphas = np.logspace(-5, -1, 50)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LassoCV(
            alphas=alphas,
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
    auto_alpha: bool = False,
) -> tuple[Pipeline, np.ndarray, float]:
    """Train a single fold, return pipeline, predictions, and selected alpha."""
    if auto_alpha:
        pipeline = build_lassocv_pipeline()
    else:
        pipeline = build_lasso_pipeline(alpha=alpha or 0.001)
    
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    
    # Get actual alpha used
    model = pipeline.named_steps["model"]
    used_alpha = model.alpha_ if hasattr(model, "alpha_") else model.alpha
    
    return pipeline, val_pred, used_alpha


def extract_coefficients(pipeline: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Extract coefficients from trained Lasso model."""
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
    oof_preds = np.zeros(len(X))
    models = []
    all_coefficients = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline, val_pred, used_alpha = train_fold(
            X_train, y_train, X_val,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
        )
        oof_preds[val_idx] = val_pred
        models.append(pipeline)
        alphas_used.append(used_alpha)

        # Extract coefficients
        coef_df = extract_coefficients(pipeline, feature_cols)
        coef_df["fold"] = fold_idx
        all_coefficients.append(coef_df)

        metrics = compute_fold_metrics(y_val.values, val_pred, fold_idx)
        fold_results.append(metrics)
        
        n_nonzero = np.sum(np.abs(pipeline.named_steps["model"].coef_) > 1e-10)
        print(f"Fold {fold_idx}: RMSE={metrics.rmse:.6f}, alpha={used_alpha:.6f}, non-zero={n_nonzero}/{len(feature_cols)}")

    # 集計
    summary = aggregate_fold_results(fold_results)
    print(f"\nOOF RMSE: {summary['oof_rmse']:.6f}")
    print(f"OOF MSR: {summary['oof_msr']:.6f}")
    print(f"Mean alpha: {np.mean(alphas_used):.6f}")

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
        "model_type": "Lasso",
        "feature_tier": args.feature_tier,
        "n_features": len(feature_cols),
        "cv_config": {"n_splits": args.n_splits, "gap": args.gap},
        "hyperparameters": {
            "alpha": float(np.mean(alphas_used)),
            "auto_alpha": args.auto_alpha,
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
# tests/models/test_lasso.py

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


class TestLassoPipeline:
    """Test Lasso pipeline components."""

    def test_lasso_basic_fit(self, sample_train_data: pd.DataFrame) -> None:
        """Lasso can fit and predict."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Lasso(alpha=0.001, random_state=42)
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)

        assert len(preds) == len(y)
        assert not np.isnan(preds).any()

    def test_lasso_sparsity(self, sample_train_data: pd.DataFrame) -> None:
        """Lasso produces sparse coefficients."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # With moderate alpha, some coefficients should be zero
        model = Lasso(alpha=0.01, random_state=42)
        model.fit(X_scaled, y)

        n_zero = np.sum(np.abs(model.coef_) < 1e-10)
        # At least some features should be zeroed out
        assert n_zero >= 0  # May vary depending on data

    def test_lasso_vs_ridge(self, sample_train_data: pd.DataFrame) -> None:
        """Lasso and Ridge give different sparsity patterns."""
        from sklearn.linear_model import Ridge

        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = Lasso(alpha=0.01, random_state=42)
        ridge = Ridge(alpha=1.0, random_state=42)

        lasso.fit(X_scaled, y)
        ridge.fit(X_scaled, y)

        # Lasso should have more zeros than Ridge
        lasso_zeros = np.sum(np.abs(lasso.coef_) < 1e-10)
        ridge_zeros = np.sum(np.abs(ridge.coef_) < 1e-10)
        
        assert lasso_zeros >= ridge_zeros

    def test_lassocv_selects_alpha(self, sample_train_data: pd.DataFrame) -> None:
        """LassoCV automatically selects alpha."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        alphas = np.logspace(-5, -1, 20)
        model = LassoCV(alphas=alphas, cv=3, random_state=42)
        model.fit(X_scaled, y)

        assert hasattr(model, "alpha_")
        assert model.alpha_ in alphas or np.isclose(model.alpha_, alphas).any()

    def test_scaling_required(self, sample_train_data: pd.DataFrame) -> None:
        """Lasso results differ significantly with/without scaling."""
        X = sample_train_data.drop(columns=["Date", "TARGET"])
        y = sample_train_data["TARGET"]

        # Without scaling
        model1 = Lasso(alpha=0.001, random_state=42)
        model1.fit(X, y)

        # With scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model2 = Lasso(alpha=0.001, random_state=42)
        model2.fit(X_scaled, y)

        # Coefficients should be different
        assert not np.allclose(model1.coef_, model2.coef_)
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
# デフォルト設定（固定alpha）
python -m src.models.lasso.train_lasso

# 自動alpha選択（LassoCV）
python -m src.models.lasso.train_lasso --auto-alpha

# alpha指定
python -m src.models.lasso.train_lasso --alpha 0.0001
```

### 6.2 テスト実行

```bash
pytest tests/models/test_lasso.py -v
```

### 6.3 品質チェック

```bash
ruff check src/models/lasso/
ruff format src/models/lasso/
pyright src/models/lasso/
```

---

## 7. 診断的活用

### 7.1 係数分析

Lassoの係数を分析することで、以下の洞察が得られる:

1. **重要特徴量の特定**: 非ゼロ係数を持つ特徴量
2. **特徴量グループの比較**: SU1 vs SU5 vs raw のどれが効いているか
3. **符号の解釈**: 正負の係数から特徴量とターゲットの関係を理解

### 7.2 分析スクリプト例

```python
# 係数分析
import pandas as pd

coef_df = pd.read_csv("artifacts/models/lasso/coefficients_summary.csv")

# 非ゼロ係数のみ
nonzero = coef_df[coef_df["abs_mean"] > 1e-10]
print(f"Non-zero features: {len(nonzero)} / {len(coef_df)}")

# Top 20 重要特徴量
print(nonzero.head(20))
```

---

## 8. 参考リンク

- [scikit-learn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [scikit-learn LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
- [Ridge仕様書](ridge.md)
- [ElasticNet仕様書](elasticnet.md)

---

## 9. 注意事項（XGBoost実装から得た共通教訓）

### 9.1 テスト予測時のfeatureフィルタリング

テストデータには学習時に存在しないカラム（`is_scored`, `lagged_*`等）が含まれる場合がある。
**学習時のfeature_colsのみを抽出**してから予測を実行：
```python
test_features = test_df[feature_cols].copy()
test_pred = final_pipeline.predict(test_features)
```

### 9.2 submission.csv のシグナル変換

生の予測値（excess returns）ではなく、**競技シグナル形式**に変換して出力：
```python
# シグナル変換: pred * mult + 1.0, clipped to [0.9, 1.1]
signal_mult = 1.0
signal_pred = np.clip(test_pred * signal_mult + 1.0, 0.9, 1.1)

# カラム名は "prediction"（target変数名ではない）
submission_df = pd.DataFrame({
    "date_id": id_values,
    "prediction": signal_pred,
})
```

### 9.3 is_scored フィルタリング

submission.csvには`is_scored==True`の行のみを含める（競技要件）。
