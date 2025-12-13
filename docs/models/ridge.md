# Ridge 回帰モデル実装仕様書

最終更新: 2025-12-12

## 実装ステータス

**Status**: ⬜ **未着手**

### 実装予定
- ⬜ `src/models/ridge/train_ridge.py`: 学習スクリプト
- ⬜ `src/models/ridge/config.py`: ハイパラ設定
- ⬜ `configs/models/ridge.yaml`: YAML設定ファイル
- ⬜ Unit tests: `tests/models/test_ridge.py`

### 成果物
- ⬜ `artifacts/models/ridge/inference_bundle.pkl`
- ⬜ `artifacts/models/ridge/oof_predictions.csv`
- ⬜ `artifacts/models/ridge/cv_fold_logs.csv`
- ⬜ `artifacts/models/ridge/model_meta.json`
- ⬜ `artifacts/models/ridge/feature_list.json`
- ⬜ `artifacts/models/ridge/submission.csv`

**Note**: 出力仕様の詳細は [README.md](README.md#成果物出力仕様kaggle-nb用) を参照。

---

## 1. 目的と位置づけ

### 1.1 モデル選定フェーズでの役割

- **目的**: 線形モデルによるベースライン確認とアンサンブル多様性の確保
- **期待効果**: 
  - 勾配ブースティングとは根本的に異なるモデルクラス
  - 高い予測の多様性（低い予測相関）
  - 解釈性が高い（係数を分析可能）
- **比較対象**: LGBM ベースライン（OOF RMSE: 0.012164, LB: 0.681）

### 1.2 Ridgeの特徴

- **L2正則化**: 係数の二乗和にペナルティを課し、過学習を抑制
- **閉形式解**: 解析的に最適解が求まるため、学習が高速
- **スケーリング依存**: 特徴量のスケールに敏感（StandardScaler必須）

### 1.3 前提条件

- **特徴セット**: FS_compact（116列）を固定（Feature Selection Phase での結論と整合）
  - **必ず FS_compact を入力として使用する**。特徴量セットの変更は行わない。
- **CV設定**: TimeSeriesSplit, n_splits=5, gap=0（LGBMと同一）
- **評価指標**:
  - **主指標**: OOF RMSE（選定フェーズの最重要指標）
  - **補助指標**: 予測相関（vs LGBM）、OOF MSR（トレード観点での監視）

---

## 2. 技術仕様

### 2.1 入出力

| 項目 | 仕様 |
|------|------|
| 入力 | `data/raw/train.csv`, `data/raw/test.csv` |
| 特徴量生成 | SU1 + SU5 → tier3除外 → 116列 |
| 出力 | `artifacts/models/ridge/` 配下に成果物 |

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
[StandardScaler]  # ★ Ridge必須：特徴量スケーリング
    ↓
[Ridge]  # ★ LGBMRegressorの代わり
```

### 2.3 初期ハイパーパラメータ

```python
ridge_params = {
    "alpha": 1.0,           # 正則化強度（CVで調整）
    "fit_intercept": True,  # 切片を学習
    "solver": "auto",       # 自動選択（データサイズに応じて最適化）
    "random_state": 42,
}
```

### 2.4 alpha のチューニング

Ridgeの主要パラメータは`alpha`のみ。CV内でグリッドサーチまたは`RidgeCV`を使用:

```python
from sklearn.linear_model import RidgeCV

# RidgeCVを使用した自動alpha選択
alphas = np.logspace(-3, 3, 20)  # 0.001 〜 1000
model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
```

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/ridge/
├── __init__.py           # モジュール初期化
├── train_ridge.py        # メイン学習スクリプト
└── config.py             # ハイパラ定義（オプション）

configs/models/
└── ridge.yaml            # YAML設定ファイル

tests/models/
└── test_ridge.py         # ユニットテスト
```

### 3.2 train_ridge.py の実装要件

#### 3.2.1 必須機能

1. **引数パース**: `argparse`で以下を受け付ける
   - `--data-dir`: データディレクトリ（デフォルト: `data/raw`）
   - `--out-dir`: 出力ディレクトリ（デフォルト: `artifacts/models/ridge`）
   - `--config-path`: feature_generation.yaml パス
   - `--preprocess-config`: preprocess.yaml パス
   - `--feature-tier`: 使用するtier（デフォルト: `tier3`）
   - `--n-splits`, `--gap`: CV設定
   - Ridgeハイパラ: `--alpha`, `--auto-alpha`（RidgeCV使用フラグ）

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

3. **前処理**: StandardScalerを追加
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.impute import SimpleImputer
   from sklearn.pipeline import Pipeline
   
   preprocess = Pipeline([
       ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
       ("scaler", StandardScaler()),
   ])
   ```

4. **CV実行**: TimeSeriesSplitで5-fold CV
   - 各foldでtrain/val分割
   - alphaの選択: 固定値 or RidgeCV
   - OOF予測を蓄積

5. **評価メトリクス計算**:
   ```python
   from src.models.common.cv_utils import (
       compute_fold_metrics,
       evaluate_oof_predictions,
   )
   ```

6. **成果物出力**:
   - `inference_bundle.pkl`: 全データで再学習したパイプライン
   - `oof_predictions.csv`: OOF予測値
   - `cv_fold_logs.csv`: フォールドごとの指標
   - `model_meta.json`: メタデータ（選択されたalphaを含む）
   - `coefficients.csv`: 特徴量係数（解釈性のため）

#### 3.2.2 コード骨格

```python
#!/usr/bin/env python
"""Ridge regression training script using the unified model framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import existing modules
from src.feature_generation.su5.train_su5 import (
    load_su1_config, load_su5_config, load_preprocess_policies,
    SU5FeatureAugmenter, _prepare_features, load_table,
    infer_train_file, infer_test_file,
)
from src.models.common.feature_loader import get_excluded_features
from src.models.common.cv_utils import compute_fold_metrics, evaluate_oof_predictions

def build_ridge_pipeline(alpha: float = 1.0, ...):
    """Build preprocessing pipeline with Ridge regressor."""
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
    ])
    return Pipeline([
        # ... existing imputers ...
        ("preprocess", preprocess),
        ("model", Ridge(alpha=alpha)),
    ])

def main(argv: Sequence[str] | None = None) -> int:
    """Main training function."""
    # 1. Parse arguments
    # 2. Load data and configs
    # 3. Generate features (SU1 + SU5)
    # 4. Apply tier3 exclusion
    # 5. Run CV with Ridge (or RidgeCV for auto alpha)
    # 6. Save artifacts (including coefficients)
    pass

if __name__ == "__main__":
    sys.exit(main())
```

### 3.3 Ridge固有の実装ポイント

#### 3.3.1 StandardScalerの適用タイミング

Ridgeはスケールに敏感なため、**前処理の最後**でスケーリングを適用:

```python
# GroupImputersの後、Ridgeの前にスケーリング
pipeline = Pipeline([
    ("augment", augmenter),
    ("m_imputer", m_imputer),
    ("e_imputer", e_imputer),
    ("i_imputer", i_imputer),
    ("p_imputer", p_imputer),
    ("s_imputer", s_imputer),
    ("final_imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("scaler", StandardScaler()),  # ★ 追加
    ("model", Ridge(alpha=1.0)),
])
```

#### 3.3.2 係数の保存と分析

Ridgeの係数は解釈に有用。学習後に保存:

```python
def save_coefficients(model: Ridge, feature_names: list, output_path: Path):
    """係数をCSVに保存."""
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_,
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_path, index=False)
```

#### 3.3.3 RidgeCVの使用

alpha自動選択を行う場合:

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-3, 3, 20)
model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
model.fit(X_train_scaled, y_train)
best_alpha = model.alpha_  # 選択されたalpha
```

### 3.4 YAML設定ファイル

`configs/models/ridge.yaml`:
```yaml
model:
  type: ridge
  version: v1

hyperparameters:
  alpha: 1.0                    # 固定alpha（auto_alphaがtrueの場合は無視）
  auto_alpha: true              # RidgeCVを使用してalphaを自動選択
  alpha_range:
    min: 0.001
    max: 1000
    n_values: 20
  fit_intercept: true
  solver: auto
  random_state: 42

preprocessing:
  scaler: standard              # StandardScaler
  imputer_strategy: constant
  imputer_fill_value: 0.0

cv:
  n_splits: 5
  gap: 0

feature_selection:
  tier: tier3
  excluded_json: configs/feature_selection/tier3/excluded.json
```

---

## 4. テスト仕様

### 4.1 ユニットテスト

`tests/models/test_ridge.py`:

```python
"""Unit tests for Ridge regression model training."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

class TestRidgeTraining:
    """Tests for Ridge training module."""

    def test_ridge_import(self):
        """Ridgeがインポートできることを確認."""
        from sklearn.linear_model import Ridge
        assert Ridge is not None

    def test_ridge_basic_fit(self, sample_data):
        """基本的なfit/predictが動作することを確認."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)
        assert hasattr(model, "coef_")

    def test_ridge_cv_alpha_selection(self, sample_data):
        """RidgeCVでalpha選択が動作することを確認."""
        X, y = sample_data
        alphas = np.logspace(-2, 2, 10)
        model = RidgeCV(alphas=alphas)
        model.fit(X, y)
        assert model.alpha_ in alphas

    def test_ridge_scaling_impact(self, sample_data):
        """スケーリングの有無で結果が変わることを確認."""
        from sklearn.preprocessing import StandardScaler
        X, y = sample_data
        
        # スケーリングなし
        model1 = Ridge(alpha=1.0)
        model1.fit(X, y)
        
        # スケーリングあり
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model2 = Ridge(alpha=1.0)
        model2.fit(X_scaled, y)
        
        # 係数が異なることを確認
        assert not np.allclose(model1.coef_, model2.coef_)

    def test_ridge_pipeline_integration(self, sample_augmented_data):
        """前処理パイプラインとの統合を確認."""
        pass

    def test_ridge_coefficient_output(self, tmp_path, sample_data):
        """係数が正しく出力されることを確認."""
        pass
```

### 4.2 統合テスト

- LGBM train_lgbm.py と同一のデータで実行し、出力フォーマットが一致することを確認
- OOF RMSE を比較（Ridgeは勾配ブースティングより劣る可能性が高い）

---

## 5. 評価基準

### 5.1 成功条件

| 優先度 | 指標 | 条件 | 備考 |
|--------|------|------|------|
| **主指標** | OOF RMSE | ≤ 0.015 | 線形モデルなので緩め |
| 補助 | 予測相関（vs LGBM） | < 0.90 | 高い多様性を期待 |
| 補助 | OOF MSR | > 0（監視のみ） | トレード観点での健全性確認 |
| 参考 | 実行時間 | < 1分 | 閉形式解のため高速 |

### 5.2 アンサンブル価値の判断

- **予測相関 < 0.85**: 高いアンサンブル効果が期待できる
- **予測相関 0.85-0.90**: 一定の効果が期待できる
- **予測相関 > 0.90**: アンサンブルへの寄与は限定的

### 5.3 LB提出判断

- Ridge単体でのLB提出は**基本的に不要**（ベースライン確認のため）
- アンサンブル時にLGBM + Ridgeの組み合わせでLB確認

---

## 6. 依存パッケージ

Ridgeは`scikit-learn`に含まれるため、追加インストール不要:

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
```

---

## 7. 実行方法

### 7.1 学習実行

```bash
# デフォルト設定で実行（RidgeCVでalpha自動選択）
uv run python src/models/ridge/train_ridge.py

# 固定alphaで実行
uv run python src/models/ridge/train_ridge.py --alpha 10.0 --no-auto-alpha

# カスタムalpha範囲でRidgeCV
uv run python src/models/ridge/train_ridge.py \
    --alpha-min 0.01 \
    --alpha-max 100 \
    --alpha-n-values 30
```

### 7.2 テスト実行

```bash
uv run pytest tests/models/test_ridge.py -v
```

### 7.3 品質チェック

```bash
uv run ruff check src/models/ridge/
uv run pyright src/models/ridge/
```

---

## 8. 参考リンク

- [LGBM実装](../../src/models/lgbm/train_lgbm.py): 参考実装
- [Feature Selection README](../feature_selection/README.md): 特徴量選定の経緯
- [Model Selection README](README.md): モデル選定戦略の全体像
- [scikit-learn Ridge Documentation](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

---

## 9. 注意事項（XGBoost実装から得た共通教訓を含む）

### 9.1 Ridge固有の注意点

1. **スケーリング必須**: Ridgeは特徴量のスケールに敏感。StandardScalerを必ず適用すること。

2. **NaN処理**: Ridgeは欠損値を受け付けない。GroupImputersの後にSimpleImputerで残余NaNを処理。

3. **多重共線性**: Ridgeは多重共線性に強いが、極端に相関が高い特徴量群ではalphaを大きくする必要がある場合がある。

4. **係数の解釈**: StandardScaler 適用後の係数であるため、特徴量間で直接比較可能。
   - 元スケールでの影響を見たい場合は、スケーラーの `mean_` と `scale_` を用いて逆変換する：
     ```python
     # 逆変換: 元スケールの係数 = スケール後の係数 / scaler.scale_
     original_coef = model.coef_ / scaler.scale_
     ```

5. **ターゲットスケーリング**: 今回はターゲット（market_forward_excess_returns）はスケーリングしない。必要に応じて検討。

### 9.2 テスト予測時のfeatureフィルタリング

テストデータには学習時に存在しないカラム（`is_scored`, `lagged_*`等）が含まれる場合がある。
**学習時のfeature_colsのみを抽出**してから予測を実行：
```python
test_features = test_df[feature_cols].copy()
test_pred = final_pipeline.predict(test_features)
```

### 9.3 submission.csv のシグナル変換

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

### 9.4 is_scored フィルタリング

submission.csvには`is_scored==True`の行のみを含める（競技要件）。

---

## 10. 追加出力: coefficients.csv

Ridgeの特徴として、係数の分析結果を追加で出力:

```
artifacts/models/ridge/
├── inference_bundle.pkl
├── oof_predictions.csv
├── cv_fold_logs.csv
├── model_meta.json
└── coefficients.csv       # ★ Ridge固有の出力
```

`coefficients.csv` の形式:
```csv
feature,coefficient,abs_coefficient
V5,0.0234,0.0234
E12,0.0198,0.0198
M7,-0.0187,0.0187
...
```

この係数分析により、どの特徴量が線形モデルで重要視されているかを確認でき、特徴量エンジニアリングの洞察が得られる。
