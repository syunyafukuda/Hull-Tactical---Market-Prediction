# CatBoost モデル実装仕様書

最終更新: 2025-12-12

## 実装ステータス

**Status**: ⬜ **未着手**

### 実装予定
- ⬜ `src/models/catboost/train_catboost.py`: 学習スクリプト
- ⬜ `src/models/catboost/config.py`: ハイパラ設定
- ⬜ `configs/models/catboost.yaml`: YAML設定ファイル
- ⬜ Unit tests: `tests/models/test_catboost.py`

### 成果物
- ⬜ `artifacts/models/catboost/inference_bundle.pkl`
- ⬜ `artifacts/models/catboost/oof_predictions.csv`
- ⬜ `artifacts/models/catboost/cv_fold_logs.csv`
- ⬜ `artifacts/models/catboost/model_meta.json`

---

## 1. 目的と位置づけ

### 1.1 モデル選定フェーズでの役割

- **目的**: 順序付きブースティング（Ordered Boosting）による過学習耐性の高いモデルを導入
- **期待効果**: 
  - LGBMとは異なるブースティング手法による多様性
  - 時系列データでの過学習抑制効果
- **比較対象**: LGBM ベースライン（OOF RMSE: 0.012164, LB: 0.681）

### 1.2 CatBoostの特徴

- **Ordered Boosting**: 時系列的な順序を考慮した学習（リーク防止に有利）
- **Symmetric Trees**: 対称木構造による高速化と正則化効果
- **Native Categorical Support**: カテゴリ変数の自動エンコーディング（今回は未使用）

### 1.3 前提条件

- **特徴セット**: FS_compact（116列）を固定
- **CV設定**: TimeSeriesSplit, n_splits=5, gap=0（LGBMと同一）
- **評価指標**: OOF RMSE, OOF MSR, 予測相関（vs LGBM）

---

## 2. 技術仕様

### 2.1 入出力

| 項目 | 仕様 |
|------|------|
| 入力 | `data/raw/train.csv`, `data/raw/test.csv` |
| 特徴量生成 | SU1 + SU5 → tier3除外 → 116列 |
| 出力 | `artifacts/models/catboost/` 配下に成果物 |

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
[ColumnTransformer: numeric + categorical]
    ↓
[CatBoostRegressor]  # ★ LGBMRegressorの代わり
```

### 2.3 初期ハイパーパラメータ

```python
catboost_params = {
    "loss_function": "RMSE",
    "iterations": 600,               # LGBMのn_estimatorsに相当
    "depth": 6,                      # LGBMより浅め
    "learning_rate": 0.05,           # LGBMと同一
    "l2_leaf_reg": 3.0,              # L2正則化（デフォルト3）
    "random_strength": 1.0,          # ランダム性の強さ
    "bagging_temperature": 1.0,      # Bayesian Bootstrap温度
    "border_count": 254,             # 数値特徴の分割数
    "random_seed": 42,
    "thread_count": -1,
    "verbose": False,
    "early_stopping_rounds": 50,
    "use_best_model": True,
}
```

### 2.4 LGBMとのパラメータ対応表

| LGBM | CatBoost | 初期値 |
|------|----------|--------|
| `num_leaves` | N/A（`depth`で制御） | - |
| `max_depth` | `depth` | 6 |
| `min_data_in_leaf` | `min_data_in_leaf` | 1（デフォルト） |
| `feature_fraction` | `rsm`（Random Subspace Method） | 1.0 |
| `bagging_fraction` | `subsample` | 1.0 |
| `learning_rate` | `learning_rate` | 0.05 |
| `n_estimators` | `iterations` | 600 |
| `lambda_l1` | N/A | - |
| `lambda_l2` | `l2_leaf_reg` | 3.0 |

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/catboost/
├── __init__.py              # モジュール初期化
├── train_catboost.py        # メイン学習スクリプト
└── config.py                # ハイパラ定義（オプション）

configs/models/
└── catboost.yaml            # YAML設定ファイル

tests/models/
└── test_catboost.py         # ユニットテスト
```

### 3.2 train_catboost.py の実装要件

#### 3.2.1 必須機能

1. **引数パース**: `argparse`で以下を受け付ける
   - `--data-dir`: データディレクトリ（デフォルト: `data/raw`）
   - `--out-dir`: 出力ディレクトリ（デフォルト: `artifacts/models/catboost`）
   - `--config-path`: feature_generation.yaml パス
   - `--preprocess-config`: preprocess.yaml パス
   - `--feature-tier`: 使用するtier（デフォルト: `tier3`）
   - `--n-splits`, `--gap`: CV設定
   - CatBoostハイパラ: `--depth`, `--learning-rate`, `--iterations` 等

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

3. **CV実行**: TimeSeriesSplitで5-fold CV
   - 各foldでtrain/val分割
   - Early stoppingを使用（`early_stopping_rounds=50`）
   - OOF予測を蓄積

4. **評価メトリクス計算**:
   ```python
   from src.models.common.cv_utils import (
       compute_fold_metrics,
       evaluate_oof_predictions,
   )
   ```

5. **成果物出力**:
   - `inference_bundle.pkl`: 全データで再学習したパイプライン
   - `oof_predictions.csv`: OOF予測値
   - `cv_fold_logs.csv`: フォールドごとの指標
   - `model_meta.json`: メタデータ

#### 3.2.2 コード骨格

```python
#!/usr/bin/env python
"""CatBoost training script using the unified model framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor, Pool

# Import existing modules
from src.feature_generation.su5.train_su5 import (
    load_su1_config, load_su5_config, load_preprocess_policies,
    SU5FeatureAugmenter, _prepare_features, load_table,
    infer_train_file, infer_test_file,
)
from src.models.common.feature_loader import get_excluded_features
from src.models.common.cv_utils import compute_fold_metrics, evaluate_oof_predictions

def build_catboost_pipeline(...):
    """Build preprocessing pipeline with CatBoostRegressor."""
    pass

def main(argv: Sequence[str] | None = None) -> int:
    """Main training function."""
    # 1. Parse arguments
    # 2. Load data and configs
    # 3. Generate features (SU1 + SU5)
    # 4. Apply tier3 exclusion
    # 5. Run CV with CatBoost
    # 6. Save artifacts
    pass

if __name__ == "__main__":
    sys.exit(main())
```

### 3.3 CatBoost固有の実装ポイント

#### 3.3.1 Poolオブジェクトの使用

CatBoostはPoolオブジェクトでデータを管理すると効率的:

```python
from catboost import Pool

train_pool = Pool(X_train, y_train)
eval_pool = Pool(X_valid, y_valid)

model.fit(
    train_pool,
    eval_set=eval_pool,
    early_stopping_rounds=50,
    verbose=False,
)
```

#### 3.3.2 特徴量名のサニタイズ

CatBoostは特徴量名に制約がないが、一貫性のため他モデルと同様にサニタイズを推奨:

```python
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量名をサニタイズ."""
    df.columns = [col.replace("/", "_").replace("__", "_") for col in df.columns]
    return df
```

### 3.4 YAML設定ファイル

`configs/models/catboost.yaml`:
```yaml
model:
  type: catboost
  version: v1

hyperparameters:
  loss_function: RMSE
  iterations: 600
  depth: 6
  learning_rate: 0.05
  l2_leaf_reg: 3.0
  random_strength: 1.0
  bagging_temperature: 1.0
  border_count: 254
  random_seed: 42
  thread_count: -1
  verbose: false
  early_stopping_rounds: 50
  use_best_model: true

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

`tests/models/test_catboost.py`:

```python
"""Unit tests for CatBoost model training."""

import pytest
import numpy as np
import pandas as pd

class TestCatBoostTraining:
    """Tests for CatBoost training module."""

    def test_catboost_import(self):
        """CatBoostがインポートできることを確認."""
        from catboost import CatBoostRegressor
        assert CatBoostRegressor is not None

    def test_catboost_basic_fit(self, sample_data):
        """基本的なfit/predictが動作することを確認."""
        from catboost import CatBoostRegressor
        X, y = sample_data
        model = CatBoostRegressor(iterations=10, depth=3, verbose=False)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)

    def test_catboost_pool(self, sample_data):
        """Poolオブジェクトが正しく作成されることを確認."""
        from catboost import Pool
        X, y = sample_data
        pool = Pool(X, y)
        assert pool.num_row() == len(y)

    def test_catboost_pipeline_integration(self, sample_augmented_data):
        """前処理パイプラインとの統合を確認."""
        pass

    def test_catboost_cv_metrics(self, sample_data):
        """CV評価メトリクスが正しく計算されることを確認."""
        pass

    def test_catboost_artifact_output(self, tmp_path, sample_data):
        """成果物が正しく出力されることを確認."""
        pass
```

### 4.2 統合テスト

- LGBM train_lgbm.py と同一のデータで実行し、出力フォーマットが一致することを確認
- OOF RMSE がベースライン（0.012164）と同等レベルであることを確認

---

## 5. 評価基準

### 5.1 成功条件

| 指標 | 条件 | 備考 |
|------|------|------|
| OOF RMSE | ≤ 0.0125 | ベースライン+3%以内 |
| 予測相関（vs LGBM） | < 0.98 | アンサンブル効果の見込み |
| 実行時間 | < 15分 | CatBoostはやや遅い傾向 |

### 5.2 LB提出判断

- OOF RMSEが0.0122以下の場合、LB提出を検討
- OOF RMSEが0.0125を超える場合、ハイパラ調整を優先

---

## 6. 依存パッケージ

```toml
# pyproject.toml に追加が必要な場合
[project.dependencies]
catboost = ">=1.2.0"
```

確認コマンド:
```bash
uv pip show catboost || uv add catboost
```

---

## 7. 実行方法

### 7.1 学習実行

```bash
# デフォルト設定で実行
uv run python src/models/catboost/train_catboost.py

# カスタム設定で実行
uv run python src/models/catboost/train_catboost.py \
    --depth 8 \
    --learning-rate 0.03 \
    --iterations 800
```

### 7.2 テスト実行

```bash
uv run pytest tests/models/test_catboost.py -v
```

### 7.3 品質チェック

```bash
uv run ruff check src/models/catboost/
uv run pyright src/models/catboost/
```

---

## 8. 参考リンク

- [LGBM実装](../../src/models/lgbm/train_lgbm.py): 参考実装
- [Feature Selection README](../feature_selection/README.md): 特徴量選定の経緯
- [Model Selection README](README.md): モデル選定戦略の全体像
- [CatBoost Documentation](https://catboost.ai/docs/)

---

## 9. 注意事項

1. **Ordered Boosting**: デフォルトで有効。時系列データに有利だが、学習が遅くなる場合がある。`boosting_type='Plain'`で無効化可能。

2. **GPUサポート**: `task_type='GPU'`で有効化可能だが、環境依存のため初期実装ではCPUのみ。

3. **モデルサイズ**: CatBoostのモデルファイルは比較的大きくなる傾向がある。

4. **sklearn互換性**: `CatBoostRegressor`はsklearn互換APIを持つが、Pipelineでの使用時に一部制約あり（fit_params の渡し方など）。

5. **特徴量重要度**: `get_feature_importance()`で取得可能。type='PredictionValuesChange'がLGBMのgainに相当。
