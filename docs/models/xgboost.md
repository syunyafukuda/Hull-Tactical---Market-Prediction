# XGBoost モデル実装仕様書

最終更新: 2025-12-12

## 実装ステータス

**Status**: ✅ **実装完了**

### 実装済み
- ✅ `src/models/xgboost/train_xgb.py`: 学習スクリプト
- ✅ `configs/models/xgboost.yaml`: YAML設定ファイル
- ✅ Unit tests: `tests/models/test_xgboost.py`

### 成果物（実行時生成）
- ✅ `artifacts/models/xgboost/inference_bundle.pkl`
- ✅ `artifacts/models/xgboost/oof_predictions.csv`
- ✅ `artifacts/models/xgboost/cv_fold_logs.csv`
- ✅ `artifacts/models/xgboost/model_meta.json`
- ✅ `artifacts/models/xgboost/feature_list.json`
- ✅ `artifacts/models/xgboost/submission.csv`

**Note**: 成果物は実際のデータで学習実行時に生成されます。
出力仕様の詳細は [README.md](README.md#成果物出力仕様kaggle-nb用) を参照。

---

## 1. 目的と位置づけ

### 1.1 モデル選定フェーズでの役割

- **目的**: LGBMと同系統の勾配ブースティングモデルだが、実装の違いによる多様性を導入
- **期待効果**: アンサンブル時に予測相関が適度に異なることで、精度向上の可能性
- **比較対象**: LGBM ベースライン（OOF RMSE: 0.012164, LB: 0.681）

### 1.2 前提条件

- **特徴セット**: FS_compact（116列）を固定（Feature Selection Phase での結論と整合）
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
| 出力 | `artifacts/models/xgboost/` 配下に成果物 |

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
[XGBRegressor]  # ★ LGBMRegressorの代わり
```

### 2.3 初期ハイパーパラメータ

```python
xgb_params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",           # 高速化
    "max_depth": 6,                  # LGBMより浅め（過学習抑制）
    "learning_rate": 0.05,           # LGBMと同一
    "n_estimators": 600,             # LGBMと同一
    "subsample": 0.8,                # LGBMのbagging_fractionに相当
    "colsample_bytree": 0.8,         # LGBMのfeature_fractionに相当
    "reg_alpha": 0.0,                # L1正則化
    "reg_lambda": 1.0,               # L2正則化
    "min_child_weight": 32,          # LGBMのmin_data_in_leafに相当
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
```

### 2.4 LGBMとのパラメータ対応表

| LGBM | XGBoost | 初期値 |
|------|---------|--------|
| `num_leaves` | N/A（`max_depth`で制御） | - |
| `max_depth` | `max_depth` | 6 |
| `min_data_in_leaf` | `min_child_weight` | 32 |
| `feature_fraction` | `colsample_bytree` | 0.8 |
| `bagging_fraction` | `subsample` | 0.8 |
| `bagging_freq` | N/A（自動） | - |
| `learning_rate` | `learning_rate` | 0.05 |
| `n_estimators` | `n_estimators` | 600 |
| `lambda_l1` | `reg_alpha` | 0.0 |
| `lambda_l2` | `reg_lambda` | 1.0 |

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/xgboost/
├── __init__.py           # モジュール初期化
├── train_xgb.py          # メイン学習スクリプト
└── config.py             # ハイパラ定義（オプション）

configs/models/
└── xgboost.yaml          # YAML設定ファイル

tests/models/
└── test_xgboost.py       # ユニットテスト
```

### 3.2 train_xgb.py の実装要件

#### 3.2.1 必須機能

1. **引数パース**: `argparse`で以下を受け付ける
   - `--data-dir`: データディレクトリ（デフォルト: `data/raw`）
   - `--out-dir`: 出力ディレクトリ（デフォルト: `artifacts/models/xgboost`）
   - `--config-path`: feature_generation.yaml パス
   - `--preprocess-config`: preprocess.yaml パス
   - `--feature-tier`: 使用するtier（デフォルト: `tier3`）
   - `--n-splits`, `--gap`: CV設定
   - XGBoostハイパラ: `--max-depth`, `--learning-rate`, `--n-estimators` 等

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
"""XGBoost training script using the unified model framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Import existing modules
from src.feature_generation.su5.train_su5 import (
    load_su1_config, load_su5_config, load_preprocess_policies,
    SU5FeatureAugmenter, _prepare_features, load_table,
    infer_train_file, infer_test_file,
)
from src.models.common.feature_loader import get_excluded_features
from src.models.common.cv_utils import compute_fold_metrics, evaluate_oof_predictions

def build_xgb_pipeline(...):
    """Build preprocessing pipeline with XGBRegressor."""
    # 既存のbuild_pipelineを参考に、モデル部分をXGBRegressorに置換
    pass

def main(argv: Sequence[str] | None = None) -> int:
    """Main training function."""
    # 1. Parse arguments
    # 2. Load data and configs
    # 3. Generate features (SU1 + SU5)
    # 4. Apply tier3 exclusion
    # 5. Run CV with XGBoost
    # 6. Save artifacts
    pass

if __name__ == "__main__":
    sys.exit(main())
```

### 3.3 YAML設定ファイル

`configs/models/xgboost.yaml`:
```yaml
model:
  type: xgboost
  version: v1

hyperparameters:
  objective: reg:squarederror
  tree_method: hist
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 600
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.0
  reg_lambda: 1.0
  min_child_weight: 32
  random_state: 42
  n_jobs: -1
  verbosity: 0
  early_stopping_rounds: 50

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

`tests/models/test_xgboost.py`:

```python
"""Unit tests for XGBoost model training."""

import pytest
import numpy as np
import pandas as pd

class TestXGBoostTraining:
    """Tests for XGBoost training module."""

    def test_xgb_import(self):
        """XGBoostがインポートできることを確認."""
        from xgboost import XGBRegressor
        assert XGBRegressor is not None

    def test_xgb_basic_fit(self, sample_data):
        """基本的なfit/predictが動作することを確認."""
        from xgboost import XGBRegressor
        X, y = sample_data
        model = XGBRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)

    def test_xgb_pipeline_integration(self, sample_augmented_data):
        """前処理パイプラインとの統合を確認."""
        # SU1/SU5特徴量生成後のデータでXGBが動作することを確認
        pass

    def test_xgb_cv_metrics(self, sample_data):
        """CV評価メトリクスが正しく計算されることを確認."""
        pass

    def test_xgb_artifact_output(self, tmp_path, sample_data):
        """成果物が正しく出力されることを確認."""
        pass
```

### 4.2 統合テスト

- LGBM train_lgbm.py と同一のデータで実行し、出力フォーマットが一致することを確認
- OOF RMSE がベースライン（0.012164）と同等レベルであることを確認

---

## 5. 評価基準

### 5.1 成功条件

| 優先度 | 指標 | 条件 | 備考 |
|--------|------|------|------|
| **主指標** | OOF RMSE | ≤ 0.0125 | ベースライン+3%以内 |
| 補助 | 予測相関（vs LGBM） | < 0.98 | アンサンブル効果の見込み |
| 補助 | OOF MSR | > 0（監視のみ） | トレード観点での健全性確認 |
| 参考 | 実行時間 | < 10分 | 同等のデータ量で |

### 5.2 LB提出判断

- OOF RMSEが0.0122以下の場合、LB提出を検討
- OOF RMSEが0.0125を超える場合、ハイパラ調整を優先

---

## 6. 依存パッケージ

```toml
# pyproject.toml に追加が必要な場合
[project.dependencies]
xgboost = ">=2.0.0"
```

確認コマンド:
```bash
uv pip show xgboost || uv add xgboost
```

---

## 7. 実行方法

### 7.1 学習実行

```bash
# デフォルト設定で実行
uv run python src/models/xgboost/train_xgb.py

# カスタム設定で実行
uv run python src/models/xgboost/train_xgb.py \
    --max-depth 8 \
    --learning-rate 0.03 \
    --n-estimators 800
```

### 7.2 テスト実行

```bash
uv run pytest tests/models/test_xgboost.py -v
```

### 7.3 品質チェック

```bash
uv run ruff check src/models/xgboost/
uv run pyright src/models/xgboost/
```

---

## 8. 参考リンク

- [LGBM実装](../../src/models/lgbm/train_lgbm.py): 参考実装
- [Feature Selection README](../feature_selection/README.md): 特徴量選定の経緯
- [Model Selection README](README.md): モデル選定戦略の全体像
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 9. 注意事項（実装から得た教訓）

### 9.1 Early Stopping と eval_set の前処理

1. **XGBoost 2.0+ API変更**: `early_stopping_rounds`はfit()ではなく**コンストラクタで指定**する必要がある
   ```python
   # ✅ 正しい（XGBoost 2.0+）
   model = XGBRegressor(early_stopping_rounds=50, ...)
   model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
   
   # ❌ エラー（XGBoost 2.0+）
   model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=...)
   ```

2. **eval_set の前処理**: CVループでeval_setを使う場合、**パイプライン経由ではなく手動でimputation**を適用する必要がある
   - パイプラインのfitではeval_setに前処理が適用されない
   - 解決策: 各imputerをclone()してfit_transform/transformを手動適用

### 9.2 特徴量名のサニタイズ

XGBoostは特徴量名に`[`, `]`, `<`, `>`を含むと警告が出る。`sanitize_feature_names()`でアンダースコアに置換。

### 9.3 テスト予測時のfeatureフィルタリング

テストデータには学習時に存在しないカラム（`is_scored`, `lagged_*`等）が含まれる場合がある。
**学習時のfeature_colsのみを抽出**してから予測を実行：
```python
test_features = test_df[feature_cols].copy()
test_pred = final_pipeline.predict(test_features)
```

### 9.4 submission.csv のシグナル変換

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

### 9.5 is_scored フィルタリング

submission.csvには`is_scored==True`の行のみを含める（競技要件）。

### 9.6 最終モデル学習時のearly_stopping無効化

全データで再学習する際は検証セットがないため、`early_stopping_rounds`を**削除**してパイプラインを再構築：
```python
model_kwargs_final = model_kwargs.copy()
model_kwargs_final.pop("early_stopping_rounds", None)
final_pipeline = build_xgb_pipeline(..., model_kwargs=model_kwargs_final)
```

### 9.7 その他

- **メモリ使用量**: `tree_method="hist"`を使用することで、GPUなしでも高速化可能
- **再現性**: `random_state`を固定しても、マルチスレッド環境では完全な再現性が保証されない場合がある

---

## 10. 実装完了レポート

### 10.1 実装概要

2025-12-12に完了したXGBoostモデル実装は、LGBM実装(`src/models/lgbm/train_lgbm.py`)をベースに以下の主要機能を実装：

1. **コマンドライン引数パース**: 
   - XGBoost固有のハイパーパラメータをサポート
   - LGBM実装と同じCV設定オプションを提供

2. **特徴量名のサニタイズ**: 
   - XGBoostが警告を出す特殊文字(`[`, `]`, `<`, `>`)を自動的にアンダースコアに置換
   - `sanitize_feature_names()`関数で実装

3. **Early Stopping対応**:
   - sklearn Pipelineを通じて`model__eval_set`パラメータで検証データを渡す
   - `early_stopping_rounds=50`をデフォルトで設定

4. **既存モジュールの再利用**:
   - `src.feature_generation.su5.train_su5`から特徴量生成パイプライン
   - `src.models.common.feature_loader`から特徴量除外ロジック
   - `src.models.common.cv_utils`からCV評価メトリクス計算

### 10.2 実装の差異（vs LGBM）

| 項目 | LGBM | XGBoost | 理由 |
|------|------|---------|------|
| モデルクラス | `LGBMRegressor` | `XGBRegressor` | - |
| 特徴量名処理 | そのまま使用 | サニタイズ実装 | XGBoostの警告回避 |
| Early Stopping | `callbacks`使用 | `early_stopping_rounds`パラメータ | XGBoost APIの違い |
| デフォルト`max_depth` | -1（制限なし） | 6 | 過学習抑制 |

### 10.3 テスト結果

すべてのユニットテストが合格：
```bash
$ pytest tests/models/test_xgboost.py -v
13 passed, 1 skipped in 0.75s
```

品質チェック結果：
- **ruff check**: All checks passed
- **ruff format**: Formatted successfully
- **pyright**: 0 errors, 0 warnings

### 10.4 次のステップ

1. **実データでの学習実行**: 
   ```bash
   uv run python -m src.models.xgboost.train_xgb
   ```

2. **評価指標の確認**:
   - OOF RMSE ≤ 0.0125 の確認
   - LGBM予測との相関係数 < 0.98 の確認

3. **LB提出判断**:
   - OOF RMSEがベースラインと同等以上であれば提出検討

4. **ハイパーパラメータ調整**:
   - 必要に応じてOptunaでの最適化を検討
