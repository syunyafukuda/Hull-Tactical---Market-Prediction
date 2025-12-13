# CatBoost モデル実装仕様書

最終更新: 2025-12-13

## 実装ステータス

**Status**: ✅ **実装完了**

### 実装済み
- ✅ `src/models/catboost/train_catboost.py`: 学習スクリプト
- ✅ `configs/models/catboost.yaml`: YAML設定ファイル
- ✅ `src/models/catboost/predict_catboost.py`: 推論スクリプト
- ✅ Unit tests: `tests/models/test_catboost.py`

### 成果物
- ✅ `artifacts/models/catboost/inference_bundle.pkl`
- ✅ `artifacts/models/catboost/oof_predictions.csv`
- ✅ `artifacts/models/catboost/cv_fold_logs.csv`
- ✅ `artifacts/models/catboost/model_meta.json`
- ✅ `artifacts/models/catboost/feature_list.json`
- ✅ `artifacts/models/catboost/submission.csv`

**Note**: 出力仕様の詳細は [README.md](README.md#成果物出力仕様kaggle-nb用) を参照。

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
- **Native Categorical Support**: カテゴリ変数の自動エンコーディング

> **本コンペでの利用方針**: 今回の FS_compact（116列）はほぼ全て数値特徴量のため、
> CatBoost のカテゴリカル自動エンコード機能は**使用しない**。
> すべての特徴量を numeric 扱いとし、不要な複雑性を避ける。

### 1.3 前提条件

- **特徴セット**: FS_compact（116列）を固定（Feature Selection Phase での結論と整合）
- **CV設定**: TimeSeriesSplit, n_splits=5, gap=0（LGBMと同一）
- **評価指標**:
  - **主指標**: OOF RMSE（選定フェーズの最重要指標）
  - **補助指標**: 予測相関（vs LGBM）、OOF MSR（トレード観点での監視）

### 1.4 実験結果と分析

**OOF Performance:**
- OOF RMSE: 0.011095
- 予測分散: 0.000495（LGBMの約9%）
- vs LGBM相関: 0.35
- vs XGBoost相関: 0.27

**Kaggle LB:**
- **LB Score: 0.602** (2025-12-13提出)
- LGBM比: -0.020 (3.2%悪化)
- XGBoost比: -0.016 (2.6%悪化)

**Critical Issues Identified:**

1. **極端な予測分散の小ささ**
   - CatBoost予測の標準偏差は0.000495で、LGBM（0.005246）の約9%しかない
   - これは市場の実際の変動を捉えられていないことを示唆
   - ハイパーパラメータチューニング（depth, l2_leaf_reg, learning_rate調整）でも改善せず

2. **予測相関の低さ**
   - LGBMとの相関0.35、XGBoostとの相関0.27は、異なるパターンを学習している証拠
   - しかしLBスコアの悪化から、そのパターンは有用でないと判断

3. **過度な正則化の可能性**
   - CatBoostのSymmetric TreeとOrdered Boostingが、このデータセットに対して過度に保守的
   - 結果として予測が平坦化し、情報量が失われている

**アンサンブルへの影響:**
- ❌ **単独での採用は推奨しない**: LBスコアがLGBM/XGBoostより明確に劣る
- ⚠️ **アンサンブルでの価値は限定的**: 予測相関が低いため多様性は提供するが、LBスコアの悪化リスクが大きい
- 💡 **代替案**: LGBM/XGBoostの異なるハイパーパラメータセットでのアンサンブルを優先すべき

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

| 優先度 | 指標 | 条件 | 備考 |
|--------|------|------|------|
| **主指標** | OOF RMSE | ≤ 0.0125 | ベースライン+3%以内 |
| 補助 | 予測相関（vs LGBM） | < 0.98 | アンサンブル効果の見込み |
| 補助 | OOF MSR | > 0（監視のみ） | トレード観点での健全性確認 |
| 参考 | 実行時間 | < 15分 | CatBoostはやや遅い傾向 |

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

## 9. 注意事項（XGBoost実装から得た共通教訓を含む）

### 9.1 CatBoost固有の注意点

1. **Ordered Boosting**: デフォルトで有効。時系列データに有利だが、学習が遅くなる場合がある。`boosting_type='Plain'`で無効化可能。

2. **GPUサポート**: `task_type='GPU'`で有効化可能だが、環境依存のため初期実装ではCPUのみ。

3. **モデルサイズ**: CatBoostのモデルファイルは比較的大きくなる傾向がある。

4. **特徴量重要度**: `get_feature_importance()`で取得可能。type='PredictionValuesChange'がLGBMのgainに相当。

### 9.2 Early Stopping と eval_set の前処理（XGBoostと同様）

1. **eval_set の前処理**: CVループでeval_setを使う場合、**パイプライン経由ではなく手動でimputation**を適用する必要がある
   - パイプラインのfitではeval_setに前処理が適用されない
   - 解決策: 各imputerをclone()してfit_transform/transformを手動適用

2. **最終モデル学習時のearly_stopping無効化**: 全データで再学習する際は検証セットがないため、`early_stopping_rounds`を**削除**してモデルを再構築

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

### 9.6 勾配ブースティング共通の教訓（XGBoost実装より）

> **重要**: XGBoost実装時に発見された問題はCatBoostでも発生する可能性があります。

#### 9.6.1 予測分散の診断

**診断指標**: `pred.std() / actual.std()` (pred/actual ratio)

- **正常範囲**: 30-70%（LGBMは約50%）
- **異常値**: < 10% は過少学習（モデルがほぼ何も予測していない）

```python
# 診断コード
ratio = oof['prediction'].std() / oof['actual'].std()
print(f"pred/actual ratio: {ratio:.1%}")
if ratio < 0.1:
    print("WARNING: Model may be underfitting - check regularization params")
```

#### 9.6.2 正則化パラメータの調整

CatBoostは `l2_leaf_reg` (L2正則化) がデフォルトで3.0と強め。
このデータセットでは以下のように調整を検討：

```python
# 過少学習が疑われる場合
l2_leaf_reg: 1.0  # 3.0 → 1.0に緩和
depth: 8          # 6 → 8に増加
iterations: 1000  # 600 → 1000に増加
```

#### 9.6.3 バージョン互換性

CatBoostもXGBoostと同様、バージョン間でモデルフォーマットが異なる場合があります。
Kaggle提出時は同一バージョンのwheelを同梱することを推奨。

#### 9.6.4 ローカル推論スクリプトとKaggleノートブックの違い

実装時は**2種類の推論コード**を用意することを推奨：

| ファイル | 用途 | 特徴 |
|----------|------|------|
| `predict_catboost.py` | ローカルでの再推論 | 既存モジュールを `import` |
| `catboost.ipynb` | Kaggle提出 | 依存クラスをすべてインライン埋込 |

**推論ロジック自体は同一**にすること：
```python
# 両方で同じロジック
prediction = pipeline.predict(X_test)
signal = to_signal(prediction, postprocess_params)
```

**注意点**:
- Python/NumPyバージョン差で ~0.02% の予測差は許容範囲
- 重要なのは **OOF RMSEが一致する** こと
- ローカル推論スクリプトは学習なしで `submission.csv` を再生成できる（デバッグ用）

