# ExtraTrees モデル実装仕様書

最終更新: 2025-12-13

## 実装ステータス

**Status**: ❌ **非採用（LB 0.500 - ベースライン同等）**

### 実装済み
- ✅ `src/models/extratrees/train_extratrees.py`: 学習スクリプト
- ✅ `src/models/extratrees/predict_extratrees.py`: 推論スクリプト
- ✅ `configs/models/extratrees.yaml`: YAML設定ファイル
- ✅ Unit tests: `tests/models/test_extratrees.py` (10テスト ALL PASS)

### 成果物（参考保存）
- ✅ `artifacts/models/extratrees/inference_bundle.pkl`
- ✅ `artifacts/models/extratrees/oof_predictions.csv`
- ✅ `artifacts/models/extratrees/cv_fold_logs.csv`
- ✅ `artifacts/models/extratrees/model_meta.json`
- ✅ `artifacts/models/extratrees/feature_list.json`
- ✅ `artifacts/models/extratrees/feature_importances.csv` （特徴量重要度）
- ✅ `artifacts/models/extratrees/submission.csv`

### LB結果

| 指標 | 値 | 評価 |
|------|------|------|
| **OOF RMSE** | 0.011440 | -5.96% vs LGBM ✅ |
| **LB Score** | 0.500 | -26.6% vs LGBM ❌❌ |

### 結論

**バギング系ツリー（ExtraTrees）はこのマーケット予測問題に不適合**
- OOFでは良好だがLBでベースライン同等（何も予測しないのと同じ）
- 予測範囲が極端に狭い（0.9999〜1.0001）
- RandomForestも同様の失敗が予想される → 試行不要

**Note**: 出力仕様の詳細は [README.md](README.md#成果物出力仕様kaggle-nb用) を参照。

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

- **特徴セット**: FS_compact（116列）を固定（Feature Selection Phase での結論と整合）
- **CV設定**: TimeSeriesSplit, n_splits=5, gap=0（LGBMと同一）
- **評価指標**:
  - **主指標**: OOF RMSE（選定フェーズの最重要指標）
  - **補助指標**: 予測相関（vs LGBM）、OOF MSR（トレード観点での監視）
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

> **📌 RandomForestとの役割分担**
>
> ExtraTreesは「よりランダムでバリアンス高め」、RandomForestは「もう少し落ち着いたランダム性」という特徴があります。
> アンサンブル観点では、ExtraTreesは「変な揺れ方」をさせたいときに採用します。
> **まずExtraTreesを試し、RandomForestはOOF RMSEと相関を見て採否を決定**する方針です。

---

## 3. 実装詳細

### 3.1 ファイル構成

```
src/models/extratrees/
├── __init__.py           # モジュール初期化
├── train_extratrees.py   # メイン学習スクリプト
└── predict_extratrees.py # 推論スクリプト

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

---

## 5. 成功基準

### 5.1 定量基準（実績）

| 優先度 | 指標 | 目標 | 実績 | 評価 |
|--------|------|------|------|------|
| **主指標** | OOF RMSE | ≤ 0.0130 | 0.011440 | ✅ |
| 主指標 | LB Score | > 0.500 | **0.500** | ❌ |
| 補助 | 予測相関（vs LGBM） | < 0.92 | N/A | - |

### 5.2 失敗分析

ExtraTreesは **OOFでは良好だがLBでベースライン同等** という結果。

- 予測値が極端に1.0付近に集中（range: 0.9999〜1.0001）
- 実質的に「何も予測しない」のと同じ
- バギング系ツリーの特性（平均化によるバリアンス削減）が裏目に出た
- 金融データの微弱シグナルを検出できない

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

---

## 7. 参考リンク

- [scikit-learn ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [Extremely Randomized Trees 論文](https://link.springer.com/article/10.1007/s10994-006-6226-1)
- [LGBM実装](../models/lgbm/train_lgbm.py)
- [CV共通モジュール](../../src/models/common/cv_utils.py)
