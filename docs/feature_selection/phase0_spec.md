# Phase 0: Tier0 固定（ベースライン凍結）仕様書

## 概要

特徴量選定フェーズの前提として、現行ベストラインを「Tier0」として凍結する。
以降の「削る・圧縮する」判断が、Sharpe 向きの指標と整合するようにし、評価軸を統一して比較可能な状態を作る。

## 目的

1. **ベースライン固定**: 現行ベストライン（生＋SU1＋SU5＋前処理＋LGBM）を再現可能な状態で保存
2. **評価軸統一**: RMSE を Primary、MSR (Sharpe proxy) を Secondary とする CV 評価体制を確立
3. **Importance 出力基盤**: 後続フェーズで使用する fold 毎の feature importance 出力機能を整備

---

## 現行ベースライン情報

| 項目 | 値 |
|------|-----|
| ベストライン | SU1 + SU5 + Brushup |
| 特徴量数 | 577 列 |
| OOF RMSE | 0.012134 |
| LB Score | 0.681 |
| ソースブランチ | `dev` |
| 学習スクリプト | `src/feature_generation/su5/train_su5.py` |
| 推論スクリプト | `src/feature_generation/su5/predict_su5.py` |
| 設定ファイル | `configs/feature_generation.yaml`, `configs/preprocess.yaml` |

---

## タスク詳細

### T0-1: Tier0 ベースラインの凍結

#### 目的
現行ベストラインを再現可能な形で固定し、以降の比較基準とする。

#### 実装内容

1. **Config スナップショット**
   - 保存先: `configs/tier0_snapshot/`
   - 対象ファイル:
     - `configs/feature_generation.yaml` → `configs/tier0_snapshot/feature_generation.yaml`
     - `configs/preprocess.yaml` → `configs/tier0_snapshot/preprocess.yaml`

2. **アーティファクト整理**
   - 保存先: `artifacts/tier0/`
   - 必須ファイル:
     - `feature_list.json`: パイプライン入力列、SU1/SU5 生成列、モデル入力列
     - `model_meta.json`: モデルパラメータ、OOF 指標、評価設定
     - `inference_bundle.pkl`: 学習済みパイプライン（Kaggle NB のインプットとして使用）

4. **特徴量リスト形式**
   ```json
   {
     "version": "tier0-v1",
     "created_at": "2025-12-06T00:00:00Z",
     "pipeline_input_columns": ["M1", "M2", ...],
     "su1_generated_columns": ["m/M1", "m/M2", ...],
     "su5_generated_columns": ["co_miss_now/M1__M2", ...],
     "total_feature_count": 577,
     "source_commit": "<commit_hash>",
     "source_branch": "dev"
   }
   ```

#### 成果物
- `configs/tier0_snapshot/feature_generation.yaml`
- `configs/tier0_snapshot/preprocess.yaml`
- `artifacts/tier0/feature_list.json`
- `artifacts/tier0/model_meta.json`
- `artifacts/tier0/inference_bundle.pkl`

---

### T0-2: 評価軸の統一

#### 目的
以降のフェーズで一貫した評価を行うため、評価軸と評価関数を明確化する。

#### 評価指標

| 優先度 | 指標 | 説明 | 使用場面 |
|--------|------|------|----------|
| Primary | RMSE | Root Mean Squared Error | 予測精度の直接評価、モデル・特徴量セットの採用判断は基本こちらに従う |
| Secondary | MSR | Mean-Sharpe-Ratio (Sharpe proxy) | リターン予測の実用性評価、補助指標として参照 |
| 補助 | MSR_down | 下方リスク考慮 MSR | リスク感度分析 |

#### 評価関数

既存の `scripts/utils_msr.py` を使用:
- `evaluate_msr_proxy(y_pred, y_true, params, eps, lam)` → dict
- `grid_search_msr(y_pred, y_true, mult_grid, lo_grid, hi_grid, ...)` → best_params, grid_all

#### CV 設定

| 項目 | 値 |
|------|-----|
| 分割方式 | TimeSeriesSplit |
| fold 数 | 5 |
| gap | 0（デフォルト）|
| 評価対象 | OOF 予測値 |

> **Note**: gap は Phase 0 では 0 とするが、将来のリーク検討（Purged/Embargoed CV 等）に応じて変更可能とする。

#### 成果物
- `artifacts/tier0/model_meta.json` 内に評価設定を記録
- 評価軸の統一ルールを本仕様書に明記（完了）

---

### T0-3: ベースライン評価スクリプト作成

#### 目的
Tier0 ベースラインの再評価と、fold 毎の feature importance 出力を行うスクリプトを作成する。

#### ファイル
- `src/feature_selection/evaluate_baseline.py`

#### 機能要件

1. **ベースライン再評価**
   - 入力: train.csv, 設定ファイル
   - 出力: OOF RMSE, OOF MSR, fold 毎の指標
   - 形式: コンソール出力 + JSON 保存

2. **Feature Importance 出力**
   - LightGBM の gain/split importance を fold 毎に算出
   - 出力形式: CSV（列: feature_name, importance_gain, importance_split, fold）
   - 保存先: `results/feature_selection/tier0_importance.csv`

3. **集計統計**
   - 各特徴量について:
     - 平均重要度（gain/split）
     - 標準偏差（fold 間のばらつき）
     - 最小・最大値
   - 保存先: `results/feature_selection/tier0_importance_summary.csv`

#### インターフェース

```bash
# 基本実行
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection

# オプション
  --n-splits 5          # fold 数
  --random-state 42     # 乱数シード
  --verbosity -1        # LightGBM verbosity
```

#### 出力ファイル

| ファイル | 内容 |
|----------|------|
| `tier0_evaluation.json` | OOF 指標サマリ |
| `tier0_importance.csv` | fold 毎の importance 詳細 |
| `tier0_importance_summary.csv` | importance 集計統計 |
| `tier0_fold_logs.csv` | fold 毎の RMSE/MSR |

#### コード構造

```python
# src/feature_selection/evaluate_baseline.py

def parse_args() -> argparse.Namespace:
    """CLI 引数パース"""
    ...

def load_tier0_pipeline(config_path: Path, preprocess_config: Path) -> Pipeline:
    """Tier0 設定からパイプラインを構築"""
    ...

def compute_fold_importance(
    model: LGBMRegressor,
    feature_names: List[str],
    fold_idx: int
) -> pd.DataFrame:
    """fold 毎の importance を DataFrame で返す"""
    ...

def aggregate_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """fold 毎の importance を集計"""
    ...

def evaluate_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_logs: List[Dict]
) -> Dict[str, Any]:
    """OOF 評価指標を算出"""
    ...

def main(argv: Sequence[str] | None = None) -> int:
    """メインエントリーポイント"""
    ...
```

#### 成果物
- `src/feature_selection/__init__.py`
- `src/feature_selection/evaluate_baseline.py`
- `results/feature_selection/tier0_evaluation.json`
- `results/feature_selection/tier0_importance.csv`
- `results/feature_selection/tier0_importance_summary.csv`

---

## ディレクトリ構造（Phase 0 完了時）

```
configs/
├── feature_generation.yaml
├── preprocess.yaml
└── tier0_snapshot/
    ├── feature_generation.yaml
    └── preprocess.yaml

artifacts/
└── tier0/
    ├── feature_list.json
    └── model_meta.json

src/
└── feature_selection/
    ├── __init__.py
    └── evaluate_baseline.py

results/
└── feature_selection/
    ├── tier0_evaluation.json
    ├── tier0_importance.csv
    ├── tier0_importance_summary.csv
    └── tier0_fold_logs.csv
```

---

## 完了条件

- [x] `configs/tier0_snapshot/` に設定ファイルがコピーされている
- [x] `artifacts/tier0/feature_list.json` が正しい形式で出力されている
- [ ] `artifacts/tier0/inference_bundle.pkl` が生成されている（データ必要）
- [x] `src/feature_selection/evaluate_baseline.py` が動作する
- [ ] `results/feature_selection/tier0_importance.csv` が出力されている（データ必要）
- [ ] OOF RMSE が 0.012134 前後で再現されている（データ必要）

> **Note**: 完全な評価とテストには `data/raw/train.csv` と `data/raw/test.csv` が必要です。
> データファイルは Git には含まれていないため、Kaggle からダウンロードするか、
> プロジェクトの data ディレクトリに配置してください。

---

## 依存関係

### 入力
- `data/raw/train.csv`
- `configs/feature_generation.yaml`
- `configs/preprocess.yaml`
- 既存の `src/feature_generation/su5/train_su5.py` のロジック

### 出力
- Phase 1 以降で使用する importance データ
- 比較基準となる Tier0 評価指標

---

## 備考

- `inference_bundle.pkl` は Kaggle NB のインプットとして必須。サイズが大きいため Git には含めず、Kaggle Dataset として管理
- 評価スクリプトは既存の `train_su5.py` のロジックを再利用し、重複を最小限にする
- 後続フェーズで importance データを頻繁に参照するため、CSV 形式で保存する
