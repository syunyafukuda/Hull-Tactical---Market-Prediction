# Phase 1: フィルタベースの雑草抜き 仕様書

## 概要

Tier0 ベースライン（577列）に対して、統計的に明らかに不要な列を機械的に除去する。
Sharpe コンペ固有の難しさとは独立な領域であり、リスクの低い最適化ステップ。

## 目的

1. **分散ほぼゼロの列除去**: 予測に寄与しない定数に近い列を削除
2. **欠損率ほぼ100%の列除去**: Imputer で常に同じ値になる列を削除
3. **ほぼ線形従属な列除去**: 他列と極めて高い相関を持つ冗長な列を削除

---

## パイプライン内での選定位置

```
生データ (94列)
    ↓
[SU1 特徴量生成] → 366列追加
    ↓
[SU5 特徴量生成] → 108列追加
    ↓
合計 568列
    ↓
[★ Phase 1 除外処理] ← ここで削除候補を drop
    ↓
[前処理 (ColumnTransformer)]
    ↓
[LightGBM]
```

### 選定位置の根拠

- **SU5 適用後・前処理前** で除外リストに基づき列を drop
- 分散・欠損率・相関の判定は **前処理適用後の値** で行う（Imputer/Scaler 適用後が正確）
- 判定と除外のタイミングが異なる点に注意:
  1. **判定**: 前処理後の特徴量を分析して削除候補を特定
  2. **除外**: SU5 適用後・前処理前で列を drop

---

## Tier0 ベースライン情報

| 項目 | 値 |
|------|-----|
| 特徴量数 | 577 列（94 input + 366 SU1 + 108 SU5 + 9 other） |
| OOF RMSE | 0.012134 |
| OOF MSR | 0.019929 |
| 特徴量リスト | `artifacts/tier0/feature_list.json` |

---

## 除去基準

| カテゴリ | 判定基準 | 理由 |
|---------|---------|------|
| 分散ほぼゼロ | `var < 1e-10` | 情報量なし、モデルに寄与しない |
| 欠損率ほぼ100% | `missing_rate > 0.99` | Imputer 後に定数化 |
| ほぼ線形従属 | `\|corr\| > 0.999` | 冗長、片方あれば十分 |

### 補足

- **分散ゼロ判定**: 前処理後（Imputer適用後）の値で判定
- **相関判定**: ペアのうち importance が低い方を削除候補とする
- **複数条件該当**: いずれか1つでも該当すれば削除候補

---

## タスク詳細

### T1-1: フィルタスクリプト作成

#### ファイル
- `src/feature_selection/filter_trivial.py`

#### 機能要件

1. **統計量算出**
   - 各列の分散、欠損率、相関行列を計算
   - 前処理後の特徴量に対して適用

2. **削除候補判定**
   - 上記3基準に基づき削除候補をマーク
   - 各列について該当理由を記録

3. **出力形式**
   ```json
   {
     "version": "phase1-v1",
     "created_at": "2025-12-06T00:00:00Z",
     "thresholds": {
       "variance_min": 1e-10,
       "missing_rate_max": 0.99,
       "correlation_max": 0.999
     },
     "candidates": [
       {
         "feature_name": "M1",
         "reason": "low_variance",
         "value": 1.2e-11
       },
       {
         "feature_name": "co_miss_now/M1__M2",
         "reason": "high_correlation",
         "correlated_with": "co_miss_now/M1__M3",
         "correlation": 0.9995
       }
     ],
     "summary": {
       "total_features": 577,
       "low_variance_count": 5,
       "high_missing_count": 2,
       "high_correlation_count": 10,
       "total_candidates": 15
     }
   }
   ```

#### CLI インターフェース

```bash
python src/feature_selection/filter_trivial.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-path results/feature_selection/phase1_filter_candidates.json \
  --variance-threshold 1e-10 \
  --missing-threshold 0.99 \
  --correlation-threshold 0.999
```

#### コード構造

```python
# src/feature_selection/filter_trivial.py

def parse_args() -> argparse.Namespace:
    """CLI 引数パース"""

def compute_statistics(X: pd.DataFrame) -> Dict[str, Any]:
    """分散、欠損率、相関行列を算出"""

def find_low_variance_features(
    X: pd.DataFrame,
    threshold: float = 1e-10
) -> List[Dict[str, Any]]:
    """分散が閾値未満の列を抽出"""

def find_high_missing_features(
    X: pd.DataFrame,
    threshold: float = 0.99
) -> List[Dict[str, Any]]:
    """欠損率が閾値超の列を抽出"""

def find_high_correlation_features(
    X: pd.DataFrame,
    threshold: float = 0.999,
    importance_df: pd.DataFrame | None = None
) -> List[Dict[str, Any]]:
    """相関が閾値超のペアから低 importance 側を抽出"""

def main(argv: Sequence[str] | None = None) -> int:
    """メインエントリーポイント"""
```

#### 成果物
- `src/feature_selection/filter_trivial.py`

---

### T1-2: Tier0 に対してフィルタ適用

#### 目的
Tier0 特徴量に対してフィルタを実行し、削除候補を特定

#### 作業内容

1. `filter_trivial.py` を実行
2. 削除候補リストを `results/feature_selection/phase1_filter_candidates.json` に出力
3. 削除候補の内訳を確認（カテゴリ別の件数）

#### 入力
- `artifacts/tier0/feature_list.json`
- `results/feature_selection/tier0_importance_summary.csv`（相関判定時の importance 参照用）

#### 出力
- `results/feature_selection/phase1_filter_candidates.json`

#### 成果物
- 削除候補リスト（JSON）

---

### T1-3: フィルタ後の評価

#### 目的
削除候補を除外した場合の Sharpe/RMSE を評価し、採用判断を行う

#### 作業内容

1. **評価スクリプト実行**
   - Tier0 特徴量から削除候補を除外
   - 同じ CV 設定で OOF RMSE/MSR を算出

2. **比較**
   | 指標 | Tier0（全特徴） | Tier1（フィルタ後） | 差分 |
   |------|----------------|-------------------|------|
   | 特徴量数 | 577 | ? | ? |
   | OOF RMSE | 0.012134 | ? | ? |
   | OOF MSR | 0.019929 | ? | ? |

3. **判断基準**
   - RMSE 同等以上 → 削除採用
   - RMSE 微増（+0.0001以内）+ 列数大幅減 → 採用検討
   - RMSE 明確悪化（+0.0002以上）→ 不採用 or 削減幅縮小

#### CLI インターフェース

```bash
# 評価スクリプトに除外リストを渡すオプションを追加
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection \
  --exclude-features results/feature_selection/phase1_filter_candidates.json
```

#### 成果物
- `results/feature_selection/tier1_evaluation.json`
- `docs/feature_selection/phase1_report.md`（結果レポート）

---

## ディレクトリ構造（完了時）

```
src/
└── feature_selection/
    ├── __init__.py
    ├── evaluate_baseline.py      # Phase 0 で作成済み
    └── filter_trivial.py         # T1-1 で新規作成

results/
└── feature_selection/
    ├── tier0_evaluation.json         # Phase 0 で作成済み
    ├── tier0_importance_summary.csv  # Phase 0 で作成済み
    ├── phase1_filter_candidates.json # T1-2 で出力
    └── tier1_evaluation.json         # T1-3 で出力

docs/
└── feature_selection/
    ├── README.md                 # 計画
    ├── phase0_spec.md            # Phase 0 仕様書
    ├── phase1_spec.md            # 本ファイル
    └── phase1_report.md          # T1-3 で作成
```

---

## 受け入れ条件

- [ ] `src/feature_selection/filter_trivial.py` が動作する
- [ ] 3種類のフィルタ（分散/欠損率/相関）が正しく動作する
- [ ] `phase1_filter_candidates.json` が正しい形式で出力される
- [ ] Tier0 vs Tier1 の比較評価が実行できる
- [ ] 結果が `phase1_report.md` に記録される

---

## 想定工数

- T1-1: 2-3h（スクリプト実装）
- T1-2: 0.5h（実行・確認）
- T1-3: 1-2h（評価・レポート作成）
- **合計: 4-6h**

---

## 注意事項

1. **相関判定の順序**
   - 相関ペアのうち、importance が低い方を削除候補とする
   - `tier0_importance_summary.csv` の `mean_gain` を参照

2. **前処理後の値で判定**
   - 分散・欠損率は Imputer 適用後の値で判定
   - 生データではなく、モデルに入力される形式で評価

3. **安全側の閾値**
   - 初回は保守的な閾値（上記デフォルト値）で実行
   - 必要に応じて閾値を緩めて再実行可能

---

## 関連ドキュメント

- 全体計画: `docs/feature_selection/README.md`
- Phase 0 仕様書: `docs/feature_selection/phase0_spec.md`
- Tier0 アーティファクト: `artifacts/tier0/`
