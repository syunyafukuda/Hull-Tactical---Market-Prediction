# Phase 2: モデルベース重要度による特徴量選定 - 実行結果レポート

## 概要

Phase 1 で統計フィルタを適用した Tier1 特徴セット（160列）を入力とし、LGBM の gain importance と Permutation Importance を用いて低重要度特徴を特定・削除した。

**目標**: 160列 → 100〜120列への削減（RMSE 維持）

---

## 実行環境

- **実行日**: [実行後に記入]
- **ブランチ**: copilot/feature-selection-phase-2
- **ベースライン**: Tier1 (160 features)

---

## Phase 2-1: LGBM Importance Analysis

### 実行コマンド

```bash
python src/feature_selection/compute_importance.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier1/excluded.json \
  --out-dir results/feature_selection \
  --n-splits 5
```

### 出力ファイル

- `results/feature_selection/tier1/importance.csv` - fold 毎の importance
- `results/feature_selection/tier1/importance_summary.csv` - 集計統計

### 重要度分布の分析

[実行後に記入]

**主要な観察点**:
- [ ] mean_gain の分布（ヒストグラム）
- [ ] mean_gain vs std_gain の散布図
- [ ] 上位・下位特徴の名前パターン

### 削除候補の抽出

**抽出基準**:
- `mean_gain < quantile(0.25)` (下位 25%)
- `std_gain < median(std_gain)` (fold 間で安定して低い)

**結果**:
- 候補数: [実行後に記入] 列
- 候補率: [実行後に記入] %
- 出力: `results/feature_selection/phase2/importance_candidates.json`

**候補の内訳**:
- SU1 特徴: [実行後に記入] 列
- SU5 特徴: [実行後に記入] 列
- 生特徴: [実行後に記入] 列

---

## Phase 2-2: Permutation Importance

### 実行コマンド

```bash
python src/feature_selection/permutation_importance.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier1/excluded.json \
  --candidates results/feature_selection/phase2/importance_candidates.json \
  --out-path results/feature_selection/phase2/permutation_results.csv \
  --n-permutations 5 \
  --random-seed 42 \
  --n-splits 5
```

### 出力ファイル

- `results/feature_selection/phase2/permutation_results.csv`

### Permutation 結果の分析

[実行後に記入]

**ΔRMSE 分布**:
- 最小値: [実行後に記入]
- 25パーセンタイル: [実行後に記入]
- 中央値: [実行後に記入]
- 75パーセンタイル: [実行後に記入]
- 最大値: [実行後に記入]

**最終閾値の決定**:
- 採用閾値: `|mean_delta_rmse| < [実行後に記入]` AND `std_delta_rmse < [実行後に記入]`
- 理由: [実行後に記入]

**削除確定特徴**:
- 削除数: [実行後に記入] 列
- 削除率: [候補数に対する割合]

### 削除特徴一覧

[実行後に記入]

| Feature Name | mean_delta_rmse | std_delta_rmse | Note |
|--------------|-----------------|----------------|------|
| ... | ... | ... | ... |

---

## Tier2 特徴セット生成

### 除外リスト作成

**入力**:
- `configs/feature_selection/tier1/excluded.json` (Phase 1 除外リスト)
- `results/feature_selection/phase2/permutation_results.csv` (Phase 2-2 削除確定列)

**出力**:
- `configs/feature_selection/tier2/excluded.json`

**内容**:
- Tier1 除外: [実行後に記入] 列
- Phase 2 削除: [実行後に記入] 列
- **合計除外**: [実行後に記入] 列
- **残存特徴**: 577 - [合計除外] = [実行後に記入] 列

---

## Tier2 評価

### 実行コマンド

```bash
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier2/excluded.json \
  --out-dir results/feature_selection \
  --artifacts-dir artifacts/tier2 \
  --n-splits 5
```

### 出力ファイル

- `results/feature_selection/tier2/evaluation.json`
- `results/feature_selection/tier2_fold_logs.csv`
- `results/feature_selection/tier2_importance.csv`
- `results/feature_selection/tier2_importance_summary.csv`

### 比較表

| 指標 | Tier1 | Tier2 | 差分 | 差分率 | 判定 |
|------|-------|-------|------|--------|------|
| **特徴量数** | 160 | [実行後に記入] | [実行後に記入] | [実行後に記入]% | - |
| **OOF RMSE** | 0.012168 | [実行後に記入] | [実行後に記入] | [実行後に記入]% | [同等以上?] |
| **OOF MSR** | 0.019201 | [実行後に記入] | [実行後に記入] | [実行後に記入]% | [同等以上?] |
| OOF MSR Down | 0.029607 | [実行後に記入] | [実行後に記入] | [実行後に記入]% | - |
| Coverage | 83.31% | [実行後に記入] | [実行後に記入]% | - | - |

### Fold 別の結果

#### Tier2

| Fold | Train Size | Val Size | RMSE | MSR | MSR Down |
|------|-----------|----------|------|-----|----------|
| 1 | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| 2 | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| 3 | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| 4 | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| 5 | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] | [実行後に記入] |

---

## 判定

### 採用基準の確認

- [ ] RMSE 同等以上（差分 < +0.0001）
- [ ] 列数削減（目標: 100〜120列）
- [ ] MSR が維持または向上（参考指標）

### 判定結果

**[実行後に記入]**

```
採用 / 不採用 / 条件付き採用
```

### 理由

```
[実行後に記入]
- RMSE: [分析]
- MSR: [分析]
- 特徴量削減効果: [分析]
```

---

## 次のステップ

### 採用の場合

1. Tier2 を新しいベースラインとして採用
2. Phase 3（相関クラスタリング）へ進むか検討

### 不採用の場合

1. 閾値を調整して再実行
2. 削除候補を個別にレビュー
3. 削除候補の一部のみを採用

---

## 成果物

| ファイル | 説明 |
|----------|------|
| `results/feature_selection/tier1/importance.csv` | fold 毎の importance |
| `results/feature_selection/tier1/importance_summary.csv` | importance 集計 |
| `results/feature_selection/phase2/importance_candidates.json` | 削除候補リスト |
| `results/feature_selection/phase2/permutation_results.csv` | Permutation 結果 |
| `configs/feature_selection/tier2/excluded.json` | Tier2 除外リスト |
| `results/feature_selection/tier2/evaluation.json` | Tier2 評価結果 |
| `results/feature_selection/tier2_fold_logs.csv` | Tier2 Fold 別ログ |
| `docs/feature_selection/phase2_report.md` | 本レポート |
| `notebooks/feature_selection/importance_analysis.ipynb` | 重要度分析ノートブック |

---

## 備考

### 学び

[実行後に記入]

### 改善点

[実行後に記入]

### 計算時間

- compute_importance.py: [実行後に記入]
- permutation_importance.py: [実行後に記入]
- evaluate_baseline.py (Tier2): [実行後に記入]

### 注意事項

- Permutation Importance は相関の強い特徴がある場合に歪む可能性がある
- Phase 1 で高相関列（|ρ| > 0.999）は既に除去済み
- 閾値は事前の提案値であり、実際の分布を見て最終決定した

---

## 参考

- 仕様書: `docs/feature_selection/phase2_spec.md`
- Phase 1 結果: `docs/feature_selection/phase1_report.md`
- 全体計画: `docs/feature_selection/README.md`
