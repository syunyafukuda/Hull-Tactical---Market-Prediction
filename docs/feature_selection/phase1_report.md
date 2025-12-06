# Phase 1: フィルタベースの雑草抜き 実行結果レポート

## 概要

Tier0 ベースライン（577列）に対して、統計的フィルタを適用し、不要な特徴量を除去した結果をまとめる。

---

## 実行環境

- **実行日**: [実行後に記入]
- **ブランチ**: feat/fs-phase1-filter
- **ベースライン**: Tier0 (commit: 5d177133bf0dbd95213cfee9621fe2aa90f38d71)

---

## フィルタ設定

### 閾値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| variance_threshold | 1e-10 | 分散がこの値未満の特徴量を除外 |
| missing_threshold | 0.99 | 欠損率がこの値超の特徴量を除外 |
| correlation_threshold | 0.999 | 相関がこの値超のペアから importance が低い方を除外 |

### 実行コマンド

```bash
# フィルタ候補の特定
python src/feature_selection/filter_trivial.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-path results/feature_selection/phase1_filter_candidates.json \
  --importance-path results/feature_selection/tier0_importance_summary.csv \
  --variance-threshold 1e-10 \
  --missing-threshold 0.99 \
  --correlation-threshold 0.999

# フィルタ適用後の評価
python src/feature_selection/evaluate_baseline.py \
  --config-path configs/tier0_snapshot/feature_generation.yaml \
  --preprocess-config configs/tier0_snapshot/preprocess.yaml \
  --data-dir data/raw \
  --out-dir results/feature_selection \
  --exclude-features results/feature_selection/phase1_filter_candidates.json
```

---

## 削除候補の内訳

### サマリー

| カテゴリ | 件数 |
|---------|------|
| 低分散 (low_variance) | [実行後に記入] |
| 高欠損率 (high_missing) | [実行後に記入] |
| 高相関 (high_correlation) | [実行後に記入] |
| **合計（重複除く）** | [実行後に記入] |

### 削除候補例

**低分散**:
```
[実行後に記入]
```

**高欠損率**:
```
[実行後に記入]
```

**高相関**:
```
[実行後に記入]
```

---

## 評価結果

### 比較表

| 指標 | Tier0（全特徴） | Tier1（フィルタ後） | 差分 | 差分率 |
|------|----------------|-------------------|------|--------|
| 特徴量数 | 577 | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| OOF RMSE | 0.012134 | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| OOF MSR | 0.019929 | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| OOF MSR Down | 0.030318 | [実行後に記入] | [実行後に記入] | [実行後に記入] |
| Coverage | 83.31% | [実行後に記入] | [実行後に記入] | [実行後に記入] |

### Fold 別の結果

#### Tier0

| Fold | Train Size | Val Size | RMSE | MSR | MSR Down |
|------|-----------|----------|------|-----|----------|
| [実行後に記入] | | | | | |

#### Tier1 (フィルタ後)

| Fold | Train Size | Val Size | RMSE | MSR | MSR Down |
|------|-----------|----------|------|-----|----------|
| [実行後に記入] | | | | | |

---

## 判定

### 採用基準の確認

- [ ] RMSE 同等以上（差分 < 0.0001）
- [ ] RMSE 微増（差分 < 0.0002）かつ列数大幅減
- [ ] MSR が維持または向上

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

1. Tier1 を新しいベースラインとして凍結
2. Phase 2（相互情報量ベースの選定）へ進む

### 不採用の場合

1. 閾値を緩和して再実行を検討
2. 削除候補を個別にレビュー
3. 削除候補の一部のみを採用することも検討

---

## 成果物

| ファイル | 説明 |
|----------|------|
| `results/feature_selection/phase1_filter_candidates.json` | 削除候補リスト |
| `results/feature_selection/tier1_evaluation.json` | Tier1 評価結果 |
| `results/feature_selection/tier1_fold_logs.csv` | Tier1 Fold 別ログ |
| `results/feature_selection/tier1_importance.csv` | Tier1 特徴量重要度 |
| `results/feature_selection/tier1_importance_summary.csv` | Tier1 重要度サマリー |
| `docs/feature_selection/phase1_report.md` | 本レポート |

---

## 備考

### 学び

```
[実行後に記入]
- フィルタリングで得られた知見
- 予想外の結果や発見
```

### 改善点

```
[実行後に記入]
- プロセスの改善点
- 次回への提案
```
