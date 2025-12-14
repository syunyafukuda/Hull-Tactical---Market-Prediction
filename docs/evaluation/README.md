# Evaluation Phase

最終更新: 2025-12-14

## 概要

本フェーズでは、現行の RMSE/MSR ベースの評価から **公式 Hull Competition Sharpe + Walk-Forward CV** に移行し、
モデル選定・ハイパラチューニングの指標を LB に近づける。

## 現状ベースライン

| 項目 | 値 |
|------|-----|
| 採用特徴セット | FS_compact（116列） |
| ベースモデル | LightGBM |
| OOF RMSE | 0.012164 |
| LB Score | 0.681 |
| 評価手法 | TimeSeriesSplit + RMSE/MSR |

## 問題意識

1. **RMSE と LB の相関が弱い**: XGBoost/CatBoost は OOF RMSE で LGBM を上回るが、LB では大幅劣化
2. **LB は Sharpe 系指標**: 公式評価は修正 Sharpe 比で、ペナルティ項を含む
3. **期間安定性が見えない**: 単一 OOF では特定期間の過学習を検出できない

## 解決策

| 施策 | 目的 |
|------|------|
| **公式 Sharpe メトリック移植** | LB と同一ロジックでローカル評価 |
| **Walk-Forward CV** | 複数期間で Sharpe の安定性を検証 |
| **ペナルティ内訳の可視化** | vol_penalty / ret_penalty の診断 |

---

## ドキュメント構成

| ファイル | 内容 |
|----------|------|
| [hull_sharpe_walkforward_plan.md](hull_sharpe_walkforward_plan.md) | 概要計画・公式ロジック転記 |
| [hull_sharpe_spec.md](hull_sharpe_spec.md) | **詳細仕様書** |
| [optimized_settings.md](optimized_settings.md) | **最適化設定（確定版）** ← 2025-12-14 追加 |

---

## フェーズ構成

```
Evaluation Phase
├── Step 1: 公式 Sharpe メトリック移植
│   ├── src/metrics/hull_sharpe.py
│   └── tests/metrics/test_hull_sharpe.py
├── Step 2: Walk-Forward Splitter 実装
│   ├── src/models/common/walk_forward.py
│   └── tests/models/test_walk_forward.py
├── Step 3: CLI 統合 & ロギング
│   ├── src/models/lgbm/train_lgbm.py 改修
│   └── artifacts 構造追加
├── Step 4: サニティチェック & LB 比較
│   └── scripts/debug/run_sharpe_sanity.py
└── Step 5: ドキュメント更新
    └── README.md / docs/models/README.md
```

---

## 成功基準

| 指標 | 基準 |
|------|------|
| メトリック再現 | 公式 NB と同一ロジックで単体テスト Pass |
| Walk-Forward | 3+ fold で Sharpe を算出可能 |
| LGBM ベースライン | Walk-Forward Sharpe が正の値（サニティ） |
| LB との相対整合 | LGBM > XGBoost の順序が維持 |

---

## 参照

- [hull_sharpe_walkforward_plan.md](hull_sharpe_walkforward_plan.md): 概要計画
- [hull_sharpe_spec.md](hull_sharpe_spec.md): 詳細仕様
