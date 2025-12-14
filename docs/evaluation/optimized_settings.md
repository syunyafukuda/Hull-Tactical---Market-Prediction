# Hull Sharpe × Walk-Forward CV 最適化設定

最終更新: 2025-12-14  
ステータス: **LB提出完了 → 方針見直し中**

---

## 1. 概要

本ドキュメントは、Hull Competition Sharpe × Walk-Forward CV 評価基盤の最適化結果と、
Public LB 3.318 達成後の分析・方針転換を記載する。

### 1.1 背景

prediction → position マッピングにおける `sharpe_mult` パラメータを調整し、
**Vol Ratio を 0.8〜1.2 の許容範囲内**に収めることで、ボラティリティペナルティを回避する。

---

## 2. 最適化結果

### 2.1 最終採用設定

| パラメータ | 値 | 説明 |
|------------|-----|------|
| **sharpe_mult** | **20.0** | 予測値→ポジションのスケール係数 |
| sharpe_offset | 1.0 | オフセット（`position = pred * mult + offset`） |
| wf_train_window | 6000 | 学習ウィンドウ（約24年相当） |
| wf_val_window | 1000 | 検証ウィンドウ（約4年相当） |
| **wf_step** | **500** | ステップ幅（4 fold を生成） |
| wf_mode | expanding | 拡大ウィンドウ方式 |

### 2.2 チューニング経緯

デフォルトの `sharpe_mult=100` では Vol Ratio が 1.39 に達し、
大きなボラティリティペナルティ（〜19.2）が発生していた。

| sharpe_mult | Mean Sharpe | Vol Ratio 範囲 | Std Sharpe | 判定 |
|-------------|-------------|----------------|------------|------|
| 100 | -9.22 | 1.03 – 1.39 | 1.57 | ❌ ペナルティ大 |
| 50 | -3.01 | 1.01 – 1.30 | 0.29 | ❌ 一部超過 |
| 30 | -0.02 | 0.99 – 1.21 | 0.12 | ⚠️ 境界ギリギリ |
| **20** | **+0.11** | **0.99 – 1.13** | **0.08** | ✅ 最適 |
| 10 | +0.09 | 0.99 – 1.06 | 0.08 | ✅ 保守的 |

**採用理由**:
- `sharpe_mult=20` は Mean Sharpe が最大（+0.11）
- Std Sharpe が最小（0.08）で安定性が高い
- 全 4 fold で Vol Ratio が 0.8〜1.2 範囲内

### 2.3 4-Fold 詳細結果

| Fold | Train Range | Val Range | Hull Sharpe | Vol Ratio | Vol Penalty |
|------|-------------|-----------|-------------|-----------|-------------|
| 1 | 0 – 5999 | 6000 – 6999 | +0.22 | 0.99 | 0.0 |
| 2 | 0 – 6499 | 6500 – 7499 | +0.07 | 1.02 | 0.0 |
| 3 | 0 – 6999 | 7000 – 7999 | +0.14 | 1.07 | 0.0 |
| 4 | 0 – 7499 | 7500 – 8989 | -0.01 | 1.13 | 0.0 |

**統計サマリ**:
- Mean Sharpe: +0.11
- Min Sharpe: -0.01
- Max Sharpe: +0.22
- Std Sharpe: 0.08

---

## 3. 採用ベースライン（パイプライン）

| 項目 | 値 |
|------|-----|
| 特徴セット | FS_compact（116列） |
| 特徴生成 | SU1 + SU5 |
| 前処理 | M/E/I/P/S group imputers |
| モデル | LightGBM |
| ハイパーパラメータ | lr=0.05, n_estimators=600, num_leaves=63 |
| OOF RMSE | 0.01140 |

---

## 4. コマンドライン再現

```bash
python -m src.metrics.lgbm.train_lgbm \
  --cv-mode walk_forward \
  --wf-train-window 6000 \
  --wf-val-window 1000 \
  --wf-step 500 \
  --wf-mode expanding \
  --eval-sharpe \
  --sharpe-mult 20 \
  --sharpe-offset 1.0 \
  --feature-tier tier3 \
  --out-dir artifacts/models/lgbm-sharpe-wf-opt
```

---

## 5. 生成アーティファクト

| ファイル | 説明 |
|----------|------|
| `inference_bundle.pkl` | 学習済みパイプライン |
| `oof_predictions.csv` | OOF 予測値 |
| `cv_fold_logs.csv` | Fold 別 RMSE/MSR |
| `walk_forward_folds.csv` | Fold 別 Hull Sharpe 詳細 |
| `hull_sharpe_summary.json` | Hull Sharpe 統計サマリ |
| `model_meta.json` | メタデータ |
| `submission.csv` | Kaggle 提出用（別途生成） |

---

## 6. Public LB 結果と分析（2025-12-14）

### 6.1 提出結果

| 項目 | 値 |
|------|-----|
| **Public LB Score** | **3.318** |
| 従来ベスト（SU5+LGBM） | 0.681 |
| 改善率 | +387% |

### 6.2 分析結論

**Public LB 3.318 は「真の実力の証明」とは言えない**

#### 根拠

1. **Kaggle 公式の警告**
   > "As the public leaderboard is based on publicly available data, the ranks are not meaningful."

2. **コミュニティの分析（discussion/611071）**
   - Public LB ヒストグラムに3つの極大:
     - ~17.5: `train[-180:]` 最適化によるリーク解
     - ~10: 別種のリーク系
     - **~0.5 (0.469 ピーク): リークなしの "do-nothing" ベースライン**

3. **ローカル検証との乖離**

   | 指標 | ローカル WF | Public LB |
   |------|-------------|-----------|
   | Sharpe | +0.11 | +3.32 |
   | 乖離 | - | **30倍** |

4. **結論**
   - 3.318 は「Public テスト切片でたまたま当たった」可能性が高い
   - 180日程度の短期 Sharpe は極めてノイジー
   - **妥当な比較軸は「ローカル WF Sharpe が 0.469 (do-nothing) を安定的に上回るか」**

---

## 7. 方針転換

### 7.1 評価軸の変更

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| 主評価指標 | Public LB | **ローカル Walk-Forward Sharpe** |
| 比較対象 | 従来 LB (0.681) | **do-nothing baseline (0.469)** |
| 採用基準 | LB 改善 | **WF Mean Sharpe ≥ 0.5 かつ Min Sharpe > 0** |

### 7.2 次のステップ: クリッピング戦略

**prediction → position 変換の最適化**で、RMSE 改善なしに Sharpe を押し上げる。

詳細: [prediction_clipping.md](prediction_clipping.md)

| パラメータ | 説明 | 探索範囲 |
|------------|------|----------|
| alpha | 予測値のスケール | 0, 0.1, 0.25, 0.5 |
| beta | オフセット（市場中立=1.0） | 0.6, 0.8, 1.0, 1.2 |
| clip_min | 最小ポジション | 0.0, 0.2, 0.4 |
| clip_max | 最大ポジション | 1.6, 1.8, 2.0 |
| winsor_pct | 外れ値除去 | None, 0.01, 0.05 |

### 7.3 サニティチェック（必須）

1. **do-nothing baseline の再現**
   - `alpha=0, beta=0.806` で Hull Sharpe ≈ 0.469 を確認
   - 実装の健全性チェック

2. **Walk-Forward 複数設定**
   - 異なるウィンドウ設計・seed で Sharpe 分布を確認
   - Mean Sharpe が一貫して 0.469 を超えるかを検証

---

## 8. 次のステップ

1. ~~Kaggle 提出~~ ✅ 完了（LB 3.318）
2. ~~LB 相関検証~~ → **ローカル WF との乖離が大きく信頼性なし**
3. **クリッピング戦略の実装**: [prediction_clipping.md](prediction_clipping.md)
4. **do-nothing baseline (0.469) の再現確認**
5. **複数 WF 設定での安定性検証**

---

## 参照

- [prediction_clipping.md](prediction_clipping.md): クリッピング戦略
- [hull_sharpe_spec.md](hull_sharpe_spec.md): Hull Sharpe 仕様書
- [configs/evaluation/walk_forward.yaml](../../configs/evaluation/walk_forward.yaml): 設定ファイル
- [../submission/submissions.md](../submission/submissions.md): 提出履歴
