# Feature Selection Phase

## 概要

戦略単位（SU1-SU11）のフェーズ完了後、特徴量選定フェーズに移行。
約 577 列の特徴量から、**Sharpe に寄与しない／揺らぎが大きい特徴を系統立てて削る・圧縮する**ことが目的。

## 現状ベースライン

| 項目 | 値 |
|------|-----|
| ベストライン | SU1 + SU5 + Brushup (生特徴 + 前処理 + LGBM) |
| 特徴量数 | 577 列 |
| OOF RMSE | 0.012134 |
| LB Score | 0.681 |
| ブランチ | `dev` |

---

## フェーズ構成

```
Phase 0: Tier0 固定（ベースライン凍結）
    ↓
Phase 1: フィルタベースの雑草抜き（統計的除去）
    ↓
Phase 2: モデルベース重要度（LGBM importance → Permutation）
    ↓
Phase 3: グルーピングと冗長性削減（相関クラスタリング）
    ↓
Phase 4: 次元圧縮（PCA等、ブロック限定・オプション）
    ↓
Phase 5: 最終評価とアーティファクト整理
```

---

## Phase 0: Tier0 固定（ベースライン凍結）

### 目的
- 以降の「削る・圧縮する」判断が、Sharpe 向きの指標と整合するようにする
- 評価軸を統一し、比較可能な状態を作る

### タスク

- [ ] **T0-1**: 現行ベストライン（生＋SU1＋SU5＋前処理＋LGBM）を「Tier0」として固定
  - config snapshot: `configs/tier0_snapshot/`
  - 特徴量リスト: `artifacts/tier0/feature_list.json`
  - 学習済みパイプライン: `artifacts/tier0/inference_bundle.pkl`

- [ ] **T0-2**: 評価軸を CV に統一
  - Primary: RMSE ― 予測精度の直接評価、モデル・特徴量セットの採用判断は基本こちらに従う
  - Secondary: MSR (Mean-Sharpe-Ratio) ― リターン予測の実用性評価、補助指標
  - 評価関数: `scripts/utils_msr.py` の既存実装を使用

- [ ] **T0-3**: ベースライン評価スクリプト作成
  - `src/feature_selection/evaluate_baseline.py`
  - fold 毎の importance 出力機能を含む

### 成果物
- `artifacts/tier0/` ディレクトリ
- `configs/tier0_snapshot/` ディレクトリ

---

## Phase 1: フィルタベースの雑草抜き

### 目的
- 統計的に明らかに不要な列を機械的に落とす
- Sharpe コンペ固有の難しさとは独立な領域

### 除去対象

| カテゴリ | 判定基準 | 備考 |
|---------|---------|------|
| 分散ほぼゼロ | `var < 1e-10` | 一定値に近い列 |
| 欠損率ほぼ100% | `missing_rate > 0.99` | Imputer で常に同じ値 |
| ほぼ線形従属 | `|corr| > 0.999` | 他列との相関が極端に高い |

### タスク

- [ ] **T1-1**: フィルタスクリプト作成
  - `src/feature_selection/filter_trivial.py`
  - 入力: 特徴量 DataFrame
  - 出力: 削除候補リスト（JSON）

- [ ] **T1-2**: Tier0 に対してフィルタ適用
  - 削除候補を `results/feature_selection/phase1_filter_candidates.json` に出力

- [ ] **T1-3**: フィルタ後の評価
  - Tier0 → Tier1 の Sharpe/RMSE 比較
  - 結果を `docs/feature_selection/phase1_report.md` に記録

### 判定基準
- Sharpe 同等以上 → 削除採用
- Sharpe 微減でも列数大幅減 → 採用検討

---

## Phase 2: モデルベース重要度

### Phase 2-1: LGBM gain/split 重要度

#### 目的
- fold 毎の feature importance を算出し、安定性を評価
- 「全 fold で一貫して低いもの」を候補としてマーク

#### タスク

- [ ] **T2-1-1**: Importance 算出スクリプト作成
  - `src/feature_selection/compute_importance.py`
  - 出力: fold 毎の gain/split importance（CSV）

- [ ] **T2-1-2**: 可視化と候補抽出
  - `notebooks/feature_selection/importance_analysis.ipynb`
  - 平均重要度 × fold 間ばらつきの散布図
  - 下位 20-30% を「Tier1 削除候補」としてマーク

- [ ] **T2-1-3**: 候補リスト出力
  - `results/feature_selection/phase2_importance_candidates.json`

### Phase 2-2: Permutation Importance（Sharpe ベース）

#### 目的
- Tier1 削除候補に対して、Sharpe への実際の影響を検証
- 「明らかに影響ゼロな列」を確定

#### 注意点
- 相関の強い特徴がある場合、Permutation Importance は歪む可能性
- 完全に信じるのではなく、「importance がほぼ0で安定している列」を探す用途に割り切る

#### タスク

- [ ] **T2-2-1**: Permutation スクリプト作成
  - `src/feature_selection/permutation_importance.py`
  - 指標: MSR (Sharpe ベース)
  - 対象: Phase 2-1 で抽出した候補列のみ（計算コスト削減）

- [ ] **T2-2-2**: 結果分析と最終候補確定
  - `results/feature_selection/phase2_permutation_results.csv`

---

## Phase 3: グルーピングと冗長性削減

### 目的
- 強く相関しているグループを発見
- グループ単位で importance を見て、代表列を残し残りを削る

### タスク

- [ ] **T3-1**: 相関クラスタリング
  - `src/feature_selection/correlation_clustering.py`
  - 閾値: `|ρ| > 0.95` で同一グループ
  - グループ可視化（ヒートマップ、デンドログラム）

- [ ] **T3-2**: グループ単位の重要度集計
  - 各グループの代表列を決定
  - 削除候補: グループ内で重要度最低の列

- [ ] **T3-3**: 削減セット確定
  - `results/feature_selection/phase3_removal_set.json`
  - Phase 1 + Phase 2 + Phase 3 の統合リスト

- [ ] **T3-4**: 削減後評価
  - Tier0 全特徴 vs Tier0 − 削減セット の CV 比較
  - Sharpe/MSR 同等以上 → 削減採用
  - 明確に悪化 → 削減幅を縮小 or 一部戻す

---

## Phase 4: 次元圧縮（PCA 等）- オプション

### 目的
- 特定ブロックに対してのみ PCA を検討
- 「最後の調整」として限定的に使用

### 対象候補
- 非常に多い同種スケールの列（M/E/I グループなど）
- SU1/SU5 の欠損構造ブロック（3〜5 次元に圧縮）

### タスク

- [ ] **T4-1**: PCA 適用スクリプト
  - `src/feature_selection/block_pca.py`
  - ブロック単位で分散説明率を確認

- [ ] **T4-2**: PCA 版評価
  - PCA 成分を「既存特徴の代替」とするか「追加の特徴」とするかを比較
  - Sharpe/安定性を検証

### 判定基準
- 特徴削減だけで Sharpe が安定/改善 → PCA 不要
- まだ列数・冗長性が気になる → ブロック限定で PCA 試行

---

## Phase 5: 最終評価とアーティファクト整理

### タスク

- [ ] **T5-1**: 最終特徴量セット確定
  - `artifacts/feature_selection/final_feature_list.json`

- [ ] **T5-2**: 最終評価
  - CV Sharpe / RMSE
  - LB 提出（必要に応じて）

- [ ] **T5-3**: ドキュメント整理
  - `docs/feature_selection/summary.md` に全体まとめ
  - `docs/submissions.md` に結果追記

---

## ブランチ戦略

```
dev (現行)
 └── feat/feature-selection
      ├── feat/fs-phase1-filter      # フィルタベース除去
      ├── feat/fs-phase2-importance  # 重要度ベース選定
      ├── feat/fs-phase3-grouping    # グルーピング・冗長性削減
      └── feat/fs-phase4-pca         # PCA（オプション）
```

### ブランチ運用ルール

1. **feat/feature-selection** をフェーズ全体の親ブランチとする
2. 各 Phase は子ブランチで作業し、完了後に親へ merge
3. Phase 完了ごとに `dev` へ統合（PR レビューは省略可）
4. 重大な変更時のみタグ付け（例: `fs-phase1-complete`）

---

## フォルダ構成

```
src/
└── feature_selection/
    ├── __init__.py
    ├── evaluate_baseline.py      # Phase 0: ベースライン評価
    ├── filter_trivial.py         # Phase 1: フィルタベース除去
    ├── compute_importance.py     # Phase 2-1: LGBM importance
    ├── permutation_importance.py # Phase 2-2: Permutation importance
    ├── correlation_clustering.py # Phase 3: 相関クラスタリング
    └── block_pca.py              # Phase 4: ブロック PCA

notebooks/
└── feature_selection/
    ├── importance_analysis.ipynb     # 重要度分析・可視化
    └── correlation_analysis.ipynb    # 相関分析・可視化

results/
└── feature_selection/
    ├── phase1_filter_candidates.json
    ├── phase2_importance_candidates.json
    ├── phase2_permutation_results.csv
    └── phase3_removal_set.json

artifacts/
├── tier0/                           # Phase 0: ベースライン凍結
│   ├── feature_list.json
│   ├── inference_bundle.pkl
│   └── model_meta.json
└── feature_selection/
    └── final_feature_list.json      # Phase 5: 最終特徴量セット

configs/
└── tier0_snapshot/                  # Phase 0: config 凍結
    ├── feature_generation.yaml
    └── preprocess.yaml

docs/
└── feature_selection/
    ├── README.md                    # 本ファイル（計画）
    ├── phase1_report.md             # Phase 1 結果
    ├── phase2_report.md             # Phase 2 結果
    ├── phase3_report.md             # Phase 3 結果
    └── summary.md                   # 最終まとめ
```

---

## 進め方の原則

### 判断基準

| 状況 | 判断 |
|------|------|
| Sharpe 同等以上 + 列数減 | 採用 |
| Sharpe 微減 + 列数大幅減 | 一時許容（後続で取り返す余地あり） |
| Sharpe 明確悪化 | 不採用 or 削減幅縮小 |

### 記録ルール

各実験ごとに以下を記録：
- 使用した特徴リスト
- CV Sharpe / RMSE
- LB スコア（試した場合）
- 判断理由

記録先:
- 詳細: `docs/feature_selection/phaseX_report.md`
- サマリ: `docs/submissions.md`

---

## 優先順位とタイムライン（目安）

| Phase | 優先度 | 想定工数 | 備考 |
|-------|--------|----------|------|
| Phase 0 | 必須 | 0.5日 | 最初に完了させる |
| Phase 1 | 必須 | 1日 | 機械的処理、リスク低 |
| Phase 2-1 | 必須 | 1日 | LGBM importance |
| Phase 2-2 | 必須 | 1-2日 | Permutation（計算コスト高め） |
| Phase 3 | 必須 | 1-2日 | グルーピング・削減確定 |
| Phase 4 | オプション | 1日 | 必要に応じて |
| Phase 5 | 必須 | 0.5日 | 整理・まとめ |

**合計目安: 5-8 日**

---

## 参考情報

### Hull Tactical コンペ関連
- ディスカッション: 次元の呪いを避けるための PCA / 冗長特徴削減の重要性
- 金融リターンの因子抽出に PCA/派生 PCA を使って Sharpe を改善した研究多数

### 注意点
- 時系列データでは、適切な CV と評価軸に沿って feature importance を取ることが重要
- 相関の強い特徴が多いと importance が「薄く割れる」現象がある
- Permutation Importance は相関が強い場合に歪む可能性

---

## 次のアクション

1. **Phase 0 開始**: `feat/feature-selection` ブランチ作成
2. Tier0 凍結（タグ・config snapshot）
3. `src/feature_selection/` ディレクトリ構造作成
4. Phase 1 フィルタスクリプト実装へ
