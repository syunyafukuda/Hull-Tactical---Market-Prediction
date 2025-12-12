# Feature Selection Phase

## 概要

戦略単位（SU1-SU11）のフェーズ完了後、特徴量選定フェーズに移行。
約 577 列の特徴量から、**Sharpe に寄与しない／揺らぎが大きい特徴を系統立てて削る・圧縮する**ことが目的。

## 現状ベースライン（Tier0）

| 項目 | 値 |
|------|-----|
| ベストライン | SU1 + SU5 + Brushup (生特徴 + 前処理 + LGBM) |
| 特徴量数 | 577 列（94 input + 366 SU1 + 108 SU5 + 9 other） |
| OOF RMSE | 0.012134 |
| OOF MSR | 0.019929 |
| LB Score | 0.681 |
| ブランチ | `dev` |
| アーティファクト | `artifacts/tier0/` |

---

## パイプライン構造と選定位置

全フェーズの特徴量選定は、以下のパイプライン内の同じ位置で適用される。

```
生データ (94列)
    ↓
[SU1 特徴量生成] → 366列追加
    ↓
[SU5 特徴量生成] → 108列追加
    ↓
合計 568列
    ↓
[★ 特徴量選定（除外処理）] ← Phase 1-3 の除外リストをここで適用
    ↓
[前処理 (ColumnTransformer)]
  - Imputer
  - Scaler
  - OneHotEncoder
    ↓
[LightGBM]
```

### 選定の判定と除外のタイミング

| 処理 | タイミング | 説明 |
|------|-----------|------|
| **判定** | 前処理後の値で分析 | Imputer/Scaler 適用後の値で分散・欠損・相関を評価 |
| **除外** | SU5 適用後・前処理前 | 除外リストに基づき DataFrame から列を drop |

この方式により、全フェーズ（Phase 1〜3）で生成された除外リストを統一的に適用できる。

---

## フェーズ構成

```
Phase 0: Tier0 固定（ベースライン凍結）                    ✅ 完了
    ↓
Phase 1: フィルタベースの雑草抜き（統計的除去）            ✅ 完了
    ↓
Phase 2: モデルベース重要度（LGBM importance）             ✅ 完了
    ↓
Phase 3: グルーピングと冗長性削減（相関クラスタリング）    ✅ 完了
    ↓
モデル選定フェーズへ
```

### 進捗サマリー

| Phase | 入力 | 出力 | 削減数 | 残列数 | OOF RMSE | LB Score | 状態 |
|-------|------|------|--------|--------|----------|----------|------|
| Phase 0 | - | Tier0 | - | 577 | 0.012134 | 0.681 | ✅ 完了 |
| Phase 1 | Tier0 | Tier1 | -417 | 160 | 0.012168 | 0.681 | ✅ 完了 |
| Phase 2 | Tier1 | Tier2 | -40 | 120 | 0.012172 | 0.681 | ✅ 完了 |
| Phase 3 | Tier2 | Tier3 | -4 | 116 | 0.012164 | 0.681 | ✅ 完了 |

### 最終Feature Set

| セット名 | 説明 | 列数 | OOF RMSE | LB Score | ステータス |
|---------|------|------|----------|----------|------------|
| FS_full | Tier2そのまま | 120 | 0.012172 | 0.681 | baseline |
| **FS_compact** | Tier3（相関クラスタ後） | 116 | 0.012164 | **0.681** | **採用** |
| FS_topK | importance上位50列 | 50 | 0.012023 | 0.589 | 非採用（過学習） |

---

## Phase 0: Tier0 固定（ベースライン凍結）✅ 完了

### 目的
- 以降の「削る・圧縮する」判断が、Sharpe 向きの指標と整合するようにする
- 評価軸を統一し、比較可能な状態を作る

### タスク

- [x] **T0-1**: 現行ベストライン（生＋SU1＋SU5＋前処理＋LGBM）を「Tier0」として固定
  - config snapshot: `configs/tier0_snapshot/`
  - 特徴量リスト: `artifacts/tier0/feature_list.json`
  - 学習済みパイプライン: `artifacts/tier0/inference_bundle.pkl`
  - モデルメタ情報: `artifacts/tier0/model_meta.json`

- [x] **T0-2**: 評価軸を CV に統一
  - Primary: RMSE ― 予測精度の直接評価、モデル・特徴量セットの採用判断は基本こちらに従う
  - Secondary: MSR (Mean-Sharpe-Ratio) ― リターン予測の実用性評価、補助指標
  - 評価関数: `scripts/utils_msr.py` の既存実装を使用

- [x] **T0-3**: ベースライン評価スクリプト作成
  - `src/feature_selection/common/evaluate_baseline.py`
  - fold 毎の importance 出力機能を含む

### 成果物

| ファイル | 説明 |
|----------|------|
| `configs/tier0_snapshot/feature_generation.yaml` | 特徴量生成設定 |
| `configs/tier0_snapshot/preprocess.yaml` | 前処理設定 |
| `artifacts/tier0/feature_list.json` | 特徴量リスト（568列） |
| `artifacts/tier0/model_meta.json` | モデルパラメータ・評価結果 |
| `artifacts/tier0/inference_bundle.pkl` | 学習済みパイプライン |
| `results/feature_selection/tier0/evaluation.json` | OOF 評価結果 |
| `results/feature_selection/tier0/importance.csv` | fold毎の importance |
| `results/feature_selection/tier0/importance_summary.csv` | importance 集計 |
| `results/feature_selection/tier0/fold_logs.csv` | fold毎の RMSE/MSR |

### 仕様書
- `docs/feature_selection/phase0_spec.md`

---

## Phase 1: フィルタベースの雑草抜き 🔜 次のステップ

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
  - 削除候補を `results/feature_selection/phase2/importance_candidates.json` に出力

- [ ] **T1-3**: フィルタ後の評価
  - Tier0 → Tier1 の Sharpe/RMSE 比較
  - 結果を `docs/feature_selection/phase1_report.md` に記録

### 判定基準
- Sharpe 同等以上 → 削除採用
- Sharpe 微減でも列数大幅減 → 採用検討

### 仕様書
- `docs/feature_selection/phase1_spec.md`

---

## Phase 2: モデルベース重要度

### Phase 2-1: LGBM gain/split 重要度

#### 目的
- fold 毎の feature importance を算出し、安定性を評価
- 「全 fold で一貫して低いもの」を候補としてマーク

#### タスク

- [ ] **T2-1-1**: Importance 算出スクリプト作成
  - `src/feature_selection/phase2/compute_importance.py`
  - 出力: fold 毎の gain/split importance（CSV）

- [ ] **T2-1-2**: 可視化と候補抽出
  - `notebooks/feature_selection/importance_analysis.ipynb`
  - 平均重要度 × fold 間ばらつきの散布図
  - 下位 20-30% を「Tier1 削除候補」としてマーク

- [ ] **T2-1-3**: 候補リスト出力
  - `results/feature_selection/phase2/importance_candidates.json`

### Phase 2-2: Permutation Importance（Sharpe ベース）

#### 目的
- Tier1 削除候補に対して、Sharpe への実際の影響を検証
- 「明らかに影響ゼロな列」を確定

#### 注意点
- 相関の強い特徴がある場合、Permutation Importance は歪む可能性
- 完全に信じるのではなく、「importance がほぼ0で安定している列」を探す用途に割り切る

#### タスク

- [ ] **T2-2-1**: Permutation スクリプト作成
  - `src/feature_selection/phase2/permutation_importance.py`
  - 指標: MSR (Sharpe ベース)
  - 対象: Phase 2-1 で抽出した候補列のみ（計算コスト削減）

- [ ] **T2-2-2**: 結果分析と最終候補確定
  - `results/feature_selection/phase2/permutation_results.csv`

---

## Phase 3: グルーピングと冗長性削減 ✅ 完了

### 目的
- Tier2 特徴セット（120列）の中から相関の高いグループを発見
- グループ単位で importance を見て、代表列を残し残りを削る
- 複数の Feature Set バリエーションを定義してモデル選定フェーズへ引き継ぐ

### 実施内容

- [x] **T3-1**: 相関クラスタリング
  - `src/feature_selection/phase3/correlation_clustering.py`
  - 閾値: `|ρ| > 0.95` で階層クラスタリング（Ward 法）
  - 出力: `results/feature_selection/phase3/correlation_clusters.json`

- [x] **T3-2**: クラスタ代表選出
  - `src/feature_selection/phase3/select_representatives.py`
  - 各クラスタから mean_gain 最大の特徴を代表として選出
  - 出力: `results/feature_selection/phase3/cluster_representatives.json`

- [x] **T3-3**: Tier3 除外リスト作成
  - `src/feature_selection/phase3/create_tier3_excluded.py`
  - Tier2 + Phase 3 削除候補を統合
  - 出力: `configs/feature_selection/tier3/excluded.json`

- [x] **T3-4**: Feature Set 定義
  - `src/feature_selection/phase3/create_feature_sets.py`
  - FS_full (Tier2): 120列、最大性能
  - FS_compact (Tier3): 80-100列、冗長性削減後
  - FS_topK: 50列、Top-K 特徴のみ
  - 出力: `configs/feature_selection/feature_sets.json`

- [x] **T3-5**: 統合パイプライン
  - `src/feature_selection/phase3/run_phase3.py`
  - 全ステップを自動実行、レポート生成
  - 出力: `docs/feature_selection/phase3_report.md`

### 使用方法

```bash
# 完全パイプラインの実行
python src/feature_selection/phase3/run_phase3.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --tier2-excluded configs/feature_selection/tier2/excluded.json \
  --tier2-importance results/feature_selection/tier2/importance_summary.csv \
  --tier2-evaluation results/feature_selection/tier2/evaluation.json

# 相関クラスタリングをスキップ（Tier2を最終セットとして使用）
python src/feature_selection/phase3/run_phase3.py --skip-clustering
```

詳細は `src/feature_selection/phase3/README.md` を参照。
実行は終わりました。たーみたーみ
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
    ├── README.md
    ├── common/                       # 共通ユーティリティ
    │   ├── __init__.py
    │   └── evaluate_baseline.py      # Tier評価共通
    ├── phase1/                       # Phase 1: フィルタベース
    │   ├── __init__.py
    │   └── filter_trivial.py         # 統計フィルタ
    ├── phase2/                       # Phase 2: モデルベース重要度
    │   ├── __init__.py
    │   ├── compute_importance.py     # LGBM importance
    │   └── permutation_importance.py # Permutation importance
    ├── phase3/                       # Phase 3: 相関クラスタリング（予定）
    │   └── correlation_clustering.py
    ├── phase4/                       # Phase 4: ブロック PCA（予定）
    │   └── block_pca.py
    └── inference/                    # 推論
        ├── __init__.py
        └── predict_tier.py

notebooks/
└── feature_selection/
    ├── importance_analysis.ipynb     # 重要度分析・可視化
    └── correlation_analysis.ipynb    # 相関分析・可視化

results/
└── feature_selection/
    ├── tier0/
    │   ├── evaluation.json
    │   ├── importance.csv
    │   ├── importance_summary.csv
    │   └── fold_logs.csv
    ├── tier1/
    │   ├── evaluation.json
    │   ├── importance.csv
    │   ├── importance_summary.csv
    │   └── fold_logs.csv
    ├── tier2/
    │   └── evaluation.json
    ├── phase2/
    │   ├── importance_candidates.json
    │   └── permutation_results.csv
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
