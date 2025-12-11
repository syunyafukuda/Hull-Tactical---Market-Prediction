# Phase 3: 相関クラスタリングと最終特徴セット確定 仕様書

## 概要

Phase 2 で LGBM importance ベースのフィルタを適用した Tier2 特徴セット（120列）を入力とし、
**相関クラスタリングによる冗長性削減と最終特徴セットの確定**を行うフェーズ。

Phase 0〜2 で 577 → 120 列（-79%）まで削減済みのため、Phase 3 では:
1. Tier2 をベースラインとして固定
2. 相関クラスタリングで「もう一段だけ」冗長性を削減（任意）
3. モデル選定フェーズへ渡す Feature Set バリエーションを定義

---

## 入力・出力

### 入力

| 項目 | 値 | ファイル |
|------|-----|----------|
| 特徴量セット | Tier2（Phase 2 後） | `configs/feature_selection/tier2/excluded.json` |
| 特徴量数 | 120 列 | - |
| OOF RMSE | 0.012172 | `results/feature_selection/tier2/evaluation.json` |
| LB Score | 0.681 | - |
| Importance 集計 | Tier2 の importance_summary | `results/feature_selection/tier2/importance_summary.csv` |
| アーティファクト | 学習済みパイプライン等 | `artifacts/tier2/` |

### 出力

| 項目 | 説明 | ファイル |
|------|------|----------|
| 相関クラスタ結果 | クラスタ割当と代表特徴 | `results/feature_selection/phase3/correlation_clusters.json` |
| Tier3 除外リスト | Phase 3 で確定した追加削除列 | `configs/feature_selection/tier3/excluded.json` |
| Tier3 評価結果 | Tier3 の OOF RMSE | `results/feature_selection/tier3/evaluation.json` |
| Feature Set 定義 | モデル選定用の特徴セット | `configs/feature_selection/feature_sets.json` |
| Phase 3 レポート | 実行結果と判定 | `docs/feature_selection/phase3_report.md` |

---

## 制約条件

1. **CV 設定は Tier0 と同一**: TimeSeriesSplit, 5 fold, gap=0
2. **評価指標**: **RMSE のみ**（MSR は参考値として記録するが判定には使用しない）
3. **判定基準**:
   - RMSE 同等以上（+0.0001 以内）+ 列数減 → 採用
   - RMSE 微増でも列数大幅減 → 採用検討
   - RMSE 明確悪化（+0.0001 超）→ 削減幅を縮小

---

## フェーズ構成

```
Phase 3-1: 相関クラスタリング
    ↓
    高相関グループの特定（|ρ| > 0.95）
    ↓
Phase 3-2: クラスタ代表選出
    ↓
    各クラスタから importance 最大の列を残す
    ↓
Phase 3-3: Tier3 評価
    ↓
    RMSE 判定（+0.0001 以内なら採用）
    ↓
Phase 3-4: Feature Set 定義
    ↓
    モデル選定フェーズへ
```

---

## Phase 3-1: 相関クラスタリング

### 目的

Tier2（120列）の中に残っている「そこそこ効いているが、互いに似た情報を持つグループ」を特定し、
冗長な列を削減する候補を抽出する。

### 方法

1. **相関行列の計算**
   - 前処理後（Imputer + Scaler 適用後）の特徴行列で相関を計算
   - Tier2 の 120 列のみを対象

2. **閾値設定**
   - Phase 1 より緩い閾値: `|ρ| > 0.95`
   - 「ほぼ同じ情報」ではなく「かなり似た情報」を捉える

3. **クラスタリング**
   - 相関行列を距離に変換: `distance = 1 - |ρ|`
   - 階層クラスタリング（Ward 法）でグループ化
   - 閾値でクラスタを切り出し

### 出力スキーマ

`results/feature_selection/phase3/correlation_clusters.json`:

```json
{
  "threshold": 0.95,
  "n_clusters": 15,
  "clusters": [
    {
      "cluster_id": 1,
      "features": ["M1", "M2", "M3"],
      "representative": "M1",
      "max_correlation": 0.98
    },
    ...
  ],
  "singleton_features": ["E1", "P3", ...]
}
```

### 実装

**ファイル**: `src/feature_selection/phase3/correlation_clustering.py`

**引数**:
- `--config-path`: 特徴量生成設定
- `--preprocess-config`: 前処理設定
- `--data-dir`: データディレクトリ
- `--exclude-features`: Tier2 除外リスト
- `--correlation-threshold`: 相関閾値（デフォルト: 0.95）
- `--out-dir`: 出力先

---

## Phase 3-2: クラスタ代表選出

### 目的

各クラスタから「代表特徴」を選び、それ以外を削除候補とする。

### 代表選択基準

1. **Importance 最大**: Phase 2 の `mean_gain` が最大の列を代表とする
2. **生特徴優先**: 同等の importance なら生特徴（D, E, I, M, P, S, V）を優先
3. **1クラスタ1代表**: 各クラスタから 1 列のみ残す

### 実装詳細

1. `results/feature_selection/tier2/importance_summary.csv` から importance を読み込み
2. 各クラスタ内で `mean_gain` 最大の列を代表に選定
3. 代表以外の列を削除候補リストに追加

### 出力

`results/feature_selection/phase3/cluster_representatives.json`:

```json
{
  "representatives": [
    {"cluster_id": 1, "feature": "M1", "mean_gain": 1234.5},
    ...
  ],
  "to_remove": [
    {"cluster_id": 1, "feature": "M2", "mean_gain": 1100.2},
    {"cluster_id": 1, "feature": "M3", "mean_gain": 980.1},
    ...
  ],
  "total_removed": 20
}
```

---

## Phase 3-3: Tier3 評価

### 目的

Phase 3-2 で選定した削除候補を除外した Tier3 を評価し、採用可否を判定する。

### 実行手順

1. **Tier3 除外リスト作成**
   - Tier2 除外リスト + Phase 3 削除候補 = Tier3 除外リスト

2. **OOF 評価**
   ```bash
   python src/feature_selection/common/evaluate_baseline.py \
     --config-path configs/feature_generation.yaml \
     --preprocess-config configs/preprocess.yaml \
     --data-dir data/raw \
     --exclude-features configs/feature_selection/tier3/excluded.json \
     --out-dir results/feature_selection/tier3
   ```

3. **判定**

   | 条件 | 判定 |
   |------|------|
   | RMSE 悪化 ≤ +0.0001 | ✅ Tier3 採用 |
   | RMSE 悪化 > +0.0001 | ❌ Tier2 のまま |

### 出力

- `configs/feature_selection/tier3/excluded.json`
- `results/feature_selection/tier3/evaluation.json`
- `results/feature_selection/tier3/fold_logs.csv`
- `results/feature_selection/tier3/importance_summary.csv`

---

## Phase 3-4: Feature Set 定義

### 目的

モデル選定フェーズへ渡すための Feature Set バリエーションを定義する。

### Feature Set 候補

| セット名 | 説明 | 列数（目安） | 用途 |
|---------|------|-------------|------|
| `FS_full` | Tier2 そのまま | 120 | 最大性能ベースライン |
| `FS_compact` | Tier3（相関クラスタ後） | 80-100 | 汎化性能・学習速度重視 |
| `FS_topK` | importance 上位 K 列 | 40-60 | 最小構成・デバッグ用 |

### 出力

`configs/feature_selection/feature_sets.json`:

```json
{
  "version": "v1",
  "created_at": "2025-12-11T...",
  "feature_sets": {
    "FS_full": {
      "description": "Tier2 full feature set",
      "excluded_json": "configs/feature_selection/tier2/excluded.json",
      "n_features": 120,
      "oof_rmse": 0.012172
    },
    "FS_compact": {
      "description": "Tier3 after correlation clustering",
      "excluded_json": "configs/feature_selection/tier3/excluded.json",
      "n_features": 95,
      "oof_rmse": 0.012180
    },
    "FS_topK": {
      "description": "Top 50 features by importance",
      "excluded_json": "configs/feature_selection/tier_topK/excluded.json",
      "n_features": 50,
      "oof_rmse": 0.012250
    }
  },
  "recommended": "FS_compact"
}
```

---

## タスク一覧

### T3-1: 相関クラスタリング

- [ ] `src/feature_selection/phase3/correlation_clustering.py` 作成
- [ ] Tier2 特徴量で相関行列を計算
- [ ] 閾値 0.95 でクラスタリング
- [ ] 結果を `results/feature_selection/phase3/correlation_clusters.json` に出力

### T3-2: クラスタ代表選出

- [ ] `src/feature_selection/phase3/select_representatives.py` 作成
- [ ] importance 最大の列を各クラスタから選出
- [ ] 削除候補リストを `results/feature_selection/phase3/cluster_representatives.json` に出力

### T3-3: Tier3 評価

- [ ] Tier3 除外リスト作成: `configs/feature_selection/tier3/excluded.json`
- [ ] `evaluate_baseline.py` で OOF 評価
- [ ] RMSE 判定（+0.0001 以内なら採用）
- [ ] 結果を `results/feature_selection/tier3/` に出力

### T3-4: Feature Set 定義

- [ ] `configs/feature_selection/feature_sets.json` 作成
- [ ] FS_full, FS_compact, FS_topK の3バリエーションを定義
- [ ] 推奨セットを決定

### T3-5: LB 検証（任意）

- [ ] Tier3 artifacts 生成
- [ ] Kaggle 提出、LB スコア確認
- [ ] 結果を `docs/submissions.md` に記録

### T3-6: ドキュメント整備

- [ ] `docs/feature_selection/phase3_report.md` 作成
- [ ] `docs/feature_selection/README.md` 更新

---

## 判定基準

### Tier3 採用判定

| 条件 | 判定 |
|------|------|
| RMSE 悪化 ≤ +0.0001 かつ 列数減 | ✅ Tier3 採用 |
| RMSE 悪化 > +0.0001 | ❌ Tier2 維持 |

### Phase 3 完了判定

- [ ] Tier3 または Tier2 のいずれかを最終ベースラインとして確定
- [ ] Feature Set 定義が完了
- [ ] モデル選定フェーズへの引き継ぎドキュメントが整備

---

## ディレクトリ構成（完了後）

```
configs/feature_selection/
├── tier0/
│   └── baseline.json
├── tier1/
│   └── excluded.json          # Phase 1 除外（417列）
├── tier2/
│   └── excluded.json          # Phase 2 除外（457列）
├── tier3/
│   └── excluded.json          # Phase 3 除外（470-480列程度）
├── tier_topK/
│   └── excluded.json          # Top K 以外を除外
└── feature_sets.json          # Feature Set 定義

results/feature_selection/
├── tier0/
├── tier1/
├── tier2/
├── tier3/
│   ├── evaluation.json
│   ├── fold_logs.csv
│   └── importance_summary.csv
└── phase3/
    ├── correlation_clusters.json
    └── cluster_representatives.json

artifacts/
├── tier0/
├── tier2/
└── tier3/                     # 必要に応じて生成
```

---

## スケジュール目安

| タスク | 工数 |
|--------|------|
| T3-1: 相関クラスタリング | 1-2時間 |
| T3-2: クラスタ代表選出 | 0.5時間 |
| T3-3: Tier3 評価 | 0.5時間 |
| T3-4: Feature Set 定義 | 0.5時間 |
| T3-5: LB 検証（任意） | 1時間 |
| T3-6: ドキュメント整備 | 0.5時間 |
| **合計** | **4-5時間** |

---

## 備考

### Phase 3 のスキップ条件

以下の場合、Phase 3-1〜3-3 をスキップして Phase 3-4 のみ実施可能:

1. Tier2（120列）で十分コンパクトと判断
2. 相関クラスタリングで有意なグループが見つからない
3. 追加削減による RMSE 悪化リスクを避けたい

この場合、`FS_full = Tier2` として Feature Set 定義のみ行い、モデル選定フェーズへ進む。

### 次フェーズへの橋渡し

Phase 3 完了後、以下のフェーズに進む:

1. **モデル選定**: LGBM 以外のモデル（XGBoost, CatBoost, Ridge 等）を Feature Set で評価
2. **ハイパラ最適化**: 確定した Feature Set でハイパラチューニング
3. **アンサンブル**: 複数モデル・Feature Set の組み合わせ

---

## 参考

- Phase 2 結果: `docs/feature_selection/phase2_report.md`
- Phase 1 結果: `docs/feature_selection/phase1_report.md`
- 全体計画: `docs/feature_selection/README.md`
- Tier2 評価: `results/feature_selection/tier2/evaluation.json`
