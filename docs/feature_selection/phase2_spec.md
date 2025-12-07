# Phase 2: モデルベース重要度による特徴量選定 仕様書

## 概要

Phase 1 で統計フィルタを適用した Tier1 特徴セット（160列）を入力とし、
**モデルベースの重要度で「使っていない特徴を安全に削る」フェーズ**。

LGBM の gain importance を主指標として削除候補をマーキングした後、
RMSE ベースの Permutation Importance で本当に効いていない列を確定削除する二段構えで進める。

---

## 入力・出力

### 入力

| 項目 | 値 | ファイル |
|------|-----|----------|
| 特徴量セット | Tier1（Phase 1 フィルタ後） | `configs/feature_selection/tier1/excluded.json` |
| 特徴量数 | 160 列 | - |
| OOF RMSE | 0.012168 | - |
| LB Score | 0.681 | - |
| ベース設定 | Tier0 snapshot | `configs/feature_generation.yaml`, `configs/preprocess.yaml` |

### 出力

| 項目 | 説明 | ファイル |
|------|------|----------|
| 削除候補リスト | Phase 2-1 で特定した低重要度列 | `results/feature_selection/phase2/importance_candidates.json` |
| Permutation 結果 | Phase 2-2 の ΔRMSE 分析 | `results/feature_selection/phase2/permutation_results.csv` |
| Tier2 除外リスト | Phase 2 で確定した削除列 | `configs/feature_selection/tier2/excluded.json` |
| 評価結果 | Tier2 の OOF RMSE | `results/feature_selection/tier2/evaluation.json` |

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
Phase 2-1: LGBM gain importance（主指標）
    ↓
    削除候補のマーキング（下位 20-30%）
    ↓
Phase 2-2: Permutation Importance（RMSE ベース）
    ↓
    ΔRMSE ≈ 0 の列を削除確定
    ↓
Tier2 特徴セット作成・評価
```

---

## Phase 2-1: LGBM gain/split 重要度

### 目的

LightGBM の gain/split importance を fold ごとに取得し、
**全 fold で一貫して重要度の低い列**を削除候補としてマークする。

### 主指標の選定

| importance_type | 説明 | 用途 |
|-----------------|------|------|
| **gain（主指標）** | 分岐による損失（RMSE）改善の合計 | 削除候補の抽出に使用 |
| split（補助指標） | 分岐に使われた回数 | 参考情報として記録 |

**理由**: gain は「どれだけ RMSE 改善に貢献したか」を直接反映し、RMSE 最適化との整合性が高い。

### 実装詳細

#### T2-1-1: Importance 算出スクリプト

**ファイル**: `src/feature_selection/phase2/compute_importance.py`

**機能要件**:

1. Tier1 特徴セットで Tier0 と同じ CV・ハイパラ設定で学習
2. 各 fold ごとに全特徴の gain/split importance を算出
3. 出力:
   - fold 毎の importance: `results/feature_selection/tier1/importance.csv`
   - 集計統計: `results/feature_selection/tier1/importance_summary.csv`

**出力スキーマ**:

`importance.csv`（fold 別）:

| 列名 | 型 | 説明 |
|------|-----|------|
| fold | int | fold 番号（1-5） |
| feature_name | str | 特徴量名 |
| importance_gain | float | gain importance |
| importance_split | int | split importance |

`importance_summary.csv`（集計）:

| 列名 | 型 | 説明 |
|------|-----|------|
| feature_name | str | 特徴量名 |
| mean_gain | float | 平均 gain importance（主指標） |
| std_gain | float | fold 間標準偏差 |
| min_gain | float | 最小 gain importance |
| max_gain | float | 最大 gain importance |
| mean_split | float | 平均 split importance（補助） |
| std_split | float | fold 間標準偏差 |
| min_split | float | 最小 split importance |
| max_split | float | 最大 split importance |
| mean_gain_normalized | float | 全体に占める割合（正規化、0-1）|

**CLI インターフェース**:

```bash
python src/feature_selection/phase2/compute_importance.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier1/excluded.json \
  --out-dir results/feature_selection/tier1 \
  --n-splits 5
```

**Note**: `evaluate_baseline.py` に既に importance 出力機能があるため、
そちらを拡張するか、専用スクリプトを作成するかは実装判断に委ねる。

---

#### T2-1-2: 可視化と候補抽出

**ファイル**: `notebooks/feature_selection/importance_analysis.ipynb`

**内容**:

1. **重要度分布の可視化**
   - ヒストグラム: 平均重要度の分布
   - 散布図: 平均重要度 × fold 間ばらつき
   - バープロット: 上位/下位 N 件の重要度

2. **削除候補の抽出ロジック**
   - 条件: `mean_gain` が下位 20-30% AND `std_gain` が小さい（安定して低い）
   - 閾値案:
     - `mean_gain < quantile(0.25)` AND `std_gain < median(std_gain)`
     - または `share_of_total < 0.001`（全体の 0.1% 未満）

3. **候補リストのレビュー**
   - 候補列の名前パターンを確認（SU1 由来？SU5 由来？生特徴？）
   - 意図せず重要な列が混入していないかチェック

---

#### T2-1-3: 候補リスト出力

**出力ファイル**: `results/feature_selection/phase2/importance_candidates.json`

**形式**:

```json
{
  "version": "phase2-v1",
  "created_at": "2025-12-07T00:00:00Z",
  "source_tier": "tier1",
  "selection_criteria": {
    "method": "lgbm_importance",
    "metric": "gain",
    "threshold_quantile": 0.25,
    "require_stable_low": true
  },
  "candidates": [
    {
      "feature_name": "co_miss_rollrate/M1__M2",
      "mean_gain": 0.0001,
      "std_gain": 0.00005,
      "share_of_total": 0.0002,
      "note": "全 fold でほぼ 0"
    }
  ],
  "summary": {
    "total_features": 160,
    "candidate_count": 40,
    "candidate_ratio": 0.25
  }
}
```

---

## Phase 2-2: Permutation Importance（RMSE ベース）

### 目的

Phase 2-1 でマークした削除候補列に対して、
**その列をシャッフルしたときの RMSE の変化を測り、
「性能にほぼ影響しない列」を削除確定**とする。

### 評価指標

**RMSE のみを使用**。MSR/Sharpe は参考値として記録するが、判定には使用しない。

理由:
- RMSE は予測精度を直接反映
- Phase 2-1 の LGBM gain importance も RMSE 最適化に基づく
- 評価指標の一貫性を確保

### モデル再学習の頻度

**重要**: 計算コスト削減のため、以下の方式を採用する。

1. **各 fold について、Tier1 特徴セットでモデルを 1 回だけ学習する**
2. **Permutation Importance 計算では、この学習済みモデルを使い回す**
3. **各候補列ごとに入力をシャッフルして OOF RMSE を再計算する**
4. **列ごとに再学習は行わない**

### 実装詳細

#### T2-2-1: Permutation スクリプト

**ファイル**: `src/feature_selection/phase2/permutation_importance.py`

**機能要件**:

1. 対象は Phase 2-1 の候補列のみ（計算コスト削減）
2. 手順:
   - 各 fold で Tier1 特徴セットを使ってモデルを 1 回学習
   - 候補列ごとに:
     - 対象列をランダムシャッフル
     - シャッフル後の RMSE を計算
     - 「シャッフル後 RMSE - 元 RMSE」を ΔRMSE として記録
3. 出力: `results/feature_selection/phase2/permutation_results.csv`

**Permutation の方式**:

```python
# 擬似コード
# Step 1: 各 fold でモデルを 1 回だけ学習（列ごとに再学習しない）
for fold_id, (train_idx, val_idx) in enumerate(tscv.split(X)):
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Step 2: ベースライン RMSE を計算
    y_pred_baseline = model.predict(X_val)
    rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred_baseline))
    
    # Step 3: 各候補列をシャッフルして ΔRMSE を計算
    for col in candidate_columns:
        X_val_shuffled = X_val.copy()
        X_val_shuffled[col] = np.random.permutation(X_val[col])
        
        y_pred_shuffled = model.predict(X_val_shuffled)
        rmse_shuffled = np.sqrt(mean_squared_error(y_val, y_pred_shuffled))
        
        # 正 = 重要（シャッフルで悪化）、0/負 = 不要
        delta_rmse = rmse_shuffled - rmse_baseline
```

**CLI インターフェース**:

```bash
python src/feature_selection/phase2/permutation_importance.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --exclude-features configs/feature_selection/tier1/excluded.json \
  --candidates results/feature_selection/phase2/importance_candidates.json \
  --out-path results/feature_selection/phase2/permutation_results.csv \
  --n-permutations 5 \
  --random-seed 42
```

**出力スキーマ** (`phase2/permutation_results.csv`):

| 列名 | 型 | 説明 |
|------|-----|------|
| feature_name | str | 特徴量名 |
| mean_delta_rmse | float | ΔRMSE の fold 平均（正=重要、0≈不要） |
| std_delta_rmse | float | ΔRMSE の fold 間標準偏差 |
| fold_1_delta | float | fold 1 の ΔRMSE |
| fold_2_delta | float | fold 2 の ΔRMSE |
| fold_3_delta | float | fold 3 の ΔRMSE |
| fold_4_delta | float | fold 4 の ΔRMSE |
| fold_5_delta | float | fold 5 の ΔRMSE |
| decision | str | "remove" または "keep"（phase2_report.md で最終決定） |

---

#### T2-2-2: 結果分析と最終候補確定

**削除確定の基準**:

> **重要**: 以下の閾値は **初期案** であり、固定値ではない。
> 実際の ΔRMSE 分布（`phase2/permutation_results.csv`）を可視化・確認した上で、
> **最終的な閾値は `phase2_report.md` 内で明示する**。

**初期閾値案**:

1. `|mean_delta_rmse| < 1e-5`（シャッフルしても RMSE がほぼ変わらない）
2. `std_delta_rmse < 1e-5`（fold 間で安定して影響なし）
3. 上記両方を満たす列を削除候補とする

**代替案**（相対比較）:
- `|mean_delta_rmse| < 0.001 * baseline_rmse`

**出力**:

1. 削除確定列リスト: `configs/feature_selection/tier2/excluded.json`
   - Phase 1 の除外リスト + Phase 2-2 の削除確定列
2. 分析レポート: `docs/feature_selection/phase2_report.md`

---

## Tier2 特徴セット評価

### T2-3: Tier2 評価

**目的**: Phase 1 + Phase 2 の削除列を一括で落とした Tier2 特徴リストで再学習し、性能を検証

**作業内容**:

1. Tier2 特徴リスト作成
   - Tier1 列 − Phase 2-2 削除確定列
   - 例: 160列 → 100〜120列（目標）

2. Tier2 で再学習
   - `evaluate_baseline.py` に `--exclude-features tier2/excluded.json` を指定

3. Tier1 との比較

| 指標 | Tier1 | Tier2 | 差分 | 判定 |
|------|-------|-------|------|------|
| 特徴量数 | 160 | ? | -? | - |
| OOF RMSE | 0.012168 | ? | ? | 同等以上? |
| OOF MSR | 0.019201 | ? | ? | 同等以上? |

4. 判定
   - 同等以上 or 微改善 → 削除採用
   - 明確悪化 → 削減幅を戻す（ΔMSR の閾値を上げる等）

5. LB 検証（オプション）
   - 最終的には Kaggle 提出で確認

---

## タスク一覧

### Phase 2-1: LGBM 重要度

| ID | タスク | 成果物 | 依存 |
|----|--------|--------|------|
| T2-1-1 | Importance 算出スクリプト作成 | `src/feature_selection/phase2/compute_importance.py` | - |
| T2-1-2 | 可視化と候補抽出 | `notebooks/feature_selection/importance_analysis.ipynb` | T2-1-1 |
| T2-1-3 | 候補リスト出力 | `results/feature_selection/phase2/importance_candidates.json` | T2-1-2 |

### Phase 2-2: Permutation Importance

| ID | タスク | 成果物 | 依存 |
|----|--------|--------|------|
| T2-2-1 | Permutation スクリプト作成 | `src/feature_selection/phase2/permutation_importance.py` | T2-1-3 |
| T2-2-2 | 結果分析と削除確定 | `configs/feature_selection/tier2/excluded.json` | T2-2-1 |

### Tier2 評価

| ID | タスク | 成果物 | 依存 |
|----|--------|--------|------|
| T2-3-1 | Tier2 評価 | `results/feature_selection/tier2/evaluation.json` | T2-2-2 |
| T2-3-2 | レポート作成 | `docs/feature_selection/phase2_report.md` | T2-3-1 |
| T2-3-3 | LB 検証（オプション） | `docs/submissions.md` 更新 | T2-3-1 |

---

## 成果物一覧

### スクリプト

| ファイル | 説明 |
|----------|------|
| `src/feature_selection/phase2/compute_importance.py` | LGBM importance 算出 |
| `src/feature_selection/phase2/permutation_importance.py` | Permutation importance 算出 |

### 設定・結果

| ファイル | 説明 |
|----------|------|
| `results/feature_selection/tier1/importance.csv` | fold 毎の importance |
| `results/feature_selection/tier1/importance_summary.csv` | importance 集計 |
| `results/feature_selection/phase2/importance_candidates.json` | 削除候補リスト |
| `results/feature_selection/phase2/permutation_results.csv` | Permutation 結果 |
| `configs/feature_selection/tier2/excluded.json` | Tier2 除外リスト |
| `results/feature_selection/tier2/evaluation.json` | Tier2 評価結果 |

**`tier2/evaluation.json` スキーマ例**:

```json
{
  "tier": "tier2",
  "n_features": 105,
  "oof_rmse": 0.012150,
  "oof_msr": 0.019180,
  "delta_rmse_from_tier1": -0.000018,
  "delta_msr_from_tier1": -0.000021,
  "cv_folds": 5,
  "random_seed": 42,
  "created_at": "2025-01-20T12:34:56Z",
  "exclude_config": "configs/feature_selection/tier2/excluded.json",
  "notes": "Phase 2-2 Permutation 削除後の評価結果"
}
```

### ドキュメント

| ファイル | 説明 |
|----------|------|
| `docs/feature_selection/phase2_spec.md` | 本仕様書 |
| `docs/feature_selection/phase2_report.md` | Phase 2 実行結果レポート |
| `notebooks/feature_selection/importance_analysis.ipynb` | 重要度分析ノートブック |

---

## 想定タイムライン

| Phase | 工数（目安） | 備考 |
|-------|-------------|------|
| T2-1-1 | 0.5日 | 既存 evaluate_baseline.py の拡張で対応可能 |
| T2-1-2 | 0.5日 | 可視化と候補抽出 |
| T2-1-3 | 0.25日 | 候補リスト出力 |
| T2-2-1 | 1-1.5日 | Permutation スクリプト（計算コスト高め） |
| T2-2-2 | 0.5日 | 結果分析 |
| T2-3 | 0.5日 | Tier2 評価 + レポート |
| **合計** | **3-4日** | - |

---

## 注意点・リスク

### Permutation Importance の歪み

相関の強い特徴がある場合、Permutation Importance は以下の問題が生じる:
- 高相関ペアの片方をシャッフルしても、もう片方が情報を保持
- 結果として両方とも「重要度が低い」と判定される可能性

**対策**:
- Phase 1 で高相関列（|ρ| > 0.999）を既に除去済み
- それでも相関が残る場合は、Phase 3 の相関クラスタリングで対処

### 計算コスト

Permutation は列数 × fold 数 × permutation 回数 の計算が必要:
- 候補 40 列 × 5 fold × 5 permutations = 1000 回の予測

**対策**:
- Phase 2-1 で候補を絞り込み（全 160 列ではなく 40 列程度に）
- 必要に応じてバッチ処理やマルチプロセス化

### 判定閾値の調整

ΔMSR の閾値は事前に決め打ちできない可能性がある。

**対策**:
- 分布を見てから閾値を決定
- 保守的に始め（閾値を厳しく）、必要に応じて緩める

---

## Phase 3 以降とのつながり

- Phase 2 の役割: **個別列レベルで「ほぼ効いていない特徴」を安全に削る**
- Phase 2 完了後もまだ列数が多く冗長性が高い場合 → Phase 3 で相関クラスタリング + 代表列選定
- PCA などの次元圧縮は Phase 2/3 で十分に削り切れなかった場合のオプションとして温存

---

## 参考

- `docs/feature_selection/README.md`: 全体計画
- `docs/feature_selection/phase1_report.md`: Phase 1 結果
- `scripts/utils_msr.py`: MSR 評価関数
- `src/feature_selection/common/evaluate_baseline.py`: 既存の評価スクリプト
