# SU3 Stage 1 採用ポリシー

**採用日**: 2025-11-22  
**ブランチ**: `feat/miss-core-su3`  
**ステータス**: ✅ 採用決定、Kaggle提出準備中

---

## エグゼクティブサマリー

SU3 Stage 1を**採用**し、Kaggle提出を進める。

**採用理由**:
1. ✅ 特徴量数444列（目標500列以下を達成）
2. ✅ SU2の過学習問題を回避（top-k選択、群集約）
3. ✅ OOF MSR=0.005772（スイープベスト構成）
4. ✅ ミニマル設計（Stage 2は提出後判断）

**Stage 2（代入影響トレース）の扱い**:
- 実装は未完了（コメントアウト状態）
- Stage 1でLBスコア確認後に判断
- リスク: 特徴量660列超→SU2の二の舞懸念

---

## 1. 採用構成の詳細

### ベストパラメータ（スイープ結果）

```yaml
su3:
  enabled: true
  reappear_top_k: 20
  temporal_top_k: 10
  holiday_top_k: 10
  include_imputation_trace: false  # Stage 1
  
  # その他の設定（デフォルト）
  include_transitions: true
  transition_group_agg: true
  include_reappearance: true
  include_temporal_bias: true
  include_holiday_interaction: true
```

### OOF性能

| 指標 | 値 | 備考 |
|------|-----|------|
| **OOF RMSE** | 0.011107 | |
| **OOF MSR** | 0.005772 | Sharpe-like指標（大きいほど良い） |
| **特徴量数** | 444列 | SU1: 368列 + SU3: 76列 |
| **実行時間** | 12.0秒/構成 | |

### 特徴量内訳

| カテゴリ | 内容 | 列数 |
|---------|------|------|
| **SU1特徴** | 欠損構造一次特徴 | 368列 |
| **SU3 - A** | 遷移フラグ（群集約） | 6列 |
| **SU3 - B** | 再出現パターン（top-20） | ~40列 |
| **SU3 - D** | 時間的欠損傾向（top-10） | ~20列 |
| **SU3 - E** | 祝日関連欠損（top-10） | ~10列 |
| **合計** | | **444列** |

---

## 2. SU2の教訓をどう反映したか

### SU2の失敗分析

| 項目 | SU2の失敗 | SU3での対策 |
|------|----------|-----------|
| **特徴量数** | 1397列（爆発） | 444列（68%削減） |
| **設計思想** | 二次特徴の大量生成 | top-k選択で枝刈り |
| **遷移フラグ** | 94列×2=188列 | 群集約で6列に圧縮 |
| **過学習** | CV良好→LB劣化 | ミニマル構成、正則化強化 |
| **LBスコア** | 0.597（SU1比-0.077） | 目標: ≥0.672（非劣化） |

### SU3の改善ポイント

1. **top-k選択**
   - 再出現パターン: 全94列→top-20列
   - 時間的傾向: 全94列→top-10列
   - 祝日関連: 全94列→top-10列

2. **群集約**
   - 遷移フラグ: 列単位（188列）→群単位（6列）に圧縮
   - M/E/I/P/S/Vの6グループで集約

3. **Stage分離**
   - Stage 1: ミニマル構成（76列追加）
   - Stage 2: 代入影響（220列追加予定）→ Stage 1の結果次第

4. **正則化強化**
   - LightGBMの`reg_alpha`, `reg_lambda`を明示的に設定
   - 過学習を予防

---

## 3. スイープ結果の分析

### 全48構成の性能分布

```
MSR範囲: -0.009764 ～ 0.005772
RMSE範囲: 0.011107 ～ 0.011116
```

### パラメータ感度分析

#### reappear_top_k の影響（最重要）

| 値 | MSR傾向 | 判断 |
|----|---------|------|
| 10 | -0.004773 | 低い |
| **20** | **0.005772** | **✅ ベスト** |
| 30 | 0.003884 | 良好 |
| 50 | 0.003818 | 良好 |

**結論**: reappear_top_k=20が最高MSR

#### temporal_top_k の影響

| 値 | MSR傾向 | 判断 |
|----|---------|------|
| **10** | **最高** | **✅ ベスト** |
| 20 | 中程度 | 許容 |
| 30 | 低下 | 過学習の兆候 |

**結論**: temporal_top_k=10が安定、30で過学習

#### holiday_top_k の影響（影響小）

| 値 | MSR傾向 | 判断 |
|----|---------|------|
| **10** | **安定** | **✅ 採用** |
| 20 | 同等 | 許容 |
| 30 | 同等 | 許容 |
| 50 | 同等 | 許容 |

**結論**: holiday_top_k=10で十分

---

## 4. 採用判断基準

### 必須条件（すべて満たす）

- [x] **特徴量数 < 500列**: 444列（✅ 達成）
- [x] **SU2の過学習回避**: top-k選択、群集約実装（✅ 達成）
- [ ] **LB性能 ≥ 0.672**: SU1比で非劣化（⏳ 提出後確認）

### 推奨条件

- [x] **解釈可能性**: 遷移、再出現、時間傾向（✅ 明確）
- [x] **実装品質**: 19テスト合格、Ruff/Pyright合格（✅ 達成）
- [ ] **ロバストネス**: CV fold間の分散確認（⏳ 要分析）

---

## 5. Stage 2の判断基準

### Stage 2の内容

**追加される特徴**（カテゴリC: 代入影響トレース）:
- `imp_used/<col>`: 代入実施フラグ（94列）
- `imp_delta/<col>`: 補完値との差分（94列）
- `imp_absdelta/<col>`: 差分の絶対値（94列）
- `imp_policy_<grp>/<policy>`: 補完ポリシー（30-50列）
- **合計**: ~220-240列追加 → **総計660-684列**

### Stage 2実行の条件

**実行する場合**:
1. ✅ Stage 1のLBスコアがSU1比で非劣化（≥0.672）
2. ✅ Stage 1の実装が安定稼働
3. ✅ `_generate_imputation_features()`の実装完了（3-4時間必要）

**実行しない場合**:
1. ❌ Stage 1のLBスコアがSU1比で劣化（<0.672）
2. ❌ 特徴量数660列超でSU2と同じリスク
3. ❌ Stage 1で十分な性能が確認された

### Stage 2の判断タイムライン

```
Stage 1提出 → LBスコア確認（24時間以内）
  ↓
LB ≥ 0.672?
  ↓ YES
Stage 2実装検討
  - 実装時間: 3-4時間
  - スイープ時間: ~30分（96構成）
  - 判断: MSR +0.001以上 & 特徴量<700列 → 採用
  
  ↓ NO
Stage 1で確定、Stage 2は保留
```

---

## 6. 実装品質の確認

### テスト状況

```bash
pytest tests/feature_generation/su3/ -v
# 結果: 19/19 passed ✅
```

### 品質チェック

```bash
./scripts/check_quality.sh
# 結果: 
# - Ruff: PASSED ✅
# - Pyright: PASSED ✅
# - Pytest: 92/92 passed ✅
```

### トラブルシューティング履歴

**MSR=0問題（2025-11-22解決済み）**:
- 原因: `PostProcessParams(lo=1.0, hi=1.0)`でシグナルが定数化
- 修正: `lo=0.0, hi=2.0`に変更（デフォルト値準拠）
- 検証: 修正後のスイープでMSR正常計算確認
- 詳細: `docs/feature_generation/troubleshooting/MSR_zero_issue.md`

---

## 7. Kaggle提出の準備

### 成果物の生成

```bash
# 1. ベスト構成でモデル学習
python src/feature_generation/su3/train_su3.py \
  --config-path configs/feature_generation.yaml \
  --preprocess-config configs/preprocess.yaml \
  --data-dir data/raw \
  --output-dir artifacts/SU3

# 2. 成果物確認
ls artifacts/SU3/
# - inference_bundle.pkl    # 前処理+SU1+SU3+スケール+モデル
# - model_meta.json         # バージョン、fold境界、性能指標
# - feature_list.json       # 生成カラム一覧
# - cv_fold_logs.csv        # 各foldの学習ログ
# - oof_predictions.csv     # OOF予測値
# - submission.csv          # 提出ファイル
```

### 設定ファイルの更新

```yaml
# configs/feature_generation.yaml
su3:
  enabled: true  # ← false から true に変更
  reappear_top_k: 20
  temporal_top_k: 10
  holiday_top_k: 10
  include_imputation_trace: false
```

### Kaggleノートブックの作成

- 場所: `notebooks/submit/su3_stage1.ipynb`
- 内容:
  1. SU1/SU3クラス定義（sys.modulesに登録）
  2. `inference_bundle.pkl`のロード
  3. テストデータでの推論
  4. `DefaultInferenceServer`での提供
- numpy==1.26.4固定（互換性確保）

---

## 8. リスク管理

### 想定リスクと対策

| リスク | 確率 | 影響 | 対策 |
|-------|------|------|------|
| **LBスコア劣化** | 中 | 高 | SU1に戻す（即座にリバート） |
| **特徴量数超過** | 低 | 中 | Stage 1は444列で安全圏 |
| **過学習** | 中 | 高 | top-k選択、群集約で予防済み |
| **numpy互換性** | 低 | 高 | 1.26.4固定で予防済み |

### 撤退基準

以下の場合、SU3を非採用としてSU1に戻る:

1. **LBスコア < 0.672**（SU1の0.674から-0.002以上劣化）
2. **LBスコア < 0.670**（SU2の0.597は上回るが、SU1に劣る）
3. **推論エラー**（Kaggleノートブックでエラー発生）

### Stage 2への移行条件

以下の**すべて**を満たす場合のみStage 2を検討:

1. ✅ Stage 1のLBスコア ≥ 0.672
2. ✅ Stage 1が安定稼働（推論エラーなし）
3. ✅ Stage 2実装完了（3-4時間必要）
4. ✅ Stage 2スイープで MSR +0.001以上
5. ✅ 特徴量数 < 700列

---

## 9. 次のアクション

### 即座に実行（本日中）

- [ ] `configs/feature_generation.yaml`を更新（su3.enabled=true）
- [ ] `src/feature_generation/su3/train_su3.py`実行
- [ ] `artifacts/SU3/`の成果物確認
- [ ] `notebooks/submit/su3_stage1.ipynb`作成
- [ ] ローカルで推論テスト

### Kaggle提出準備（明日）

- [ ] Private Datasetにアップロード（inference_bundle.pkl + scikit-learn wheel）
- [ ] Kaggleノートブックで推論検証（インターネットOFF）
- [ ] Kaggle提出（1日最大2回）
- [ ] LBスコア確認（24時間以内）

### 結果に応じた判断（提出後）

#### LB ≥ 0.674（SU1と同等以上）
- ✅ SU3 Stage 1を正式採用
- ⏳ Stage 2の実装検討（時間があれば）
- 📝 `docs/submissions.md`更新（LBスコア記録）

#### 0.672 ≤ LB < 0.674（許容範囲）
- ✅ SU3 Stage 1を採用
- ⏸️ Stage 2は保留（リスク高）
- 📝 ドキュメント更新

#### LB < 0.672（非採用ライン）
- ❌ SU3を非採用
- 🔄 SU1に戻る
- 📝 `docs/submissions.md`に非採用理由を記録

---

## 10. ドキュメント参照

### SU3関連

- **仕様**: `docs/feature_generation/SU3.md`
- **実装**: `src/feature_generation/su3/`
- **テスト**: `tests/feature_generation/su3/`
- **スイープ結果**: `results/ablation/SU3/sweep_2025-11-22_110535.json`
- **トラブルシューティング**: `docs/feature_generation/troubleshooting/MSR_zero_issue.md`

### 全体戦略

- **ロードマップ**: `docs/feature_generation/README.md`
- **提出履歴**: `docs/submissions.md`
- **SU1仕様**: `docs/feature_generation/SU1.md`
- **SU2分析**: `docs/feature_generation/SU2.md`（非採用）

---

**最終更新**: 2025-11-22  
**次回レビュー**: Kaggle提出後（LBスコア確認時）  
**承認者**: Feature Generation Team
