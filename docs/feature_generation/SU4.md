# 欠損構造特徴 SU4（代入影響トレース・Imputation Trace）

最終更新: 2025-11-23

## ⚠️ 重要: SU4削除決定（2025-11-23）

**Status**: ❌ **実装完了後、削除決定** - 予測性能への寄与がほぼゼロと判明

### 削除決定の経緯

#### 実装完了状況（削除前）
- ✅ `feature_su4.py`: 完全実装（539行）
- ✅ `SU4Config`, `SU4FeatureGenerator`, `SU4FeatureAugmenter`: 実装完了
- ✅ `train_su4.py`: OOF学習パイプライン（1157行）
- ✅ Unit tests: 7テストケース全通過、85%カバレッジ
- ✅ `configs/feature_generation.yaml`: su4セクション（enabled=false に変更済み）
- ✅ OOF評価完了: RMSE=0.012141, MSR=0.023319、138特徴生成

#### 削除決定の3つの根拠

**1. 特徴重要度分析結果**（決定的証拠）
- **138個のSU4特徴のうち136個が重要度0**
- わずか2個（`imp_method/missforest`, `imp_method/ridge_stack`）のみ重要度1
- SU4合計重要度: 2.00（全体の0.0%）、非SU4: 37,198.00（100.0%）
- SU4平均重要度: 0.01 vs 非SU4平均: 65.60
- **結論**: LightGBMがSU4特徴をほぼ使用していない

**2. ハイパーパラメータスイープ結果**
- 18設定（top_k_imp_delta × top_k_holiday_cross × winsor_p）で**完全に同一のRMSE/MSR**
- 全設定でOOF RMSE=0.012141, MSR=0.023319
- **結論**: SU4パラメータが予測性能に一切影響していない

**3. Ablation Study結果**
- **SU4なし**（SU1+SU5+GroupImputersのみ）: OOF RMSE **0.012284**
- **SU4あり**（baseline）: OOF RMSE **0.012141**
- 差分: **+0.000143** (+1.2%、統計的に有意でない誤差範囲内)
- **結論**: SU4削除の性能への影響はほぼゼロ

#### コンセプトの問題点

**なぜSU4が機能しなかったか**:
1. **補完トレース情報がtargetと無相関**: 欠損値がどう補完されたかは、市場リターン予測に寄与しない
2. **情報の冗長性**: 既存特徴（元データ+SU1+SU5）で予測に必要な情報は網羅されている
3. **特徴設計の問題**: `imp_delta`などが補完率の低い箇所では0になり、分散が小さく情報量が少ない

**他のデータセットでも同様の問題が予想される**: 補完トレース情報が予測に寄与するユースケースは限定的

#### 削除によるメリット

- 138特徴分の計算コスト削減
- 訓練時間短縮（SU4生成・補完トレース計算の削減）
- メモリ使用量削減（2.1GB削減）
- コードベース簡略化・保守性向上

#### 今後の方向性

**標準パイプライン**（SU4削除後）:
```
入力データ（94列）
  ↓
SU1FeatureAugmenter（+368列） → 462列
  ↓
SU5FeatureAugmenter（+105列） → 567列
  ↓
GroupImputers（M/E/I/P/S欠損値補完）
  ↓
Preprocessing（StandardScaler）
  ↓
LightGBMRegressor
```

**期待される性能**:
- OOF RMSE: 0.01228程度（SU4ありの0.01214から+0.00014）
- LB score: SU5の0.681と同等またはわずかに低下（推定0.680程度）

#### 学んだ教訓

1. **特徴重要度分析を早期に実施すべき**: スイープ前に実施していれば無駄な計算を避けられた
2. **Ablation Studyは基本**: 新特徴追加時は常に「なし」バージョンとの比較が必要
3. **コンセプトの妥当性検証**: 実装前に「なぜこの特徴がtargetと相関するか」の仮説を立て、簡易検証すべき
4. **LightGBMの特徴選択能力を信頼**: 重要度0の特徴が多い場合、設計を見直す

#### 参照ドキュメント
- **特徴重要度分析結果**: `results/ablation/SU4/feature_importance_analysis.csv`
- **スイープ結果**: `results/ablation/SU4/sweep_summary.csv`
- **Ablation実行スクリプト**: `src/feature_generation/su4/ablation_no_su4.py`
- **設定ファイル**: `configs/feature_generation.yaml` (su4.enabled=false)
- **提出履歴**: `docs/submissions.md` (SU4エントリ参照)

---

## 以下は参考: SU4の当初設計仕様（実装済み、削除決定）

この仕様書は、SU5（共欠損構造）のパターンを踏襲しつつ、「代入影響（imputation trace）」を系統的に扱うための設計ドキュメントです。実装は完了しましたが、予測性能への寄与がゼロのため削除が決定されました。

---

## 1. SU4 の目的と位置づけ

### 1.1 SU シリーズの中での SU4

- **SU1**: 単列レベルの欠損構造（`m/<col>`, `gap_ffill/<col>`, `run_na/<col>` など）
- **SU2**: 欠損率のローリング統計・履歴率（→ 過学習により非採用、LB 0.597）
- **SU3**: 欠損の遷移・再出現パターン（→ OOF/LB 劣化により完全放棄、LB 0.461）
- **SU4**: **代入影響トレース（imputation trace）** — 「欠損補完によってどれだけ値が変わったか」を特徴量化
- **SU5**: 共欠損（co-miss）構造（→ 正式採用、LB 0.681）

SU4 は、これまでの SU1-3 が「欠損そのもののパターン」を扱ってきたのに対し、**前処理（欠損補完）の副作用**を明示的に特徴化するユニットです。

### 1.2 コンペ全体の文脈での狙い

#### 前処理パイプラインの現状

本プロジェクトでは、M/E/I/P/S/V 各グループに対して異なる欠損補完ポリシーを採用しています：

| グループ | 採用ポリシー | LB Score | 特徴 |
|---------|------------|----------|------|
| M | ridge_stack | 0.629 | 多変量回帰ベース補完 |
| E | ridge_stack | - | 同上 |
| I | ridge_stack | 0.623 | 同上 |
| P | mice | 0.625 | 多重代入法（反復的補完） |
| S | missforest | 0.616 | ランダムフォレストベース補完 |
| V | ffill_bfill | 0.590 (非採用) | 時系列補完 |

これらの補完手法は、欠損値を「推定値」で埋めますが、その推定の「確からしさ」や「元の値からのズレ」は特徴として明示されていません。

#### SU4 の狙い

- **補完の副作用を可視化**: `imp_delta/<col> = x_imputed - x_raw`
- **補完実施の有無を明示**: `imp_used/<col> = 1` if 当日補完された
- **補完手法の情報を付与**: 代入種別 One-hot（`ffill`, `mice`, `missforest` 等）
- **欠損パターンとの交差**: `holiday_bridge * m/<col>` で祝日起因の欠損を強調

**評価方針**:

- SU5（LB 0.681）が現在のベストスコア
- SU4 で **+0.003-0.008 の改善**を目標（期待値: +0.005）
- SU2/SU3 の失敗（特徴量爆発、過学習）を回避するため、**列数制御を厳格化**
  - `imp_delta` は上位 K 列のみ（K=20-30）
  - 代入種別 One-hot は主要ポリシーのみ（5-6 種類）
- OOF MSR で **+0.5σ** 以上、または LB で明確な改善が見える場合のみ採用

---

## 2. 設計方針

### 2.1 入出力と依存関係

#### 入力

1. **生データ** (`raw_data`):
   - 欠損補完前の生データ（`train.csv`, `test.csv`）
   - M/E/I/P/S/V 列に NaN が含まれる状態

2. **補完済みデータ** (`imputed_data`):
   - 前処理パイプライン（M/E/I/P/S GroupImputer）を通過後のデータ
   - 欠損が埋められた状態

3. **SU1 特徴** (オプション):
   - `m/<col>`: 欠損フラグ（SU4 の `imp_used` と整合性チェック用）

4. **補完メタ情報** (オプション):
   - 各グループのポリシー名（`configs/preprocess.yaml` から取得）
   - カレンダー情報（holiday_bridge 使用時）

#### 出力（SU4 特徴の例）

| 特徴カテゴリ | 列名例 | 型 | 範囲 | 説明 |
|------------|--------|-----|------|------|
| 代入実施フラグ | `imp_used/<col>` | uint8 | {0, 1} | 当日補完されたか |
| 代入差分 | `imp_delta/<col>` | float32 | [-∞, +∞] → winsorize | 補完値 - 生値 |
| 代入絶対差分 | `imp_absdelta/<col>` | float32 | [0, +∞] → winsorize | \|imp_delta\| |
| 代入種別 One-hot | `imp_method/ffill` | uint8 | {0, 1} | ffill で補完された行 |
| 代入種別 One-hot | `imp_method/mice` | uint8 | {0, 1} | mice で補完された行 |
| 代入種別 One-hot | `imp_method/missforest` | uint8 | {0, 1} | missforest で補完 |
| 代入種別 One-hot | `imp_method/ridge_stack` | uint8 | {0, 1} | ridge_stack で補完 |
| 代入種別 One-hot | `imp_method/holiday_bridge` | uint8 | {0, 1} | holiday_bridge 使用 |
| 交差特徴 | `holiday_bridge_x_m/<col>` | uint8 | {0, 1} | holiday_bridge かつ欠損 |

#### 依存関係

- **SU1 を前提とする**: `m/<col>` と `imp_used/<col>` の整合性チェックに使用
- **前処理パイプラインに依存**: M/E/I/P/S GroupImputer の出力が必要
- **モデル本体には依存しない**: 特徴生成のみ

### 2.2 列数制御と top-k 方針

SU2 の失敗（935 列による過学習）を踏まえ、**列数を厳格に制御**します：

#### A. imp_delta/imp_absdelta の列数制限

1. **補完頻度による選択**:
   ```python
   # 学習期間での補完実施率を計算
   imputation_rate = (raw_data.isna() & ~imputed_data.isna()).mean()
   
   # 補完頻度が高い列のみ選択（top_k）
   top_k_cols = imputation_rate.nlargest(top_k_imp_delta).index
   ```

2. **推奨値**: `top_k_imp_delta = 20-30`
   - SU2 の 935 列と比較して 1/30 に削減
   - SU5 の 105 列と同等規模

#### B. imp_method One-hot の制限

- **主要ポリシーのみ**: 5-6 種類に限定
  - `ffill`, `mice`, `missforest`, `ridge_stack`, `holiday_bridge`, `other`
- **行レベル One-hot**: グループ単位で 1 つのみ active
  - 例: M グループは `ridge_stack=1`, 他は 0

#### C. 交差特徴の制限

- `holiday_bridge_x_m/<col>` は **top_k_holiday_cross** 列のみ
- 推奨値: `top_k_holiday_cross = 10`

**合計特徴量数の見積もり**:
- `imp_used`: ~85 列（M/E/I/P/S の全列）
- `imp_delta`: ~25 列（top_k_imp_delta=25）
- `imp_absdelta`: ~25 列
- `imp_method`: 6 列（One-hot）
- `holiday_bridge_x_m`: ~10 列
- **合計**: ~151 列 ← SU5 の 105 列より多いが、SU2 の 935 列よりは大幅削減

### 2.3 リーク防止と CV 一貫性

#### 時系列リークの防止

1. **補完は時刻 t のみを参照**:
   - `imp_delta[t] = imputed_data[t] - raw_data[t]`
   - 未来情報は使用しない

2. **fold 境界でのリセット**:
   - SU2/SU5 と同様に `fold_indices` を受け取る
   - validation 区間のみに fold 番号を振る（train は 0）
   - ローリング統計（後述）を fold 境界でリセット

#### fold_indices の運用仕様（SU2/SU5 との整合）

```python
fold_indices_full = np.zeros(len(X), dtype=int)
for fold_idx, (_, val_idx) in enumerate(splitter.split(X)):
    fold_indices_full[val_idx] = fold_idx + 1
```

- **意図**: validation 区間の境界でリセット、train 側は連続履歴

### 2.4 Winsorization（外れ値処理）

`imp_delta/<col>` は補完手法によって極端な値を取る可能性があります：

```python
# ±p99 でクリップ（configs/feature_generation.yaml で設定）
winsor_p = 0.99  # デフォルト
lower_bound = np.percentile(imp_delta, (1 - winsor_p) * 100)
upper_bound = np.percentile(imp_delta, winsor_p * 100)
imp_delta_clipped = np.clip(imp_delta, lower_bound, upper_bound)
```

- **理由**: mice や missforest は外れ値を生成しやすい
- **設定可能**: `winsor_p` は YAML で調整可能（0.95, 0.99, 0.999）

---

## 3. 生成する特徴の詳細仕様

### カテゴリ A: 代入実施フラグ

#### A-1. 単列代入フラグ `imp_used/<col>`

- **対象**: M/E/I/P/S グループの全列（V は非採用のため除外）
- **定義**:
  ```python
  imp_used[col][t] = 1 if (raw_data[col][t] is NaN and imputed_data[col][t] is not NaN)
                     else 0
  ```
- **型**: `uint8`
- **範囲**: {0, 1}
- **意図**: 「その日、この列が補完されたか」を直接示す

#### A-2. 群内代入カウント `imp_cnt_day/<grp>`（オプション）

- **対象**: M/E/I/P/S の 5 グループ
- **定義**:
  ```python
  imp_cnt_day[grp][t] = sum(imp_used[col][t] for col in group[grp])
  ```
- **型**: `int16`
- **範囲**: [0, 群内列数]
- **意図**: 「その日、そのグループで何列補完されたか」

#### A-3. 全体代入率 `imp_rate_day/ALL`（オプション）

- **定義**:
  ```python
  imp_rate_day_all[t] = imp_cnt_day_all[t] / total_columns
  ```
- **型**: `float32`
- **範囲**: [0.0, 1.0]

### カテゴリ B: 代入差分・絶対差分

#### B-1. 代入差分 `imp_delta/<col>`

- **対象**: top_k_imp_delta 列（補完頻度上位）
- **定義**:
  ```python
  imp_delta[col][t] = imputed_data[col][t] - raw_data[col][t]
                      if imp_used[col][t] == 1
                      else 0.0
  ```
- **型**: `float32`
- **範囲**: winsorize 後 `[p1, p99]`
- **意図**: 「補完によってどれだけ値が変わったか」

#### B-2. 代入絶対差分 `imp_absdelta/<col>`

- **対象**: top_k_imp_delta 列
- **定義**:
  ```python
  imp_absdelta[col][t] = |imp_delta[col][t]|
  ```
- **型**: `float32`
- **範囲**: winsorize 後 `[0, p99]`

### カテゴリ C: 代入種別 One-hot

#### C-1. 代入手法フラグ `imp_method/<method>`

- **対象**: 主要 5-6 手法
  - `ffill`: 前方埋め（時系列補完の基本）
  - `mice`: 多重代入法（P グループ）
  - `missforest`: RF ベース補完（S グループ）
  - `ridge_stack`: 多変量回帰（M/E/I グループ）
  - `holiday_bridge`: カレンダー補完
  - `other`: 上記以外
- **定義**:
  ```python
  # 行レベルで「どの手法が使われたか」を記録
  # 各グループの採用ポリシーに基づく
  imp_method[method][t] = 1 if row t で method が使用された
                          else 0
  ```
- **型**: `uint8`
- **範囲**: {0, 1}
- **排他性**: 各行で最大 1 つのみ 1（複数グループがあるため厳密には非排他）

#### C-2. 実装ノート: グループ別ポリシーの取得

```python
# configs/preprocess.yaml から読み込み
group_policies = {
    'M': 'ridge_stack',
    'E': 'ridge_stack',
    'I': 'ridge_stack',
    'P': 'mice',
    'S': 'missforest',
    # V は enabled=false のため除外
}

# 各列のグループを推定（SU1 と同じロジック）
col_to_method = {}
for col in columns:
    group = _infer_group(col)  # "M", "E", "I", "P", "S"
    if group in group_policies:
        col_to_method[col] = group_policies[group]
```

### カテゴリ D: 交差特徴（holiday_bridge 限定）

#### D-1. holiday_bridge × 欠損フラグ `holiday_bridge_x_m/<col>`

- **対象**: top_k_holiday_cross 列（holiday_bridge が効いている列）
- **定義**:
  ```python
  holiday_bridge_x_m[col][t] = 1 if (
      imp_method['holiday_bridge'][t] == 1 and
      m[col][t] == 1  # SU1 の欠損フラグ
  ) else 0
  ```
- **型**: `uint8`
- **範囲**: {0, 1}
- **意図**: 「祝日関連で欠損していた列」を強調

---

## 4. パイプライン統合

### 4.1 データフロー

```
生データ (raw_data)
    ↓
[SU1FeatureGenerator]  ← 欠損構造層（生データの欠損パターン）
    ↓ m/<col>, gap_ffill/<col>, ...
    ↓
[SU5FeatureGenerator]  ← 共欠損構造層（m/<col>のみ使用、補完前）
    ↓ co_miss_now/<a>__<b>, co_miss_rollrate_5/<a>__<b>, ...
    ↓
[MGroupImputer, EGroupImputer, ..., SGroupImputer]  ← 補完層
    ↓ imputed_data
    ↓
[SU4FeatureGenerator]  ← 補完副作用層（raw vs imputed の差分）
    ↓ imp_used/<col>, imp_delta/<col>, imp_method/<method>, ...
    ↓
[ColumnTransformer + Ridge]  ← 前処理+モデル層
```

**重要**: SU5はSU1の`m/<col>`のみを使用し、GroupImputersやSU4の出力には依存しません。
したがって、SU5はGroupImputersの前に配置するのが役割分離的に自然です。

### 4.2 sklearn Pipeline への統合

```python
from sklearn.pipeline import Pipeline
from src.feature_generation.su1.feature_su1 import SU1FeatureAugmenter
from src.feature_generation.su4.feature_su4 import SU4FeatureAugmenter
from src.feature_generation.su5.feature_su5 import SU5FeatureAugmenter

pipeline = Pipeline([
    # 欠損パターン層（生データの欠損構造）
    ('su1', SU1FeatureAugmenter(su1_config)),
    ('su5', SU5FeatureAugmenter(su5_config)),  # m/<col>のみ使用
    
    # 補完層（欠損を埋める）
    ('m_imputer', MGroupImputer(...)),
    ('e_imputer', EGroupImputer(...)),
    ('i_imputer', IGroupImputer(...)),
    ('p_imputer', PGroupImputer(...)),
    ('s_imputer', SGroupImputer(...)),
    
    # 補完副作用層（raw vs imputed）
    ('su4', SU4FeatureAugmenter(su4_config, raw_data)),
    
    # 前処理+モデル層
    ('preprocess', ColumnTransformer(...)),
    ('model', LGBMRegressor(...))
])
```

**注意**: SU5は補完前（GroupImputersの前）に配置します。SU5は`m/<col>`（SU1の欠損フラグ）
のみを使用し、補完後のデータには依存しないため、役割分離的にこの位置が適切です。

### 4.3 SU4FeatureAugmenter の役割

- **入力**: 
  - `X`: 補完済みデータ（M/E/I/P/S GroupImputer を通過後）
  - `raw_data_`: 補完前の生データ（fit 時に保存）
- **出力**: 
  - `X_with_su4`: X + SU4 特徴（imp_used, imp_delta, ...）

```python
class SU4FeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, config: SU4Config, raw_data: pd.DataFrame):
        self.config = config
        self.raw_data_ = raw_data  # 補完前データを保持
        self.generator_ = SU4FeatureGenerator(config)
    
    def fit(self, X, y=None):
        # top-k 列の選定などを実施
        self.generator_.fit(self.raw_data_, X)
        return self
    
    def transform(self, X):
        su4_features = self.generator_.transform(self.raw_data_, X)
        return pd.concat([X, su4_features], axis=1)
```

---

## 5. 設定ファイル仕様（`configs/feature_generation.yaml`）

```yaml
su4:
  enabled: true  # SU4 を有効化
  base_features: su1  # SU1 を前提とする
  id_column: date_id
  output_prefix: su4
  
  # 列数制御パラメータ
  top_k_imp_delta: 25  # imp_delta を生成する列数
  top_k_holiday_cross: 10  # holiday_bridge 交差の列数
  
  # Winsorization
  winsor_p: 0.99  # ±p99 でクリップ
  
  # 代入手法リスト（主要ポリシーのみ）
  imp_methods:
    - ffill
    - mice
    - missforest
    - ridge_stack
    - holiday_bridge
    - other
  
  # fold リセット
  reset_each_fold: true
  
  # データ型
  dtype:
    flag: uint8
    int: int16
    float: float32
  
  # メタデータ
  metadata:
    artifacts_dir: artifacts/SU4
    depends_on: [SU1, M/E/I/P/S GroupImputers]
    expected_usage: "Imputation trace features"
    numpy_version: 1.26.4
    status: development
    target_feature_count: ~151  # imp_used(85) + delta(25) + absdelta(25) + method(6) + cross(10)
```

---

## 6. 実装タスク分解

### フェーズ 1: コア機能実装（2-3 日）

#### タスク 1.1: SU4Config と load_su4_config
- `SU4Config` dataclass 実装
- `load_su4_config()` 関数（YAML 読み込み）
- 設定バリデーション

#### タスク 1.2: SU4FeatureGenerator（基本）
- `imp_used/<col>` の生成
- `imp_delta/<col>` の生成（winsorize 含む）
- `imp_absdelta/<col>` の生成
- top-k 列選択ロジック

#### タスク 1.3: SU4FeatureGenerator（One-hot）
- グループ→ポリシーマッピング（`configs/preprocess.yaml` 読込）
- `imp_method/<method>` の生成
- 排他性チェック（デバッグ用）

#### タスク 1.4: SU4FeatureGenerator（交差特徴）
- `holiday_bridge_x_m/<col>` の生成
- SU1 の `m/<col>` との結合

### フェーズ 2: パイプライン統合（1-2 日）

#### タスク 2.1: SU4FeatureAugmenter
- sklearn Transformer 実装
- `raw_data_` の保持ロジック
- fit/transform メソッド

#### タスク 2.2: train_su4.py
- SU1 + GroupImputers + SU4 + Ridge の Pipeline 構築
- OOF 学習ループ
- `inference_bundle.pkl` 出力

#### タスク 2.3: predict_su4.py
- バンドル読込・推論
- `submission.csv` 出力

### フェーズ 3: テストとスイープ（1-2 日）

#### タスク 3.1: tests/feature_generation/test_su4.py
- `test_su4_config_loading`: YAML 読込
- `test_su4_imp_used_generation`: imp_used フラグ
- `test_su4_imp_delta_winsorize`: winsorize 動作
- `test_su4_imp_method_onehot`: One-hot 排他性
- `test_su4_holiday_cross`: 交差特徴
- `test_su4_output_shape`: 列数チェック
- `test_su4_dtype`: データ型チェック

#### タスク 3.2: sweep_oof.py
- ハイパーパラメータグリッド:
  - `top_k_imp_delta`: [20, 25, 30]
  - `winsor_p`: [0.95, 0.99]
  - `top_k_holiday_cross`: [5, 10, 15]
- OOF RMSE/MSR 計算
- 結果を `results/ablation/SU4/sweep_summary.csv` に出力

### フェーズ 4: LB 検証（1 日）

#### タスク 4.1: Kaggle Notebook 作成
- `notebooks/submit/su4.ipynb` 作成（su5.ipynb ベース）
- SU4 セクション追加
- artifact 自動検出

#### タスク 4.2: LB 提出・判断
- ベスト構成で学習
- Kaggle 提出
- LB スコア記録（目標: 0.684-0.689）

---

## 7. テストシナリオ

### 7.1 単体テストケース

#### ケース 1: 全列補完なし
- 入力: `raw_data = imputed_data`（NaN なし）
- 期待: `imp_used` 全て 0, `imp_delta` 全て 0

#### ケース 2: 特定列のみ補完
- 入力: `raw_data['M1']` に NaN, `imputed_data['M1']` は埋まっている
- 期待: `imp_used/M1 = 1`, `imp_delta/M1 != 0`

#### ケース 3: winsorize 動作
- 入力: `imp_delta` に極端な値（±10σ）
- 期待: ±p99 でクリップされる

#### ケース 4: imp_method One-hot
- 入力: M グループは `ridge_stack`, P グループは `mice`
- 期待: 
  - M 列の行で `imp_method/ridge_stack = 1`
  - P 列の行で `imp_method/mice = 1`

#### ケース 5: holiday_bridge 交差
- 入力: 
  - `imp_method/holiday_bridge = 1`
  - `m/M1 = 1`（SU1 の欠損フラグ）
- 期待: `holiday_bridge_x_m/M1 = 1`

### 7.2 統合テストケース

#### ケース 6: Pipeline 通過
- SU1 → GroupImputers → SU4 → SU5 の順で実行
- 各ステップで列数増加を確認
- 最終出力が Ridge に投入可能か確認

#### ケース 7: fold_indices リセット
- `reset_each_fold=True` で validation 境界をまたぐ
- ローリング統計（後述）がリセットされるか確認

---

## 8. パフォーマンス最適化

### 8.1 メモリ削減

- `imp_used/<col>`: `uint8` で 1/4 サイズ
- `imp_delta/<col>`: `float32` で 1/2 サイズ（`float64` と比較）
- top-k 選択で列数を 85 → 25 に削減

### 8.2 計算コスト

- `imp_delta` 計算: O(N × K) （N=行数, K=top_k 列数）
- winsorize: O(N × K) （パーセンタイル計算）
- imp_method 判定: O(N × グループ数) = O(N × 5)

**見積もり**:
- 学習データ: N=1000 行, K=25 列 → 0.1 秒程度
- SU2 の 935 列と比較して 1/37 の計算量

---

## 9. 品質基準とリリース判断

### 9.1 実装品質

- ✅ Ruff（lint/format）通過
- ✅ Pyright（型チェック）通過
- ✅ Pytest 全テスト通過（7 テストケース以上）
- ✅ カバレッジ 80% 以上

### 9.2 OOF 性能基準

| 指標 | 基準 | 備考 |
|------|------|------|
| OOF RMSE | SU5 比 +0.5% 以内 | 0.01214 → 0.01220 以下 |
| OOF MSR | SU5 比 +0.5σ 以上 | 0.02407 → 0.02500 以上 |
| 特徴量数 | 180 列以下 | SU5 の 567 列 + SU4 の ~151 列 = 718 列以下 |

### 9.3 LB 採用基準

| LB Score | 判断 | アクション |
|----------|------|-----------|
| **0.684 以上** | ✅ 採用 | SU4 を正式採用、ドキュメント更新 |
| **0.681-0.683** | ⚠️ 保留 | SU5 単独と比較、追加スイープ検討 |
| **0.681 未満** | ❌ 非採用 | SU5 をベースラインとして継続 |

### 9.4 失敗時の対応

- **OOF 劣化**: top_k パラメータを削減（25 → 15）
- **LB 劣化**: SU2/SU3 と同様に非採用判断
- **特徴量爆発**: カテゴリ D（交差特徴）を削除

---

## 10. 参考資料

### 10.1 関連ドキュメント

- `docs/feature_generation/SU1.md`: 基本設計パターン
- `docs/feature_generation/SU5.md`: 最新の実装例
- `docs/preprocessing.md`: GroupImputer の仕様
- `configs/preprocess.yaml`: 採用ポリシー一覧

### 10.2 コード参照

- `src/feature_generation/su1/feature_su1.py`: SU1 実装
- `src/feature_generation/su5/feature_su5.py`: SU5 実装（テンプレート）
- `src/preprocess/M_group/m_group.py`: MGroupImputer 実装

### 10.3 テスト参照

- `tests/feature_generation/test_su5.py`: 7 テストケースの例

---

## 11. 実装チェックリスト

### コア実装
- [ ] `src/feature_generation/su4/__init__.py`
- [ ] `src/feature_generation/su4/feature_su4.py`
  - [ ] `SU4Config` dataclass
  - [ ] `load_su4_config()` 関数
  - [ ] `SU4FeatureGenerator` クラス
    - [ ] `_compute_imp_used()`
    - [ ] `_compute_imp_delta()`
    - [ ] `_compute_imp_method_onehot()`
    - [ ] `_compute_holiday_cross()`
    - [ ] `fit()` メソッド
    - [ ] `transform()` メソッド
  - [ ] `SU4FeatureAugmenter` クラス

### Pipeline
- [ ] `src/feature_generation/su4/train_su4.py`
- [ ] `src/feature_generation/su4/predict_su4.py`
- [ ] `src/feature_generation/su4/sweep_oof.py`

### 設定
- [ ] `configs/feature_generation.yaml` に `su4` セクション追加

### テスト
- [ ] `tests/feature_generation/test_su4.py`
  - [ ] `test_su4_config_loading`
  - [ ] `test_su4_imp_used_generation`
  - [ ] `test_su4_imp_delta_winsorize`
  - [ ] `test_su4_imp_method_onehot`
  - [ ] `test_su4_holiday_cross`
  - [ ] `test_su4_output_shape`
  - [ ] `test_su4_dtype`

### ドキュメント
- [x] `docs/feature_generation/SU4.md`（本仕様書）
- [ ] `docs/submissions.md` に SU4 結果追記（LB 検証後）
- [ ] `README.md` に SU4 セクション追加（採用時）

### 品質チェック
- [ ] `./scripts/check_quality.sh` 全通過
- [ ] カバレッジ 80% 以上
- [ ] Kaggle Notebook 作成（`notebooks/submit/su4.ipynb`）

---

## 12. 期待される成果

### 定量的目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| LB Score | **0.684-0.689** | SU5（0.681）から +0.003-0.008 |
| OOF RMSE | 0.01220 以下 | SU5（0.01214）から +0.5% 以内 |
| OOF MSR | 0.02500 以上 | SU5（0.02407）から +0.5σ |
| 特徴量数 | ~151 列 | imp_used(85) + delta(50) + method(6) + cross(10) |
| 実装期間 | 4-6 日 | フェーズ 1-4 の合計 |

### 定性的目標

- ✅ SU2/SU3 の失敗（特徴量爆発、過学習）を回避
- ✅ 補完の「副作用」を明示的に特徴化
- ✅ 前処理パイプラインとの整合性を保持
- ✅ 再現可能な実装（numpy 1.26.4, scikit-learn 1.7.2）

---

## 変更履歴

- 2025-11-23: 初版作成（SU5 の成功を踏まえた設計）
