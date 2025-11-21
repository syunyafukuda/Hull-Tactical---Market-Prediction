# 欠損構造特徴 SU3 仕様（遷移・再出現・代入影響トレース）

最終更新: 2025-11-21

## 概要

SU3は**欠損パターンの時間的変化と代入の影響**を捕捉する三次特徴量群です。SU1が欠損の「静的構造」、SU2が「統計的派生」を扱ったのに対し、SU3は以下を重視します:

1. **遷移検知**: NaN↔観測の切り替わりタイミング
2. **再出現パターン**: 欠損後の観測復帰までの間隔とその正規化
3. **代入影響トレース**: 前処理で補完された値の変化量と信頼度

### SU2の教訓を反映

SU2は935個の二次特徴量により過学習（LB 0.597 vs SU1 0.674）したため、SU3では以下を徹底:

- **特徴量数の厳格な制限**: 最大200列まで（SU1の368列の半分以下）
- **単純で解釈可能な特徴**: 複雑な統計量を避け、遷移フラグと単純集計のみ
- **スイープによる枝刈り**: 初期実装後、permutation importanceで下位50%を削除
- **正則化の強化**: LightGBMの`reg_alpha=0.1`, `reg_lambda=0.1`を標準設定

---

## 設計方針

### 1. SU1との関係

SU3はSU1の出力を**前提**とします:

- **入力**: `m/<col>`, `gap_ffill/<col>`, `run_na/<col>`, `run_obs/<col>`（SU1生成済み）
- **追加情報**: 前処理後の補完値（M/E/I/P/Sグループのimputer出力）
- **依存**: SU1 → SU3 の順で適用（SU2はスキップ）

### 2. パイプライン統合

```
生データ
  ↓
[SU1FeatureAugmenter]  # 欠損構造一次特徴（368列）
  ↓
[SU3FeatureAugmenter]  # 遷移・代入影響（最大200列）
  ↓
[MGroupImputer] → [EGroupImputer] → [IGroupImputer] → [PGroupImputer] → [SGroupImputer]
  ↓
[ColumnTransformer]  # スケーリング
  ↓
[LGBMRegressor]
```

### 3. リーク防止原則

- **未来参照禁止**: すべて時刻`t`以前の情報のみ使用
- **fold境界リセット**: CV時は各foldで状態を初期化（SU1と同様）
- **expanding統計**: burn-in期間（20日）を設けて初期の不安定を回避

---

## 生成する特徴（詳細仕様）

### カテゴリA: 遷移フラグ（Binary Indicators）

過去1時点との比較により、欠損状態の変化を0/1で記録します。

#### A-1. 基本遷移フラグ

各特徴量列`<col>`について:

- **`na_to_obs/<col>`** (uint8)
  - 定義: 前日NaN かつ 当日観測 → 1、それ以外 → 0
  - 計算: `(m[t-1]==1) & (m[t]==0)`
  - 初日処理: `t=0`では常に0（比較対象なし）
  
- **`obs_to_na/<col>`** (uint8)
  - 定義: 前日観測 かつ 当日NaN → 1、それ以外 → 0
  - 計算: `(m[t-1]==0) & (m[t]==1)`
  - 初日処理: `t=0`では常に0

#### A-2. 群集約遷移率

各グループ`<grp>`（M/E/I/P/S/V）について:

- **`trans_rate/<grp>`** (float32)
  - 定義: 当日、群内で遷移（0→1 or 1→0）した列の比率
  - 計算: `(sum(na_to_obs) + sum(obs_to_na)) / n_cols_grp`
  - 範囲: [0, 1]
  - 意図: 群全体の「不安定度」を測る

**特徴量数**: 94列×2 (na_to_obs, obs_to_na) + 6群×1 (trans_rate) = **194列**

> **削減方針**: 初期実装後、permutation importanceで下位50列を削除 → **144列**に圧縮

---

### カテゴリB: 再出現パターン（Reappearance Dynamics）

欠損が終わり観測が復帰するまでの時間パターンを捕捉します。

#### B-1. 再出現間隔

- **`reappear_gap/<col>`** (int16)
  - 定義: 直前の「観測→欠損→再観測」サイクルでのNaN継続日数
  - 計算:
    ```python
    if m[t] == 0 and m[t-1] == 1:  # 今日観測、昨日NaN（復帰点）
        reappear_gap[t] = run_na[t-1]  # 昨日までのNaN連続長
    else:
        reappear_gap[t] = 0
    ```
  - clip: ≤60
  - 意図: 「欠損が何日続いた後に復帰したか」を記録

#### B-2. 再出現位置の正規化

- **`pos_since_reappear/<col>`** (float32)
  - 定義: 直前の再出現からの経過日数を[0, 1]に正規化
  - 計算:
    ```python
    days_since_reappear = 0  # 初期値
    if m[t] == 0 and m[t-1] == 1:  # 復帰点
        days_since_reappear = 0
    elif m[t] == 0:  # 観測継続中
        days_since_reappear += 1
    else:  # NaN中
        days_since_reappear = 0  # リセット
    
    pos_since_reappear[t] = min(days_since_reappear / 60.0, 1.0)
    ```
  - 範囲: [0, 1]
  - 意図: 「復帰後どれくらい安定しているか」を測る

**特徴量数**: 94列×2 (reappear_gap, pos_since_reappear) = **188列**

> **削減方針**: 相関≥0.95のペアで片方削除 + permutation importance下位30% → **約120列**に圧縮

---

### カテゴリC: 代入影響トレース（Imputation Impact）

前処理で補完された値の「確からしさ」と変化量を追跡します。

#### C-1. 代入実施フラグ

- **`imp_used/<col>`** (uint8)
  - 定義: 当日が補完で埋まったか（元データがNaNだったか）
  - 計算: `m[t]`と同値（1=補完あり、0=元から観測）
  - 意図: どの日が「推定値」かを明示

#### C-2. 代入差分

前処理の各imputerが出力する補完値と、元の生データの差分を記録します。

- **`imp_delta/<col>`** (float32)
  - 定義: `x_imputed[t] - x_raw[t]`
  - 観測日: 0（補完なし）
  - 欠損日: imputerが生成した値 - NaN（NaNは0として扱う）
  - winsorize: ±99パーセンタイル（外れ値を抑制）
  - 意図: 「補完がどれだけ大きな値を入れたか」を測る

- **`imp_absdelta/<col>`** (float32)
  - 定義: `|imp_delta[t]|`
  - 意図: 方向を無視した変化の大きさ

#### C-3. 代入種別（One-Hot）

M/E/I/P/Sの各グループで採用されているimputerポリシーに応じて:

- **`imp_policy_<policy>/<col>`** (uint8)
  - policy候補: `ffill`, `bfill`, `missforest`, `ridge_stack`, `knn_k`, `mice`, `kalman_local_level`, `holiday_bridge`, `state_space_custom`
  - 定義: 当日がそのポリシーで補完されたか → 1、それ以外 → 0
  - 計算: `configs/preprocess.yaml`から各列のポリシーを読み取り、`imp_used[t]==1`の日にフラグを立てる
  - 制約: 1列につき1つのポリシーのみ（排他的）

**特徴量数**:
- `imp_used`: 94列
- `imp_delta`, `imp_absdelta`: 94列×2 = 188列
- `imp_policy_*`: 94列×9ポリシー = 846列（理論上最大）

> **問題**: 846列は過剰 → **削減策**:
> 1. 実際に使用されているポリシーのみ生成（M=ridge_stack, E=ridge_stack, I=ridge_stack, P=mice, S=missforest → 5種類）
> 2. グループ単位でOne-Hot化（列単位ではなく、`imp_policy_M=ridge_stack` のような6特徴）
> 
> **最終**: グループ単位One-Hot（6グループ×5ポリシー = 30列）+ `imp_delta/absdelta`（188列）= **218列**
> 
> **さらに削減**: `imp_policy`はグループ単位のみ（30列）、`imp_delta/absdelta`は上位50列のみ（importance順）→ **80列**

---

### カテゴリD: 欠損の曜日・月次パターン（Temporal Missingness Bias）

特定の曜日や月に欠損が多い場合、その傾向を捕捉します。

#### D-1. 曜日別欠損率（Expanding）

- **`dow_m_rate/<col>`** (float32)
  - 定義: 過去の同じ曜日でのNaN比率（expanding平均）
  - 計算:
    ```python
    # 曜日ごとにカウンタを保持
    dow = date_id % 7  # 0=月曜, 6=日曜
    dow_na_count[dow] += m[t]
    dow_total_count[dow] += 1
    dow_m_rate[t] = dow_na_count[dow] / dow_total_count[dow]
    ```
  - burn-in: 最低3サンプル（同曜日が3回以上出現するまでNaN）
  - 範囲: [0, 1]

#### D-2. 月次欠損率（Expanding）

- **`month_m_rate/<col>`** (float32)
  - 定義: 過去の同じ月でのNaN比率
  - 計算: 曜日別と同様、月（1-12）ごとにカウンタ
  - burn-in: 最低2サンプル

**特徴量数**: 94列×2 (dow_m_rate, month_m_rate) = **188列**

> **削減方針**: 標準偏差≤0.05の列（ほぼ一定）を除外 → **約100列**に圧縮

---

### カテゴリE: 祝日との交差（Holiday Bridge Interaction）

祝日前後で欠損パターンが変わる可能性を捕捉します。

#### E-1. 祝日×欠損フラグ

- **`holiday_bridge_m/<col>`** (uint8)
  - 定義: `holiday_bridge[t] * m[t]`（祝日ブリッジかつNaN → 1）
  - 意図: 祝日特有の欠損を識別

**特徴量数**: 94列 = **94列**

> **削減方針**: 祝日が少ない場合、ほぼ0になる列を除外 → **約50列**に圧縮

---

## 特徴量数の集計と最終調整

### 初期実装時の列数

| カテゴリ | 詳細 | 初期列数 | 削減後列数 |
|---------|------|----------|-----------|
| A. 遷移フラグ | na_to_obs, obs_to_na, trans_rate | 194 | 144 |
| B. 再出現パターン | reappear_gap, pos_since_reappear | 188 | 120 |
| C. 代入影響 | imp_used, imp_delta, imp_absdelta, imp_policy | 282 | 80 |
| D. 曜日・月次 | dow_m_rate, month_m_rate | 188 | 100 |
| E. 祝日交差 | holiday_bridge_m | 94 | 50 |
| **合計** | | **946** | **494** |

> **問題**: 削減後も494列は多すぎる（SU2の935列に近い）

### 最終削減戦略

1. **Phase 1: 実装時削減**
   - グループ単位集約（trans_rate, imp_policy）
   - 定数列・低分散列の除外（σ≤0.05）
   - 相関≥0.95のペア削減
   - → **300列**に圧縮

2. **Phase 2: 学習後削減**
   - Permutation importance計測
   - 下位50%の特徴を削除
   - SHAP値で寄与度確認
   - → **150-200列**に圧縮

### 推奨実装順序

**Stage 1: ミニマル実装（100列以下）**
- A. 遷移フラグ（群集約のみ、6列）
- B. 再出現パターン（上位20列のみ、importance事前推定）
- C. 代入影響（グループ単位imp_policy 30列 + delta上位20列）
- D. 曜日・月次（上位20列のみ）
- E. 祝日交差（上位20列のみ）
- **合計: 約96列**

**Stage 2: 拡張（200列まで）**
- importanceで有効と判明した特徴を列単位で追加
- 最大200列でキャップ

---

## 実装ガイドライン

### 1. ディレクトリ構成

```
src/feature_generation/su3/
    __init__.py
    feature_su3.py          # SU3FeatureGenerator（sklearn Transformer互換）
    train_su3.py            # SU1 → SU3 → 前処理 → 学習 → バンドル出力
    predict_su3.py          # バンドルによる推論
    sweep_oof.py            # OOFスイープ（提出なし）

tests/feature_generation/
    test_su3.py             # 単体テスト

configs/feature_generation.yaml  # SU3設定追加

artifacts/SU3/
    inference_bundle.pkl    # 前処理+SU1+SU3+スケール+モデル
    model_meta.json
    feature_list.json
    cv_fold_logs.csv
    oof_grid_results.csv
    submission.csv
```

### 2. `feature_su3.py` のクラス設計

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class SU3Config:
    """SU3特徴生成の設定"""
    # 基本設定
    id_column: str = "date_id"
    output_prefix: str = "su3"
    
    # 遷移フラグ
    include_transitions: bool = True
    transition_group_agg: bool = True  # 群集約のみ
    
    # 再出現パターン
    include_reappearance: bool = True
    reappear_clip: int = 60
    reappear_top_k: int = 20  # importance上位K列のみ
    
    # 代入影響
    include_imputation_trace: bool = True
    imp_delta_winsorize_p: float = 0.99
    imp_delta_top_k: int = 20
    imp_policy_group_level: bool = True  # グループ単位One-Hot
    
    # 曜日・月次
    include_temporal_bias: bool = True
    temporal_burn_in: int = 3  # 最低サンプル数
    temporal_top_k: int = 20
    
    # 祝日交差
    include_holiday_interaction: bool = True
    holiday_top_k: int = 20
    
    # データ型
    dtype_flag: str = "uint8"
    dtype_int: str = "int16"
    dtype_float: str = "float32"
    
    # fold境界リセット
    reset_each_fold: bool = True

class SU3FeatureGenerator(BaseEstimator, TransformerMixin):
    """SU3特徴量生成器
    
    SU1の出力（m/<col>, gap_ffill/<col>, run_na/<col>, run_obs/<col>）
    を入力として、遷移・再出現・代入影響・曜日月次・祝日交差を生成する。
    """
    
    def __init__(self, config: SU3Config):
        self.config = config
    
    def fit(self, X: pd.DataFrame, y=None) -> "SU3FeatureGenerator":
        """特徴名の抽出とメタデータの保存"""
        # SU1特徴列を識別
        self.m_columns_ = [c for c in X.columns if c.startswith("m/")]
        self.gap_ffill_columns_ = [c for c in X.columns if c.startswith("gap_ffill/")]
        self.run_na_columns_ = [c for c in X.columns if c.startswith("run_na/")]
        self.run_obs_columns_ = [c for c in X.columns if c.startswith("run_obs/")]
        
        # グループマッピング
        self.groups_ = self._extract_groups(X.columns)
        
        # 特徴名リストを生成
        self.feature_names_ = self._generate_feature_names()
        
        return self
    
    def transform(self, X: pd.DataFrame, fold_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """SU3特徴を生成
        
        Args:
            X: SU1特徴を含むDataFrame
            fold_indices: CV用のfoldインデックス（fold境界でリセット）
        
        Returns:
            SU3特徴のDataFrame
        """
        # fold境界の準備
        fold_boundaries = self._compute_fold_boundaries(len(X), fold_indices)
        
        features = {}
        
        # A. 遷移フラグ
        if self.config.include_transitions:
            trans_features = self._generate_transition_features(X, fold_boundaries)
            features.update(trans_features)
        
        # B. 再出現パターン
        if self.config.include_reappearance:
            reappear_features = self._generate_reappearance_features(X, fold_boundaries)
            features.update(reappear_features)
        
        # C. 代入影響（※前処理後の値が必要なため、別途処理）
        # → train_su3.pyでパイプライン統合時に追加
        
        # D. 曜日・月次
        if self.config.include_temporal_bias:
            temporal_features = self._generate_temporal_features(X, fold_boundaries)
            features.update(temporal_features)
        
        # E. 祝日交差
        if self.config.include_holiday_interaction:
            holiday_features = self._generate_holiday_features(X, fold_boundaries)
            features.update(holiday_features)
        
        return pd.DataFrame(features, index=X.index)
    
    def _generate_transition_features(self, X: pd.DataFrame, fold_boundaries: List[tuple]) -> Dict[str, np.ndarray]:
        """遷移フラグの生成"""
        features = {}
        
        if self.config.transition_group_agg:
            # 群集約のみ
            for grp in self.groups_:
                grp_cols = [c for c in self.m_columns_ if self._get_group(c) == grp]
                trans_rate = self._compute_group_trans_rate(X, grp_cols, fold_boundaries)
                features[f"{self.config.output_prefix}/trans_rate/{grp}"] = trans_rate
        else:
            # 列単位（初期検証用）
            for col in self.m_columns_:
                base_col = col[2:]  # "m/" を除去
                na_to_obs, obs_to_na = self._compute_transitions(X[col].values, fold_boundaries)
                features[f"{self.config.output_prefix}/na_to_obs/{base_col}"] = na_to_obs
                features[f"{self.config.output_prefix}/obs_to_na/{base_col}"] = obs_to_na
        
        return features
    
    def _compute_transitions(self, m_values: np.ndarray, fold_boundaries: List[tuple]) -> tuple:
        """1列の遷移フラグを計算"""
        n = len(m_values)
        na_to_obs = np.zeros(n, dtype=self.config.dtype_flag)
        obs_to_na = np.zeros(n, dtype=self.config.dtype_flag)
        
        for start_idx, end_idx in fold_boundaries:
            for i in range(start_idx + 1, end_idx):
                prev_val = m_values[i - 1]
                curr_val = m_values[i]
                
                if prev_val == 1 and curr_val == 0:  # NaN → 観測
                    na_to_obs[i] = 1
                elif prev_val == 0 and curr_val == 1:  # 観測 → NaN
                    obs_to_na[i] = 1
        
        return na_to_obs, obs_to_na
    
    def _compute_group_trans_rate(self, X: pd.DataFrame, grp_cols: List[str], fold_boundaries: List[tuple]) -> np.ndarray:
        """群内遷移率を計算"""
        n = len(X)
        trans_rate = np.zeros(n, dtype=self.config.dtype_float)
        
        if not grp_cols:
            return trans_rate
        
        for start_idx, end_idx in fold_boundaries:
            for i in range(start_idx + 1, end_idx):
                trans_count = 0
                for col in grp_cols:
                    prev_val = X[col].iloc[i - 1]
                    curr_val = X[col].iloc[i]
                    if prev_val != curr_val:
                        trans_count += 1
                trans_rate[i] = trans_count / len(grp_cols)
        
        return trans_rate
    
    # ... 他のメソッド省略（実装詳細は同様のパターン）
```

### 3. `train_su3.py` のパイプライン構成

```python
def build_pipeline(
    su1_config: SU1Config,
    su3_config: SU3Config,
    preprocess_settings: Dict[str, Any],
    *,
    numeric_fill_value: float,
    model_kwargs: Dict[str, Any],
    random_state: int,
) -> Pipeline:
    """SU1 → SU3 → 前処理 → モデルのパイプライン構築"""
    
    # SU1特徴生成器
    su1_augmenter = SU1FeatureAugmenter(su1_config, fill_value=numeric_fill_value)
    
    # SU3特徴生成器
    su3_augmenter = SU3FeatureAugmenter(su1_config, su3_config, fill_value=numeric_fill_value)
    
    # 各グループのimputer
    m_imputer = MGroupImputer(...)
    e_imputer = EGroupImputer(...)
    i_imputer = IGroupImputer(...)
    p_imputer = PGroupImputer(...)
    s_imputer = SGroupImputer(...)
    
    # 前処理（スケーリング）
    preprocess = _build_preprocess(numeric_fill_value)
    
    # モデル
    model = LGBMRegressor(
        **model_kwargs,
        reg_alpha=0.1,  # SU3では正則化を強化
        reg_lambda=0.1,
    )
    
    steps = [
        ("su1_augment", su1_augmenter),
        ("su3_augment", su3_augmenter),  # SU1後に追加
        ("m_imputer", m_imputer),
        ("e_imputer", e_imputer),
        ("i_imputer", i_imputer),
        ("p_imputer", p_imputer),
        ("s_imputer", s_imputer),
        ("preprocess", preprocess),
        ("model", model),
    ]
    
    return Pipeline(steps=steps)
```

### 4. テストケース（`test_su3.py`）

```python
def test_su3_transition_flags():
    """遷移フラグの基本動作確認"""
    # 入力: m/<col>が [0, 0, 1, 1, 0, 0] の場合
    # na_to_obs: [0, 0, 0, 0, 1, 0]（4番目で NaN→観測）
    # obs_to_na: [0, 0, 1, 0, 0, 0]（2番目で 観測→NaN）
    ...

def test_su3_reappear_gap():
    """再出現間隔の計算"""
    # run_na が [0, 1, 2, 3, 0, 0] の場合
    # reappear_gap: [0, 0, 0, 0, 3, 0]（4番目で復帰、その時のrun_na[3]=3）
    ...

def test_su3_fold_reset():
    """fold境界でのリセット確認"""
    # fold_indices が [0, 0, 0, 1, 1] の場合
    # 3番目と4番目の間で状態がリセットされる
    ...

def test_su3_all_nan_column():
    """全NaN列の扱い"""
    # 全NaN列は遷移なし、再出現なしで0埋め
    ...

def test_su3_output_shape():
    """出力特徴量数の確認"""
    # Stage 1実装で約100列、Stage 2で最大200列
    ...
```

---

## スイープ戦略

### Phase 1: ミニマル構成の検証

**目的**: SU3の基本有効性を確認

**構成**:
```yaml
su3:
  enabled: true
  include_transitions: true
  transition_group_agg: true  # 群集約のみ（6列）
  include_reappearance: true
  reappear_top_k: 20
  include_imputation_trace: false  # 初期はオフ
  include_temporal_bias: true
  temporal_top_k: 20
  include_holiday_interaction: true
  holiday_top_k: 20
```

**期待特徴量数**: 約96列

**評価指標**:
- OOF RMSE（SU1比で±0.0005以内なら許容）
- OOF MSR（SU1比で+0.001以上なら採用候補）
- 予測分散（SU1比で±10%以内）

**閾値**:
- ✅ 採用: OOF MSR +0.001以上 かつ 分散非悪化
- ⚠️ 保留: OOF MSR ±0.0005（LB待ち）
- ❌ 却下: OOF MSR -0.001以下 または 分散+20%以上

### Phase 2: 特徴追加とアブレーション

**追加候補**:
1. 代入影響トレース（imp_delta, imp_policy）
2. 列単位遷移フラグ（重要度上位30列）
3. 曜日・月次の拡張（標準偏差≥0.1の列のみ）

**スイープパラメータ**:
```yaml
su3_sweep:
  reappear_top_k: [10, 20, 30, 50]
  imp_delta_top_k: [0, 10, 20, 50]  # 0=オフ
  temporal_top_k: [10, 20, 30]
  holiday_top_k: [10, 20, 30, 50]
```

**実行**:
```bash
uv run python src/feature_generation/su3/sweep_oof.py \
    --data-dir data/raw \
    --config-path configs/feature_generation.yaml \
    --preprocess-config configs/preprocess.yaml \
    --n-splits 5 \
    --gap 0 \
    --sweep-mode grid
```

**出力**:
- `results/ablation/SU3/sweep_yyyy-mm-dd.csv`
- 各構成のOOF RMSE, MSR, 特徴量数, 学習時間

### Phase 3: 最終選定とKaggle提出

**選定基準**:
1. OOF MSRが上位3構成
2. 特徴量数≤200列
3. SU1比でMSR +0.001以上

**提出前チェック**:
- [ ] numpy==1.26.4で再学習
- [ ] ローカル推論で特徴量数確認
- [ ] `model_meta.json`に全設定記録
- [ ] `feature_list.json`に列名一覧

**Kaggle提出**:
- 1日最大2提出
- LB劣化閾値: SU1比で-0.002以下なら即リバート

---

## 成功基準と採用判断

### 必須条件（すべて満たす必要あり）

1. **OOF性能**: SU1比でMSR +0.001以上
2. **特徴量数**: 最大200列（SU1の368列より少ない）
3. **LB性能**: SU1の0.674から-0.002以内（0.672以上）
4. **計算コスト**: 学習時間がSU1比で+50%以内

### 推奨条件（望ましい）

1. **解釈可能性**: 特徴の意味が明確（遷移、再出現、代入影響）
2. **ロバストネス**: CV fold間でのMSR分散が小さい
3. **PSI**: train後期 vs test で PSI≤0.3

### 採用シナリオ

#### ✅ シナリオA: SU1単独継続

- SU3がすべての必須条件を満たせない場合
- SU1（LB 0.674）をベースラインとして維持
- SU3は参考実装として`enabled: false`

#### ✅ シナリオB: SU1+SU3統合

- SU3がすべての必須条件を満たした場合
- パイプライン: 生データ → SU1 → SU3 → 前処理 → モデル
- `configs/feature_generation.yaml`で`su3.enabled: true`

#### ❌ シナリオC: SU3単独（非推奨）

- SU3はSU1を前提とするため、SU3単独は不可
- SU1なしでのSU3実装は行わない

---

## 実装チェックリスト

### Phase 1: 基本実装（1-2日）

- [ ] `src/feature_generation/su3/__init__.py`作成
- [ ] `src/feature_generation/su3/feature_su3.py`作成
  - [ ] `SU3Config`クラス
  - [ ] `SU3FeatureGenerator`クラス
  - [ ] 遷移フラグ生成（群集約のみ）
  - [ ] 再出現パターン生成（top-k）
  - [ ] 曜日・月次パターン生成（top-k）
  - [ ] 祝日交差生成（top-k）
- [ ] `configs/feature_generation.yaml`にSU3セクション追加
- [ ] `tests/feature_generation/test_su3.py`作成
  - [ ] 遷移フラグのテスト
  - [ ] 再出現間隔のテスト
  - [ ] fold境界リセットのテスト
  - [ ] 全NaN列のテスト
- [ ] 品質チェック（Ruff, Pyright, Pytest）

### Phase 2: パイプライン統合（1日）

- [ ] `src/feature_generation/su3/train_su3.py`作成
  - [ ] `SU3FeatureAugmenter`クラス（SU1+SU3統合）
  - [ ] `build_pipeline`関数
  - [ ] CV実装（TimeSeriesSplit + fold_indices）
  - [ ] バンドル保存（`artifacts/SU3/inference_bundle.pkl`）
- [ ] `src/feature_generation/su3/predict_su3.py`作成
  - [ ] バンドルロード
  - [ ] 推論実行
  - [ ] submission生成
- [ ] ローカル学習・推論テスト
- [ ] 特徴量数確認（目標: 96列）

### Phase 3: スイープ実装（1日）

- [ ] `src/feature_generation/su3/sweep_oof.py`作成
  - [ ] パラメータグリッド定義
  - [ ] OOF評価ループ
  - [ ] 結果CSV出力
- [ ] スイープ実行（n_splits=5, gap=0）
- [ ] 結果分析（OOF MSR, 特徴量数, 学習時間）
- [ ] 最良構成の選定

### Phase 4: Kaggle提出（1日）

- [ ] numpy==1.26.4で再学習
- [ ] `artifacts/SU3/`に全成果物生成
- [ ] `model_meta.json`確認
- [ ] `feature_list.json`確認
- [ ] Kaggleノートブック作成（`notebooks/submit/su3.ipynb`）
  - [ ] 全クラス定義（SU1, SU3, Imputers）
  - [ ] sys.modules登録
  - [ ] バンドルロード
  - [ ] 推論実行
- [ ] ローカルでノートブック検証
- [ ] Kaggle Private Datasetアップロード
- [ ] Kaggle提出（インターネットOFF）
- [ ] LBスコア確認

### Phase 5: 採用判断とドキュメント（1日）

- [ ] LBスコア評価（SU1比）
- [ ] 採用/非採用の決定
- [ ] `docs/submissions.md`更新
- [ ] `docs/feature_generation/SU3.md`更新（本ファイル）
- [ ] `configs/feature_generation.yaml`最終調整
- [ ] PR作成（テンプレート使用）
- [ ] 品質チェック最終確認

---

## トラブルシューティング

### 問題1: 特徴量数が200列を超える

**原因**: top-kの設定が大きすぎる

**対策**:
1. `reappear_top_k`, `temporal_top_k`, `holiday_top_k`を10に削減
2. `transition_group_agg: true`を維持（列単位にしない）
3. Permutation importanceで下位を削除

### 問題2: OOF MSRがSU1より悪化

**原因**: 過学習または不要な特徴の追加

**対策**:
1. 正則化を強化（`reg_alpha=0.2`, `reg_lambda=0.2`）
2. `feature_fraction=0.8`に削減
3. 特徴量数を50列以下に制限
4. 代入影響トレースをオフ（`include_imputation_trace: false`）

### 問題3: LBスコアがOOFより大幅に悪い

**原因**: 時系列分割への過適合（SU2と同じ問題）

**対策**:
1. fold境界リセットを確認（`reset_each_fold: true`）
2. expanding統計のburn-inを増やす（20 → 30）
3. 遷移フラグのみに絞る（他をオフ）
4. SU3を非採用とし、SU1継続

### 問題4: 学習時間が長すぎる

**原因**: 特徴生成の計算量が大きい

**対策**:
1. top-kを削減（50 → 20 → 10）
2. 曜日・月次をオフ
3. 祝日交差をオフ
4. JIT編集（numbaなど）を検討

---

## 参考情報

### SU1との比較

| 項目 | SU1 | SU3 |
|------|-----|-----|
| 入力 | 生データ | SU1出力 |
| 特徴タイプ | 静的構造 | 動的変化 |
| 特徴量数 | 368列 | 96-200列 |
| LBスコア | 0.674 | TBD |
| 実装複雑度 | 中 | 高 |

### SU2の教訓

| 問題 | SU2での発生 | SU3での対策 |
|------|------------|-----------|
| 特徴量爆発 | 935列 | 最大200列に制限 |
| 過学習 | OOF良好、LB悪化 | 正則化強化 + 枝刈り |
| 時系列過適合 | fold境界最適化 | ミニマル構成で検証 |
| 複雑な統計 | rolling/EWMA多数 | 単純な遷移フラグ優先 |

### 関連ドキュメント

- [SU1仕様](./SU1.md) - 欠損構造一次特徴
- [SU2仕様](./SU2.md) - 非採用の経緯と分析
- [特徴量生成ロードマップ](./README.md) - 全体戦略
- [提出履歴](../submissions.md) - LBスコア一覧

---
