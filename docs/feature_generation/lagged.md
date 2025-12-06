# Lagged Features 仕様書

最終更新: 2025-12-06

---

## 0. ステータス

| 項目 | 状態 |
|------|------|
| 実装状況 | 🚧 **実装中** |
| ベースライン | SU1+SU5+Brushup (LB 0.681, OOF RMSE 0.012134, 577列) |
| 目標 | +3〜5列の最小構成で検証 |
| リスク評価 | 低（列数が少なく、SU7の轍を踏まない設計） |

---

## 1. 背景と目的

### 1.1 lagged_* カラムとは

Kaggle公式の説明より:

> **Test set only.**
> - `lagged_forward_returns` – The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.
> - `lagged_risk_free_rate` – The risk-free rate with a lag of one day.
> - `lagged_market_forward_excess_returns` – The forward excess return with a lag of one day.

つまり、ある日 t の予測を行うとき、**t-1 日の確定情報**として:
- `lagged_forward_returns` = 前日の forward_returns
- `lagged_risk_free_rate` = 前日の risk_free_rate
- `lagged_market_forward_excess_returns` = 前日の market_forward_excess_returns

が test.csv に提供されている。

### 1.2 リークではない理由

- 未来（t+1 以降）の情報は含まない
- t-1 時点で既に確定している過去情報
- 公式が「使ってよい特徴量」として明示的に提供

### 1.3 train/test での違い

| データ | 状況 |
|--------|------|
| **test.csv** | `lagged_*` カラムが存在（公式提供） |
| **train.csv** | `lagged_*` は存在しない → **自前で再現が必要** |

再現方法:
```python
train_df["lagged_forward_returns"] = train_df["forward_returns"].shift(1)
train_df["lagged_risk_free_rate"] = train_df["risk_free_rate"].shift(1)
train_df["lagged_market_forward_excess_returns"] = train_df["market_forward_excess_returns"].shift(1)
```

### 1.4 これまでのSUとの関係

| SU | 列数 | 結果 | lagged_*との違い |
|----|------|------|-----------------|
| SU7 (モメンタム) | 72〜120列 | LB 0.476 ❌ | 多ラグ×多集約で爆発 |
| SU8 (ボラ) | 20〜30列 | LB 0.624 ❌ | 間接的にリターン使用 |
| SU10 (外部) | 14列 | LB 0.597 ❌ | 外部リターン使用 |
| **lagged** | **3〜5列** | ? | **ターゲット直近値のみ、最小構成** |

**SU7との決定的な違い**:
- SU7: 補完後の特徴行列から派生 → 間接的
- lagged: ターゲット関連値の直近1日 → 直接的、かつ最小

---

## 2. 設計方針

### 2.1 基本原則

```
✅ やること
- lagged_* を 3列だけそのまま追加
- オプションで sign/abs 派生を +2列
- 合計 +3〜5列に厳密に抑える
- ON/OFF を設定で簡単に切り替え

❌ やらないこと
- 2ラグ以上 (k=2,3,5,10...) → SU7の轍
- rolling統計 (window=5,20...) → SU7の轍
- 複雑な組み合わせ → SU2/SU7の轍
- 新しいSUディレクトリを作成 → 既存SU5に組み込み
```

### 2.2 追加する特徴量

#### 基本構成（3列）

| 特徴名 | 定義 | 型 |
|--------|-----|-----|
| `lagged_forward_returns` | 前日の forward_returns | float32 |
| `lagged_risk_free_rate` | 前日の risk_free_rate | float32 |
| `lagged_market_forward_excess_returns` | 前日の market_forward_excess_returns | float32 |

#### 拡張構成（オプション、+2列）

| 特徴名 | 定義 | 型 | 備考 |
|--------|-----|-----|------|
| `sign_lagged_fwd_excess` | sign(lagged_market_forward_excess_returns) | int8 | {-1, 0, 1} |
| `abs_lagged_fwd_excess` | abs(lagged_market_forward_excess_returns) | float32 | ショック検知 |

### 2.3 列数まとめ

| 構成 | 追加列数 | 合計列数 |
|------|---------|---------|
| SU5+Brushup (現行) | 0 | 577列 |
| + lagged基本 | +3 | 580列 |
| + lagged拡張 | +5 | 582列 |

---

## 3. 実装計画

### 3.1 ファイル構成

```
src/feature_generation/
  su5/
    feature_su5.py     # lagged特徴生成メソッドを追加
    train_su5.py       # train側でlagged_*を再現
    predict_su5.py     # test側はそのまま使用

configs/
  feature_generation.yaml  # lagged_features セクション追加
```

### 3.2 設定パラメータ

```yaml
su5:
  # ... 既存設定 ...
  
  lagged_features:
    enabled: true
    columns:
      - lagged_forward_returns
      - lagged_risk_free_rate
      - lagged_market_forward_excess_returns
    source_columns:  # train側で再現する際のソース
      lagged_forward_returns: forward_returns
      lagged_risk_free_rate: risk_free_rate
      lagged_market_forward_excess_returns: market_forward_excess_returns
    include_sign: true   # sign_lagged_fwd_excess を追加
    include_abs: false   # abs_lagged_fwd_excess は最初はOFF
```

### 3.3 実装詳細

#### train側での再現ロジック

```python
def _generate_lagged_features(
    self, 
    df: pd.DataFrame, 
    is_train: bool = True
) -> pd.DataFrame:
    """lagged特徴を生成（trainでは再現、testではそのまま使用）"""
    result = df.copy()
    
    if is_train:
        # train側: shift(1)で再現
        for lagged_col, source_col in self.config.lagged_source_columns.items():
            if source_col in df.columns:
                result[lagged_col] = df[source_col].shift(1)
    else:
        # test側: 既存のlagged_*をそのまま使用
        pass
    
    # オプション: sign/abs派生
    if self.config.lagged_include_sign:
        excess_col = "lagged_market_forward_excess_returns"
        if excess_col in result.columns:
            result["sign_lagged_fwd_excess"] = np.sign(result[excess_col]).astype("int8")
    
    if self.config.lagged_include_abs:
        excess_col = "lagged_market_forward_excess_returns"
        if excess_col in result.columns:
            result["abs_lagged_fwd_excess"] = np.abs(result[excess_col]).astype("float32")
    
    return result
```

#### 欠損処理

- train側: 先頭1行は shift(1) で NaN になる
- 既存の GroupImputer / 欠損補完で処理（新規ロジック不要）

---

## 4. タスクチェックリスト

### Sprint 1: 実装

- [ ] `src/feature_generation/su5/feature_su5.py` に lagged 特徴生成メソッド追加
- [ ] `src/feature_generation/su5/train_su5.py` に train 側での lagged 再現ロジック追加
- [ ] `configs/feature_generation.yaml` に lagged_features セクション追加
- [ ] `tests/feature_generation/test_su5.py` に lagged テスト追加

### Sprint 2: 評価

- [ ] OOF評価実行（SU5+Brushup+lagged）
- [ ] ベースライン比較
  - SU5+Brushup: OOF RMSE 0.012134
  - SU5+Brushup+lagged: ?
- [ ] LB提出（OOF改善時のみ）
- [ ] 採否判断・ドキュメント更新

---

## 5. 評価基準

### 5.1 採用条件

| 指標 | 条件 |
|------|------|
| OOF RMSE | ベースライン (0.012134) と同等以下 |
| LB Score | ベースライン (0.681) 以上 |

### 5.2 非採用条件（即座にロールバック）

- OOF RMSE が +1% 以上悪化
- LB Score が -0.005 以上悪化

### 5.3 期待値

- **楽観シナリオ**: OOF/LB ともに微改善（+0.001〜0.003）
- **現実的シナリオ**: OOF微改善、LB維持（±0）
- **悲観シナリオ**: 効果なし or 微悪化 → lagged.enabled=false で即座に無効化

---

## 6. リスクと対策

| リスク | 対策 |
|--------|------|
| 自己相関が弱く効果なし | 3列だけなので損失も最小 |
| SU7と同じ轍 | 列数を厳密に抑制（+5列以下） |
| train/test不整合 | shift(1)再現を必ず実装 |
| 先頭行のNaN | 既存の欠損補完で対応 |

---

## 7. 参考

- [submissions.md](../submissions.md) - SU7/SU8/SU9/SU10の失敗履歴
- [SU7.md](./SU7.md) - モメンタム系の失敗分析
- [SU1_5_brushup.md](./SU1_5_brushup.md) - 列数抑制アプローチの成功例
- Kaggle公式: https://www.kaggle.com/competitions/hull-tactical-market-prediction/data
