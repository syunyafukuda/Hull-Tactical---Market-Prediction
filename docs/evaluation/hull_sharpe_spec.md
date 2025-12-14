# Hull Competition Sharpe × Walk-Forward CV 詳細仕様書

最終更新: 2025-12-14  
ステータス: **実装完了**

---

## 1. 目的

現行の FS_compact + LGBM ベースライン（LB 0.681）に対し、**公式 Hull Competition Sharpe** をローカルで再現し、
**Walk-Forward CV** で期間別の Sharpe 安定性を検証する評価基盤を構築する。

### 1.1 現行課題

| 課題 | 具体例 |
|------|--------|
| OOF と LB の乖離 | XGBoost: OOF -0.84% 改善 → LB -8.7% 悪化 |
| 期間依存の過学習 | 特定期間だけ良いモデルを検出できない |
| ペナルティ不明 | vol_penalty / ret_penalty が LB に影響している可能性 |

### 1.2 目標

1. **公式 Sharpe と同一ロジック**でローカル評価
2. **Walk-Forward 3-4 fold** で Sharpe の分布を確認
3. **ペナルティ内訳**を可視化し、改善方向を特定

---

## 2. 採用ベースライン（前提）

| 項目 | 値 | 備考 |
|------|-----|------|
| 特徴セット | FS_compact（116列） | `artifacts/tier3/excluded.json` で除外 |
| モデル | LightGBM | `src/models/lgbm/train_lgbm.py` |
| 提出 NB | `notebooks/submit/LGBM.ipynb` | Kaggle 用 |
| Artifacts | `artifacts/models/lgbm-artifacts/` | inference_bundle.pkl 等 |
| OOF RMSE | 0.012164 | TimeSeriesSplit 5-fold |
| LB Score | 0.681 | Public LB |

---

## 3. 公式 Hull Competition Sharpe 仕様

### 3.1 入力

| カラム | 型 | 説明 |
|--------|-----|------|
| `date_id` | int | 日付識別子 |
| `prediction` | float | 投資比率 [0, 2] |
| `forward_returns` | float | S&P500 の翌日リターン |
| `risk_free_rate` | float | 無リスク金利（Federal Funds Rate） |

### 3.2 計算ロジック

```python
# 定数
ANNUALIZATION = sqrt(252)
VOL_TOLERANCE = 0.20   # 市場ボラとの許容比率 ±20%

# 1. ポジションクリップ
position = clip(prediction, 0, 2)

# 2. 戦略リターン計算
#    無リスク資産 (1 - position) + リスク資産 (position)
strategy_returns = risk_free_rate * (1 - position) + position * forward_returns
strategy_excess = strategy_returns - risk_free_rate
market_excess = forward_returns - risk_free_rate

# 3. Raw Sharpe
mean_strategy = mean(strategy_excess)
std_strategy = std(strategy_excess)
raw_sharpe = ANNUALIZATION * mean_strategy / (std_strategy + 1e-12)

# 4. Volatility Penalty
vol_ratio = std(strategy_excess) / (std(market_excess) + 1e-12)
if vol_ratio > 1.2:
    vol_penalty = (vol_ratio - 1.2) * 100
elif vol_ratio < 0.8:
    vol_penalty = (0.8 - vol_ratio) * 100
else:
    vol_penalty = 0

# 5. Return Penalty
mean_market = mean(market_excess)
if mean_strategy < mean_market:
    return_penalty = (mean_market - mean_strategy) * 1000
else:
    return_penalty = 0

# 6. Final Score
final_score = raw_sharpe - vol_penalty - return_penalty
```

### 3.3 出力

| フィールド | 型 | 説明 |
|------------|-----|------|
| `final_score` | float | 最終評価スコア |
| `raw_sharpe` | float | ペナルティ前の Sharpe |
| `vol_ratio` | float | 戦略ボラ / 市場ボラ |
| `vol_penalty` | float | ボラティリティペナルティ |
| `return_penalty` | float | リターンペナルティ |
| `strategy_mean` | float | 戦略超過リターン平均 |
| `strategy_std` | float | 戦略超過リターン標準偏差 |
| `market_mean` | float | 市場超過リターン平均 |
| `market_std` | float | 市場ボラティリティ |

---

## 4. Walk-Forward CV 仕様

### 4.1 設計パラメータ

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `train_window` | 6000 | 学習日数 |
| `val_window` | 1000 | 検証日数 |
| `step` | 1000 | ウィンドウ移動量 |
| `min_folds` | 3 | 最小 fold 数 |

### 4.2 Fold 構成例（train 8,990 日）

```
Fold 1: train [0, 5999]      val [6000, 6999]   → 1000 days
Fold 2: train [0, 6999]      val [7000, 7999]   → 1000 days
Fold 3: train [0, 7999]      val [8000, 8989]   →  990 days
```

※ 上記は Expanding Window 形式。Rolling Window も設定で切り替え可能とする。

### 4.3 モード

| モード | 説明 |
|--------|------|
| `expanding` | train_start 固定、train_end のみ拡大 |
| `rolling` | train_start も step で移動 |

### 4.4 出力

各 fold で以下を出力:

| フィールド | 説明 |
|------------|------|
| `fold_idx` | Fold 番号 |
| `train_range` | (start_date_id, end_date_id) |
| `val_range` | (start_date_id, end_date_id) |
| `rmse` | 検証 RMSE |
| `hull_sharpe` | 検証 Hull Sharpe |
| `raw_sharpe` | ペナルティ前 Sharpe |
| `vol_ratio` | ボラ比 |
| `vol_penalty` | vol ペナルティ |
| `return_penalty` | ret ペナルティ |

集計:

| 指標 | 説明 |
|------|------|
| `mean_sharpe` | 全 fold の平均 |
| `min_sharpe` | 最悪 fold の値（安定性指標） |
| `std_sharpe` | Sharpe のばらつき |
| `mean_rmse` | 参考用 |

---

## 5. 実装仕様

### 5.1 ファイル構成

```
src/
├── metrics/
│   ├── __init__.py
│   └── hull_sharpe.py          ← NEW: 公式メトリック関数
├── models/
│   └── common/
│       ├── cv_utils.py         ← 既存
│       ├── walk_forward.py     ← NEW: Walk-Forward Splitter
│       └── signals.py          ← NEW: prediction→position マッピング

tests/
├── metrics/
│   └── test_hull_sharpe.py     ← NEW
└── models/
    └── test_walk_forward.py    ← NEW

configs/
└── evaluation/
    └── walk_forward.yaml       ← NEW: Walk-Forward 設定

artifacts/
└── models/
    └── lgbm-artifacts/
        ├── inference_bundle.pkl  ← 既存
        ├── oof_predictions.csv   ← 既存
        ├── walk_forward_folds.csv ← NEW
        └── hull_sharpe_summary.json ← NEW
```

### 5.2 モジュール仕様

#### 5.2.1 `src/metrics/hull_sharpe.py`

```python
def validate_prediction(prediction: np.ndarray) -> None:
    """prediction が [0, 2] 範囲内か検証"""
    ...

def compute_hull_sharpe(
    prediction: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
) -> dict:
    """
    Returns:
        {
            "final_score": float,
            "raw_sharpe": float,
            "vol_ratio": float,
            "vol_penalty": float,
            "return_penalty": float,
            "strategy_mean": float,
            "strategy_std": float,
            "market_mean": float,
            "market_std": float,
        }
    """
    ...

def hull_sharpe_score(prediction, forward_returns, risk_free_rate) -> float:
    """final_score のみ返すショートカット"""
    return compute_hull_sharpe(...)["final_score"]
```

#### 5.2.2 `src/models/common/walk_forward.py`

```python
@dataclass
class WalkForwardConfig:
    train_window: int = 6000
    val_window: int = 1000
    step: int = 1000
    mode: str = "expanding"  # "expanding" or "rolling"
    min_folds: int = 3

def make_walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    date_col: str = "date_id",
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """
    Returns:
        [(train_idx, val_idx, metadata), ...]
        metadata: {"train_range": (start, end), "val_range": (start, end)}
    """
    ...
```

#### 5.2.3 `src/models/common/signals.py`

```python
def map_to_position(
    y_pred: np.ndarray,
    mult: float = 1.0,
    offset: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 2.0,
) -> np.ndarray:
    """
    予測値を投資比率 [0, 2] に変換
    
    prediction = clip(y_pred * mult + offset, clip_min, clip_max)
    """
    ...
```

### 5.3 CLI 改修

`src/models/lgbm/train_lgbm.py` に以下を追加:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--cv-mode` | `timeseries_split` | `walk_forward` も選択可 |
| `--wf-train-window` | 6000 | Walk-Forward 学習窓 |
| `--wf-val-window` | 1000 | Walk-Forward 検証窓 |
| `--wf-step` | 1000 | Walk-Forward ステップ |
| `--wf-mode` | `expanding` | `rolling` も選択可 |
| `--eval-sharpe` | `false` | Hull Sharpe 評価を有効化 |

### 5.4 Artifact 出力

#### walk_forward_folds.csv

```csv
fold_idx,train_start,train_end,val_start,val_end,rmse,hull_sharpe,raw_sharpe,vol_ratio,vol_penalty,return_penalty
0,0,5999,6000,6999,0.0121,0.543,0.612,1.05,0.0,6.9
1,0,6999,7000,7999,0.0119,0.621,0.698,0.98,0.0,7.7
2,0,7999,8000,8989,0.0123,0.489,0.556,1.12,0.0,6.7
```

#### hull_sharpe_summary.json

```json
{
  "cv_mode": "walk_forward",
  "config": {
    "train_window": 6000,
    "val_window": 1000,
    "step": 1000,
    "mode": "expanding"
  },
  "n_folds": 3,
  "metrics": {
    "mean_sharpe": 0.551,
    "min_sharpe": 0.489,
    "max_sharpe": 0.621,
    "std_sharpe": 0.066,
    "mean_rmse": 0.0121,
    "mean_vol_ratio": 1.05,
    "mean_vol_penalty": 0.0,
    "mean_return_penalty": 7.1
  }
}
```

---

## 6. テスト仕様

### 6.1 単体テスト

| テスト | ファイル | 内容 |
|--------|----------|------|
| Sharpe 計算 | `tests/metrics/test_hull_sharpe.py` | 既知のサンプルで再現性確認 |
| Walk-Forward | `tests/models/test_walk_forward.py` | fold 生成、境界条件 |
| Signal 変換 | `tests/models/test_signals.py` | クリップ、マッピング |

### 6.2 統合テスト

| テスト | 内容 |
|--------|------|
| LGBM + Walk-Forward | `train.tail(2000)` で E2E 実行 |
| CV モード互換性 | `--cv-mode timeseries_split` で既存動作確認 |

### 6.3 サニティチェック

`scripts/debug/run_sharpe_sanity.py`:

1. `train[-1000:]` を擬似 LB として使用
2. LGBM で学習・予測
3. Hull Sharpe を計算
4. 正の Sharpe が出ることを確認

---

## 7. 実装フロー

### Phase 1: メトリック移植（Day 1）

1. `src/metrics/hull_sharpe.py` 作成
2. `tests/metrics/test_hull_sharpe.py` 作成
3. 公式 NB のサンプルで再現性確認

### Phase 2: Walk-Forward 実装（Day 1-2）

1. `src/models/common/walk_forward.py` 作成
2. `src/models/common/signals.py` 作成
3. 単体テスト作成・Pass

### Phase 3: CLI 統合（Day 2）

1. `train_lgbm.py` に `--cv-mode` 等を追加
2. Sharpe ロギング実装
3. Artifact 出力実装

### Phase 4: サニティ & ドキュメント（Day 2-3）

1. `run_sharpe_sanity.py` で動作確認
2. LGBM Walk-Forward 実行
3. README / docs 更新
4. コミット & プッシュ

---

## 8. リスク & 軽減策

| リスク | 軽減策 |
|--------|--------|
| ローカル Sharpe と LB が乖離 | 絶対値ではなく相対比較に使用 |
| Walk-Forward で計算コスト増 | fold 数を 3-4 に制限 |
| ペナルティ係数が公式と異なる | 公式 NB コードを 1:1 で移植 |
| 短期間で Sharpe が不安定 | val_window ≥ 500 を推奨 |

---

## 9. 期待される成果

1. **LGBM Walk-Forward Sharpe**: 0.5〜0.7 程度（LB 0.681 と同等オーダー）を確認
2. **XGBoost との比較**: Walk-Forward Sharpe でも LGBM > XGBoost を確認
3. **ペナルティ診断**: vol_penalty / return_penalty の内訳を可視化
4. **今後の指標**: ハイパラチューニング・アンサンブルの主指標として使用可能

---

## 10. 参照

- [hull_sharpe_walkforward_plan.md](hull_sharpe_walkforward_plan.md): 概要計画
- [README.md](README.md): Evaluation Phase 概要
- `scripts/utils_msr.py`: 既存 MSR ユーティリティ（参考）
- `src/models/common/cv_utils.py`: 既存 CV ユーティリティ
