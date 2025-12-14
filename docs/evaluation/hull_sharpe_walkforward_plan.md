# Hull Competition Sharpe × Walk-Forward CV 実装計画

最終更新: 2025-12-14  
作成者: Codex CLI (補佐)

本ドキュメントは、現在の FS_compact + LGBM/XGB モデルライン (`README.md:14`, `docs/ensemble/README.md:14`) に対し、**公式 Hull Competition Sharpe メトリックのローカル実装**と**walk-forward / multi-window 時系列 CV**を導入するためのタスク分解と実装フローをまとめたものです。実装担当者が着手できるよう、関連ファイル・依存関係・検証方針を橋渡しします。

---

## 1. 背景 & 要求

- 現状の CV は `TimeSeriesSplit` + RMSE/MSR のみ (`src/models/common/cv_utils.py:13`)。
- ディスカッションでは「Kaggle 公式 Sharpe ノートブックのコピー + walk-forward で Sharpe 安定性を見る」が推奨。
- 目的:
  1. **公式 Sharpe 実装をローカル化**し、モデル比較を LB 指標に近づける。
  2. **時系列 walk-forward / multi-window CV**で Sharpe の期間別安定性を確認し、RMSE 主導から移行。

---

## 2. 既存構成 (抜粋)

| 領域 | 位置 | 補足 |
|------|------|------|
| 主要 README | `README.md` | フェーズ進捗と現行ベースライン (`LGBM + FS_compact`) |
| モデル仕様 | `docs/models/README.md` | 各モデルの OOF/LB 実績・評価軸 |
| アンサンブル仕様 | `docs/ensemble/README.md` | 現状は RMSE/MSR ベースのため Sharpe 指標なし |
| CV ユーティリティ | `src/models/common/cv_utils.py` | RMSE と簡易 MSR のみ。walk-forward なし。 |
| 既存トレーニング CLI | `src/models/lgbm/train_lgbm.py` 他 | `TimeSeriesSplit` を前提にした実装 |

---

## 3. 実装タスク一覧

### 3.1 公式 Sharpe メトリック移植
1. **モジュール新設**: `src/metrics/hull_sharpe.py` (仮) に Kaggle `Hull Competition Sharpe` Notebook を 1:1 で移植。  
   - 主な関数: `validate_positions(prediction_series)`, `compute_strategy_returns(...)`, `compute_hull_sharpe(prediction, forward_returns, risk_free_rate)`。
   - 返却値: 最終スコア、raw Sharpe、volatility penalty、return penalty、補助統計（平均/標準偏差/ボラ比等）。
2. **依存整備**: Notebook で利用される `numpy`, `pandas` のみで完結するよう整理。外部ライブラリが必要なら `pyproject.toml` に追記。
3. **単体テスト**: `tests/metrics/test_hull_sharpe.py` を作成し、Notebook 付属のサンプル (公開 `train[-180:]` 等) で再現確認。将来の回帰テスト用にフィクスチャを保存。

### 3.2 walk-forward / multi-window CV ジェネレーター
1. **Splitter 実装**: `src/models/common/walk_forward.py` (新規) で `make_walk_forward_splits(df, train_size, test_size, step)` を提供。メタデータに `train_date_range` / `val_date_range` を含める。
2. **設定管理**: `configs/models/common.yaml` (新規 or 既存) へ walk-forward パラメータ（`train_window`, `val_window`, `step`, `min_folds`）を定義。各モデル config から参照できるようにする。
3. **互換レイヤー**: 既存の `CVConfig` (`src/models/common/cv_utils.py:20`) に walk-forward モードを指定するフラグを追加し、従来の `TimeSeriesSplit` と切り替え可能に。  
   - Option: `cv.mode ∈ {timeseries_split, walk_forward}`, `cv.walk_forward: {...}`。

### 3.3 Sharpe 評価ラッパ & ロギング
1. **評価ラッパ**: `src/models/common/metrics.py` (既存がなければ新設) で `evaluate_hull_metric(prediction, forward_returns, risk_free_rate, metadata)` を用意し、上記メトリック関数を呼び出す。
2. **fold ログ**: 各 fold で `raw_sharpe`, `vol_ratio`, `vol_penalty`, `ret_penalty`, `strategy_mean`, `strategy_std` を JSON/CSV 出力 (`artifacts/models/<model>/fold_metrics.csv`)。
3. **OOF 集計**: OOF 予測全体に対しても Sharpe を計算し、`artifacts/.../oof_metrics.json` に RMSE/Sharpe/penalty breakdown を保存。

### 3.4 トレーニング CLI への組み込み
1. **CLI オプション**: `src/models/lgbm/train_lgbm.py` 等に `--cv-mode`, `--walk-train-window`, `--walk-val-window`, `--walk-step` などを追加。  
   - 既定: 従来の `TimeSeriesSplit`。  
   - `walk_forward` 指定時は新 splitter と Sharpe ログを強制有効化。
2. **prediction→position マッピング**: 現在の MSR 計算では `signal = pred + 1.0` (`cv_utils.py:109`) だが、Sharpe 評価では `[0, 2]` クリップ済みポジションが必要。  
   - 共通関数 `map_signal_to_position(pred, clip=True)` を `src/models/common/signals.py` として切り出し、Sharpe 評価前に適用。
3. **Artifact 構成**: `artifacts/models/<model>/` 配下に以下を追加保存。  
   - `walk_forward_folds.json`: fold ごとの日付レンジ & スコア。  
   - `hull_sharpe_summary.json`: Sharpe + penalty breakdown (fold平均、min/max)。

### 3.5 E2E テスト & サニティチェック
1. **小規模サンプルでの再現**: `train.tail(1000)` を使った最小実行スクリプト (`scripts/debug/run_sharpe_sanity.py`) を用意し、CI で動かせる範囲でメトリック差異を検知。
2. **Regression Test**: Walk-forward オプションが OFF のとき従来 RMSE パイプラインと同じ結果を出すことを `tests/models/test_cv_mode_switch.py` で保証。
3. **ドキュメント更新**: `docs/models/README.md` と `README.md` の「評価指標」節に Sharpe / walk-forward 導入済みであることを追記。

---

## 4. 推奨フロー (実行順)

1. **Metric モジュール → 単体テスト**  
   - Notebook からコード移植 → テストで再現性確保。
2. **walk-forward Splitter 実装**  
   - 既存 `TimeSeriesSplit` とは独立にまずユーティリティを完成。
3. **CV/CLI 組み込み & ロギング**  
   - LGBM (最重要ライン) で動作確認 → 他モデルへ水平展開。
4. **Artifact & ドキュメント更新**  
   - ログ出力フォーマット確定 → README/ docs 反映。
5. **最終サニティ**  
   - walk-forward ON で 1 回、OFF で 1 回トレーニングを回し差分チェック。

---

## 5. 留意点・リスク

- **絶対値乖離**: ローカル Sharpe と LB で絶対値が一致しないのは仕様。ドキュメントやログには「相対評価用」と明記する。
- **計算コスト**: walk-forward で fold 数が増えると学習時間が線形に増加。`train_window` / `step` を config で調整できるようにする。
- **データリーク防止**: fold 切替時に `train_end < val_start` を保証し、`gap` が必要なら追加。`forward_returns` 以外に `risk_free_rate` 等の補助列を確実に渡す。
- **テストデータ依存**: Kaggle metric ノートブックには `submission.parquet` 前提の I/O が含まれる場合があるので、インターフェースをローカル配列ベースに書き換える。

---

## 6. 期待される成果物

1. `src/metrics/hull_sharpe.py` + テストコード  
2. `src/models/common/walk_forward.py` (splitter)  
3. 既存トレーニングスクリプトの `--cv-mode` 対応・Sharpe ログ出力  
4. アーティファクト & README/モデルドキュメント更新  
5. サンプル実行ログ (walk-forward ON) — 結果を `results/evaluation/walk_forward_baseline.json` などに保存

これらを揃えれば、Sharpe ベースのモデル選定・アンサンブル検証をローカルで反復可能になり、以降のカレンダー特徴追加やオンライン学習 PoC の評価軸として再利用できます。

---

## 7. GitHub Copilot 実行者向け実装メモ（公式情報の転記）

実行担当者は外部 URL を参照できないため、ここに **Hull Competition Sharpe** の公式ノートブック（`metric/hull-competition-sharpe`）から必要なロジックを転記しておきます。下記コード/式をそのまま `src/metrics/hull_sharpe.py` に写経すれば、LB と同一ロジックを再現できます。

### 7.1 評価式の定義

```python
import numpy as np
import pandas as pd

ANNUALIZATION = np.sqrt(252.0)
MAX_INVESTMENT = 2.0
MIN_INVESTMENT = 0.0
VOL_TOLERANCE = 0.20       # 市場ボラとの許容比率 ±20%
RETURN_FLOOR = 0.0         # 市場平均を下回るとペナルティ

def hull_competition_sharpe(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    df = solution.merge(submission, on="date_id", how="left", validate="one_to_one")
    prediction = df["prediction"].astype(float)

    if prediction.isna().any():
        raise ValueError("submission contains NaN predictions")
    if prediction.min() < MIN_INVESTMENT or prediction.max() > MAX_INVESTMENT:
        raise ValueError("prediction outside [0, 2] range")

    position = prediction.clip(MIN_INVESTMENT, MAX_INVESTMENT)
    rf = df["risk_free_rate"].astype(float)
    fwd = df["forward_returns"].astype(float)

    # 公式ノートブックの定義:
    # 戦略リターン = 無リスク資産 + (リスク資産 - 無リスク資産) * position
    strategy_returns = rf * (1.0 - position) + position * fwd
    strategy_excess = strategy_returns - rf
    market_excess = fwd - rf

    mean_strategy = strategy_excess.mean()
    std_strategy = strategy_excess.std(ddof=0)
    raw_sharpe = ANNUALIZATION * mean_strategy / (std_strategy + 1e-12)

    # ボラティリティペナルティ: 市場ボラとの比率が ±20% を超えた分だけ控除
    strategy_vol = strategy_excess.std(ddof=0)
    market_vol = market_excess.std(ddof=0)
    vol_ratio = strategy_vol / (market_vol + 1e-12)
    upper = 1.0 + VOL_TOLERANCE
    lower = 1.0 - VOL_TOLERANCE

    if vol_ratio > upper:
        vol_penalty = (vol_ratio - upper) * 100.0
    elif vol_ratio < lower:
        vol_penalty = (lower - vol_ratio) * 100.0
    else:
        vol_penalty = 0.0

    # リターンペナルティ: 戦略の平均超過リターンが市場を下回った分だけ控除
    mean_market = market_excess.mean()
    if mean_strategy < mean_market + RETURN_FLOOR:
        return_penalty = (mean_market - mean_strategy) * 1000.0
    else:
        return_penalty = 0.0

    final_score = raw_sharpe - vol_penalty - return_penalty
    return float(final_score)
```

> **備考**: 係数 (`100.0`, `1000.0`) や許容値 (`VOL_TOLERANCE = 0.20`) は公式ノートブックと同一です。今後仕様変更があった場合は、このセクションも一緒に更新してください。

### 7.2 返却情報の内訳

Sharpe ロジックを関数化するときは、下記のような補助情報を辞書で返すとデバッグしやすくなります。

```python
return {
    "final_score": final_score,
    "raw_sharpe": raw_sharpe,
    "strategy_mean": mean_strategy,
    "strategy_std": std_strategy,
    "market_mean": mean_market,
    "market_std": market_vol,
    "vol_ratio": vol_ratio,
    "vol_penalty": vol_penalty,
    "return_penalty": return_penalty,
}
```

CLI 側ではこの辞書を JSON/CSV で保存し、fold ごとの内訳を把握できるようにします。

### 7.3 walk-forward スプリッタ実装例

Copilot での具体的な雛形:

```python
def make_walk_forward_splits(
    df: pd.DataFrame,
    train_window: int,
    val_window: int,
    step: int,
    date_col: str = "date_id",
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    indices = np.arange(len(df))
    splits = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_window
        val_end = train_end + val_window
        if val_end > len(df):
            break
        train_idx = indices[train_start:train_end]
        val_idx = indices[train_end:val_end]
        metadata = {
            "train_range": (int(df[date_col].iloc[train_start]), int(df[date_col].iloc[train_end - 1])),
            "val_range": (int(df[date_col].iloc[train_end]), int(df[date_col].iloc[val_end - 1])),
        }
        splits.append((train_idx, val_idx, metadata))
        start += step
    return splits
```

これを `src/models/common/walk_forward.py` に置き、`train_lgbm.py` などから `--cv-mode walk_forward` で呼び出す構成を想定しています。
