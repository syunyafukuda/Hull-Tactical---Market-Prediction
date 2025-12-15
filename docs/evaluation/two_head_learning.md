# Two-Head Learning Strategy (forward_returns + risk_free_rate)

最終更新: 2025-12-14  
作成者: Codex CLI  
ステータス: 実装準備完了

GitHub Copilot 実行者向けに、外部 URL を参照せずに 2 ヘッド学習（`forward_returns` と `risk_free_rate` の同時予測）を実装できるよう、本ドキュメントでディスカッション要点とタスクをすべて記載します。

---

## 1. 背景（公開情報からの要約）

### 1.1 Kaggle Discussion: Public leaderboard: 17 is possible (ID: 608349)

> ```
> solution = train[-180:].copy()
>
> market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
> market_excess_cumulative = (1 + market_excess_returns).prod()
> market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
>
> c = (1 + market_mean_excess_return) ** (1 / (market_excess_returns > 0).mean()) - 1
> submission = pd.DataFrame({'prediction': (c / market_excess_returns).clip(0, 2)})
> score(solution, submission, '')
> ```
>
> コメントでは、**未知のケースでは `positions = ((x - risk_free_rates) / (forward_returns - risk_free_rates)).clamp(0.0, 2.0)` の 1 次元最適化に帰着できる**という指摘があり、「`forward_returns` と `risk_free_rate` を別々に予測してからポジションを計算する」アプローチが示唆されています。

### 1.2 Kaggle Discussion: The REAL "do-nothing" baseline is 0.469 (ID: 611071)

> 目的変数 `pred_excess` を `position = clip(beta + alpha * pred_excess)` で写像し、`alpha=0, beta=0.806` で Public LB ≈ 0.469。  
> **わずかでも予測精度があれば 0.5〜0.6+ が cutting-edge**。

これらから、「`forward_returns` と `risk_free_rate` を別個に予測し、公式評価式に近い形でポジションを算出する」戦略が有効と考えられます。

---

## 2. ゴール

1. LightGBM など既存パイプラインを拡張し、`forward_returns` と `risk_free_rate` をそれぞれ予測できるようにする（＝2ヘッド）。  
2. 予測結果から `positions = ((x - \hat{rf}) / (\hat{fr} - \hat{rf}))` を計算し、`[0, 2]` にクリップ。  
3. `x` の最適値（スカラー）を walk-forward / grid search で探索し、Sharpe 指標に沿って選択。  
4. OOF・テスト双方で新ポジションを生成し、`submission.csv` に反映。  
5. ローカル walk-forward Sharpe で 1 ヘッド（既存 α/β クリッピング）との比較評価を実施。

---

## 3. 実装タスク

| # | 項目 | 内容 |
|---|------|------|
| 1 | データロード拡張 | `src/metrics/lgbm/train_lgbm.py` などで `forward_returns` と `risk_free_rate` の列を確実に取得。`train_df` / `test_df` に存在しない場合は `market_forward_excess_returns`＋近似で補完。 |
| 2 | モデル構成 | 2 つのターゲットを同時に扱う仕組みを実装。<br>案A: LightGBM を 2 本（別々に学習）→ `forward_model`, `rf_model`。<br>案B: 1 つの multi-output モデル（CatBoost, NN）。 |
| 3 | 推論・OOF 出力 | `artifacts/models/.../` に `forward_predictions.csv` `riskfree_predictions.csv` を保存。OOF と test で同じ形式を出力。 |
| 4 | ポジション算出ロジック | 下記関数を `src/models/common/signals.py` に追加：<br>```python<br>def map_positions_from_forward_rf(forward_pred, rf_pred, x, clip_min=0.0, clip_max=2.0):<br>    denom = forward_pred - rf_pred<br>    raw = (x - rf_pred) / np.where(np.abs(denom) < 1e-8, np.sign(denom) * 1e-8, denom)<br>    return np.clip(raw, clip_min, clip_max)<br>``` |
| 5 | x の最適化 | `scripts/tune_position_mapping.py` を拡張 or 新規 `scripts/tune_two_head_positions.py` を作成。α/β の代わりに `x`（および optional λ for smoothing）をグリッドサーチし、walk-forward Sharpe で最良を記録。 |
| 6 | トレーニング CLI | `train_lgbm.py`（もしくは新ファイル `train_lgbm_two_head.py`）に `--two-head` オプションを追加し、2 モデルをまとめて学習・保存。`model_meta.json` に `two_head: true` 等のメタ情報を残す。 |
| 7 | submission 生成 | `predict_lgbm.py` に 2 ヘッド推論パスを追加。`submission.csv` の `prediction` 列は `map_positions_from_forward_rf` の結果を用いる。 |
| 8 | 評価 | walk-forward Sharpe 設定を流用し、1 ヘッド vs 2 ヘッドで `mean_sharpe`, `min_sharpe`, `vol_ratio`, `penalty` を比較。 |
| 9 | ドキュメント・ログ | 本ファイルと `docs/evaluation/optimized_settings.md` に結果を追記。チューニング結果は `results/two_head/` へ保存。 |

---

## 4. 参考イメージ

### 4.1 2 モデルを並列に学習する例（擬似コード）

```python
# forward_returns モデル
forward_model = clone(core_pipeline_template)
forward_model.fit(X_train, y_forward_train)
forward_pred = forward_model.predict(X_valid)

# risk_free_rate モデル
rf_model = clone(core_pipeline_template)
rf_model.fit(X_train, y_rf_train)
rf_pred = rf_model.predict(X_valid)
```

### 4.2 ポジション算出

```python
from src.models.common.signals import map_positions_from_forward_rf

positions = map_positions_from_forward_rf(
    forward_pred=forward_pred,
    rf_pred=rf_pred,
    x=0.0008,          # グリッドサーチで決まるスカラー
    clip_min=0.0,
    clip_max=2.0,
)
```

### 4.3 x の探索

```python
for x in np.linspace(-0.001, 0.001, 21):
    positions = map_positions_from_forward_rf(forward_oof, rf_oof, x)
    sharpe = hull_sharpe_score(positions, fwd_true, rf_true)
    log_result(x, sharpe)
```

---

## 5. リスクと検討事項

1. **分母ゼロ問題**  
   - `forward_pred ≈ rf_pred` の場合に分母が小さくなり暴れる。`np.where(np.abs(denom) < 1e-8, np.sign(denom)*1e-8, denom)` などでガードする。  
2. **x のスケール**  
   - `forward_returns` が 0.001 前後なので、`x` も同オーダー（例: -0.001〜0.001）でグリッド。  
3. **モデル相関**  
   - `forward` と `rf` を同じ特徴で学習すると高相関になりがち。必要なら特徴 subset や正則化を調整。  
4. **Sharpe vs RMSE のトレードオフ**  
   - 2 ヘッドにすると RMSE が悪化する恐れ。Sharpe 向上が見られるか Walk-Forward で検証し、差分をドキュメント化する。  

---

## 6. 完了条件

- [ ] `forward_returns` / `risk_free_rate` の 2 ターゲット学習・推論パイプラインを実装。  
- [ ] `map_positions_from_forward_rf` と `x` チューニング CLI を追加。  
- [ ] walk-forward Sharpe で 1 ヘッドより改善 or 同等の結果（少なくとも `mean_sharpe` が非劣、`min_sharpe` が悪化しない）。  
- [ ] 新戦略のドキュメントと結果（LB/Sharpe）を `docs/evaluation/optimized_settings.md` に追記。  
- [ ] `submission.csv` が 2 ヘッドポジションで生成されることを確認。  

この手順に沿って実装を進めれば、2 ヘッド学習の検証を効率的に行えます。

