# 前処理・特徴量生成ガイド

最終更新: 2025-10-26

## 特徴量エンジニアリングの標準プロセス

| フェーズ | 内容 | 目的 | 出力 |
| --- | --- | --- | --- |
| **1. データ理解 (EDA)** | 分布・欠損・相関の把握 | 特徴設計の方向を決定 | 統計概要・可視化ノート |
| **2. 前処理 (Preprocessing)** | 欠損補完・スケーリング・符号化 | 学習に適した形式へ統一 | 正規化済みデータ |
| **3. 特徴量生成 (Feature Generation)** | ドメイン知識・時系列処理・組合せ・統計量作成 | モデルの表現力を拡張 | 生成特徴セット |
| **4. 特徴量削減 (Feature Selection)** | 冗長・ノイズ特徴の除去 | 安定性と汎化向上 | 精選特徴セット |
| **5. モデル適合・評価** | CVとLBで性能評価 | 仮説検証・改善判断 | スコア・重要度 |
| **6. メンテナンス** | 変数追跡・コード整備 | 再現性・追試性確保 | ログ・ドキュメント |

## ポリシー単位でのブランチ運用

- 「1つの前処理 = 1ブランチ = 1PR」の原則で進める。
- 粒度は列単位ではなく**ポリシー単位**（同じ性質を持つ列群へ共通適用する方針）とする。
  - 例: 数値欠損の既定方針を「時系列分割対応 Median」に統一。
  - 例: 時系列の ffill/bfill を「銘柄単位＋CV境界リセット」に統一。
  - 例: カテゴリ欠損を「専用トークン + OneHot」に統一。
  - 例: 外れ値処理を「分位クリップ p1–p99」に統一。
- ブランチ内では同一ポリシー内の候補を比較し、勝者のみを PR に反映する。
- PR は最終採用案のみ設定を反映し、その他候補は結果 CSV だけ残す。

## 実装ルール（リーク対策と再現性）

- Split-aware: `fit(train)` で統計量を算出し、`transform(valid/test)` へ適用する。
- ffill/bfill はグループ単位（例: 銘柄）で実施し、CV 境界でリセットする。
- 乱数シード固定、NaN を増やさず形状不変。
- `sklearn` の `Pipeline` や `ColumnTransformer` にまとめる。
- CV 境界リセットを扱う自作 `TimeAwareImputer` を活用する。

```yaml
preprocess:
  numeric_missing:
    policy: "group_median"   # ["ffill_bfill","mean","median","group_median"]
    group_keys: ["ticker"]
  cat_missing:
    policy: "missing_token"  # ["missing_token"]
  outlier:
    policy: "quantile_clip"  # ["none","quantile_clip"]
    q_low: 0.01
    q_high: 0.99
```

## 評価と採用基準

- 指標: コンペ公式 Sharpe 派生。
- 採用条件: 平均スコアが向上 **または** 分散が低下し、かつ各 Fold 非悪化率 ≥ 60%。
- LB チェックは最終候補のみ実施。
- PR 要件（前処理用）:
  - 変更ファイルを `src/preprocess/<policy>.py`、`configs/preprocess.yaml`、`tests/preprocess/test_<policy>.py` の 3 点に限定。
  - 本文には目的、リーク対策、再現コマンド、アブレーション表、影響列の説明を含める。

## 最新 P 系欠損補完方針 (2025-10-21)

- **採用ポリシー**: `mice` を既定設定として `configs/preprocess.yaml` の `p_group` に反映。`results/ablation/P_group/p_sweep_20251021_p_group_summary.csv` で最小 OOF RMSE 0.012060 を記録。
- **次点 (RMSE 観点)**: `ridge_stack` は OOF RMSE 0.012110 と僅差の次点で、同一カレンダー設定のまま切り替え可能。
- **Sharpe 系最優秀 (MSR 観点)**: `kalman_local_level` と `state_space_custom` が MSR 0.029623 で同値首位。msr ベースでの切り替え時は `statsmodels` 依存を確認。
- **概要表**:

| 観点 | 1位ポリシー | 指標値 | 2位ポリシー | 指標値 |
| --- | --- | --- | --- | --- |
| OOF RMSE | `mice` | 0.012060 | `ridge_stack` | 0.012110 |
| MSR | `kalman_local_level` | 0.029623 | `state_space_custom` | 0.029623 |

- **運用メモ**: いずれも `rolling_window=5`, `ema_alpha=0.3`, `calendar_col=date_id` を共有。`kalman_local_level` と `state_space_custom` は `statsmodels` の状態空間モジュールを使用するため、ランタイム環境での依存確認を推奨。

### P 系スイープで遭遇したエラーと対処

- **statsmodels の ImportError**: `kalman_local_level`, `arima_auto`, `state_space_custom` をローカル Python で直接実行した際に `ModuleNotFoundError: No module named 'statsmodels'` が発生。`uv pip install statsmodels` を行い、以降は `uv run` 経由でスクリプトを実行することで仮想環境の依存関係を統一し解消。
- **ランタイム不一致**: VS Code の別インタープリタから実行した場合に依然として `statsmodels` が見つからないため、すべてのスイープコマンドを `uv run python ...` に統一して再現性を確保。
- **missforest のリソース過多**: デフォルト設定では計算時間・メモリが肥大したため、`missforest_max_iter=3`, `missforest_estimators=100`, `missforest_max_depth=12` に調整して完走可能化。スイープ再実行時も `--p-policy-param` で同パラメータを付加する。

## 最新 M 系欠損補完方針 (2025-10-15)

- **採用ポリシー**: `ridge_stack` を既定設定として `configs/preprocess.yaml` に反映済み。`results/ablation/20251015-full_consolidated_m_group_summary.csv` で Sharpe 系指標 (MSR 0.027205) が全候補中トップ。
- **バックアップ**: `knn_k` は最小 RMSE (0.012175) を記録したため、本番環境で `ridge_stack` が外部要因で動作不可となった場合の切り替え候補とする。
- **運用メモ**: いずれも `m_policy_params` は空指定で再現可能。切り替え時は `configs/preprocess.yaml` の `m_group.policy` を差し替え、再学習ジョブを再実行する。
- **今後の改善余地**: `missforest` は設定調整により完走するが計算コストが高いため保留。必要に応じて推定木本数・反復回数をさらに最適化する。
- **サマリー全体** (`results/ablation/20251015-full_consolidated_m_group_summary.csv` 手動転記):

| policy | status | rolling_window | ema_alpha | calendar_col | pp_aggregate | n_splits | gap | min_val_size | optimize_for | m_column_count | m_post_impute_nan_ratio | oof_rmse | coverage | msr | msr_down | vmsr | duration_sec | error | m_imputer_warning_count | m_imputer_warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arima_auto | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124765670119824 | 0.8331479421579533 | 0.0266248968212061 | 0.0455748441795106 | 0.0266248968212061 | 1106.0900243160004 |  | 0.0 | [] |
| backfill_robust | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 24.98068392199912 |  | 0.0 | [] |
| dom_median | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.689574097999866 |  | 0.0 | [] |
| dow_median | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.60012001799987 |  | 0.0 | [] |
| ema_alpha | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 35.45198599299965 |  | 0.0 | [] |
| ffill_bfill | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.012232261965822 | 0.8331479421579533 | 0.0200279734726796 | 0.0311000924818628 | 0.0200279734726796 | 15.48830048200034 |  | 0.0 | [] |
| ffill_only | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.766584827999395 |  | 0.0 | [] |
| holiday_bridge | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.012232261965822 | 0.8331479421579533 | 0.0200279734726796 | 0.0311000924818628 | 0.0200279734726796 | 15.777719927999897 |  | 0.0 | [] |
| kalman_local_level | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124907837947078 | 0.8331479421579533 | 0.0206233505975375 | 0.0354610343953481 | 0.0206233505975375 | 59.00385652400018 |  | 0.0 | [] |
| knn_k | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.012174873938772783 | 0.8331479421579533 | 0.024661435789453574 | 0.03741689137941374 | 0.024661435789453574 | 24.650195952000104 |  | 0.0 | [] |
| linear_interp | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.9858271840003 |  | 0.0 | [] |
| mask_plus_mean | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0122624008542696 | 0.8331479421579533 | 0.0252216529925805 | 0.0435273875728548 | 0.0252216529925805 | 17.217486819999976 |  | 0.0 | [] |
| mice | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.012326114117501612 | 0.8331479421579533 | 0.02466924011267032 | 0.041387855751072004 | 0.02466924011267032 | 18.66563667599985 |  | 0.0 | [] |
| missforest | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.012304822991452514 | 0.8331479421579533 | 0.023356245256741764 | 0.04137318601687986 | 0.023356245256741764 |  |  | 0.0 | [] |
| month_median | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.826452643000266 |  | 0.0 | [] |
| pca_reconstruct_r | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0123450141764596 | 0.8331479421579533 | 0.023222922137266 | 0.0400836566148608 | 0.023222922137266 | 16.368858057999205 |  | 0.0 | [] |
| quantile_fill | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 44.047776032999536 |  | 0.0 | [] |
| ridge_stack | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0123208380219478 | 0.8331479421579533 | 0.0272054995931105 | 0.0424602066150114 | 0.0272054995931105 | 15.889731966999534 |  | 0.0 | [] |
| rolling_mean_k | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 28.66748592699969 |  | 0.0 | [] |
| rolling_median_k | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 25.501611500000763 |  | 0.0 | [] |
| spline_interp_deg | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.583311454000068 |  | 0.0 | [] |
| state_space_custom | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124907837947078 | 0.8331479421579533 | 0.0206233505975375 | 0.0354610343953481 | 0.0206233505975375 | 58.45718927800044 |  | 0.0 | [] |
| time_interp | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 15.758772857999247 |  | 0.0 | [] |
| two_stage | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 48.35515317699992 |  | 0.0 | [] |
| winsorized_median_k | ok | 5.0 | 0.3 | date_id | refit | 5.0 | 0.0 | 0.0 | msr | 18.0 | 0.0 | 0.0124491947720859 | 0.8331479421579533 | 0.0240726096760416 | 0.0421208166656981 | 0.0240726096760416 | 64.40469194700017 |  | 0.0 | [] |

## 推奨ポリシー単位

| ポリシー単位 | 対象列群 | 前処理手法 | 理由・補足 |
| --- | --- | --- | --- |
| **P1: 数値欠損補完（連続特徴）** | M*, E*, I*, P*, V*, MOM* | 1) `groupby(date_id).ffill().bfill()` → 2) 残存 NaN を列中央値補完 | 時系列方向の自然継続性を保持。初期期間の欠損多発に対応。 |
| **P2: ダミー / Binary 監視** | D* | 補完不要（欠損無し） | 欠損が発生しないため監視ログのみで十分。 |
| **P3: 外れ値処理** | M*, P*, V*, MOM* | 分位クリップ（1–99%） | 初期データや異常ボラ期間の極値を緩和。 |
| **P4: スケーリング** | 全連続列 | RobustScaler（中央値・IQR 基準） | 分布歪みが大きく、標準化より堅牢。 |
| **P5: 定常化処理（検討）** | MOM*, P* | 一次差分またはローリング 60 日 z-score | モメンタム系のトレンド除去に有効。後段実験用。 |
| **P6: カテゴリ符号化** | 国・業種など（存在する場合） | OneHot または Target Encoding | 経済カテゴリがある場合に限定。 |
| **P7: 欠損率閾値削除** | 全列 | 欠損率 > 0.9 を削除 | 初期期間に極端な欠損を持つ列への対処。 |

## 実行順序

1. 欠損補完（P1, P2）
2. 外れ値処理（P3）
3. スケーリング（P4）
4. 欠損率削除（P7）
5. 定常化または特殊変換（P5）
6. 符号化（P6）

## M 系欠損補完の運用フロー

- **ステップ1: スイープ実行**
  - コマンド例: `python src/preprocess/M_group/sweep_m_policy.py --policies ffill_bfill linear_interp`。
  - `results/ablation/<timestamp>_m_group_*.json` とサマリ `..._summary.csv` が生成され、各ポリシーの OOF 指標（RMSE, MSR など）と適用パラメータが記録される。
- **ステップ2: 結果比較**
  - `summary.csv` を基にスコア・coverage・季節列有無を比較し、採用候補を決定する。
  - 追加パラメータは `param_*` 列で確認できるため、設定値の抜け漏れを防げる。
- **ステップ3: 設定反映**
  - 採用するポリシーとパラメータを `configs/preprocess.yaml` の `m_group` セクションへ反映し、後続処理に共有する。
  - カレンダー参照列（例: `date_id`）を利用する場合は `calendar_column` を必ず揃える。
- **ステップ4: 本番学習**
  - コマンド例: `python src/preprocess/M_group/train_pre_m.py --m-policy linear_interp --m-policy-param deg=2 --m-calendar-col date_id`。
  - モデル・メタ情報が `artifacts/Preprocessing_M/` に保存され、OOF のグリッド探索ログと fold ログも併せて出力される。
- **ステップ5: 推論・提出物生成**
  - コマンド例: `python src/preprocess/M_group/predict_pre_m.py --artifacts-dir artifacts/Preprocessing_M --data-dir data/raw`。
  - 学習時と同じ特徴構成で整形され、`submission.parquet` / `submission.csv` が再生成される。
- **ステップ6: 検証/共有**
  - `model_meta.json` に保存された `m_policy_params`, `m_calendar_col`, `m_mask_cols` を確認し、再学習や他メンバーとの共有時に参照する。
  - 必要に応じて `results/ablation` のログと併せて PR に貼り付け、意思決定の根拠とする。

## E 系欠損補完方針

- **採用ポリシー**: `ridge_stack` を E 系特徴量の既定設定として採用する。`results/ablation/E_group/e_sweep_20251017174140_e_group_summary.csv` にて OOF RMSE 0.012147・coverage 0.8331・MSR 0.0248 を記録し、全候補中最良の誤差指標を達成した。
- **次点候補**: `pca_reconstruct_r` (OOF RMSE 0.012153, MSR 0.0277) と `kalman_local_level` (OOF RMSE 0.012170, MSR 0.0302) は Sharpe 系指標が高く、主要切り替え候補として保持する。
- **missforest の軽量設定**: `missforest` はデフォルトだとリソース要求が高いが、`missforest_max_iter=3`, `missforest_estimators=100`, `missforest_max_depth=12` の調整で完走し (OOF RMSE 0.012417, runtime 約 475 秒)、マルチモデル比較時のベンチマークとして再現可能。
- **再現手順**: `uv run python src/preprocess/E_group/sweep_e_policy.py --suite full --resume --tag e_sweep_20251017174140 --skip-on-error` を基点に、上記パラメータを `--policy-param` で指定すると結果を再生成できる。
- **運用メモ**: Runner-up へ切り替える際は `configs/preprocess.yaml` の `e_group` セクションで `policy` を変更し、アブレーション結果 CSV を添付した PR を作成する。
- **Submit 実績**: `kaggle_preprocessing_e.ipynb` を Kaggle Notebook として提出し、Public LB スコア 0.625 を記録。
- **M+E LB 検証**: M 系単独 (`ridge_stack`) の直近 LB 0.629 に対し、M+E 併用 (`ridge_stack` + E 欠損補完) では 0.625 と僅差の後退にとどまる。Sharpe 系 OOF 指標と安定性が改善したため、E 系を残した統合パイプラインを正式採用とする。
- **成果物の整備**: `artifacts/Preprocessing_E/model_meta.json` と `artifacts/Preprocessing_E/model_simple.pkl` に M/E 両方のポリシー・カラム構成を同梱済み。推論・提出スクリプトは同アーティファクトを参照することで、E 系列活用時も追加設定なしに再現できる。

## 最新 I 系欠損補完方針 (2025-10-19)

- **採用ポリシー**: Public LB 検証の結果、`ridge_stack` を I 系特徴量の既定設定として確定した（`configs/preprocess.yaml` へ反映済み）。
- **LB 比較**: Kaggle Private Notebook（Dataset: preprocess-i-group-hull-tactical）で `ridge_stack` と `knn_k` がいずれも LB 0.623、`missforest` は 0.561。安定性と再現性の観点から `ridge_stack` を優先採用。
- **次点候補**: `knn_k` は LB 同率ながら補完時の距離計算コストが高いため、バックアップポリシーとして待機。切り替え時は `configs/preprocess.yaml` の `i_group.policy` を差し替え、再学習ジョブを再実行する。
- **missforest 所感**: メモリ節約のための軽量設定（max_iter=3, estimators=100, max_depth=12）で完走するが、今回の LB では Sharpe が伸びず次点以下。必要に応じてさらなる特徴エンジニアリング併用で再検証する。
- **成果物**: `artifacts/Preprocessing_I/` に更新済みの `model_meta.json` と Ridge Stack ベースの学習パイプラインを保管し、推論 Notebook (`kaggle_preprocessing_i_ridge_stack.ipynb`) から直接再学習・提出可能な構成とした。

## 最新 S 系欠損補完方針 (2025-10-26)

- **LB 検証結果** (Kaggle Private Notebooks, Dataset: preprocess-s-group-hull-tactical)
  - `kaggle_s_missforest.ipynb` — LB 0.616
  - `kaggle_s_kalman_local_level.ipynb` — LB 0.616
  - `kaggle_s_holiday_bridge.ipynb` — LB 0.557
- **意思決定**: `missforest` を S 系特徴量の既定ポリシーとして採択。`kalman_local_level` は同率ながらメタ安定性が missforest より劣後、`holiday_bridge` は Sharpe が低下したためアーカイブ扱いとする。
- **互換性対応**: 3 ポリシーすべてを `numpy==1.26.4`, `scikit-learn==1.7.2` 環境で再学習し、`model_pre_s.pkl` の MT19937 互換性を確認済み。Kaggle Notebook では `sys.path.append('src')` ののち `joblib.load` を実行することでロードエラーを解消。
- **Artifacts**: `artifacts/Preprocessing_S/missforest/` を最新とし、`kalman_local_level` および `holiday_bridge` は比較用に保持（`submission.csv` へ採択ポリシーの注記を付与）。
- **分析メモ**:
  - Public LB は直近ベースライン（0.625）から 0.616 に微後退したが、OOF RMSE / MSR は悪化せず fold 別の安定性も維持。Public のばらつき要因と推定し、S 系は維持したまま後続で列削減・post-process 最適化を進める方針。
  - 特徴重要度の上位は S5, S7, S10 など holiday gap 系指標が多く、S3/S12 などミスの大きい列は削減候補。次フェーズで `feature_list.json` を基に Sharpe 貢献度を再評価する。

## グループ別前処理ポリシー（推奨）

単一の手法を全列に適用すると歪みを招くため、列プレフィクスごとに前処理方針を分ける。以下は現時点の推奨案。D 系特徴量（プレフィクス `D`）は train/test ともに欠損が確認されておらず、補完処理は不要。

| グループ | 欠損補完 | スケーリング | 外れ値処理 | 備考 |
| --- | --- | --- | --- | --- |
| D* | 不要（欠損無し） | 不要 | 不要 | バイナリ列で欠損は発生せず、監視ログのみ実施。 |
| M* | ffill → bfill | StandardScaler | 分位クリップ (1–99%) | 時系列継続性重視、欠損は少なめ。 |
| E* | group median（低頻度更新） | RobustScaler | 欠損率高 → 一部列は削除対象 | 経済統計の更新遅延に対応。 |
| I* | ffill → bfill | StandardScaler | 分位クリップ (0.1–99.9%) | 小振幅・負値あり。 |
| P* | median | RobustScaler | MAD×4 ウィンザー化 | 極端値が出やすいバリュエーション指標。 |
| V* | median | `log1p` 後 StandardScaler | 分位クリップ (1–99%) | 非負・歪な分布。 |
| MOM* | ffill → bfill | ローリング 60 日 z-score | 不要（差分で吸収） | トレンド除去が目的。 |

## 推奨優先度

1. **P1 数値欠損補完**（最重要）: 欠損構造が複雑なため、安定補完を先確定。
2. **P3 外れ値クリップ**: ノイズ抑制と安定学習に寄与。
3. **P4 スケーリング**: 特徴群ごとの差を吸収。
4. **P7 欠損率削除**: 情報量の乏しい列を整理。

上記で「データ整形フェーズ」を完了させ、次に特徴量生成 → 削減（相関・VIF）→ 学習チューニングへ進む。

## TODO: 欠損補完ポリシー候補

下記の補完ポリシーを順次検証する。各カテゴリで代表案を選定し、`prep/<policy>` ブランチで CV を走らせて採用判断する。

### A. シンプル時系列系

- `ffill_bfill`（別名: `ffill_train_bfill_in_fit`）: 学習時は前方→後方でウォームスタート、推論時は前方方向のみ（fold境界でリセット）
- `ffill_only`: 後方へは広げず初期 NaN は残す（後続処理で対応）
- `rolling_median_k`: 過去 k 日の中央値で補完
- `rolling_mean_k`: 過去 k 日の平均で補完（外れ値に弱い）
- `ema_alpha`: 過去の指数加重平均で補完

### B. 補間系

- `linear_interp`: 欠損区間を直線補間（端点は別処理）
- `spline_interp_deg`: スプライン補間（過学習リスクあり）
- `time_interp`: 営業日間隔を考慮した時間基準補間

### C. ロバスト統計系

- `backfill_robust`: 直後値で埋め、無ければ `rolling_median`
- `winsorized_median_k`: 過去 k 日のウィンザー処理後中央値
- `quantile_fill(q)`: 過去 k 日の分位点 q で補完

### D. 季節性・カレンダー系

- `dow_median`: 同曜日の過去中央値
- `dom_median`: 同日（1–31）の過去中央値
- `month_median`: 同月の過去中央値
- `holiday_bridge`: 連休前後のギャップを直前ロバスト平均で補完

### E. 多変量・学習系

- `knn_k`: 同日の他 M* から KNN 補完（標準化必須）
- `pca_reconstruct_r`: PCA の低次元再構成で欠損埋め
- `mice`: MICE（多重代入）で反復補完（計算コスト大）
- `missforest`: ランダムフォレスト補完（外れ値に強い）
- `ridge_stack`: 他 M* からのリッジ回帰で補完

### F. 時系列モデル系

- `kalman_local_level`: ローカルレベル・カルマンフィルタ（statsmodels の fittedvalues = filter 出力のみ使用）
- `arima_auto`: ARIMA で予測補完（statsmodels の fittedvalues = filter 出力のみ使用）
- `state_space_custom`: 局所線形トレンドモデル

### G. マスク活用・二段構え

- `mask_plus_mean`: 欠損フラグ列追加 + mean/median 補完
- `two_stage`: まず `linear_interp`、残存を `rolling_median`

### H. 制約・境界系

- `clip_then_fill`: 先に分位クリップ → ロバスト補完
- `bounded_zero`: 非負指標は 0 下限へ投影後補完

### I. ベースライン / 制御

- `drop_if_missing_ratio_gt_t`: 欠損率閾値超の列を除外
- `leave_na`: 補完せず（ツリーモデル限定の検証用）
