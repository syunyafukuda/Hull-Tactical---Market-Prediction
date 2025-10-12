# 前処理・特徴量生成ガイド

最終更新: 2025-10-12

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

## 推奨ポリシー単位

| ポリシー単位 | 対象列群 | 前処理手法 | 理由・補足 |
| --- | --- | --- | --- |
| **P1: 数値欠損補完（連続特徴）** | M*, E*, I*, P*, V*, MOM* | 1) `groupby(date_id).ffill().bfill()` → 2) 残存 NaN を列中央値補完 | 時系列方向の自然継続性を保持。初期期間の欠損多発に対応。 |
| **P2: ダミー / Binary 補完** | D* | 欠損を 0 で埋める | D 列は 0/1 系、欠損は 0 扱いが自然。 |
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

## ブランチ運用例

| ブランチ例 | 内容 | 検証方法 |
| --- | --- | --- |
| `prep/numeric-impute` | P1 パターン比較（ffill.bfill vs mean vs median） | CV で Sharpe 派生評価 |
| `prep/outlier-clip` | 分位クリップ vs log 変換 | CV 比較 |
| `prep/scale-robust` | Robust vs StandardScaler | CV 比較 |
| `prep/dropna-cols` | 高欠損列削除閾値 0.9 vs 0.8 | CV 比較 |

各ブランチで候補方針を比較し、勝者のみ PR で採用する。

## グループ別前処理ポリシー（推奨）

単一の手法を全列に適用すると歪みを招くため、列プレフィクスごとに前処理方針を分ける。以下は現時点の推奨案。

| グループ | 欠損補完 | スケーリング | 外れ値処理 | 備考 |
| --- | --- | --- | --- | --- |
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

- `ffill_bfill`: 前方→後方補完（fold 境界でリセット）
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

- `kalman_local_level`: ローカルレベル・カルマン平滑
- `arima_auto`: ARIMA で予測補完（初期区間に難）
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
